"""The sweep's bookkeeping, no torch needed: materialise a run's training config from sweep.json
(resolving "best of" choices from finished runs), keep /workspace/jeb/sweep-status.json current,
and collect one results.json per run from the stage outputs.

  python sweep_lib.py list [--include-optional]         run names in sweep order
  python sweep_lib.py config <run>                      write runs/<run>/config.json (exit 3 = skip)
  python sweep_lib.py status <run> <state> [note]       update sweep-status.json
  python sweep_lib.py collect <run>                     write runs/<run>/results.json and refresh status
  python sweep_lib.py summary                           print a table of finished runs

Environment: JEB_ROOT (default /workspace/jeb) holds sweep.json, data/, runs/, sweep-status.json.
JEB_DATA_DIR (default $JEB_ROOT/data) is where train.jsonl and calib.jsonl are read from; a run may
override it with "data_dir" in sweep.json (relative names resolve under JEB_ROOT), which is how the
v1 runs train on data-v1 while the v0 runs and the eval sets under data/test stay as they are.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("JEB_ROOT", "/workspace/jeb")
SWEEP = os.path.join(ROOT, "sweep.json")
STATUS = os.path.join(ROOT, "sweep-status.json")
RUNS = os.path.join(ROOT, "runs")
DATA = os.environ.get("JEB_DATA_DIR") or os.path.join(ROOT, "data")

CHANGE_FIELDS = ("epochs", "rank", "lr", "score_targets", "temperature_target", "score_ordinal_adjacent")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_sweep() -> dict:
    return json.load(open(SWEEP))


def read_json(path):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def write_json(path, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def gpu_name() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                       text=True, timeout=20).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


def run_spec(sweep: dict, name: str) -> dict:
    for r in sweep["runs"]:
        if r["name"] == name:
            return r
    raise SystemExit(f"no run named {name} in sweep.json")


def results_of(name: str):
    return read_json(os.path.join(RUNS, name, "results.json"))


def headline_of(name: str):
    r = results_of(name)
    if r and r.get("stage") == "complete":
        return r.get("headline")
    return None


# ------------------------------------------------------------------ headline

def compute_headline(sweep: dict, eval_summary: dict) -> dict:
    """The sweep's one number per run plus the components it is made of (see sweep.json)."""
    spec = sweep["headline"]
    comps = {}
    for s in spec["zero_shot_sets"]:
        if s in eval_summary and eval_summary[s].get("accuracy") is not None:
            comps[s] = 100 * eval_summary[s]["accuracy"]
    pub = {k: 100 * v["accuracy"] for k, v in eval_summary.items()
           if k.startswith(spec["nimble_public_prefix"]) and v.get("accuracy") is not None}
    if pub:
        comps["nimble-public (macro over %d)" % len(pub)] = sum(pub.values()) / len(pub)
    headline = sum(comps.values()) / len(comps) if comps else None
    in_dist = {k: 100 * eval_summary[k]["accuracy"] for k in ("typed-decisions-test", "kev-decision-v7__test")
               if k in eval_summary and eval_summary[k].get("accuracy") is not None}
    score_sets = {k: {"ece": v.get("ece"), "ece_raw": v.get("ece_raw"), "score_mae": v.get("score_mae"),
                      "decision_score_jevals": v.get("decision_score_jevals")}
                  for k, v in eval_summary.items() if v.get("score_mae") is not None}
    return {"headline": round(headline, 3) if headline is not None else None, "components": {k: round(v, 3) for k, v in comps.items()},
            "nimble_public": {k: round(v, 3) for k, v in pub.items()}, "in_distribution": {k: round(v, 3) for k, v in in_dist.items()},
            "score_sets": score_sets}


# ------------------------------------------------------------------ config

def resolve_best_of(choice: dict, spec_default):
    """{"best_of": [runs], "field": f, "default": d}: the field's value in the finished run with the
    highest headline; the default when none has finished."""
    best, best_name = None, None
    for name in choice["best_of"]:
        h = headline_of(name)
        if h is not None and h.get("headline") is not None and (best is None or h["headline"] > best):
            best, best_name = h["headline"], name
    if best_name is None:
        return choice.get("default", spec_default), f"no finished run among {choice['best_of']}; default {choice.get('default', spec_default)}"
    val = results_of(best_name)["config_resolved"].get(choice["field"], choice.get("default", spec_default))
    return val, f"{choice['field']}={val} from {best_name} (headline {best})"


def resolve_best_change(spec: dict, sweep: dict) -> tuple[dict, str]:
    """The change fields of the candidate whose headline beats the anchor by the most. Returns
    ({}, reason) when nothing beats it (the run is skipped) or when the anchor never finished
    (then the best absolute candidate is taken, and the reason says so)."""
    anchor = headline_of(spec["against"])
    anchor_h = anchor["headline"] if anchor else None
    cands = []
    for name in spec["apply_best_change_of"]:
        h = headline_of(name)
        if h is not None and h.get("headline") is not None:
            cands.append((h["headline"], name))
    if not cands:
        return {}, "no finished 4B candidate: nothing to apply"
    cands.sort(reverse=True)
    top_h, top = cands[0]
    if anchor_h is not None and top_h <= anchor_h:
        return {}, f"no candidate beats {spec['against']} ({anchor_h}); best was {top} ({top_h})"
    cfg = results_of(top)["config_resolved"]
    change = {k: cfg[k] for k in CHANGE_FIELDS if k in cfg and cfg[k] != sweep["defaults"].get(k)}
    why = f"{top} (headline {top_h} vs anchor {anchor_h}): {change}" if anchor_h is not None else \
        f"{top} (headline {top_h}; anchor {spec['against']} never finished, best absolute taken): {change}"
    return change, why


def materialise(sweep: dict, name: str) -> tuple[dict, dict, str]:
    """Returns (training config for train_jebadiah.py, the resolved run fields, a note)."""
    d = dict(sweep["defaults"])
    spec = run_spec(sweep, name)
    notes = []
    fields = {k: v for k, v in spec.items() if k in d}
    if "apply_best_change_of" in spec:
        change, why = resolve_best_change(spec, sweep)
        notes.append(f"best 4B change: {why}")
        if not change:
            return {}, {}, "SKIP " + why
        fields.update(change)
    r = {**d, **fields}
    for k, v in list(r.items()):
        if isinstance(v, dict) and "best_of" in v:
            r[k], why = resolve_best_of(v, d.get(k))
            notes.append(f"{k}: {why}")
    base = sweep["bases"][spec["base"]]
    run_dir = os.path.join(RUNS, name)
    data_dir = spec.get("data_dir") or DATA
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)
    attn = "sdpa"
    if os.path.exists(os.path.join(ROOT, "ATTN")):
        attn = open(os.path.join(ROOT, "ATTN")).read().strip() or "sdpa"
    cfg = {
        "run_name": name,
        "description": spec.get("purpose", ""),
        "base_model": base["model"],
        "base_revision": base["revision"],
        "data_dir": data_dir,
        "dataset_path": os.path.join(data_dir, "train.jsonl"),
        "eval_dataset_path": os.path.join(data_dir, "calib.jsonl"),
        "output_dir": run_dir,
        "method": "lora",
        "num_epochs": r["epochs"],
        "batch_size": r["batch_size"],
        "gradient_accumulation_steps": r["grad_accum"],
        "learning_rate": r["lr"],
        "lr_scheduler_type": r["lr_scheduler_type"],
        "lora_rank": r["rank"],
        "lora_alpha": int(r["rank"] * r["alpha_ratio"]),
        "max_seq_length": r["max_seq_length"],
        "warmup_steps": r["warmup_steps"],
        "weight_decay": r["weight_decay"],
        "max_grad_norm": r["max_grad_norm"],
        "use_gradient_checkpointing": r["gradient_checkpointing"],
        "attn_implementation": attn,
        "eval_steps": r["eval_steps"],
        "logging_steps": 10,
        "seed": r["seed"],
        "decide": {
            "objective": "candidate_ce",
            "shuffle_choice_options": r["shuffle_choice_options"],
            "lora_dropout": r["lora_dropout"],
            "calib_eval_limit": r["calib_eval_limit"],
            "target_modules": r["target_modules"],
            "score_targets": r["score_targets"],
            "score_ordinal_adjacent": r["score_ordinal_adjacent"],
        },
        "prompt_source_sha256": "d2660ebec28bd3f1704235bda88d24a397c1c62475e740519cb8ef2d08f25fdd",
        "prompt_source_commit": "e5c089386e0239c9eb270eeb490d181722b8da5b",
        "sweep": {"letter": spec.get("letter"), "smoke": bool(spec.get("smoke")), "eval_only": bool(spec.get("eval_only")),
                  "max_steps": spec.get("max_steps", -1), "eval_limit": spec.get("eval_limit", 0),
                  "eval_repeats": r["eval_repeats"], "eval_repeats_for": r["eval_repeats_for"],
                  "eval_batch_size": r["eval_batch_size"], "eval_latency_sample": r["eval_latency_sample"],
                  "temperature_target": r["temperature_target"], "notes": notes},
    }
    resolved = {k: r[k] for k in ("epochs", "rank", "lr", "score_targets", "temperature_target", "score_ordinal_adjacent",
                                  "batch_size", "grad_accum", "eval_repeats", "eval_repeats_for")}
    resolved["base"] = spec["base"]
    resolved["data_dir"] = data_dir
    return cfg, resolved, "; ".join(notes)


# ------------------------------------------------------------------ status

def load_status(sweep: dict) -> dict:
    st = read_json(STATUS) or {}
    if "runs" not in st:
        st = {"created": now(), "gpu": gpu_name(), "runs": []}
    known = {r["name"] for r in st["runs"]}
    for spec in sweep["runs"]:
        if spec["name"] not in known:
            st["runs"].append({"name": spec["name"], "letter": spec.get("letter"), "status": "pending",
                               "optional": bool(spec.get("optional")), "est_min_h100_pcie": spec.get("est_min_h100_pcie"),
                               "est_min_a100": spec.get("est_min_a100")})
    return st


def set_status(name: str, state: str, note: str = "") -> None:
    sweep = load_sweep()
    st = load_status(sweep)
    for r in st["runs"]:
        if r["name"] == name:
            r["status"] = state
            if state == "running" and not r.get("started"):
                r["started"] = now()
                r["started_epoch"] = time.time()
            if state in ("done", "failed", "skipped"):
                r["finished"] = now()
                if r.get("started_epoch"):
                    r["minutes"] = round((time.time() - r["started_epoch"]) / 60, 1)
            if note:
                r["note"] = note
            res = results_of(name)
            if res:
                r["stage"] = res.get("stage")
                r["headline"] = (res.get("headline") or {}).get("headline")
                r["minutes_by_stage"] = res.get("minutes")
    st["updated"] = now()
    st["gpu"] = st.get("gpu") or gpu_name()
    if os.path.exists(os.path.join(ROOT, "ATTN")):
        st["attn"] = open(os.path.join(ROOT, "ATTN")).read().strip()
    st["done"] = [r["name"] for r in st["runs"] if r["status"] == "done"]
    st["failed"] = [r["name"] for r in st["runs"] if r["status"] == "failed"]
    st["pending"] = [r["name"] for r in st["runs"] if r["status"] == "pending"]
    write_json(STATUS, st)


# ------------------------------------------------------------------ collect

def collect(name: str) -> dict:
    """Assemble runs/<run>/results.json from whatever stages have finished. Safe to call after
    every stage: the `stage` field says how far the run got."""
    sweep = load_sweep()
    run_dir = os.path.join(RUNS, name)
    cfg = read_json(os.path.join(run_dir, "config.json")) or {}
    stages = read_json(os.path.join(run_dir, "stages.json")) or {}
    res = {"run": name, "letter": (cfg.get("sweep") or {}).get("letter"), "purpose": cfg.get("description"),
           "gpu": gpu_name(), "attn": cfg.get("attn_implementation"), "collected": now(),
           "config_resolved": (read_json(os.path.join(run_dir, "resolved.json")) or {}),
           "config": {k: v for k, v in cfg.items() if k != "sweep"}, "sweep": cfg.get("sweep"),
           "stages": stages, "minutes": {}}
    for st, v in stages.items():
        if isinstance(v, dict) and v.get("seconds") is not None:
            res["minutes"][st] = round(v["seconds"] / 60, 1)
    if res["minutes"]:
        res["minutes"]["total"] = round(sum(res["minutes"].values()), 1)
    ts = read_json(os.path.join(run_dir, "train_summary.json"))
    if ts:
        res["train"] = {k: v for k, v in ts.items() if k != "config"}
    temps = read_json(os.path.join(run_dir, "adapter", "temperatures.json"))
    if temps:
        res["temperatures"] = temps
    ev = read_json(os.path.join(run_dir, "eval-adapter", "summary.json")) or read_json(os.path.join(run_dir, "eval-base", "summary.json"))
    if ev:
        res["eval"] = ev
        res["headline"] = compute_headline(sweep, ev)
        # per-set ECE / flips from the postprocessed records when present
        extra = {}
        for path in glob.glob(os.path.join(run_dir, "eval-*", "*.json")):
            if path.endswith("summary.json"):
                continue
            rec = read_json(path)
            if rec and "decide" in rec:
                o = rec["decide"]["overall"]
                extra[os.path.basename(path)[:-5]] = {k: o.get(k) for k in ("majority_floor_per_question", "decision_score_acc_per_question", "flip_max_top2_gap")}
                extra[os.path.basename(path)[:-5]]["flips"] = len(o.get("flips", []))
        if extra:
            res["postprocess"] = extra
    failed = [s for s, v in stages.items() if isinstance(v, dict) and v.get("status") == "failed"]
    order = ["train", "temps", "eval", "postprocess"]
    done = [s for s in order if isinstance(stages.get(s), dict) and stages[s].get("status") == "ok"]
    eval_only = bool((cfg.get("sweep") or {}).get("eval_only"))
    required = ["eval", "postprocess"] if eval_only else order
    if failed:
        res["stage"] = "failed:" + ",".join(failed)
    elif all(s in done for s in required):
        res["stage"] = "complete"
    else:
        res["stage"] = "partial:" + ",".join(done) if done else "started"
    write_json(os.path.join(run_dir, "results.json"), res)
    return res


def summary_table() -> str:
    sweep = load_sweep()
    lines = [f"{'run':14s} {'status':10s} {'min':>6s} {'headline':>9s}  components"]
    st = load_status(sweep)
    for r in st["runs"]:
        res = results_of(r["name"]) or {}
        h = res.get("headline") or {}
        comps = " ".join(f"{k.replace('jevals-', 'jev-').replace('__test', '').replace('nimble-public', 'np').split(' ')[0][:14]}={v:.1f}"
                         for k, v in (h.get("components") or {}).items())
        lines.append(f"{r['name']:14s} {r['status']:10s} {str(r.get('minutes', '')):>6s} {str(h.get('headline', '')):>9s}  {comps}")
    return "\n".join(lines)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    sweep = load_sweep()
    if cmd == "list":
        inc = "--include-optional" in rest
        print("\n".join(r["name"] for r in sweep["runs"] if inc or not r.get("optional")))
        return 0
    if cmd == "config":
        name = rest[0]
        run_dir = os.path.join(RUNS, name)
        os.makedirs(run_dir, exist_ok=True)
        marker = os.path.join(os.path.dirname(RUNS), "SKIP-" + name)
        if os.path.exists(marker):
            note = "SKIP: trimmed for budget (marker " + marker + ")"
            print(note)
            write_json(os.path.join(run_dir, "skipped.json"), {"reason": note, "at": now()})
            return 3
        cfg, resolved, note = materialise(sweep, name)
        if note.startswith("SKIP"):
            print(note)
            write_json(os.path.join(run_dir, "skipped.json"), {"reason": note, "at": now()})
            return 3
        write_json(os.path.join(run_dir, "config.json"), cfg)
        write_json(os.path.join(run_dir, "resolved.json"), resolved)
        print(json.dumps({"run": name, "resolved": resolved, "attn": cfg["attn_implementation"], "note": note}))
        return 0
    if cmd == "status":
        set_status(rest[0], rest[1], " ".join(rest[2:]))
        return 0
    if cmd == "collect":
        res = collect(rest[0])
        set_status(rest[0], {"complete": "done"}.get(res["stage"], "running") if not res["stage"].startswith("failed") else "failed")
        print(json.dumps({"run": rest[0], "stage": res["stage"], "headline": (res.get("headline") or {}).get("headline"),
                          "minutes": res.get("minutes")}))
        return 0
    if cmd == "summary":
        print(summary_table())
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
