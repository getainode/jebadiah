#!/usr/bin/env python3
"""Watch the Jev Decision Index board for grading changes and report where Jebadiah stands.

    python3 eval/decision-index/board_check.py              # diff against the stored baseline, print our rows
    python3 eval/decision-index/board_check.py --update     # same, then store today's copy as the new baseline
    python3 eval/decision-index/board_check.py --notify run:run_34e895c78127   # also send an Orca status on a change
    python3 eval/decision-index/board_check.py --self-test

Pulls the Space's data/index.json, data/v03.json and data/methodology.json plus the kit's
hub/0.3/manifest.json and README (Rules and Submitting sections). What counts as a grading change:
the edition, changelog, rules, complete-run gate, superseded policy, latency protocol, Full-score
weights and formula, the frozen-suite manifest, or the kit's Rules/Submitting text. Board movement
(our Full score, rank, tie-rank, latency, metadata) is reported separately.

Exit 0 nothing changed, 1 something changed, 2 a source could not be fetched. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

SPACE = "https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/"
KIT = "https://raw.githubusercontent.com/apolinario/decision-index/main/"
STATE = pathlib.Path(os.environ.get("JEVDI_STATE", "~/.local/state/jevdi-board")).expanduser()
OURS = "Jebadiah"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "jebadiah-board-check/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def readme_sections(text, names=("Rules", "Submitting a model to the leaderboard")):
    out = {}
    for name in names:
        m = re.search(r"^## " + re.escape(name) + r"\n(.*?)(?=^## |\Z)", text, re.S | re.M)
        out[name] = m.group(1).strip() if m else ""
    return out


def grading(sources):
    """The parts of the sources whose change means the grading or the submission rules moved."""
    m, v = sources["methodology"], sources["v03"]
    return {
        "edition": m.get("edition"),
        "changelog": m.get("changelog"),
        "rules": m.get("rules"),
        "latency_protocol": (m.get("latency") or {}).get("reproductions", {}).get("protocol"),
        "full_weights": v.get("weights"),
        "full_formula": v.get("formula"),
        "manifest": sources["manifest"],
        "kit_rules_and_submitting": readme_sections(sources["readme"]),
    }


def board(sources):
    """Our rows plus Jev, keyed by engine, with what the board shows for each."""
    meta = {m["engine"]: m for m in sources["index"]["models"]}
    rows = {}
    for m in sources["v03"]["models"]:
        if OURS in m.get("name", "") or m.get("jev"):
            i = meta.get(m["engine"], {})
            rows[m["engine"]] = {
                "name": m["name"], "full": m.get("v"), "rank": m.get("rank_v"), "tie_rank": m.get("rank_tie"),
                "public": m.get("pub"), "same_skills": m.get("same_skills"), "new_domains": m.get("new_domains"),
                "latency": i.get("latency"),
                "kind": (i.get("meta") or {}).get("kind"), "code_url": (i.get("meta") or {}).get("code_url"),
                "weights_repo": (i.get("meta") or {}).get("weights_repo"),
            }
    return {"generated_utc": sources["v03"].get("generated_utc"), "models": len(sources["v03"]["models"]), "rows": rows}


def diff(old, new, path=""):
    """Flat list of 'path: old -> new' lines for every leaf that differs."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new), key=str):
            out += diff(old.get(k), new.get(k), f"{path}.{k}" if path else str(k))
        return out
    if old == new:
        return []
    short = lambda x: (json.dumps(x, sort_keys=True)[:160] + ("..." if len(json.dumps(x)) > 160 else ""))
    return [f"{path}: {short(old)} -> {short(new)}"]


def load_sources():
    return {
        "index": json.loads(fetch(SPACE + "data/index.json")),
        "v03": json.loads(fetch(SPACE + "data/v03.json")),
        "methodology": json.loads(fetch(SPACE + "data/methodology.json")),
        "manifest": json.loads(fetch(KIT + "hub/0.3/manifest.json")),
        "readme": fetch(KIT + "README.md"),
    }


def report(cur, base):
    g = diff(base["grading"], cur["grading"]) if base else []
    b = diff(base["board"]["rows"], cur["board"]["rows"]) if base else []
    lines = [f"board generated {cur['board']['generated_utc']}, {cur['board']['models']} models; "
             f"baseline {base['board']['generated_utc'] if base else 'none (first run)'}"]
    lines.append("GRADING CHANGED:" if g else "grading: unchanged")
    lines += ["  " + x for x in g]
    for r in sorted(cur["board"]["rows"].values(), key=lambda r: r["rank"] or 999):
        lat = r["latency"] or {}
        lines.append(f"  #{r['rank']} (tie {r['tie_rank']}) {r['name']}: Full {r['full']} = public {r['public']}, "
                     f"same {r['same_skills']}, new {r['new_domains']}; latency median {lat.get('median')} "
                     f"mean {lat.get('mean')} ms; kind {r['kind']}, code_url {r['code_url']}")
    lines.append("our rows moved:" if b else "our rows: unchanged")
    lines += ["  " + x for x in b]
    return lines, bool(g or b)


def self_test():
    a = {"edition": {"id": "v0.3"}, "rules": {"hard_rules": ["x"]}}
    b = {"edition": {"id": "v0.3.3"}, "rules": {"hard_rules": ["x"]}}
    assert diff(a, a) == []
    assert diff(a, b) == ['edition.id: "v0.3" -> "v0.3.3"'], diff(a, b)
    s = readme_sections("## Rules\nno truncation\n## Engines\nx\n## Submitting a model to the leaderboard\nPR a line\n")
    assert s == {"Rules": "no truncation", "Submitting a model to the leaderboard": "PR a line"}, s
    print("board_check self-test: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--update", action="store_true", help="store this run as the new baseline")
    ap.add_argument("--notify", metavar="RUN", help="Orca target for a status message when anything changed")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    try:
        src = load_sources()
    except Exception as e:
        print(f"fetch failed: {e}")
        return 2
    cur = {"grading": grading(src), "board": board(src)}
    bp = STATE / "baseline.json"
    base = json.loads(bp.read_text()) if bp.exists() else None
    lines, changed = report(cur, base)
    print("\n".join(lines))
    if a.notify and changed:
        subj = "Jev Decision Index: grading changed" if base and diff(base["grading"], cur["grading"]) else "Jev Decision Index: our rows moved"
        subprocess.run(["orca", "orchestration", "send", "--to", a.notify, "--type", "status",
                        "--subject", subj, "--body", "\n".join(lines)[:6000]], check=False)
    if a.update or base is None:
        STATE.mkdir(parents=True, exist_ok=True)
        stamp = (cur["board"]["generated_utc"] or "unknown").replace(":", "")
        (STATE / f"sources-{stamp}.sha256").write_text(
            "".join(f"{hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()}  {k}\n" for k, v in src.items()))
        bp.write_text(json.dumps(cur, indent=1, sort_keys=True))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
