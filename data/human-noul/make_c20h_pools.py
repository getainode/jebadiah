"""Assemble data-dose-c20h (c20 + human-noul) and data-dose-c20h-alt (c20 + human-noul-alt).

    python3 make_c20h_pools.py

For each: pool.jsonl = data-dose-c20/pool.jsonl (read only, copied line for line) + the human file,
then train/split_pool.py pool.jsonl train.jsonl calib.jsonl, then train/lint_data.py --train on all
three and on the human files. Prints composition by set and type and the step count (8 questions a
step, 1 epoch); writes it to pools-manifest.json here.
"""
import collections
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# where the data-* pool directories live (the sweep root on the training box, or any local directory)
PKG = os.environ.get("JEB_ROOT", "/workspace/jeb")
C20 = os.path.join(PKG, "data-dose-c20", "pool.jsonl")
TRAIN = os.path.join(HERE, "..")  # split_pool.py and lint_data.py
PY = sys.executable
HUMAN_SETS = {"human-condaqa", "human-contractnli", "human-sharc", "human-unfairtos"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def comp(path):
    t = collections.Counter()
    recs = 0
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs += 1
            for q in r["questions"].values():
                t[(r["set"], q["type"])] += 1
    return recs, t


def main():
    c20_sha = sha256_file(C20)
    out = {"c20_pool_sha256": c20_sha, "pools": {}}
    lint_all = []
    for name, human in (("data-dose-c20h", "human-noul.jsonl"), ("data-dose-c20h-alt", "human-noul-alt.jsonl")):
        d = os.path.join(PKG, name)
        os.makedirs(d, exist_ok=True)
        pool = os.path.join(d, "pool.jsonl")
        with open(pool, "w", encoding="utf-8") as f:
            for src in (C20, os.path.join(HERE, human)):
                for line in open(src, encoding="utf-8"):
                    if line.strip():
                        f.write(line if line.endswith("\n") else line + "\n")
        sp = subprocess.run([PY, os.path.join(TRAIN, "split_pool.py"), pool, os.path.join(d, "train.jsonl"),
                             os.path.join(d, "calib.jsonl")], capture_output=True, text=True, check=True)
        split = json.loads(sp.stdout)
        lint = subprocess.run([PY, os.path.join(TRAIN, "lint_data.py"), "--train", pool,
                               os.path.join(d, "train.jsonl"), os.path.join(d, "calib.jsonl")],
                              capture_output=True, text=True, cwd=TRAIN)
        lint_all.append(lint.returncode)
        recs, t = comp(pool)
        tot = sum(t.values())
        by_type = collections.Counter()
        for (_, ty), n in t.items():
            by_type[ty] += n
        noul_h = sum(n for (s, ty), n in t.items() if ty == "noul" and s in HUMAN_SETS)
        # family straddle check across the split
        sides = collections.defaultdict(set)
        for side in ("train", "calib"):
            for line in open(os.path.join(d, f"{side}.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                sides[f"{r['set']}:{r.get('family')}"].add(side)
        hum_split = collections.Counter()
        for side in ("train", "calib"):
            for line in open(os.path.join(d, f"{side}.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                if r["set"] in HUMAN_SETS:
                    hum_split[side] += len(r["questions"])
        tr_q = split["train"]["questions"]
        out["pools"][name] = {
            "human_file": human, "records": recs, "questions": tot,
            "by_set_type": {f"{s} {ty}": n for (s, ty), n in sorted(t.items())},
            "by_type": dict(by_type), "noul_share": round(by_type["noul"] / tot, 4),
            "noul_human_new": noul_h, "noul_other": by_type["noul"] - noul_h,
            "split": split, "human_questions_by_side": dict(hum_split),
            "families_on_both_sides": sum(1 for s in sides.values() if len(s) > 1),
            "steps_1_epoch_8_per_step": -(-tr_q // 8),
            "sha256": {f: sha256_file(os.path.join(d, f)) for f in ("pool.jsonl", "train.jsonl", "calib.jsonl")},
            "lint_exit": lint.returncode, "lint": lint.stdout.strip().splitlines(),
        }
        print(name, json.dumps({k: v for k, v in out["pools"][name].items() if k not in ("split", "sha256")}, indent=1))
    hl = subprocess.run([PY, os.path.join(TRAIN, "lint_data.py"), "--train"] +
                        [os.path.join(HERE, f) for f in ("human-condaqa.jsonl", "human-contractnli.jsonl", "human-sharc.jsonl",
                                                         "human-unfairtos.jsonl", "human-noul.jsonl", "human-noul-alt.jsonl")],
                        capture_output=True, text=True, cwd=TRAIN)
    out["human_files_lint_exit"] = hl.returncode
    out["human_files_lint"] = hl.stdout.strip().splitlines()
    print(hl.stdout)
    out["c20_pool_sha256_after"] = sha256_file(C20)
    json.dump(out, open(os.path.join(HERE, "pools-manifest.json"), "w"), indent=1)
    return 1 if any(lint_all) or hl.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
