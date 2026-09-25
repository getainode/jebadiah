"""Compose the v3 pools from pools and parts that already exist, then split and lint them.

These three pools were made by plain concatenation during the sweep; this script writes the same
bytes (checked against data/manifests/pools.json with pool_manifest.py).

  data-v3-c20fh-alt  = data-v3-c20f/pool.jsonl + human-noul/human-noul-alt.jsonl
  data-v3-full       = data-v3-c20fh-alt/pool.jsonl + human/prolific-r1.jsonl
  data-v3-full-s     = data-v3-full/pool.jsonl + a second copy of every helpsteer2-train and
                       summeval-train record, id suffixed with "#dup2" (the score share v1 had)

Inputs: ROOT/data-v3-c20f (experiments/fable-targets/finalize_fable_targets.py) and
data/human-noul/human-noul-alt.jsonl (data/human-noul/convert_human_noul.py).

  python data/compose_pools.py ROOT          # ROOT defaults to $JEB_ROOT or /workspace/jeb
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE_SETS = ("helpsteer2-train", "summeval-train")


def cat(out_dir: str, parts: list[str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pool = os.path.join(out_dir, "pool.jsonl")
    with open(pool, "wb") as g:
        for p in parts:
            with open(p, "rb") as f:
                g.write(f.read())
    return pool


def dup_score(src: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    lines = open(src, encoding="utf-8").readlines()
    out = list(lines)
    for line in lines:
        r = json.loads(line)
        if r["set"] in SCORE_SETS:
            r["id"] += "#dup2"
            out.append(json.dumps(r, ensure_ascii=False) + "\n")
    pool = os.path.join(out_dir, "pool.jsonl")
    with open(pool, "w", encoding="utf-8") as g:
        g.writelines(out)
    return pool


def split_and_lint(d: str) -> None:
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "split_pool.py"), f"{d}/pool.jsonl", f"{d}/train.jsonl", f"{d}/calib.jsonl"],
                   check=True)
    subprocess.run([py, os.path.join(HERE, "lint_data.py"), "--train", f"{d}/pool.jsonl", f"{d}/train.jsonl",
                    f"{d}/calib.jsonl"], check=True)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("JEB_ROOT", "/workspace/jeb")
    j = lambda *p: os.path.join(root, *p)  # noqa: E731
    cat(j("data-v3-c20fh-alt"), [j("data-v3-c20f", "pool.jsonl"), os.path.join(HERE, "human-noul", "human-noul-alt.jsonl")])
    split_and_lint(j("data-v3-c20fh-alt"))
    cat(j("data-v3-full"), [j("data-v3-c20fh-alt", "pool.jsonl"), os.path.join(HERE, "human", "prolific-r1.jsonl")])
    split_and_lint(j("data-v3-full"))
    dup_score(j("data-v3-full", "pool.jsonl"), j("data-v3-full-s"))
    split_and_lint(j("data-v3-full-s"))
    subprocess.run([sys.executable, os.path.join(HERE, "pool_manifest.py"), "check", root,
                    "data-v3-c20fh-alt", "data-v3-full", "data-v3-full-s"], check=False)


if __name__ == "__main__":
    main()
