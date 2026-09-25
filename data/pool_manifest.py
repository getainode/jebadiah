"""Record or check the sha256 and composition of built training pools.

The pools themselves are not in this repository. Every pool we trained on is described in
data/manifests/pools.json: for pool.jsonl, train.jsonl and calib.jsonl, the sha256, the number of
records and questions, and the counts by set and by question type. A rebuilt pool can be checked
against it byte for byte.

  python data/pool_manifest.py write ROOT [DIR ...]   record ROOT/<DIR>/ (default every ROOT/data-*/)
  python data/pool_manifest.py check ROOT [DIR ...]   compare ROOT/<DIR>/ against the manifest

Exit 1 from check when any recorded file is present and differs.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifests", "pools.json")
FILES = ("pool.jsonl", "train.jsonl", "calib.jsonl")


def describe(path: str) -> dict:
    h = hashlib.sha256()
    sets, types = collections.Counter(), collections.Counter()
    records = questions = 0
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            r = json.loads(line)
            records += 1
            sets[r.get("set", "?")] += 1
            for q in (r.get("questions") or {}).values():
                questions += 1
                types[q.get("type", "?")] += 1
    return {"sha256": h.hexdigest(), "bytes": os.path.getsize(path), "records": records,
            "questions": questions, "by_set": dict(sorted(sets.items())), "by_type": dict(sorted(types.items()))}


def dirs(root: str, names: list[str]) -> list[str]:
    if names:
        return names
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "data-*")) if os.path.isdir(d))


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in ("write", "check"):
        print(__doc__)
        return 2
    mode, root, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    bad = 0
    for d in dirs(root, names):
        for fn in FILES:
            p = os.path.join(root, d, fn)
            if not os.path.exists(p):
                continue
            got = describe(p)
            if mode == "write":
                man.setdefault(d, {})[fn] = got
                print(f"recorded {d}/{fn} {got['sha256'][:16]} {got['records']} records {got['questions']} questions")
            else:
                want = (man.get(d) or {}).get(fn)
                if want is None:
                    print(f"?    {d}/{fn} not in the manifest")
                elif want["sha256"] == got["sha256"]:
                    print(f"ok   {d}/{fn} {got['sha256'][:16]}")
                else:
                    bad += 1
                    print(f"DIFF {d}/{fn} got {got['sha256'][:16]} want {want['sha256'][:16]}")
    if mode == "write":
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w") as f:
            json.dump(dict(sorted(man.items())), f, indent=1)
            f.write("\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
