"""Build a "dose" pool: the v1 pool unchanged plus synthetic CHOICE questions only (no synthetic noul).

    python make_dose_pool.py --share 0.20 --out DIR [--dry-run]

Inputs (read only):
  pool-v2/v1-rebuild/data-v1/pool.jsonl        the rebuilt v1 pool (11,013 records, 15,813 questions)
  pool-v2/data-v2/synth-<family>.jsonl x24     the synth-v2 records (choice and noul mixed per state)

What it does:
  1. From every synth record, drop the noul questions (from questions, label, target and
     provenance.questions). Records left with no choice question are dropped.
  2. Target synthetic choice count S = share / (1 - share) x 15,813, so that synthetic choice is
     `share` of the final pool. S is split evenly over the 24 families (water-filled: a family with
     fewer choice questions than its share gives all of them and the remainder goes to the others).
  3. Inside a family, records are taken in a fixed pseudo-random order (sha256 of "dose:<id>"), whole
     records at a time (a state's questions stay together), until the family quota is reached. With
     --share at or above the available total, every synthetic choice question is taken.
  4. Writes DIR/pool.jsonl (v1 records first, then the synth records) and nothing else; split and lint
     are separate commands (split_pool.py, lint_data.py), as for data-v2.

--dry-run prints the composition (counts by set and type, per family) and the expected train/calib
split using split_pool.py's own hash rule, and writes nothing.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# the pool-v2 working directory (data/pool-v2 after generate.py and finalize.py ran there)
POOL_V2 = os.environ.get("JEB_POOL_V2", os.path.join(_HERE, "..", "pool-v2"))
V1_POOL = f"{POOL_V2}/v1-rebuild/data-v1/pool.jsonl"
SYNTH_GLOB = f"{POOL_V2}/data-v2/synth-*.jsonl"
sys.path.insert(0, os.path.join(_HERE, "..", "..", "train"))
sys.path.insert(0, os.path.join(_HERE, ".."))
from split_pool import CALIB_PERCENT, key_of  # noqa: E402  same hash rule as the real split


def choice_only(r: dict) -> dict | None:
    keep = [q for q, v in r["questions"].items() if v["type"] == "choice"]
    if not keep:
        return None
    out = dict(r)
    out["questions"] = {q: r["questions"][q] for q in keep}
    out["label"] = {q: r["label"][q] for q in keep}
    out["target"] = {q: r["target"][q] for q in keep}
    prov = dict(r.get("provenance") or {})
    if isinstance(prov.get("questions"), dict):
        prov["questions"] = {q: prov["questions"][q] for q in keep if q in prov["questions"]}
    out["provenance"] = prov
    return out


def order_key(r: dict) -> str:
    return hashlib.sha256(f"dose:{r['id']}".encode()).hexdigest()


def stats(rows):
    t = collections.Counter()
    for r in rows:
        for q in r["questions"].values():
            t[(r["set"], q["type"])] += 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, required=True, help="synthetic choice share of the final pool, e.g. 0.20")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    v1 = [json.loads(l) for l in open(V1_POOL, encoding="utf-8") if l.strip()]
    v1_q = sum(len(r["questions"]) for r in v1)
    target = round(a.share / (1 - a.share) * v1_q)

    fams = {}
    for path in sorted(glob.glob(SYNTH_GLOB)):
        fam = os.path.basename(path)[len("synth-"):-len(".jsonl")]
        recs = [c for c in (choice_only(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()) if c]
        fams[fam] = sorted(recs, key=order_key)
    avail = {f: sum(len(r["questions"]) for r in rs) for f, rs in fams.items()}
    available = sum(avail.values())
    # Even split with water-filling: a family smaller than its share gives its whole set and the
    # rest of its share goes to the others, so a share at or above `available` takes everything.
    quota, left = {}, float(target)
    for i, f in enumerate(sorted(fams, key=lambda f: avail[f])):
        quota[f] = min(avail[f], left / (len(fams) - i))
        left -= quota[f]

    picked, per_fam = [], {}
    for fam, recs in fams.items():
        n = 0
        for r in recs:
            if n >= quota[fam]:
                break
            picked.append(r)
            n += len(r["questions"])
        per_fam[fam] = (n, sum(len(r["questions"]) for r in recs))

    pool = v1 + picked
    s = stats(pool)
    tot = sum(s.values())
    syn = sum(n for (st, t), n in s.items() if st == "synth-v2")
    print(f"v1 questions {v1_q}; target synthetic choice {target} ({a.share:.0%}); available {available}")
    print(f"pool: {len(pool)} records, {tot} questions; synthetic choice {syn} = {syn / tot:.1%}")
    for (st, t), n in sorted(s.items()):
        print(f"  {st:24s} {t:7s} {n:6d}")
    by_type = collections.Counter()
    for (_, t), n in s.items():
        by_type[t] += n
    print("  by type:", dict(by_type), f"score share {by_type['score'] / tot:.1%}")
    print("  per family (taken / available):", ", ".join(f"{f} {x}/{y}" for f, (x, y) in per_fam.items()))

    tr = cal = 0
    trt = collections.Counter()
    for r in pool:
        b = int.from_bytes(hashlib.sha256(key_of(r).encode("utf-8")).digest()[:4], "big") % 100
        k = len(r["questions"])
        if b < CALIB_PERCENT:
            cal += k
        else:
            tr += k
            for q in r["questions"].values():
                trt[q["type"]] += 1
    print(f"split (split_pool rule): train {tr} questions ({dict(trt)}), calib {cal}; "
          f"optimizer steps at batch 8, 1 epoch: {-(-tr // 8)}")

    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pool.jsonl"), "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", os.path.join(a.out, "pool.jsonl"))


if __name__ == "__main__":
    main()
