"""Split the training pool into train (95 percent) and calib (5 percent) by a stable hash.

The hash is over the record's family when the source gives one (Nimble's contrastive pair,
Kev's pair or group id), else over the canonical JSON of the state and questions, so
sibling records that differ by one fact land on the same side and the calibration slice is
not a near-copy of the training set. The calib slice fits temperatures; it never trains.
"""
import collections
import hashlib
import json
import sys

CALIB_PERCENT = 5


def key_of(r: dict) -> str:
    fam = r.get("family")
    if fam:
        return f"{r['set']}:{fam}"
    return json.dumps({"state": r["state"], "questions": r["questions"]}, sort_keys=True, ensure_ascii=False)


def main(pool_path, train_path, calib_path):
    train, calib = [], []
    for line in open(pool_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        h = hashlib.sha256(key_of(r).encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % 100
        (calib if bucket < CALIB_PERCENT else train).append(r)
    for path, rows in ((train_path, train), (calib_path, calib)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    def stats(rows):
        t = collections.Counter(q["type"] for r in rows for q in r["questions"].values())
        s = collections.Counter(r["set"] for r in rows)
        return {"records": len(rows), "questions": sum(t.values()), "types": dict(t), "sets": dict(s)}
    out = {"train": stats(train), "calib": stats(calib), "rule": f"sha256(family or canonical record)[:4] mod 100 < {CALIB_PERCENT} -> calib"}
    print(json.dumps(out, indent=1))
    json.dump(out, open(calib_path.replace(".jsonl", "-split.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
