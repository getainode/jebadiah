"""Post-processing over the saved eval records (no GPU):
  - majority_floor_per_question: the majority class computed per question id (the honest floor
    for a set that asks several different questions), and the Decision Score against it
  - flips: for every question whose pick changed between identical repeats, the top-2 probability
    gap in each repeat (what moved, and by how much)
Rewrites each record's decide.overall with the extra fields and prints a compact table.
"""
import collections
import glob
import json
import os
import sys


def per_question_floor(rows0):
    by_q = collections.defaultdict(collections.Counter)
    for r in rows0:
        by_q[r["qid"]][r["label"]] += 1
    return sum(c.most_common(1)[0][1] for c in by_q.values()) / len(rows0)


def flips(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["id"], r["qid"])].append(r)
    out = []
    for key, rs in by.items():
        same = [r for r in rs if r.get("order_seed", 0) == 0]
        if len({r["pick"] for r in same}) > 1:
            gaps = []
            for r in same:
                p = sorted(r["probs"], reverse=True)
                gaps.append(round(p[0] - p[1], 4))
            out.append({"id": key[0], "qid": key[1], "picks": [r["pick"] for r in same], "top2_gaps": gaps,
                        "label": same[0]["label"]})
    return out


def main(dirs):
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            if path.endswith("summary.json"):
                continue
            rec = json.load(open(path))
            rows = rec["decide"]["rows"]
            rows0 = [r for r in rows if r["repeat"] == 0]
            o = rec["decide"]["overall"]
            fq = per_question_floor(rows0)
            o["majority_floor_per_question"] = fq
            o["decision_score_acc_per_question"] = (o["accuracy"] - fq) / (1 - fq) if fq < 1 else None
            fl = flips(rows)
            o["flips"] = fl
            o["flip_max_top2_gap"] = max((max(f["top2_gaps"]) for f in fl), default=0.0)
            json.dump(rec, open(path, "w"), indent=1)
            name = os.path.basename(path)[:-5]
            def fmt(v, spec):
                return "n/a" if v is None else format(v, spec)
            print(f"{os.path.basename(d):14s} {name:32s} n={o['n']:5d} acc={fmt(o.get('accuracy'), '.3f')} floor={fmt(o.get('majority_floor'), '.3f')} "
                  f"floor_q={fmt(fq, '.3f')} DS={fmt(o.get('decision_score_acc'), '.3f')} DS_q={fmt(o.get('decision_score_acc_per_question'), '.3f')} "
                  f"DSj={fmt(o.get('decision_score_jevals'), '.1f')} ece={fmt(o.get('ece'), '.3f')} flips={len(fl)} max_gap={fmt(o.get('flip_max_top2_gap'), '.3f')}")


if __name__ == "__main__":
    main(sys.argv[1:])
