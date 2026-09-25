"""Overlap between a Jebadiah training pool and the 231 public JevBench items.

For each public item: its text = state + instructions + criteria texts, whitespace- and
case-normalized. Reported: (a) exact normalized-state matches against training states,
(b) items sharing any 13-word span with the training text, (c) items sharing more than three
8-word spans (the threshold JPT-4B's request used).

  python3 overlap_check.py OUT.json POOL.jsonl [POOL2 ...]
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "repo" / "datasets" / "public"


def words(x):
    if isinstance(x, str):
        return re.findall(r"[a-z0-9$%.,']+", x.lower())
    if isinstance(x, dict):
        return [w for k, v in x.items() for w in words(str(k)) + words(v)]
    if isinstance(x, list):
        return [w for v in x for w in words(v)]
    return words(str(x)) if x is not None else []


def grams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main(out, *pools):
    items = [json.loads(l) for f in ("easy.jsonl", "original.jsonl", "hard.jsonl") for l in open(REPO / f)]
    item_text = {d["id"]: words(d["state"]) + words(d["question"]["instructions"]) + words(d["question"].get("criteria")) for d in items}
    item_state = {d["id"]: " ".join(words(d["state"])) for d in items}
    g8 = {i: grams(w, 8) for i, w in item_text.items()}
    g13 = {i: grams(w, 13) for i, w in item_text.items()}
    want8 = set().union(*g8.values())
    want13 = set().union(*g13.values())
    states = set(item_state.values())
    hit8, hit13, exact, rows = set(), set(), set(), 0
    for p in pools:
        for line in open(p):
            r = json.loads(line)
            rows += 1
            s = " ".join(words(r.get("state")))
            if s in states:
                exact.add(s)
            ws = words(r.get("state")) + words(r.get("question") or r.get("questions") or r.get("instructions"))
            for g in grams(ws, 8) & want8:
                hit8.add(g)
            for g in grams(ws, 13) & want13:
                hit13.add(g)
    res = {
        "pools": [str(p) for p in pools], "training_rows": rows, "public_items": len(items),
        "exact_state_matches": [i for i, s in item_state.items() if s in exact],
        "items_with_any_13gram": sorted(i for i in g13 if g13[i] & hit13),
        "items_with_more_than_3_8grams": sorted(i for i in g8 if len(g8[i] & hit8) > 3),
        "items_with_any_8gram": len([i for i in g8 if g8[i] & hit8]),
        "example_shared_8grams": sorted(hit8)[:15],
    }
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 20 else f"{len(v)} items") for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
