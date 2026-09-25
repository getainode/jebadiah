"""Build questions.jsonl: 1,000 human-labelled yes/no questions (250 per source, 125 true / 125 false,
sha256(id) order) plus 400 Measuring Hate Speech items (150 clear-no, 150 clear-yes, 100 close calls)."""
import collections
import hashlib
import json
import os

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
HN = os.environ.get("JEB_HUMAN_NOUL", os.path.join(HERE, "..", "..", "data", "human-noul"))  # convert_human_noul.py output
SOURCES = ["condaqa", "contractnli", "sharc", "unfairtos"]
MHS_PARQUET = os.path.join(HERE, "raw", "mhs-train.parquet")
MHS_SOURCE = ("huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech @5468f6e11839 "
              "data/train-00000-of-00001.parquet sha256=0d65ce8cfb857f5db90e74ce295067bba7473fe12115715f9dd18d2a5f87e976")
MHS_LICENSE = "CC BY 4.0 (checked 2026-09-24: HF card front matter 'license: cc-by-4.0', raw/mhs-README.md)"
MOD_INSTR = ("Is this comment hate speech, meaning it attacks or dehumanises people because of a protected "
             "characteristic such as race, religion, gender, sexuality, disability or origin?")
MOD_CRIT = {"true": "Yes: the comment attacks or dehumanises a group or person on a protected characteristic.",
            "false": "No: the comment may be rude or critical but does not attack a protected characteristic."}


def h(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


out = []
for src in SOURCES:
    recs = [json.loads(l) for l in open(f"{HN}/human-{src}.jsonl")]
    recs.sort(key=lambda r: h(r["id"]))
    got = {True: [], False: []}
    for r in recs:
        (qid, lab), = r["label"].items()
        if len(got[lab]) < 125:
            got[lab].append(r)
    for r in sorted(got[True] + got[False], key=lambda r: h(r["id"])):
        (qid, q), = r["questions"].items()
        out.append({"id": r["id"], "set": r["set"], "subset": r["subset"], "source_key": src, "family": r["family"],
                    "state": r["state"], "questions": {qid: q}, "label": r["label"], "target": r["target"],
                    "slice": "human"})

t = pq.read_table(MHS_PARQUET, columns=["comment_id", "hatespeech", "text"]).to_pydict()
ann = collections.defaultdict(list)
text = {}
for cid, hs, tx in zip(t["comment_id"], t["hatespeech"], t["text"]):
    ann[cid].append(hs)
    text.setdefault(cid, tx)
pool = {"clear_no": [], "clear_yes": [], "close": [], "clear_yes_4": []}
for cid in sorted(ann, key=h):
    v = ann[cid]
    if len(v) < 4 or not text[cid] or not text[cid].strip():
        continue
    share = sum(1 for x in v if x == 2) / len(v)
    if len(v) == 4:
        # the >=5-annotator rule leaves only 56 clear-yes comments; unanimous 4-annotator yes fills the rest
        if share >= 0.8:
            pool["clear_yes_4"].append((cid, share, len(v)))
        continue
    if share <= 0.2:
        pool["clear_no"].append((cid, share, len(v)))
    elif share >= 0.8:
        pool["clear_yes"].append((cid, share, len(v)))
    elif 0.3 <= share <= 0.7 and share != 0.5:   # exactly 0.5 has no human side, so it cannot score a close call
        pool["close"].append((cid, share, len(v)))
print("MHS comments with >=5 annotators by bucket:", {k: len(v) for k, v in pool.items()})
pool["clear_yes"] = pool["clear_yes"] + pool["clear_yes_4"][:150 - len(pool["clear_yes"])]
take = {"clear_no": 150, "clear_yes": 150, "close": 100}   # close: only 60 exist, all are taken
for k, n in take.items():
    for cid, share, na in pool[k][:n]:
        out.append({"id": f"mhs:{cid}", "set": "mhs", "subset": "mhs-close" if k == "close" else "mhs-clear",
                    "source_key": "mhs-close" if k == "close" else "mhs-clear", "family": f"mhs/{cid}",
                    "state": text[cid], "questions": {"answer": {"type": "noul", "instructions": MOD_INSTR,
                                                                 "criteria": MOD_CRIT}},
                    "label": {"answer": share > 0.5}, "target": {"answer": {"true": round(share, 6),
                                                                              "false": round(1 - share, 6)}},
                    "human_share": round(share, 6), "n_annotators": na, "slice": "moderation",
                    "source": MHS_SOURCE, "license": MHS_LICENSE})

with open(os.path.join(HERE, "questions.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = collections.Counter((r["source_key"], r["label"]["answer"]) for r in out)
print(len(out), sorted(c.items()))
