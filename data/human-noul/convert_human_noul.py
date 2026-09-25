"""Convert four human-labelled yes/no datasets into Jebadiah noul training records.

    python3 convert_human_noul.py            # downloads into raw/ if missing, writes human-*.jsonl here

Sources and conversion follow workers-2026-09-23/yesno-data/REPORT.md section 4:
  CondaQA      HF lasha-nlp/CONDAQA (train.json + test.json, pinned revision), YES/NO only,
               family = the original passage (all edits and questions of a passage together)
  ContractNLI  stanfordnlp.github.io/contract-nli/resources/contract-nli.zip (sha256 pinned), train.json,
               Entailment -> true, Contradiction and NotMentioned -> false, family = the NDA
  ShARC        sharc-data.github.io/data/sharc1-official.zip (sha256 pinned; the file the HF UCLNLP/sharc
               loader script downloads), train, Yes/No utterances only, at most 2 per rule snippet,
               family = the rule snippet
  UNFAIR-ToS   HF coastalcph/lex_glue config unfair_tos split train, fetched through the datasets-server
               rows API (no pyarrow on this Mac), true when any unfairness label is set,
               family = the normalised clause text

Every state goes through the eval overlap check of pool-v2/finalize.py (same code, same 20 rebuilt
test sets): drop on a normalised hash match or on 5-gram containment >= 0.5. Then each source is
balanced (500 true / 500 false; ContractNLI 500 Entailment / 250 Contradiction / 250 NotMentioned)
in sha256 order of the family, whole families first, then remaining quota filled from the next
families in the same order. Targets: 0.95/0.05 for CondaQA and ShARC, one-hot for the other two.

Writes human-<source>.jsonl x4, human-noul.jsonl (CondaQA + ContractNLI + ShARC),
human-noul-alt.jsonl (CondaQA + ContractNLI + UNFAIR-ToS) and human-noul-manifest.json (every count).
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import statistics
import sys
import types
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
POOL_V2 = os.environ.get("JEB_POOL_V2", os.path.join(HERE, "..", "pool-v2"))
TEST_DIR = os.path.join(POOL_V2, "v1-rebuild", "data-v1", "test")

# Reuse pool-v2's own text normalisation (generate.py imports aiohttp at module level for its
# network code; it is not installed here and not needed for these two pure functions).
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "pool-v2"))  # generate.py
from generate import norm_words, state_hash  # noqa: E402

CONDAQA_REV = "3c9caa2f2f6960711e7f4d2e800581def2b6c183"
LEXGLUE_REV = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
SHARC_URL = "https://sharc-data.github.io/data/sharc1-official.zip"
SHARC_SHA = "72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e"
CNLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
CNLI_SHA = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
UNFAIR_ROWS = "https://datasets-server.huggingface.co/rows?dataset=coastalcph/lex_glue&config=unfair_tos&split=train"

CHECKED = "re-checked 2026-09-24"
LICENSE = {
    "condaqa": f"Apache-2.0 ({CHECKED}: HF lasha-nlp/CONDAQA cardData.license apache-2.0 and card 'license: apache-2.0'; "
               "github.com/AbhilashaRavichander/CondaQA LICENSE spdx Apache-2.0)",
    "contractnli": f"CC BY 4.0 ({CHECKED}: archive LICENSE is the CC Attribution 4.0 International text, TERMS grants use "
                   "'in accordance with ... the Creative Commons Attribution 4.0 International Public License', "
                   "README 'Our dataset is released under CC BY 4.0'; stanfordnlp.github.io/contract-nli says CC BY 4.0)",
    "sharc": f"CC BY-SA 3.0, UNCONFIRMED ({CHECKED}: only the HF UCLNLP/sharc cardData.license tag cc-by-sa-3.0; the card's "
             "Licensing section says 'More Information Needed'; sharc-data.github.io has the CC BY-SA 3.0 line only inside "
             "an HTML comment; the archive README and the paper arXiv 1809.01494 state no license)",
    "unfairtos": f"CC BY 4.0 ({CHECKED}: HF coastalcph/lex_glue cardData.license cc-by-4.0, Licensing section 'More "
                 "Information Needed'; LexGLUE paper arXiv 2110.00976: the datasets 'are available for re-use and "
                 "re-share with appropriate attribution')",
}

SPEC = {
    "condaqa": {
        "instructions": "Answer the question about the passage. Read negations carefully. Question: {q}",
        "true": "The passage implies the answer is yes.",
        "false": "The passage implies the answer is no.",
        "smooth": 0.95, "quota": {True: 500, False: 500},
    },
    "contractnli": {
        "instructions": "Decide whether this excerpt of a non-disclosure agreement provides that: {q}",
        "true": "The excerpt provides this.",
        "false": "The excerpt contradicts it or does not provide it.",
        "smooth": 1.0, "quota": {"Entailment": 500, "Contradiction": 250, "NotMentioned": 250},
    },
    "sharc": {
        "instructions": "A user asks whether a rule applies to them. Decide from the rule text, their scenario and their answers.",
        "true": "Yes: under the rule, the answer to the user's question is yes.",
        "false": "No: under the rule, the answer is no.",
        "smooth": 0.95, "quota": {True: 500, False: 500}, "per_family_cap": 2,
    },
    "unfairtos": {
        # not in the section 4 table (it names UNFAIR-ToS as the ShARC fallback without wording)
        "instructions": "Is this terms-of-service clause potentially unfair to the consumer?",
        "true": "Yes: the clause is potentially unfair to the consumer.",
        "false": "No: the clause is not potentially unfair.",
        "smooth": 1.0, "quota": {True: 500, False: 500},
    },
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, want_sha=None):
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        print("download", url, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "jebadiah-human-noul/1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    got = sha256_file(path)
    if want_sha and got != want_sha:
        raise SystemExit(f"{path}: sha256 {got} != pinned {want_sha}")
    return got


def words(s: str) -> int:
    return len(s.split())


# ----------------------------------------------------------------------------- loaders
# Each loader returns (items, counts). An item: {id, family, label (bool), cls, state, q, extra}.

def load_condaqa():
    items, raw = [], 0
    files = {}
    for split in ("train", "test"):
        p = os.path.join(RAW, f"condaqa-{split}.json")
        files[split] = fetch(f"https://huggingface.co/datasets/lasha-nlp/CONDAQA/resolve/{CONDAQA_REV}/{split}.json", p)
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            raw += 1
            if r["label"] not in ("YES", "NO"):
                continue
            items.append({
                "id": f"{split}/p{r['PassageID']}/e{r['PassageEditID']}/{r['QuestionID']}/s{r['SampleID']}",
                # one passage can appear under two PassageIDs; key the family on the original text
                "family": "passage-" + sha(" ".join(norm_words(r["original passage"])))[:16],
                "label": r["label"] == "YES", "cls": r["label"] == "YES",
                "state": r["sentence1"].strip(), "q": r["sentence2"].strip(),
            })
    src = f"huggingface.co/datasets/lasha-nlp/CONDAQA train.json+test.json @{CONDAQA_REV[:12]}"
    return items, {"raw": raw, "files_sha256": files}, src


def cnli_excerpt(doc, idxs):
    """Text of the given span indices, contiguous runs kept in original formatting, gaps marked."""
    text, spans = doc["text"], doc["spans"]
    runs, cur = [], [idxs[0]]
    for i in idxs[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return "\n...\n".join(text[spans[r[0]][0]:spans[r[-1]][1]].strip() for r in runs)


def load_contractnli():
    p = os.path.join(RAW, "contract-nli.zip")
    zsha = fetch(CNLI_URL, p, CNLI_SHA)
    d = json.loads(zipfile.ZipFile(p).read("contract-nli/train.json"))
    hyp = {k: v["hypothesis"] for k, v in d["labels"].items()}
    items, raw, pending_nm = [], 0, []
    for doc in d["documents"]:
        n = len(doc["spans"])
        ec_len = []
        for key, a in sorted(doc["annotation_sets"][0]["annotations"].items()):
            raw += 1
            it = {"id": f"doc{doc['id']}/{key}", "family": f"nda-{doc['id']}", "cls": a["choice"],
                  "label": a["choice"] == "Entailment", "q": hyp[key], "doc": doc}
            if a["choice"] == "NotMentioned":
                pending_nm.append(it)
                continue
            ev = sorted(set(a["spans"]))
            idxs = sorted({j for i in ev for j in (i - 1, i, i + 1) if 0 <= j < n})
            it["state"] = cnli_excerpt(doc, idxs)
            ec_len.append(words(it["state"]))
            items.append(it)
        doc["_ec_len"] = ec_len
    ec_all = [words(i["state"]) for i in items]
    glob_med = statistics.median(ec_all)
    for it in pending_nm:
        doc = it["doc"]
        n = len(doc["spans"])
        target = statistics.median(doc["_ec_len"]) if doc["_ec_len"] else glob_med
        # a contiguous window of this NDA's spans, start fixed by sha256 of the item id, grown until it
        # reaches the length of this NDA's Entailment/Contradiction excerpts (median)
        start = int(sha(f"cnli-nm:{it['id']}")[:8], 16) % n
        lo = hi = start
        while words(cnli_excerpt(doc, list(range(lo, hi + 1)))) < target and (lo > 0 or hi < n - 1):
            if hi < n - 1:
                hi += 1
            else:
                lo -= 1
        it["state"] = cnli_excerpt(doc, list(range(lo, hi + 1)))
        items.append(it)
    for it in items:
        it.pop("doc", None)
    for doc in d["documents"]:
        doc.pop("_ec_len", None)
    src = f"stanfordnlp.github.io/contract-nli/resources/contract-nli.zip sha256={zsha} train.json"
    return items, {"raw": raw, "archive_sha256": zsha, "ec_excerpt_median_words": glob_med}, src


def sharc_state(r):
    hist = "\n".join(f"- {h['follow_up_question'].strip()} {h['follow_up_answer'].strip()}" for h in r["history"])
    return (f"Rule text:\n{r['snippet'].strip()}\n\n"
            f"User scenario: {r['scenario'].strip() or '(none given)'}\n\n"
            f"Follow-up questions already answered:\n{hist or '(none)'}\n\n"
            f"User question: {r['question'].strip()}")


def load_sharc():
    p = os.path.join(RAW, "sharc1-official.zip")
    zsha = fetch(SHARC_URL, p, SHARC_SHA)
    z = zipfile.ZipFile(p)
    rows = json.loads(z.read("sharc1-official/json/sharc_train.json"))
    neg = set()
    for f in ("sharc_negative_question_utterance_ids.txt", "sharc_negative_scenario_utterance_ids.txt"):
        neg |= set(z.read(f"sharc1-official/negative_sample_utterance_ids/{f}").decode().split())
    items = []
    for r in rows:
        if r["answer"] not in ("Yes", "No"):
            continue
        items.append({"id": r["utterance_id"], "family": "snippet-" + sha(r["snippet"].strip())[:16],
                      "tree_id": r["tree_id"], "label": r["answer"] == "Yes", "cls": r["answer"] == "Yes",
                      "state": sharc_state(r), "q": None, "negative_sample": r["utterance_id"] in neg})
    src = f"sharc-data.github.io/data/sharc1-official.zip sha256={zsha} json/sharc_train.json (the file HF UCLNLP/sharc's loader downloads)"
    return items, {"raw": len(rows), "archive_sha256": zsha,
                   "yes_no_negative_sample_ids": sum(i["negative_sample"] for i in items)}, src


def load_unfairtos():
    p = os.path.join(RAW, "unfair_tos-train.rows.jsonl")
    if not os.path.exists(p):
        out, off, total = [], 0, None
        while total is None or off < total:
            req = urllib.request.Request(f"{UNFAIR_ROWS}&offset={off}&length=100",
                                         headers={"User-Agent": "jebadiah-human-noul/1"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            total = d["num_rows_total"]
            out += [x["row"] | {"row_idx": x["row_idx"]} for x in d["rows"]]
            off += 100
        with open(p, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fsha = sha256_file(p)
    items, raw = [], 0
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        raw += 1
        text = r["text"].strip()
        if not text:
            continue
        items.append({"id": f"train/{r['row_idx']}", "family": "clause-" + sha(" ".join(norm_words(text)))[:16],
                      "label": bool(r["labels"]), "cls": bool(r["labels"]), "state": text, "q": None})
    src = (f"huggingface.co/datasets/coastalcph/lex_glue config=unfair_tos split=train @{LEXGLUE_REV[:12]} "
           f"via datasets-server rows API (rows file sha256={fsha})")
    return items, {"raw": raw, "rows_file_sha256": fsha}, src


# ----------------------------------------------------------------------------- eval overlap (pool-v2/finalize.py)

def grams(obj, n=5):
    w = norm_words(obj)
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def build_eval_overlap():
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jsonl")))
    inv = collections.defaultdict(set)
    test_hash = {}
    test_ids = []
    for tf in test_files:
        for line in open(tf, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            i = len(test_ids)
            test_ids.append(f"{os.path.basename(tf)[:-6]}:{r.get('id')}")
            test_hash[state_hash(r["state"])] = test_ids[i]
            for g in grams(r["state"]):
                inv[g].add(i)

    def eval_overlap(state):
        h = state_hash(state)
        if h in test_hash:
            return 1.0, test_hash[h]
        gs = grams(state)
        hits = collections.Counter()
        for g in gs:
            for i in inv.get(g, ()):
                hits[i] += 1
        if not hits:
            return 0.0, None
        i, n = hits.most_common(1)[0]
        return n / max(1, len(gs)), test_ids[i]

    return eval_overlap, [os.path.basename(t) for t in test_files], len(test_ids)


# ----------------------------------------------------------------------------- balance

def balance(name, items):
    """Fill each class quota in sha256 order of the family: whole families that fit first, then the
    remaining quota from the families left over, in the same order (records inside a family in
    sha256 order of their id). ShARC keeps at most 2 utterances per rule snippet."""
    spec = SPEC[name]
    quota = dict(spec["quota"])
    fams = collections.defaultdict(list)
    for it in items:
        fams[it["family"]].append(it)
    order = sorted(fams, key=lambda f: sha(f"human-noul:{name}:{f}"))
    for f in order:
        fams[f].sort(key=lambda it: sha(f"human-noul:{name}:{it['id']}"))
    cap = spec.get("per_family_cap")
    left = dict(quota)
    taken = collections.defaultdict(list)
    whole = partial = 0
    for f in order:
        cand = fams[f][:cap] if cap else fams[f]
        need = collections.Counter(it["cls"] for it in cand)
        if all(left.get(c, 0) >= k for c, k in need.items()):
            taken[f] = list(cand)
            for c, k in need.items():
                left[c] -= k
            whole += 1
        if not any(left.values()):
            break
    for f in order:
        if not any(left.values()):
            break
        if f in taken:
            continue
        got = []
        for it in fams[f]:
            if cap and len(got) >= cap:
                break
            if left.get(it["cls"], 0) > 0:
                got.append(it)
                left[it["cls"]] -= 1
        if got:
            taken[f] = got
            partial += 1
    out = [it for f in order for it in taken.get(f, [])]
    return out, {"families_whole": whole, "families_partial": partial, "unfilled": {str(k): v for k, v in left.items() if v}}


# ----------------------------------------------------------------------------- records

def to_record(name, it, src):
    spec = SPEC[name]
    p = spec["smooth"] if it["label"] else 1 - spec["smooth"]
    ins = spec["instructions"].format(q=it["q"]) if it["q"] else spec["instructions"]
    return {
        "id": f"human-{name}:{it['id']}",
        "set": f"human-{name}",
        "subset": name if name != "contractnli" else f"contractnli-{it['cls'].lower()}",
        "source": src,
        "license": LICENSE[name],
        "family": f"{name}/{it['family']}",
        "state": it["state"],
        "questions": {"answer": {"type": "noul", "instructions": ins,
                                 "criteria": {"true": spec["true"], "false": spec["false"]}}},
        "label": {"answer": it["label"]},
        "target": {"answer": {"true": round(p, 6), "false": round(1 - p, 6)}},
    }


def label_counts(items):
    c = collections.Counter(it["label"] for it in items)
    return {"true": c[True], "false": c[False]}


def main():
    eval_overlap, test_sets, n_test = build_eval_overlap()
    loaders = {"condaqa": load_condaqa, "contractnli": load_contractnli, "sharc": load_sharc, "unfairtos": load_unfairtos}
    manifest = {"eval_overlap": {"test_dir": TEST_DIR, "test_sets": test_sets, "test_states": n_test,
                                 "rule": "drop on normalised hash match or 5-gram containment >= 0.5 (pool-v2/finalize.py)"},
                "sources": {}}
    written = {}
    for name, loader in loaders.items():
        items, counts, src = loader()
        st = {"source": src, **counts, "yes_no": len(items), "yes_no_labels": label_counts(items),
              "yes_no_by_class": dict(collections.Counter(str(it["cls"]) for it in items))}
        fam_of = {it["id"]: it["family"] for it in items}
        kept, dropped, maxov = [], [], (0.0, None, None)
        cache = {}
        for it in items:
            if it["state"] not in cache:
                cache[it["state"]] = eval_overlap(it["state"])
            ov, which = cache[it["state"]]
            if ov >= 0.5:
                dropped.append({"id": it["id"], "containment": round(ov, 3), "test_item": which,
                                "state_head": it["state"][:300]})
                continue
            kept.append(it)
        # a family (passage, NDA, rule, clause) with any dropped member is dropped whole, so an edited
        # copy of an eval passage cannot stay in through a small wording change
        bad_fams = {fam_of[d["id"]] for d in dropped}
        fam_extra = [it for it in kept if it["family"] in bad_fams]
        kept = [it for it in kept if it["family"] not in bad_fams]
        st["overlap_dropped"] = len(dropped)
        st["overlap_dropped_states"] = len({state_hash(s) for s, (ov, _) in cache.items() if ov >= 0.5})
        st["overlap_family_extra_dropped"] = len(fam_extra)
        st["overlap_dropped_by_test_set"] = dict(collections.Counter(d["test_item"].split(":")[0] for d in dropped))
        st["overlap_examples"] = dropped[:3]
        for it in kept:
            ov, which = cache[it["state"]]
            if ov > maxov[0]:
                maxov = (ov, which, it["id"])
        st["max_containment_kept"] = {"value": round(maxov[0], 3), "test_item": maxov[1], "record": maxov[2]}
        st["after_overlap"] = len(kept)
        st["after_overlap_labels"] = label_counts(kept)
        st["after_overlap_by_class"] = dict(collections.Counter(str(it["cls"]) for it in kept))
        picked, bst = balance(name, kept)
        st["balance"] = bst
        st["after_balance"] = len(picked)
        st["after_balance_labels"] = label_counts(picked)
        st["after_balance_by_class"] = dict(collections.Counter(str(it["cls"]) for it in picked))
        st["families"] = len({it["family"] for it in picked})
        if name == "sharc":
            st["negative_sample_ids_picked"] = sum(it["negative_sample"] for it in picked)
            per_tree = collections.Counter(it["tree_id"] for it in picked)
            st["max_per_tree_id"] = max(per_tree.values())
        lens = [words(it["state"]) for it in picked]
        st["state_words"] = {"mean": round(statistics.mean(lens), 1), "median": statistics.median(lens),
                             "min": min(lens), "max": max(lens)}
        recs = [to_record(name, it, src) for it in picked]
        path = os.path.join(HERE, f"human-{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st["file"] = {"path": os.path.basename(path), "records": len(recs), "sha256": sha256_file(path)}
        written[name] = recs
        manifest["sources"][name] = st
        print(name, json.dumps({k: st[k] for k in ("raw", "yes_no", "overlap_dropped", "after_overlap", "after_balance",
                                                   "after_balance_labels", "state_words")}), flush=True)
    for out, names in (("human-noul.jsonl", ("condaqa", "contractnli", "sharc")),
                       ("human-noul-alt.jsonl", ("condaqa", "contractnli", "unfairtos"))):
        path = os.path.join(HERE, out)
        with open(path, "w", encoding="utf-8") as f:
            for n in names:
                for r in written[n]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[out] = {"sources": list(names), "records": sum(len(written[n]) for n in names), "sha256": sha256_file(path)}
    json.dump(manifest, open(os.path.join(HERE, "human-noul-manifest.json"), "w"), indent=1, ensure_ascii=False)
    write_sample(written)
    print("wrote", ", ".join(f"human-{n}.jsonl" for n in written), "human-noul.jsonl human-noul-alt.jsonl "
          "human-noul-manifest.json SAMPLE.md")


def write_sample(written):
    L = ["# Human noul: converted record sample", "",
         "8 records per source (4 true, 4 false), the first in sha256 order of `sample:<id>`. Written by "
         "convert_human_noul.py. States are shown in full.", ""]
    for name, recs in written.items():
        L += [f"## {name}", ""]
        order = sorted(recs, key=lambda r: sha(f"sample:{r['id']}"))
        pick = [r for r in order if r["label"]["answer"]][:4] + [r for r in order if not r["label"]["answer"]][:4]
        for i, r in enumerate(pick, 1):
            q = r["questions"]["answer"]
            L += [f"### {name} {i}. `{r['id']}`", "", f"family `{r['family']}`, subset `{r['subset']}`", "",
                  "**State:**", "", "```text", r["state"], "```", "",
                  f"**Question:** {q['instructions']}", "",
                  f"- `true`: {q['criteria']['true']}", f"- `false`: {q['criteria']['false']}", "",
                  f"**Label:** `{str(r['label']['answer']).lower()}`, target {json.dumps(r['target']['answer'])}", ""]
    with open(os.path.join(HERE, "SAMPLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
