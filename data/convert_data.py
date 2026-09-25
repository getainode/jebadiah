"""Build Jebadiah v1's PUBLIC data on the rented box: one training pool and the test sets, all in
the Jev wire shape with the label OUTSIDE the question object:

  {"id", "set", "subset", "source", "license", "family",
   "state", "questions": {qid: {type, instructions, criteria}}, "label": {qid: answer}}

Label values: choice -> option key (str); noul -> true/false (bool); score -> level index (int).
Optional record-level `target`: {qid: {key: weight}}, a soft label where the source gives a
distribution (typed-decisions' gold, the mean of three teacher samples).

This is the v0 converter with every non-public source removed. Everything it reads is fetched
from a public URL by this script; nothing is bundled and nothing private exists on the box.

  pool   HelpSteer2's TRAIN split (CC BY-4.0): helpfulness in the Jevals task's shape plus the other
         four attributes on a subset; SummEval (MIT, the mteb/summeval file Nimble's converter reads):
         coherence, consistency, fluency and relevance from the expert mean. Both are held out of the
         reported eval sets by assertion (Jevals items, Nimble's committed id manifests),
         LocalLLaMA/typed-decisions train (Apache-2.0, 6,000 decisions with gold distributions),
         Kev decision-v7 train.jsonl (huggingface.co/datasets/jaredpalmer/kev-suites) filtered to
         the sources whose license permits derived weights and that are not a Jevals test source.
         Nimble's train.jsonl is NOT in the pool: the repo has no LICENSE file.
  tests  typed-decisions test; Kev decision-v7 test and transfer-v4 test (every question);
         Nimble data/eval.jsonl (324) and Nimble's 13 public human-labelled subsets, rebuilt from
         the upstream files with Nimble's own converter and its committed id manifests;
         the three Jevals suite 0.1.0 tasks rebuilt from their pinned Hugging Face revisions.

Every test source is independent: one that cannot be fetched is logged under manifest.failures
and the rest are still written. --limit N caps every source at N records (smoke tests).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from jebadiah_prompt import flatten_text  # noqa: E402

# Pinned upstream revisions (resolved 2026-09-22; the manifest records what was actually used).
TYPED_DECISIONS = "LocalLLaMA/typed-decisions"
TYPED_DECISIONS_REV = "ea9306458d6e9563628369a3d1e72e362fb381d2"
KEV_SUITES = "jaredpalmer/kev-suites"
KEV_SUITES_REV = "a88f56db5341397299137cb68775c2ea6e3f68cb"
KEV_FILES = {
    "v7-train": ("v7/decision-v7/train.jsonl", "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad"),
    "v7-test": ("v7/decision-v7/test.jsonl", "cd7d129a84232e0c7a4e92b4840a6dfe14c8bb93c5d1904b526fa8c085325d2d"),
    "transfer-v4-test": ("v4/transfer-v4/test.jsonl", "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517"),
}
NIMBLE_REPO = "bespokelabsai/nimble"
NIMBLE_COMMIT = "f136b3f75721fda4ea961f73993cc50b08488835"
NIMBLE_LICENSE = ("no license stated: the nimble repo has no LICENSE file and GitHub reports license null "
                  "(2026-09-22); the README calls it open data and the adapter on HF is Apache-2.0. "
                  "Evaluation only (data/eval.jsonl): its training set is excluded from the pool")

# Nimble's 13 public subsets: how to get the raw upstream file each converter reads
# (docs/PUBLIC_BENCHMARKS.md in the nimble repo), in the order the subsets are built.
# kind: "hf" = Hugging Face dataset exported to JSONL with the original field names;
#       "url" = a direct download (archive members are read in place by Nimble's reader or extracted).
NIMBLE_PUBLIC = [
    {"name": "boolq", "dataset": "boolq", "kind": "hf", "repo": "google/boolq", "config": None, "split": "validation",
     "raw": "boolq/validation.jsonl", "license": "CC BY-SA 3.0", "url": "https://huggingface.co/datasets/google/boolq"},
    {"name": "pubmedqa", "dataset": "pubmedqa", "kind": "hf", "repo": "qiaojin/PubMedQA", "config": "pqa_labeled",
     "split": "train", "raw": "pubmedqa/train.jsonl", "license": "MIT", "url": "https://huggingface.co/datasets/qiaojin/PubMedQA"},
    {"name": "paws", "dataset": "paws", "kind": "hf", "repo": "google-research-datasets/paws", "config": "labeled_final",
     "split": "test", "raw": "paws/test.jsonl", "license": "Google PAWS license: may be freely used for any purpose",
     "url": "https://huggingface.co/datasets/google-research-datasets/paws"},
    {"name": "squad2", "dataset": "squad2", "kind": "hf", "repo": "rajpurkar/squad_v2", "config": None, "split": "validation",
     "raw": "squad_v2/validation.jsonl", "license": "CC BY-SA 4.0", "url": "https://huggingface.co/datasets/rajpurkar/squad_v2"},
    {"name": "helpsteer2", "dataset": "helpsteer2", "kind": "hf", "repo": "nvidia/HelpSteer2", "config": None,
     "split": "validation", "raw": "helpsteer2/validation.jsonl", "license": "CC BY-4.0",
     "url": "https://huggingface.co/datasets/nvidia/HelpSteer2"},
    {"name": "aegis2", "dataset": "aegis2", "kind": "hf", "repo": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "config": None,
     "split": "test", "raw": "aegis2/test.jsonl", "license": "CC BY 4.0",
     "url": "https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0"},
    {"name": "civil_comments", "dataset": "civil_comments", "kind": "hf", "repo": "google/civil_comments", "config": None,
     "split": "test", "raw": "civil_comments/test.jsonl", "license": "CC0 1.0",
     "url": "https://huggingface.co/datasets/google/civil_comments"},
    {"name": "summeval-relevance", "dataset": "summeval", "subset": "relevance", "kind": "hf", "repo": "mteb/summeval",
     "config": None, "split": "test", "raw": "summeval/test.jsonl", "license": "MIT",
     "url": "https://huggingface.co/datasets/mteb/summeval"},
    {"name": "summeval-consistency", "dataset": "summeval", "subset": "consistency", "kind": "hf", "repo": "mteb/summeval",
     "config": None, "split": "test", "raw": "summeval/test.jsonl", "license": "MIT",
     "url": "https://huggingface.co/datasets/mteb/summeval"},
    {"name": "massive-en-US", "dataset": "massive", "subset": "en-US", "kind": "url",
     "url": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
     "raw": "massive/amazon-massive-dataset-1.1.tar.gz", "license": "CC BY 4.0"},
    {"name": "massive-de-DE", "dataset": "massive", "subset": "de-DE", "kind": "url",
     "url": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
     "raw": "massive/amazon-massive-dataset-1.1.tar.gz", "license": "CC BY 4.0"},
    {"name": "multinli", "dataset": "multinli", "kind": "url", "url": "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
     "raw": "multinli/multinli_1.0.zip", "member": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
     "license": "Mixed per genre: OANC public domain; fiction and other sources CC BY-SA 3.0 or similar"},
    {"name": "vitaminc-dev", "dataset": "vitaminc", "kind": "url",
     "url": "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip",
     "raw": "vitaminc/vitaminc.zip", "member": "dev.jsonl", "license": "CC BY-SA 3.0"},
]

# Kev decision-v7 question sources: keep / drop, with the license that decides it.
KEV_SOURCES = {
    "boolq": ("keep", "CC BY-SA 3.0 (google/boolq)"),
    "mnli": ("keep", "CC BY-SA 3.0 / CC BY 3.0 / OANC (nyu-mll/multi_nli)"),
    "dbpedia14": ("keep", "CC BY-SA 3.0 (fancyzhx/dbpedia_14)"),
    "contrastive_*": ("keep", "Apache-2.0 (generated by Kev's pipeline, jaredpalmer/kev)"),
    "composition_rand:*": ("keep", "Apache-2.0 (generated by Kev's pipeline, jaredpalmer/kev)"),
    "banking77": ("drop", "CC BY 4.0, but Banking77 is a Jevals test source: excluded so the Jevals choice number stays clean"),
    "agnews": ("drop", "no license on the HF card (unknown); the AG corpus page limits use to non-commercial"),
    "agnews_yn": ("drop", "same as agnews"),
    "trec": ("drop", "no license stated (CogComp/trec: unknown)"),
    "sst5": ("drop", "no license stated (SetFit/sst5)"),
    "yelp": ("drop", "Yelp dataset terms: academic use only, no redistribution"),
    "yelp_yn": ("drop", "same as yelp"),
    "amazon": ("drop", "upstream Amazon Customer Reviews terms: research only, no redistribution; the SetFit mirror's Apache tag cannot relicense it"),
    "imdb": ("drop", "IMDb terms: personal, non-commercial use (HF card: other)"),
}


def kev_source_rule(src: str):
    if src in KEV_SOURCES:
        return KEV_SOURCES[src]
    for pat, rule in KEV_SOURCES.items():
        if pat.endswith("*") and src.startswith(pat[:-1]):
            return rule
    return ("drop", f"unlisted Kev source {src!r}")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_jsonl(path):
    # split on newline only: real abstracts contain U+2028/U+2029, which splitlines() would cut
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wire_question(q: dict) -> dict:
    """The three fields a judge may see, and nothing else, with instructions and descriptions as
    strings (AINode's /v1/systemone takes strings; Kev ships some as objects)."""
    out = {"type": q["type"], "instructions": flatten_text(q.get("instructions"))}
    crit = q.get("criteria")
    if crit is not None:
        if isinstance(crit, dict):
            out["criteria"] = {k: (flatten_text(v) or None) for k, v in crit.items()}
        elif q["type"] == "choice":
            # AINode's choice wire is {name: description|null}; a bare option list becomes that
            out["criteria"] = {flatten_text(c): None for c in crit}
        else:
            out["criteria"] = [flatten_text(c) for c in crit]
    return out


def norm_label(q: dict, label):
    t = q["type"]
    if t == "noul":
        if isinstance(label, bool):
            return label
        return str(label).strip().lower() == "true"
    if t == "score":
        return int(label)
    return str(label)


def summarize(rows):
    types = collections.Counter()
    subsets = collections.Counter()
    nq = 0
    for r in rows:
        for qid, q in r["questions"].items():
            types[q["type"]] += 1
            nq += 1
        subsets[r.get("subset", "")] += 1
    return {"records": len(rows), "questions": nq, "types": dict(types), "subsets": dict(subsets)}


def capped(rows, limit):
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------ downloads

def hf_token():
    """The token from the environment only (rate limits); never written anywhere."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def download_url(url: str, dest: str, retries: int = 3) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jebadiah-v1-sweep/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed {url}: {last}")


def hf_file(repo: str, filename: str, revision: str, raw_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, token=hf_token(),
                           cache_dir=os.path.join(raw_dir, "hf-cache"))


def hf_to_jsonl(repo: str, config, split: str, dest: str, revision=None, raw_dir: str = "") -> str:
    """Export one Hugging Face split to JSON Lines with the original field names and row order
    (what Nimble's converters read)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    from datasets import load_dataset
    kwargs = {"split": split, "token": hf_token()}
    if revision:
        kwargs["revision"] = revision
    if raw_dir:
        kwargs["cache_dir"] = os.path.join(raw_dir, "hf-datasets")
    ds = load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, dest)
    return dest


def fetch_nimble_repo(raw_dir: str) -> str:
    """The nimble repo at the pinned commit, as a tarball from GitHub (no git needed)."""
    dest = os.path.join(raw_dir, "nimble")
    if os.path.isdir(os.path.join(dest, "nimble", "datasets", "public_sources")) and os.path.exists(os.path.join(dest, "data", "eval.jsonl")):
        return dest
    url = f"https://codeload.github.com/{NIMBLE_REPO}/tar.gz/{NIMBLE_COMMIT}"
    tgz = download_url(url, os.path.join(raw_dir, f"nimble-{NIMBLE_COMMIT[:12]}.tar.gz"))
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(tmp)
    inner = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
    if len(inner) != 1:
        raise RuntimeError(f"unexpected nimble tarball layout: {inner}")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(os.path.join(tmp, inner[0]), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def extract_member(archive: str, member: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n == member or n.endswith("/" + member)]
        if not names:
            raise RuntimeError(f"{member} not in {archive}: {z.namelist()[:10]}")
        with z.open(names[0]) as src, open(dest + ".part", "wb") as out:
            shutil.copyfileobj(src, out)
    os.replace(dest + ".part", dest)
    return dest


# ------------------------------------------------------------------ Nimble (eval.jsonl and the public suite)

def nimble_rows(path, set_name, subset, license_note, commit):
    rows = []
    for r in read_jsonl(path):
        qs = {qid: wire_question(q) for qid, q in r["input"]["questions"].items()}
        labels = {qid: norm_label(q, r["reference"]["target"]) for qid, q in qs.items()}
        rows.append({
            "id": f"{set_name}:{r['id']}",
            "set": set_name,
            "subset": subset or r.get("domain", ""),
            "source": f"github.com/{NIMBLE_REPO} @{commit[:12]}: {os.path.basename(path)}",
            "license": license_note,
            "family": r.get("family") or r.get("source_family") or r["id"],
            "state": r["input"]["state"],
            "questions": qs,
            "label": labels,
        })
    return rows


NIMBLE_TOKENIZER = ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404")


def nimble_tokenizer_path(raw_dir: str) -> str:
    """Nimble measured its public subsets' prompt lengths with the Qwen3.5-9B tokenizer and
    dropped a few over-long rows before selecting; four subsets (aegis2, helpsteer2, summeval x2)
    can only be rebuilt id-for-id with that filter on. The Base repo ships the same tokenizer."""
    from huggingface_hub import snapshot_download
    return snapshot_download(NIMBLE_TOKENIZER[0], revision=NIMBLE_TOKENIZER[1], token=hf_token(),
                             allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json", "*.txt"],
                             cache_dir=os.path.join(raw_dir, "hf-cache"))


def build_nimble_public(spec: dict, repo: str, raw_dir: str, build_dir: str) -> tuple[str, dict]:
    """Rebuild one public subset with Nimble's own converter and its committed id manifest (first
    without, then with Nimble's tokenizer length filter); on an id mismatch (upstream drift) fall
    back to Nimble's documented seeded draw. Returns the path of all.jsonl and a note about how it
    was selected."""
    name = spec["name"]
    if spec["kind"] == "hf":
        raw = hf_to_jsonl(spec["repo"], spec.get("config"), spec["split"], os.path.join(raw_dir, spec["raw"]), raw_dir=raw_dir)
    else:
        raw = download_url(spec["url"], os.path.join(raw_dir, spec["raw"]))
        if spec.get("member"):
            raw = extract_member(raw, spec["member"], os.path.join(raw_dir, os.path.dirname(spec["raw"]), spec["member"]))
    ids_manifest = os.path.join(repo, "docs", "assets", "public-benchmarks", "subsets", f"{name}-manifest.json")
    out_dir = os.path.join(build_dir, name)
    env = {**os.environ, "PYTHONPATH": repo + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    base = [sys.executable, "-m", "nimble.datasets.public_benchmarks", "--dataset", spec["dataset"], "--source", raw]
    if spec.get("subset"):
        base += ["--subset", spec["subset"]]
    attempts = []
    committed = json.load(open(ids_manifest)) if os.path.exists(ids_manifest) else {}
    if committed:
        attempts.append(("ids-from committed manifest", base + ["--ids-from", ids_manifest, "--output-dir", out_dir]))
        if committed.get("max_input_tokens"):
            try:
                tokp = nimble_tokenizer_path(raw_dir)
                attempts.append(("ids-from committed manifest with Nimble's tokenizer length filter",
                                 base + ["--ids-from", ids_manifest, "--tokenizer", tokp,
                                         "--max-input-tokens", str(committed["max_input_tokens"]), "--output-dir", out_dir + "-tok"]))
            except Exception as e:  # noqa: BLE001
                log(f"nimble tokenizer unavailable ({e}); skipping the length-filtered attempt")
    limit = committed.get("limit")
    if limit:
        attempts.append((f"seeded draw, limit {limit} (Nimble's documented build)", base + ["--limit", str(limit), "--output-dir", out_dir + "-reselected"]))
    errors = []
    for how, cmd in attempts:
        shutil.rmtree(cmd[-1], ignore_errors=True)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        if p.returncode == 0 and os.path.exists(os.path.join(cmd[-1], "all.jsonl")):
            built = json.load(open(os.path.join(cmd[-1], "manifest.json")))
            expected = committed.get("dataset_sha256")
            note = {"selection": how, "dataset_sha256": built.get("dataset_sha256"), "committed_dataset_sha256": expected,
                    "byte_identical_to_committed": bool(expected) and built.get("dataset_sha256") == expected,
                    "count": built.get("count"), "raw_sha256": sha256_file(raw)}
            return os.path.join(cmd[-1], "all.jsonl"), note
        errors.append(f"{how}: rc={p.returncode} {p.stderr.strip()[-400:]}")
    raise RuntimeError(f"nimble converter failed for {name}: " + " | ".join(errors))


# ------------------------------------------------------------------ Kev

def kev_rows(path, set_name, filter_sources: bool):
    """Kev records -> wire records. The pool keeps only licensed sources; the test splits keep
    every question (evaluation, no derived weights)."""
    rows, dropped = [], collections.Counter()
    for i, r in enumerate(read_jsonl(path)):
        meta = r.get("_meta", {})
        qs, labels, kept_src = {}, {}, []
        for qid, q in r["questions"].items():
            src = q.get("src", "?")
            rule, why = kev_source_rule(src)
            if filter_sources and rule != "keep":
                dropped[src] += 1
                continue
            qs[qid] = wire_question(q)
            labels[qid] = norm_label(q, q["label"])
            kept_src.append(src)
        if not qs:
            continue
        src0 = kept_src[0]
        subset = src0.split(":")[0] if src0.startswith("composition_rand") else src0
        family = meta.get("family_id") or meta.get("pair_id") or meta.get("group_id") or meta.get("id") or f"row{i}"
        rows.append({
            "id": f"{set_name}:{meta.get('id', i)}",
            "set": set_name,
            "subset": subset,
            "source": f"huggingface.co/datasets/{KEV_SUITES} {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} (src={src0}) @{KEV_SUITES_REV[:12]}",
            "license": kev_source_rule(src0)[1] if filter_sources else "Apache-2.0 repo; each upstream dataset has its own license (see kev docs/model-cards); evaluation only",
            "family": str(family),
            "state": r["state"],
            "questions": qs,
            "label": labels,
        })
    return rows, dropped


# ------------------------------------------------------------------ Jevals suite 0.1.0

def js_stringify(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def jevals_state(task_id: str, row: dict) -> dict:
    """The state object Jevals sends, recovered by matching every item's state_sha256 (SHA-256 of
    JSON.stringify(state)): banking77 {message}, pubmedqa {question, context: [passages]},
    helpsteer2 {prompt, response}."""
    if task_id == "banking77":
        return {"message": row["text"]}
    if task_id == "pubmedqa":
        return {"question": row["question"], "context": list(row["context"]["contexts"])}
    if task_id == "helpsteer2":
        return {"prompt": row["prompt"], "response": row["response"]}
    raise ValueError(task_id)


def jevals_rows(task_file, raw_dir: str, limit=None):
    from datasets import load_dataset
    task = json.load(open(task_file))
    ds = load_dataset(task["dataset"], task["config"], split=task["split"], revision=task["hf_revision"],
                      token=hf_token(), cache_dir=os.path.join(raw_dir, "hf-datasets"))
    options = task["options"]
    rows, sha_ok = [], 0
    for item in capped(task["items"], limit):
        row = ds[item["row_idx"]]
        state = jevals_state(task["id"], row)
        if sha256_text(js_stringify(state)) == item["state_sha256"]:
            sha_ok += 1
        prim = task["primitive"]
        if prim == "noul":
            q = {"type": "noul", "instructions": task["instructions"], "criteria": task["criteria"]}
            label = options[item["target"]] == "yes"
        elif prim == "choice":
            q = {"type": "choice", "instructions": task["instructions"], "criteria": task["criteria"]}
            label = options[item["target"]]
        else:
            q = {"type": "score", "instructions": task["instructions"], "criteria": task["criteria"]}
            label = int(item["target"])
        rows.append({
            "id": f"jevals-{task['id']}:{item['item_id']}",
            "set": f"jevals-{task['id']}",
            "subset": task["id"],
            "source": f"{task['source_url']} config={task['config']} split={task['split']} rev={task['hf_revision'][:12]} row={item['row_idx']}",
            "license": task["license"],
            "family": item["item_id"],
            "state": state,
            "questions": {"decision": q},
            "label": {"decision": label},
        })
    return rows, sha_ok, task


# ------------------------------------------------------------------ typed-decisions

def typed_decisions_rows(split: str, raw_dir: str, limit=None):
    from datasets import load_dataset
    kwargs = {"split": split, "token": hf_token(), "cache_dir": os.path.join(raw_dir, "hf-datasets")}
    try:
        ds = load_dataset(TYPED_DECISIONS, "all", revision=TYPED_DECISIONS_REV, **kwargs)
        rev = TYPED_DECISIONS_REV
    except Exception as e:  # noqa: BLE001
        log(f"typed-decisions pinned revision failed ({e}); falling back to main")
        ds = load_dataset(TYPED_DECISIONS, "all", **kwargs)
        rev = "main"
    rows = []
    for n, r in enumerate(ds):
        if limit and n >= limit:
            break
        state = json.loads(r["state"])
        qs = {qid: wire_question(q) for qid, q in json.loads(r["questions"]).items()}
        gold = json.loads(r["gold"])
        labels, targets = {}, {}
        for qid, q in qs.items():
            g = gold[qid]
            probs = {str(k): float(v) for k, v in g["probabilities"].items()}
            keys = list(q["criteria"]) if q["type"] == "choice" else (["true", "false"] if q["type"] == "noul"
                                                                     else [str(i) for i in range(len(q["criteria"]))])
            target = {k: probs.get(k, 0.0) for k in keys}
            s = sum(target.values())
            target = {k: v / s for k, v in target.items()} if s > 0 else target
            best = max(keys, key=lambda k: target[k])
            labels[qid] = (best == "true") if q["type"] == "noul" else (int(best) if q["type"] == "score" else best)
            targets[qid] = {k: round(v, 6) for k, v in target.items()}
        rows.append({
            "id": f"typed-decisions-{split}:{r['id']}",
            "set": f"typed-decisions-{split}",
            "subset": r["workflow"],
            "source": f"huggingface.co/datasets/{TYPED_DECISIONS} config=all split={split} rev={rev[:12]}",
            "license": "Apache-2.0 (synthetic; gold = mean of three teacher samples, label = its argmax)",
            "family": r["id"],
            "state": state,
            "questions": qs,
            "label": labels,
            "target": targets,
        })
    return rows


# ------------------------------------------------------------------ v1 pool sources (HelpSteer2 train, SummEval)

HELPSTEER2 = "nvidia/HelpSteer2"
HELPSTEER2_REV = "990b2711a36180dd19d9c94b8627844866f8982a"   # the revision jevals/helpsteer2.json pins
HELPSTEER2_LICENSE = "CC BY-4.0 (nvidia/HelpSteer2)"
SUMMEVAL = "mteb/summeval"
SUMMEVAL_LICENSE = "MIT (mteb/summeval: the file Nimble's public_benchmarks converter reads)"
SUMMEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")

V1_SETS = ("helpsteer2-train", "summeval-train")
POOL_SCORE_CAP = float(os.environ.get("JEB_SCORE_CAP", "0.45"))  # score questions as a share of the whole pool
SUMMEVAL_SHARE = 0.40        # of the new score budget; HelpSteer2 takes the rest
HS2_EXTRA_FRACTION = 0.15    # share of the used HelpSteer2 rows that also get the other four attributes
V0_POOL_BASE = {"questions": 11576, "score": 2848}  # the v0 pool, used only when pool.jsonl is not there yet

# HelpSteer2's other four attributes (the helpfulness question's wording comes from the Jevals task
# file, so it is that task's shape by construction). Levels follow HelpSteer2's annotation guide:
# correctness and coherence are quality scales, complexity and verbosity describe the response.
HS2_ATTRIBUTES = {
    "correctness": {
        "instructions": "How correct and complete is the response? Count both errors and pertinent facts left out.",
        "criteria": [
            "Completely incorrect: what it states is wrong, or it answers something that was not asked.",
            "Mostly incorrect: some relevant material, but the errors and omissions outweigh it.",
            "Partially correct: the main claims are right, but important facts are wrong or missing.",
            "Mostly correct: accurate, with a minor error or a small omission.",
            "Completely correct: every pertinent fact is there and nothing stated is wrong.",
        ]},
    "coherence": {
        "instructions": "How clear and self-consistent is the response? Judge the expression, not whether it is right.",
        "criteria": [
            "Incoherent: contradicts itself or cannot be followed.",
            "Mostly incoherent: the thread is hard to follow, with contradictions or abrupt jumps.",
            "Partially coherent: understandable, but the structure or the style gets in the way.",
            "Mostly coherent: clear and consistent, with a small lapse in flow or style.",
            "Perfectly coherent: consistent, well organised and easy to follow throughout.",
        ]},
    "complexity": {
        "instructions": "How much expertise does writing this response take? This describes the response; it is not a judgement of it.",
        "criteria": [
            "Basic: simple language that anyone who can read could have written.",
            "Simple: everyday language written with a little care; no domain knowledge needed.",
            "Intermediate: the language and content of someone with some schooling in the topic.",
            "Advanced: needs real domain knowledge or a specialist vocabulary.",
            "Expert: only someone with deep expertise in the field could have written it.",
        ]},
    "verbosity": {
        "instructions": "How long is the response for what was asked? This describes the response; it is not a judgement of it.",
        "criteria": [
            "Terse: as short as it can be, or shorter than the request needs.",
            "Brief: shorter than average for the request, with little elaboration.",
            "Average: about as long as the request calls for.",
            "Full: longer than needed, with extra detail or some repetition.",
            "Verbose: much longer than the request needs, padded or repetitive.",
        ]},
}

# SummEval. consistency and relevance are copied verbatim from Nimble's converter at the pinned
# commit (nimble/datasets/public_sources/summeval.py), so a training question has the same shape as
# the reported eval question; coherence and fluency follow SummEval's own definitions in that style.
SUMMEVAL_QUESTIONS = {
    "consistency": {
        "instructions": ("How factually consistent is the summary with the article? Every statement in the"
                         " summary should be supported by the article."),
        "criteria": [
            "Multiple statements contradict or are absent from the article.",
            "At least one clear unsupported or contradicted statement.",
            "Mostly supported, with a minor unsupported detail.",
            "Supported, with at most a small imprecision.",
            "Every statement is supported by the article.",
        ]},
    "relevance": {
        "instructions": ("How well does the summary capture the important content of the article, without"
                         " unimportant or redundant material?"),
        "criteria": [
            "Misses the main points or is mostly about minor details.",
            "Captures some key content but omits important points or includes much that is unimportant.",
            "Captures the main points with noticeable omissions or filler.",
            "Captures the main points with minor omissions.",
            "Captures all key content and only key content.",
        ]},
    "coherence": {
        "instructions": ("How coherent is the summary? Judge whether its sentences build one well organised"
                         " body of information rather than a heap of related facts."),
        "criteria": [
            "A heap of disconnected facts, in no order that helps the reader.",
            "Mostly disordered: some sentences relate, but it does not read as a whole.",
            "Partly organised: readable, with noticeable jumps or repetition.",
            "Well organised, with a small lapse in order or connection.",
            "Fully coherent: the sentences build one well organised body of information.",
        ]},
    "fluency": {
        "instructions": ("How fluent are the summary's sentences? Judge grammar, word choice and formatting,"
                         " one sentence at a time."),
        "criteria": [
            "Many sentences are ungrammatical or unreadable.",
            "Frequent grammatical or formatting errors that slow the reader down.",
            "Understandable throughout, with several awkward or malformed sentences.",
            "Fluent, with a single awkward sentence or a minor error.",
            "Every sentence is well formed and reads naturally.",
        ]},
}


def score_question(instructions: str, criteria: list) -> dict:
    return wire_question({"type": "score", "instructions": instructions, "criteria": list(criteria)})


def state_hash(state: dict) -> str:
    """The hash Jevals records for an item: SHA-256 of JSON.stringify(state)."""
    return sha256_text(js_stringify(state))


def assert_no_overlap(name: str, keys, forbidden: dict) -> int:
    """The v1 invariant: nothing a reported eval set uses may be in the training pool. The selection
    already drops these, so this firing means the filter is broken. Raises AssertionError."""
    keys = list(keys)
    hits = sorted({f"{k} -> {forbidden[k]}" for k in keys if k in forbidden})
    assert not hits, f"{name}: {len(hits)} training items overlap a reported eval set, e.g. {hits[:3]}"
    return len(keys)


def pool_base_counts(out: str) -> dict:
    """Questions and score questions already in the pool, ignoring the two v1 sources."""
    path = os.path.join(out, "pool.jsonl")
    q = s = 0
    if os.path.exists(path):
        for r in read_jsonl(path):
            if r.get("set") in V1_SETS:
                continue
            for quest in r["questions"].values():
                q += 1
                s += quest["type"] == "score"
    if q == 0:   # the pool stage has not run into this directory yet (or holds only v1 rows)
        log(f"no base pool in {path}: sizing the v1 sources against the recorded v0 pool {V0_POOL_BASE}")
        return dict(V0_POOL_BASE)
    return {"questions": q, "score": s}


def score_budget(out: str, limit) -> tuple:
    """How many new score questions keep score questions at or under POOL_SCORE_CAP of the whole
    pool: (cap*Q - S) / (1 - cap) for the pool's current Q questions of which S are scores."""
    if limit:
        return None, {"mode": f"--limit {limit}: the per-source record cap replaces the pool budget"}
    base = pool_base_counts(out)
    n = max(0, int((POOL_SCORE_CAP * base["questions"] - base["score"]) / (1 - POOL_SCORE_CAP)))
    note = {"base_pool": base, "score_cap": POOL_SCORE_CAP, "new_score_budget": n,
            "split": {"summeval": SUMMEVAL_SHARE, "helpsteer2": round(1 - SUMMEVAL_SHARE, 2)},
            "pool_questions_at_budget": base["questions"] + n}
    return n, note


def pool_write(out: str, set_name: str, rows: list, manifest: dict) -> dict:
    """Replace this set's rows in pool.jsonl and refresh the manifest's pool entry. Idempotent:
    re-running one source does not append a second copy."""
    path = os.path.join(out, "pool.jsonl")
    kept = [r for r in read_jsonl(path) if r.get("set") != set_name] if os.path.exists(path) else []
    allrows = kept + rows
    write_jsonl(path, allrows)
    summary = summarize(allrows)
    manifest["files"]["pool.jsonl"] = {"sha256": sha256_file(path), **summary,
                                       "sets": dict(collections.Counter(r["set"] for r in allrows)),
                                       "score_share": round(summary["types"].get("score", 0) / max(1, summary["questions"]), 4),
                                       **{k: v for k, v in (manifest["files"].get("pool.jsonl") or {}).items() if k == "upstream"}}
    return manifest["files"]["pool.jsonl"]


# ------------------------------------------------------------------ HelpSteer2 train split

def helpsteer2_forbidden(task: dict, val_path: str, repo: str) -> tuple:
    """Every (prompt, response) a reported eval set can draw, by state hash: the Jevals items, the
    ids in Nimble's helpsteer2 manifest, and the whole validation split both of them sample from
    (the superset, so a train row that duplicates any validation row is caught too)."""
    forbidden, prov = {}, {}
    for item in task["items"]:
        forbidden[item["state_sha256"]] = f"jevals-helpsteer2:{item['item_id']}"
    prov["jevals_items"] = len(task["items"])
    val = list(read_jsonl(val_path))
    manifest_path = os.path.join(repo, "docs", "assets", "public-benchmarks", "subsets", "helpsteer2-manifest.json") if repo else ""
    ids, resolved = [], 0
    if manifest_path and os.path.exists(manifest_path):
        ids = json.load(open(manifest_path)).get("ids") or []
        for ident in ids:
            try:  # nimble ids are helpsteer2-<row position in the exported validation file>
                row = val[int(str(ident).rsplit("-", 1)[-1])]
            except (ValueError, IndexError):
                continue
            forbidden.setdefault(state_hash({"prompt": row["prompt"], "response": row["response"]}),
                                 f"nimble-public__helpsteer2:{ident}")
            resolved += 1
    prov["nimble_manifest_ids"] = len(ids)
    prov["nimble_manifest_ids_resolved"] = resolved
    for row in val:
        forbidden.setdefault(state_hash({"prompt": row["prompt"], "response": row["response"]}), "helpsteer2-validation-split")
    prov["validation_rows"] = len(val)
    prov["forbidden_pairs"] = len(forbidden)
    return forbidden, prov


def helpsteer2_train_rows(task: dict, train_path: str, forbidden: dict, budget, limit) -> tuple:
    """Helpfulness for every used row in the Jevals task's shape, plus the other four attributes on
    a subset. Rows are drawn round robin over the helpfulness level (HelpSteer2 leans high) in a
    stable hash order, so the draw is deterministic and does not follow the file order."""
    stats = collections.Counter()
    usable = []
    for idx, row in enumerate(read_jsonl(train_path)):
        stats["rows"] += 1
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("prompt", "response")) or \
           not all(type(row.get(a)) is int and 0 <= row[a] <= 4 for a in ("helpfulness", *HS2_ATTRIBUTES)):
            stats["dropped_malformed"] += 1
            continue
        h = state_hash({"prompt": row["prompt"], "response": row["response"]})
        if h in forbidden:
            stats["dropped_eval_overlap"] += 1
            continue
        usable.append((idx, row, h))
    buckets = collections.defaultdict(list)
    for idx, row, h in usable:
        buckets[row["helpfulness"]].append((h, idx, row))
    for b in buckets.values():
        b.sort(key=lambda t: t[0])
    if limit:
        n_rows = limit
    else:
        n_rows = max(0, int(round(budget / (1 + 4 * HS2_EXTRA_FRACTION))))
    order, cursor = [], 0
    while len(order) < n_rows and any(cursor < len(b) for b in buckets.values()):
        for level in sorted(buckets):
            if cursor < len(buckets[level]) and len(order) < n_rows:
                order.append(buckets[level][cursor])
        cursor += 1
    n_extra = min(len(order), int(round(HS2_EXTRA_FRACTION * len(order))) if not limit else max(1, len(order) // 4))
    rows = []
    question_of = {"helpfulness": score_question(task["instructions"], task["criteria"])}
    for attr, spec in HS2_ATTRIBUTES.items():
        question_of[attr] = score_question(spec["instructions"], spec["criteria"])
    for n, (h, idx, row) in enumerate(order):
        attrs = ["helpfulness"] + (list(HS2_ATTRIBUTES) if n < n_extra else [])
        for attr in attrs:
            rows.append({
                "id": f"helpsteer2-train:{idx}-{attr}",
                "set": "helpsteer2-train",
                "subset": f"helpsteer2-{attr}",
                "source": f"huggingface.co/datasets/{HELPSTEER2} split=train rev={HELPSTEER2_REV[:12]} row={idx}",
                "license": HELPSTEER2_LICENSE,
                "family": "helpsteer2-" + sha256_text(row["prompt"])[:16],  # both responses to a prompt stay together
                "state": {"prompt": row["prompt"], "response": row["response"]},
                "questions": {"decision": question_of[attr]},
                "label": {"decision": int(row[attr])},
            })
        stats["extra_attribute_rows"] += n < n_extra
    stats["used_rows"] = len(order)
    stats["usable_rows"] = len(usable)
    return rows, stats


# ------------------------------------------------------------------ SummEval

def summeval_excluded(repo: str) -> tuple:
    """What Nimble's two reported summeval subsets use: their item ids, the (article, summary index)
    pairs behind them and those articles. Nimble draws whole articles (one family per article), so
    the whole article is held out of training, not just the rated dimension."""
    ids, items, articles = {}, set(), set()
    for sub in ("consistency", "relevance"):
        path = os.path.join(repo, "docs", "assets", "public-benchmarks", "subsets", f"summeval-{sub}-manifest.json")
        man = json.load(open(path))
        ids[sub] = list(man.get("ids") or [])
        if not ids[sub]:
            raise RuntimeError(f"nimble summeval-{sub} manifest has no ids: cannot prove the exclusion")
        prefix = f"summeval-{sub}-"
        for ident in ids[sub]:
            body = ident[len(prefix):] if ident.startswith(prefix) else ident
            article, _, index = body.rpartition("-")
            items.add((article, int(index)))
            articles.add(article)
    return ids, items, articles


def summeval_train_rows(raw_path: str, ids: dict, items: set, articles: set, budget, limit) -> tuple:
    """One record per (article, summary, dimension), whole articles at a time so an article is never
    split across train and calib (split_pool.py keys on the family, which is the article id)."""
    stats = collections.Counter()
    by_article = {}
    for raw in read_jsonl(raw_path):
        article, text, summaries = raw["id"], raw["text"], raw["machine_summaries"]
        stats["articles"] += 1
        stats["items"] += len(summaries)
        if article in articles:
            stats["dropped_eval_articles"] += 1
            stats["dropped_eval_items"] += len(summaries)
            continue
        by_article[article] = raw
    rows, all_ids = [], set()
    for article in sorted(by_article):
        raw = by_article[article]
        block = []
        for index, summary in enumerate(raw["machine_summaries"]):
            if (article, index) in items:      # belt and braces: the article filter already dropped these
                stats["dropped_eval_items"] += 1
                continue
            for dim in SUMMEVAL_DIMENSIONS:
                mean = raw[dim][index]
                level = min(4, max(0, math.floor(float(mean) + 0.5) - 1))   # Nimble's rounding, half up
                block.append({
                    "id": f"summeval-train:{dim}-{article}-{index}",
                    "set": "summeval-train",
                    "subset": f"summeval-{dim}",
                    "source": f"huggingface.co/datasets/{SUMMEVAL} split=test article={article} summary={index} "
                              f"dimension={dim} (the file nimble public_benchmarks reads @{NIMBLE_COMMIT[:12]})",
                    "license": SUMMEVAL_LICENSE,
                    "family": article,        # every summary of one article, every dimension, on one side
                    "state": {"article": raw["text"], "summary": summary},
                    "questions": {"decision": score_question(SUMMEVAL_QUESTIONS[dim]["instructions"],
                                                             SUMMEVAL_QUESTIONS[dim]["criteria"])},
                    "label": {"decision": level},
                })
        if limit:
            rows += block
            if len(rows) >= limit:
                rows = rows[:limit]
                break
        else:
            if rows and len(rows) + len(block) > budget:
                break
            rows += block
            stats["used_articles"] += 1
    stats["used_questions"] = len(rows)
    # the (article, summary) pairs actually written, whatever the budget or the cap cut short
    all_ids = {(r["family"], int(r["id"].rsplit("-", 1)[-1])) for r in rows}
    stats["used_items"] = len(all_ids)
    return rows, stats, all_ids


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="data directory: pool.jsonl, test/*.jsonl, manifest.json")
    ap.add_argument("--raw", required=True, help="download cache (raw upstream files, HF caches, the nimble repo)")
    ap.add_argument("--jevals", required=True, help="directory holding the three Jevals task files")
    ap.add_argument("--limit", type=int, default=0, help="cap every source at N records (smoke test)")
    ap.add_argument("--only", default="", help="comma list of stages to run: pool,helpsteer2-train,summeval-train,typed-test,kev-test,nimble-eval,nimble-public,jevals (default all)")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    ap.add_argument("--nimble-subsets", default="", help="comma list of Nimble public subsets to build (default all 13)")
    args = ap.parse_args()
    out, raw, limit = args.out, args.raw, args.limit or None
    os.makedirs(os.path.join(out, "test"), exist_ok=True)
    os.makedirs(raw, exist_ok=True)
    stages = ["pool", "helpsteer2-train", "summeval-train", "typed-test", "kev-test", "nimble-eval", "nimble-public", "jevals"]
    wanted = [s for s in (args.only.split(",") if args.only else stages) if s and s not in args.skip.split(",")]
    manifest_path = os.path.join(out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest.setdefault("files", {})
    manifest.setdefault("failures", {})
    manifest.setdefault("notes", [])
    manifest["limit"] = limit
    manifest["pins"] = {"typed-decisions": TYPED_DECISIONS_REV, "kev-suites": KEV_SUITES_REV, "nimble": NIMBLE_COMMIT}
    manifest["licenses"] = {"typed-decisions-train": "Apache-2.0", "kev-v7-train": dict(KEV_SOURCES),
                            "nimble-train (excluded)": NIMBLE_LICENSE}
    tests = {}

    def stage(name, fn):
        if name not in wanted:
            return
        log(f"== {name}")
        try:
            fn()
            manifest["failures"].pop(name, None)
        except Exception as e:  # noqa: BLE001
            manifest["failures"][name] = f"{type(e).__name__}: {e}"
            log(f"FAILED {name}: {e}")
            traceback.print_exc()
        json.dump(manifest, open(manifest_path, "w"), indent=1)

    def do_pool():
        pool = typed_decisions_rows("train", raw, limit)
        path = hf_file(KEV_SUITES, KEV_FILES["v7-train"][0], KEV_SUITES_REV, raw)
        actual = sha256_file(path)
        if actual != KEV_FILES["v7-train"][1]:
            manifest["notes"].append(f"kev v7 train.jsonl sha256 {actual[:12]} differs from the v0 record {KEV_FILES['v7-train'][1][:12]}")
        kev, dropped = kev_rows(path, "kev-v7-train", filter_sources=True)
        pool += capped(kev, limit)
        kept = [r for r in read_jsonl(f"{out}/pool.jsonl") if r.get("set") in V1_SETS] if os.path.exists(f"{out}/pool.jsonl") else []
        pool += kept   # a rebuilt base pool keeps the v1 sources that were already written
        write_jsonl(f"{out}/pool.jsonl", pool)
        summary = summarize(pool)
        manifest["files"]["pool.jsonl"] = {"sha256": sha256_file(f"{out}/pool.jsonl"), **summary,
                                           "sets": dict(collections.Counter(r["set"] for r in pool)),
                                           "score_share": round(summary["types"].get("score", 0) / max(1, summary["questions"]), 4),
                                           "upstream": {"kev-suites/v7/decision-v7/train.jsonl": actual}}
        manifest["exclusions"] = {"kev-v7-train": {src: {"dropped_questions": n, "why": kev_source_rule(src)[1]}
                                                   for src, n in sorted(dropped.items())},
                                  "nimble-train": {"why": NIMBLE_LICENSE}}
        log(f"pool: {summarize(pool)}; kev dropped {dict(dropped)}")

    def do_helpsteer2_train():
        """HelpSteer2 train split: helpfulness in the Jevals task's shape for every used row, the
        other four attributes on a subset, nothing a reported eval set draws from."""
        task_file = os.path.join(args.jevals, "helpsteer2.json")
        task = json.load(open(task_file))
        if task["dataset"] != HELPSTEER2 or task["primitive"] != "score" or len(task["criteria"]) != 5 \
                or task["options"] != [str(i) for i in range(5)]:
            raise RuntimeError("jevals/helpsteer2.json is not the five-level HelpSteer2 score task")
        val = hf_to_jsonl(HELPSTEER2, None, "validation", os.path.join(raw, "helpsteer2", "validation.jsonl"),
                          revision=HELPSTEER2_REV, raw_dir=raw)
        train_path = hf_to_jsonl(HELPSTEER2, None, "train", os.path.join(raw, "helpsteer2", "train.jsonl"),
                                 revision=HELPSTEER2_REV, raw_dir=raw)
        try:
            repo = fetch_nimble_repo(raw)
        except Exception as e:  # noqa: BLE001
            repo = ""
            log(f"nimble repo unavailable ({e}); the exclusion falls back to the Jevals items and the whole validation split")
        forbidden, prov = helpsteer2_forbidden(task, val, repo)
        budget, bnote = score_budget(out, limit)
        share = None if budget is None else int(budget * (1 - SUMMEVAL_SHARE))
        rows, stats = helpsteer2_train_rows(task, train_path, forbidden, share, limit)
        assert_no_overlap("helpsteer2-train", [state_hash(r["state"]) for r in rows], forbidden)
        entry = pool_write(out, "helpsteer2-train", rows, manifest)
        manifest.setdefault("sources", {})["helpsteer2-train"] = {
            "dataset": HELPSTEER2, "split": "train", "revision": HELPSTEER2_REV, "license": HELPSTEER2_LICENSE,
            "question_shape": {"helpfulness": "jevals/helpsteer2.json verbatim: instructions, five levels",
                               "other_attributes": sorted(HS2_ATTRIBUTES)},
            "jevals_task_file_sha256": sha256_file(task_file),
            "train_export_sha256": sha256_file(train_path), "validation_export_sha256": sha256_file(val),
            "budget": bnote, "budget_questions": share, "counts": summarize(rows),
            "by_question": dict(collections.Counter(r["subset"] for r in rows)),
            "labels": dict(collections.Counter(r["label"]["decision"] for r in rows if r["subset"] == "helpsteer2-helpfulness")),
            "exclusions": {**prov, **dict(stats)},
        }
        log(f"helpsteer2-train: {len(rows)} score questions over {stats['used_rows']} rows "
            f"({stats['extra_attribute_rows']} rows with all five attributes), "
            f"{stats['dropped_eval_overlap']} rows dropped as eval overlap; "
            f"pool now {entry['questions']} questions, score share {entry['score_share']}")

    def do_summeval_train():
        """SummEval: all four dimensions from the expert mean, whole articles, none of them an
        article Nimble's reported summeval-consistency or summeval-relevance subset uses."""
        repo = fetch_nimble_repo(raw)
        ids, items, articles = summeval_excluded(repo)
        nimble_ids = {ident: f"nimble-public__summeval-{sub}" for sub, lst in ids.items() for ident in lst}
        raw_path = hf_to_jsonl(SUMMEVAL, None, "test", os.path.join(raw, "summeval", "test.jsonl"), raw_dir=raw)
        budget, bnote = score_budget(out, limit)
        share = None if budget is None else int(budget * SUMMEVAL_SHARE)
        rows, stats, produced = summeval_train_rows(raw_path, ids, items, articles, share, limit)
        assert_no_overlap("summeval-train", [f"summeval-{sub}-{a}-{i}" for (a, i) in produced for sub in ids], nimble_ids)
        assert not {a for a, _ in produced} & articles, "summeval-train: an article Nimble reports on reached the pool"
        entry = pool_write(out, "summeval-train", rows, manifest)
        man = json.load(open(os.path.join(repo, "docs", "assets", "public-benchmarks", "subsets", "summeval-consistency-manifest.json")))
        raw_sha = sha256_file(raw_path)
        manifest.setdefault("sources", {})["summeval-train"] = {
            "dataset": SUMMEVAL, "split": "test", "nimble_commit": NIMBLE_COMMIT, "license": SUMMEVAL_LICENSE,
            "dimensions": list(SUMMEVAL_DIMENSIONS),
            "question_shape": {"consistency, relevance": f"nimble public_sources/summeval.py verbatim @{NIMBLE_COMMIT[:12]}",
                               "coherence, fluency": "SummEval's own definitions in the same five-level style"},
            "target": "the three-expert mean rounded half up to a level 1-5, stored as index 0-4 (Nimble's rule)",
            "source_sha256": raw_sha, "nimble_recorded_source_sha256": man.get("source_sha256"),
            "matches_nimble_source": raw_sha == man.get("source_sha256"),
            "budget": bnote, "budget_questions": share, "counts": summarize(rows),
            "by_question": dict(collections.Counter(r["subset"] for r in rows)),
            "split_rule": "family = article id, so split_pool.py keeps every summary of an article on one side",
            "exclusions": {"nimble_manifest_ids": {k: len(v) for k, v in ids.items()},
                           "excluded_articles": len(articles), "excluded_items": len(items), **dict(stats)},
        }
        log(f"summeval-train: {len(rows)} score questions over {stats['used_articles']} articles "
            f"(of {stats['articles']}), {stats['dropped_eval_articles']} articles held out for "
            f"nimble-public__summeval; pool now {entry['questions']} questions, score share {entry['score_share']}")

    def do_typed_test():
        tests["typed-decisions-test"] = typed_decisions_rows("test", raw, limit)

    def do_kev_test():
        for key, set_name in (("v7-test", "kev-decision-v7__test"), ("transfer-v4-test", "kev-transfer-v4__test")):
            path = hf_file(KEV_SUITES, KEV_FILES[key][0], KEV_SUITES_REV, raw)
            actual = sha256_file(path)
            if actual != KEV_FILES[key][1]:
                manifest["notes"].append(f"kev {key} sha256 {actual[:12]} differs from the v0 record {KEV_FILES[key][1][:12]}")
            rows, _ = kev_rows(path, set_name, filter_sources=False)
            tests[set_name] = capped(rows, limit)

    def do_nimble_eval():
        repo = fetch_nimble_repo(raw)
        tests["nimble-eval"] = capped(nimble_rows(os.path.join(repo, "data", "eval.jsonl"), "nimble-eval", None, NIMBLE_LICENSE, NIMBLE_COMMIT), limit)

    def do_nimble_public():
        repo = fetch_nimble_repo(raw)
        build_dir = os.path.join(out, "nimble-public-build")
        wanted_subsets = [x for x in args.nimble_subsets.split(",") if x]
        for spec in NIMBLE_PUBLIC:
            name = spec["name"]
            if wanted_subsets and name not in wanted_subsets:
                continue
            try:
                path, note = build_nimble_public(spec, repo, raw, build_dir)
                rows = nimble_rows(path, "nimble-public", name, spec["license"], NIMBLE_COMMIT)
                for r in rows:
                    r["source"] = f"{spec['url']} via github.com/{NIMBLE_REPO} public_benchmarks @{NIMBLE_COMMIT[:12]}"
                tests[f"nimble-public__{name}"] = capped(rows, limit)
                manifest["files"].setdefault("nimble_public_builds", {})[name] = note
                manifest["failures"].pop(f"nimble-public__{name}", None)
                log(f"nimble-public {name}: {len(rows)} records, {note['selection']}, identical={note['byte_identical_to_committed']}")
            except Exception as e:  # noqa: BLE001
                manifest["failures"][f"nimble-public__{name}"] = f"{type(e).__name__}: {e}"
                log(f"FAILED nimble-public {name}: {e}")

    def do_jevals():
        for t in ("pubmedqa", "banking77", "helpsteer2"):
            try:
                rows, sha_ok, task = jevals_rows(os.path.join(args.jevals, f"{t}.json"), raw, limit)
                tests[f"jevals-{t}"] = rows
                manifest["files"].setdefault("jevals", {})[t] = {
                    "suite": task["version"], "dataset": task["dataset"], "config": task["config"],
                    "split": task["split"], "hf_revision": task["hf_revision"], "seed": task["seed"],
                    "n_items": task["n_items"], "state_sha256_matches": sha_ok, "license": task["license"],
                    "task_file_sha256": sha256_file(os.path.join(args.jevals, f"{t}.json"))}
                manifest["failures"].pop(f"jevals-{t}", None)
                log(f"jevals {t}: {len(rows)} rows, {sha_ok} state hashes match")
            except Exception as e:  # noqa: BLE001
                manifest["failures"][f"jevals-{t}"] = f"{type(e).__name__}: {e}"
                log(f"FAILED jevals {t}: {e}")

    stage("pool", do_pool)
    stage("helpsteer2-train", do_helpsteer2_train)
    stage("summeval-train", do_summeval_train)
    stage("typed-test", do_typed_test)
    stage("kev-test", do_kev_test)
    stage("nimble-eval", do_nimble_eval)
    stage("nimble-public", do_nimble_public)
    stage("jevals", do_jevals)

    for set_name, rows in tests.items():
        fname = set_name + ".jsonl"
        path = f"{out}/test/{fname}"
        write_jsonl(path, rows)
        manifest["files"][f"test/{fname}"] = {"sha256": sha256_file(path), **summarize(rows),
                                              "license": rows[0]["license"] if rows else ""}
    for note in (
        "Labels live in the record-level `label` map; no question object carries a label (lint_data.py enforces it).",
        "Kev's boolq and mnli training rows come from the train splits; Nimble's public boolq (validation) and multinli (dev matched) "
        "subsets share those datasets, so those two test numbers are in-distribution by dataset, not by item.",
        "Banking77 is excluded from training so jevals-banking77 is a clean zero-shot number.",
        "Public data only: no private or customer material exists on this box.",
        "v1 pool sources: HelpSteer2 train (never validation, which Jevals and Nimble sample) and SummEval minus every "
        "article Nimble's summeval-consistency and summeval-relevance manifests use. Both exclusions are asserted in code.",
    ):
        if note not in manifest["notes"]:
            manifest["notes"].append(note)
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest["files"].items() if isinstance(v, dict) and "records" in v}, indent=1))
    if manifest["failures"]:
        print("FAILURES:", json.dumps(manifest["failures"], indent=1))
    required = [s for s in ("pool", "helpsteer2-train", "summeval-train", "typed-test") if s in wanted]
    missing = [s for s in required if s in manifest["failures"]]
    if missing:
        print("CONVERT_FAILED", missing)
        sys.exit(1)
    print("CONVERT_DONE")


if __name__ == "__main__":
    main()
