"""Build private A1/A3 ablations from hash-pinned legacy and v2.1 inputs.

No training text or labels belong in the repository. Caps count questions and
aggregate new task variants by upstream dataset, never by variant name.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

LEGACY_HASHES = {
    "train.jsonl": "2069602f17b53761f607c29e9cf0404aa16a301e9f12cf41bb0ab56eb5a72687",
    "calib.jsonl": "7da25d87eaf08ca58aa78cc96192d867865c0f3e09bd14bdf0163db8a3da0469",
}
V21_MANIFEST_HASH = "079617f615065a39549faa1b8676ce6a720a9efe8c31c04ebe39df024ca3d172"
REMOVED = {"boolq", "dbpedia14", "mnli", "summeval-coherence",
           "summeval-consistency", "summeval-fluency", "summeval-relevance"}
SEED = "item32-20261008"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Input hash mismatch: {path}")


def source_check(subset, manifest):
    info = manifest["source_manifests"].get(subset)
    if not info or not info.get("license") or not info.get("allowed_reason"):
        raise ValueError(f"Source missing license/approval: {subset}")
    ancestry = json.dumps([subset, info.get("source"), info.get("source_id"),
                          info.get("upstream_originals"), info.get("upstream", {})]).lower()
    for term in manifest["source_exclusions"]["source_substrings"]:
        if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", ancestry):
            raise ValueError(f"Blocked source ancestry: {subset}: {term}")
    return info


def root_source(subset, manifest):
    info = source_check(subset, manifest)
    # Legacy provenance includes row/config suffixes. Normalize to repository.
    identity = info["source_id"]
    match = re.search(r"(?:huggingface.co/datasets/)?([\w.-]+/[\w.-]+)", identity)
    if match:
        return match.group(1).casefold()
    return info["family"].casefold()


def state_key(row):
    # Compare across legacy/new provenance and IDs without consulting gold.
    return digest(" ".join(re.findall(r"\w+", json.dumps(row["state"],
                          sort_keys=True, ensure_ascii=False).casefold())))


def write_split(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"sha256": sha(path), "rows": len(rows),
            "questions": sum(len(r["questions"]) for r in rows)}


def build(legacy, v21, output, overlap_hits, tokenizer_path):
    check_hash(v21 / "manifest.json", V21_MANIFEST_HASH)
    manifest = json.loads((v21 / "manifest.json").read_text())
    for name in ("train.jsonl", "calib.jsonl"):
        check_hash(legacy / name, LEGACY_HASHES[name])
    check_hash(v21 / "train.jsonl", manifest["files"]["train.jsonl"]["sha256"])
    source_roots = {s: root_source(s, manifest) for s in manifest["source_manifests"]}
    hits = json.loads(overlap_hits.read_text())
    excluded_ids = {h["id"] for h in hits}
    old, old_sources = {}, set()
    before_overlap = Counter()
    removed_overlap = Counter()
    for split in ("train", "calib"):
        old[split] = []
        for r in read(legacy / (split + ".jsonl")):
            if r["subset"] in REMOVED:
                continue
            before_overlap[split] += len(r["questions"])
            if r["id"] in excluded_ids:
                removed_overlap[split] += len(r["questions"])
                continue
            old_sources.add(source_roots[r["subset"]])
            r["family_id"] = r["family"]
            old[split].append(r)
    if before_overlap["train"] != 10632:
        raise ValueError("A1 no longer isolates the expected four-source scrub")
    if {r["family_id"] for r in old["train"]} & {r["family_id"] for r in old["calib"]}:
        raise ValueError("Legacy family leakage")
    forbidden_states = {state_key(r) for rows in old.values() for r in rows}
    forbidden_families = {r["family_id"] for rows in old.values() for r in rows}
    candidates = defaultdict(list)
    excluded = Counter()
    for r in read(v21 / "train.jsonl"):
        source = source_roots[r["subset"]]
        if source in old_sources:
            excluded["existing_upstream_source"] += len(r["questions"])
            continue
        if state_key(r) in forbidden_states or r["family_id"] in forbidden_families:
            excluded["legacy_duplicate_or_calibration_family"] += len(r["questions"])
            continue
        candidates[source].append(r)
    # Check option-only prompts before selection. Long state can be truncated,
    # but long instructions/options cannot fit the fixed v2 sequence budget.
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    from jebadiah_prompt import Renderer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if sha(tokenizer_path / "tokenizer.json") != manifest["tokenizer_hashes"]["9b"]:
        raise ValueError("Tokenizer does not match the frozen 9B data manifest")
    eligibility_renderer = Renderer(tokenizer, max_tokens=1984)  # 64-token permutation reserve
    unrenderable = Counter()
    over_2048 = Counter()
    reserve_only = Counter()
    full_budget_renderer = Renderer(tokenizer, max_tokens=2048)
    for source, rows in list(candidates.items()):
        valid = []
        for r in rows:
            try:
                for question in r["questions"].values():
                    eligibility_renderer.render("", question)
            except ValueError as error:
                if "exceed max sequence length" not in str(error):
                    raise
                unrenderable[source] += len(r["questions"])
                try:
                    for question in r["questions"].values():
                        full_budget_renderer.render("", question)
                except ValueError:
                    over_2048[r["subset"]] += len(r["questions"])
                else:
                    reserve_only[r["subset"]] += len(r["questions"])
                continue
            valid.append(r)
        candidates[source] = valid
        print("eligibility", source, len(rows), "records checked", flush=True)
    legacy_count = sum(len(r["questions"]) for r in old["train"])
    # Maximal new budget at <=50%; 10% cap on each new upstream source.
    cap = (legacy_count * 2) // 10
    pools = {s: sorted(rs, key=lambda r: digest(SEED + ":" + r["id"]))
             for s, rs in candidates.items()}
    positions = Counter(); counts = Counter(); selected = []; n = 0
    # Equal-source round robin, deterministic and label-blind. Keep records whole.
    sources = sorted(pools, key=lambda s: digest(SEED + ":source:" + s))
    while n < legacy_count:
        progressed = False
        for s in sources:
            while positions[s] < len(pools[s]):
                r = pools[s][positions[s]]; positions[s] += 1
                q = len(r["questions"])
                if counts[s] + q > cap or n + q > legacy_count:
                    continue
                selected.append(r); counts[s] += q; n += q
                progressed = True
                break
        if not progressed:
            break
    total = legacy_count + n
    if n * 2 > total or any(c * 10 > total for c in counts.values()):
        raise ValueError("Actual final mixture violates caps")
    for name, train in (("a1", old["train"]), ("a3", old["train"] + selected)):
        dest = output / name; dest.mkdir(parents=True, exist_ok=True)
        families = {r["family_id"] for r in train}
        if families & {r["family_id"] for r in old["calib"]}:
            raise ValueError("Training/calibration families overlap")
        entries = {"train.jsonl": write_split(dest / "train.jsonl", train),
                   "calib.jsonl": write_split(dest / "calib.jsonl", old["calib"])}
        used = sorted({r["subset"] for r in train + old["calib"]})
        report = {"name": "item32-" + name, "private": True, "seed": SEED,
                  "files": entries, "legacy_input_hashes": LEGACY_HASHES,
                  "v21_manifest_sha256": V21_MANIFEST_HASH,
                  "removed_sources": sorted(REMOVED), "source_policy": manifest["source_exclusions"],
                  "overlap_hits_sha256": sha(overlap_hits),
                  "overlap_removed_questions": dict(removed_overlap),
                  "diagnostic_overlap_exception": "Lead authorized record-only removal for private diagnostics; shipping requires whole-family removal",
                  "source_manifests": {s: manifest["source_manifests"][s] for s in used},
                  "selection": "whole-record equal-upstream-source hash round robin; gold not consulted",
                  "new_source_question_counts": dict(counts) if name == "a3" else {},
                  "new_questions": n if name == "a3" else 0,
                  "new_fraction": n / total if name == "a3" else 0,
                  "candidate_exclusions": dict(excluded),
                  "unrenderable_new_questions_by_upstream": dict(unrenderable),
                  "over_2048_new_questions_by_subset": dict(over_2048),
                  "permutation_reserve_excluded_questions_by_subset": dict(reserve_only),
                  "option_only_prompt_budget": 1984,
                  "tokenizer_sha256": sha(tokenizer_path / "tokenizer.json"),
                  "source_variant_question_counts": dict(Counter({s: sum(len(r["questions"]) for r in train if r["subset"] == s) for s in used})),
                  "type_question_counts": dict(Counter(q["type"] for r in train for q in r["questions"].values()))}
        (dest / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, entries, "new questions", report["new_questions"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--v21", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="local pinned Qwen3.5-9B tokenizer directory")
    p.add_argument("--overlap-hits", type=Path, required=True, help="private full-suite scanner record-ID ledger")
    a = p.parse_args()
    build(a.legacy, a.v21, a.output, a.overlap_hits, a.tokenizer)
