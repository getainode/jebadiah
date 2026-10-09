"""Gated Corr2Cause rung 2: matched replacement in pinned private A3."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

from item33_skill_data import digest, sha, write_json, write_rows
from item32_ablation_mix import read, state_key
from item36_skill_mix import A3_HASHES, A3_REVISION
from item47_rung1_mix import A3_MANIFEST_SHA256, histograms
import item49_corr2cause as C

PRESENTATIONS = 21190
CALIBRATION = 512
CAP = 2119
REPLACEMENTS = 1000


def is_upstream(row):
    # Count descendants and inherited source-family metadata, not just a name
    # chosen by the new converter. All variants share this single source cap.
    return "corr2cause" in json.dumps({k: row.get(k) for k in (
        "source", "subset", "source_family", "provenance")}).lower()


def replace_slots(rows, candidates):
    before = histograms(rows)
    total = sum(len(r["questions"]) for r in rows)
    existing = sum(len(r["questions"]) for r in rows if is_upstream(r))
    limit = min(REPLACEMENTS, total // 10 - existing, CAP - existing)
    if limit <= 0:
        raise ValueError("No aggregate source cap headroom")
    pools = {}
    for kind in ("choice", "noul"):
        pools[kind] = sorted(((i, qid) for i, row in enumerate(rows)
                             for qid, q in row["questions"].items()
                             if row.get("area") == "knowledge" and q["type"] == kind),
                            key=lambda slot: digest(C.SEED + ":donor:" + rows[slot[0]]["id"] + ":" + slot[1]))
    n = min(limit, sum(map(len, pools.values())))
    if n == 0:
        raise ValueError("No knowledge binary donor slots")
    allocation = {k: min(n // 2, len(v)) for k, v in pools.items()}
    for kind in pools:
        allocation[kind] += min(n - sum(allocation.values()), len(pools[kind]) - allocation[kind])
    selected = {}
    for kind, number in allocation.items():
        incoming = sorted((r for r in candidates if r["questions"]["decision"]["type"] == kind),
                          key=lambda r: digest(C.SEED + ":replace:" + r["id"]))
        if len(incoming) < number:
            raise ValueError("Insufficient unique incoming questions")
        for slot, candidate in zip(pools[kind][:number], incoming[:number]):
            if candidate["area"] != "knowledge" or not is_upstream(candidate):
                raise ValueError("Unmatched incoming source/area")
            selected[slot] = copy.deepcopy(candidate)
    result = []; mapping = []
    for i, row in enumerate(rows):
        kept = copy.deepcopy(row)
        for qid, q in row["questions"].items():
            if (i, qid) not in selected:
                continue
            candidate = selected[(i, qid)]
            result.append(candidate)
            mapping.append({"original_id": row["id"], "question_id": qid,
                            "replacement_id": candidate["id"], "area": "knowledge",
                            "type": q["type"], "donor_source": row["source"]})
            for field in ("questions", "label", "target"):
                kept.get(field, {}).pop(qid, None)
        if kept["questions"]:
            result.append(kept)
    exposure = sum(len(r["questions"]) for r in result if is_upstream(r))
    if exposure > min(CAP, total // 10):
        raise ValueError("Aggregate upstream cap exceeded")
    if (sum(len(r["questions"]) for r in result) != total
            or histograms(result)["area_type"] != before["area_type"]):
        raise ValueError("Presentation count or area/type histogram changed")
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate incoming or original ID")
    return result, {"requested_replacements": REPLACEMENTS, "replaced_slots": n,
                    "existing_upstream_questions": existing, "upstream_questions": exposure,
                    "aggregate_cap_questions": min(CAP, total // 10),
                    "fraction_of_train": n / total, "selected_types": allocation,
                    "matched_donor_pool": {k: len(v) for k, v in pools.items()},
                    "selected_source_counts": C.counts(list(selected.values())),
                    "before": before, "after": histograms(result), "replacements": mapping}


def stream_hash(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    return h.hexdigest()


def validate_artifacts(raw, causal, licenses):
    manifest = json.loads((causal / "manifest.json").read_text())
    report = manifest["overlap_scan"]
    # Fail before reading the raw release or touching any output directory.
    if report.get("status") != "passed":
        raise ValueError("Full-source and converted Studio scan required before composition")
    if json.loads((causal / "overlap-scan-report.json").read_text()) != report:
        raise ValueError("Scan report mismatch")
    policy = C.validate_policy(raw, licenses)
    source = policy["sources"][C.SOURCE]
    if (manifest["generator_sha256"] != sha(Path(C.__file__))
            or manifest["license_manifest_sha256"] != sha(licenses)
            or manifest["source_revision"] != C.REVISION
            or manifest["source_files"] != source["complete_files"]
            or set(manifest["files"]) != {"train.jsonl", "diagnostic.jsonl"}):
        raise ValueError("Wrong frozen source/converter artifacts")
    splits = {}
    for s in ("train", "diagnostic"):
        if sha(causal / f"{s}.jsonl") != manifest["files"][f"{s}.jsonl"]["sha256"]:
            raise ValueError("Converted data hash mismatch")
        splits[s] = read(causal / f"{s}.jsonl")
    C.audit(splits)
    if (len(splits["train"]) != 6000 or len(splits["diagnostic"]) != 300
            or manifest["splits"] != {s: C.counts(rs) for s, rs in splits.items()}
            or C.counts(splits["train"])["types"] != {"choice": 3000, "noul": 3000}
            or C.counts(splits["diagnostic"])["types"] != {"choice": 150, "noul": 150}):
        raise ValueError("Wrong candidate counts")
    # Verify every candidate label, state and provenance against original train
    # bytes, rather than trusting a converter manifest or binary label alone.
    indexed = {r["provenance"]["row"]: r for rows in splits.values() for r in rows}
    verified = 0
    for i, row in enumerate(C.upstream_rows(raw / "train.csv")):
        if i in indexed:
            expected = C.convert(row, i, indexed[i]["questions"]["decision"]["type"])
            if expected != indexed[i]:
                raise ValueError("Candidate disagrees with original train row")
            verified += 1
    if verified != 6300:
        raise ValueError("Duplicate or invalid original row provenance")
    raw_count = sum(f["records"] for f in source["complete_files"].values())
    if (report.get("raw_files_sha256") != digest(json.dumps(source["complete_files"], sort_keys=True))
            or report.get("converted_sha256") != {s: sha(causal / f"{s}.jsonl") for s in splits}
            or report.get("scanned_records") != raw_count + 6300
            or report.get("source_level") is not True
            or report.get("protected_index_sha256") != C.ITEM25_INDEX_SHA256):
        raise ValueError("Incomplete or unbound full-source scan")
    for phase, number, rows in (("raw", raw_count, C.raw_scan_rows(raw, policy)),
                                ("converted", 6300, C.converted_scan_rows(splits))):
        stage = report.get(phase, {})
        if (stage.get("status") != "passed" or stage.get("candidates_sha256") != stream_hash(rows)
                or stage.get("scanned_records") != number or stage.get("retained_records") != number
                or any(stage.get(k) != 0 for k in ("removed_records", "direct_hits", "invalid_empty_state_records", "removed_families"))
                or stage.get("source_level") is not True
                or stage.get("protected_index_sha256") != C.ITEM25_INDEX_SHA256
                or stage.get("scanner_sha256") != sha(Path(C.__file__).with_name("item33_full_suite_scan.py"))):
            raise ValueError("Incomplete or unbound " + phase + " source scan")
    from jebadiah_prompt import PROMPT_SOURCE_SHA256
    contract = json.loads((Path(__file__).resolve().parents[1] / "results/runs/9b-chat-v1/adapter/prompt_contract.json").read_text())
    render = manifest["render_validation"]
    if (render.get("truncated") != 0 or render.get("max_seq_length") != 2048
            or render.get("prompt_budget") != 1984 or render.get("padding_reserve") != 64
            or render.get("checked_renders") != 9450
            or render.get("choice_orders") != ["canonical", "reversed"]
            or render.get("noul_order") != "true,false"
            or render.get("tokenizer_sha256") != C.TOKENIZER_SHA256
            or render.get("prompt_source_sha256") != PROMPT_SOURCE_SHA256
            or render.get("chat_template_sha256") != contract["chat_template_sha256"]
            or set(render.get("max_prompt_tokens_by_type", {})) != {"choice", "noul"}
            or any(not 0 < v <= 1984 for v in render["max_prompt_tokens_by_type"].values())):
        raise ValueError("Invalid render checks")
    return manifest, splits


def compose(raw, base, causal, output, licenses=C.POLICY):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty")
    for filename, expected in {**A3_HASHES, "manifest.json": A3_MANIFEST_SHA256}.items():
        if sha(base / filename) != expected:
            raise ValueError("Wrong pinned A3 input: " + filename)
    manifest, splits = validate_artifacts(raw, causal, licenses)
    original = read(base / "train.jsonl"); calib = read(base / "calib.jsonl")
    if (sum(len(r["questions"]) for r in original) != PRESENTATIONS
            or sum(len(r["questions"]) for r in calib) != CALIBRATION):
        raise ValueError("Wrong A3 presentation counts")
    forbidden = C.existing_families(base)
    if forbidden & {r["family_id"] for rows in splits.values() for r in rows}:
        raise ValueError("A3-exposed causal graph family")
    for key in (lambda r: r["id"], state_key):
        if {key(r) for r in original + calib} & {key(r) for rows in splits.values() for r in rows}:
            raise ValueError("A3 candidate collision")
    result, receipt = replace_slots(original, splits["train"])
    for key in (lambda r: r["id"], state_key):
        if {key(r) for r in result} & {key(r) for r in calib + splits["diagnostic"]}:
            raise ValueError("Held-out diagnostic or calibration leakage")
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "train.jsonl", result)
    shutil.copyfile(base / "calib.jsonl", output / "calib.jsonl")
    shutil.copyfile(causal / "diagnostic.jsonl", output / "causal-diagnostic.jsonl")
    shutil.copyfile(licenses, output / "source-licenses.json")
    (output / "NOTICE-Corr2Cause.txt").write_text(json.loads(licenses.read_text())["sources"][C.SOURCE]["notice_text"])
    write_json(output / "replacement-slots.json", receipt)
    names = ("train.jsonl", "calib.jsonl", "causal-diagnostic.jsonl", "replacement-slots.json", "source-licenses.json", "NOTICE-Corr2Cause.txt")
    report = {"name": "item49-rung2-a3", "private": True, "a3_revision": A3_REVISION,
              "a3_input_hashes": A3_HASHES, "a3_manifest_sha256": A3_MANIFEST_SHA256,
              "causal_manifest_sha256": sha(causal / "manifest.json"),
              "source_scan_sha256": sha(causal / "overlap-scan-report.json"),
              "slots": {k: v for k, v in receipt.items() if k != "replacements"},
              "questions": PRESENTATIONS, "calibration_questions": CALIBRATION,
              "calibration": "Original A3 bytes unchanged; causal diagnostic separate; temperatures unchanged",
              "base_policy": "Existing A3 private exceptions unchanged; no shipping clearance",
              "files": {n: {"sha256": sha(output / n)} for n in names}, "training_launched": False}
    write_json(output / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "base", "causal", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--licenses", type=Path, default=C.POLICY)
    args = parser.parse_args()
    print(json.dumps(compose(args.raw, args.base, args.causal, args.out, args.licenses), indent=2))
