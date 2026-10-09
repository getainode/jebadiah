"""Gated fixed-size, area/type-matched replacement in the immutable private A3."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
from item33_skill_data import digest, sha, write_json, write_rows
from item32_ablation_mix import read, state_key
from item36_skill_mix import A3_HASHES, A3_REVISION
import item47_evidence_worlds as E

A3_MANIFEST_SHA256 = "2269f3721576869e7be1df1d945259d1bd6a644557fc6bb0eae2138dff0ae7a4"
TRAIN_QUESTIONS = 21190
CALIB_QUESTIONS = 512
REPLACEMENTS = 1000
CAP = 2119


def provenance_source(row):
    # Preserve upstream name/revision, aggregate only per-record locators.
    return row["source"].split(" row=", 1)[0]


def histograms(rows):
    return {
        "area_type": dict(
            sorted(
                Counter(
                    f"{r.get('area','unassigned')}:{q['type']}"
                    for r in rows
                    for q in r["questions"].values()
                ).items()
            )
        ),
        "source_questions": dict(
            sorted(
                Counter(
                    provenance_source(r) for r in rows for q in r["questions"].values()
                ).items()
            )
        ),
        "type_questions": dict(
            sorted(
                Counter(
                    q["type"] for r in rows for q in r["questions"].values()
                ).items()
            )
        ),
    }


def replace_slots(rows, candidates):
    if any(
        "item47" in str(r.get(k, ""))
        for r in rows
        for k in ("source", "subset", "family_id")
    ):
        raise ValueError("Incoming upstream already exposed in A3; fresh rung required")
    pools = {
        (area, kind): []
        for area, kind in (("language", "choice"), ("language", "noul"))
    }
    for i, r in enumerate(rows):
        for qid, q in r["questions"].items():
            key = (r.get("area"), q["type"])
            if key in pools:
                pools[key].append((i, qid))
    presentations = sum(len(r["questions"]) for r in rows)
    limit = min(REPLACEMENTS, CAP, presentations // 10)
    selected = {}
    by_type = Counter()
    donors = []
    # 500 matched slots per type. Reduce only a type's exhausted donor pool.
    # Select siblings together across types, rotating status/rule before worlds.
    for area, kind in pools:
        slots = sorted(
            pools[(area, kind)],
            key=lambda p: digest("item47:donor:" + rows[p[0]]["id"] + ":" + p[1]),
        )
        n = min(limit // 2, len(slots))
        incoming = sorted(
            (
                r
                for r in candidates
                if r["area"] == area and r["questions"]["decision"]["type"] == kind
            ),
            key=lambda r: (
                int(r["group_id"].rsplit(":", 1)[1]),
                E.STATUSES.index(r["evidence_status"]),
                E.TRAIN_RULES.index(r["rule"]),
            ),
        )
        if len(incoming) < n:
            raise ValueError("Insufficient unique incoming questions")
        for slot, r in zip(slots[:n], incoming[:n]):
            selected[slot] = copy.deepcopy(r)
        by_type[kind] = n
        donors.extend(slots[:n])
    if len(selected) > min(CAP, presentations // 10):
        raise ValueError("Upstream exposure exceeds 10 percent cap")
    if not selected:
        raise ValueError("No matched donor slots")
    result = []
    mapping = []
    for i, row in enumerate(rows):
        kept = copy.deepcopy(row)
        for qid, q in row["questions"].items():
            if (i, qid) not in selected:
                continue
            incoming = selected[(i, qid)]
            if (row.get("area"), q["type"]) != (
                incoming["area"],
                incoming["questions"]["decision"]["type"],
            ):
                raise ValueError("Area/type mismatch")
            result.append(incoming)
            mapping.append(
                {
                    "original_id": row["id"],
                    "question_id": qid,
                    "replacement_id": incoming["id"],
                    "area": incoming["area"],
                    "type": q["type"],
                    "donor_source": row["source"],
                }
            )
            for field in ("questions", "label", "target"):
                kept.get(field, {}).pop(qid, None)
        if kept["questions"]:
            result.append(kept)
    if sum(len(r["questions"]) for r in result) != sum(
        len(r["questions"]) for r in rows
    ):
        raise ValueError("Question presentations changed")
    before = histograms(rows)
    after = histograms(result)
    if before["area_type"] != after["area_type"]:
        raise ValueError("Area/type histogram changed")
    incoming = [r for r in result if r["source"] == E.SOURCE]
    if len({r["id"] for r in incoming}) != len(incoming):
        raise ValueError("Duplicate incoming ID")
    receipt = {
        "requested_replacements": REPLACEMENTS,
        "replaced_slots": len(selected),
        "cap_questions": min(CAP, presentations // 10),
        "upstream_questions": len(incoming),
        "fraction_of_train": len(selected) / sum(len(r["questions"]) for r in rows),
        "matched_donor_pool": {f"{a}:{k}": len(v) for (a, k), v in pools.items()},
        "selected_types": dict(by_type),
        "selected_source_counts": E.counts(incoming),
        "donor_sources": dict(Counter(rows[i]["source"] for i, qid in donors)),
        "before": before,
        "after": after,
        "replacements": mapping,
    }
    return result, receipt


def validate_artifacts(evidence, licenses):
    manifest = json.loads((evidence / "manifest.json").read_text())
    report = manifest["overlap_scan"]
    if report["status"] != "passed":
        raise ValueError("Full-suite Studio scan required before composition")
    if json.loads((evidence / "overlap-scan-report.json").read_text()) != report:
        raise ValueError("Scan report mismatch")
    if manifest["generator_sha256"] != sha(Path(E.__file__)):
        raise ValueError("Wrong generator version")
    if manifest["license_manifest_sha256"] != sha(licenses):
        raise ValueError("Wrong license manifest")
    policy, source = E.validate_policy(licenses)
    if set(manifest["files"]) != {"train.jsonl", "diagnostic.jsonl"}:
        raise ValueError("Incomplete source files")
    for name, info in manifest["files"].items():
        if (
            sha(evidence / name) != info["sha256"]
            or info["sha256"]
            != source["complete_split_sha256"][name.removesuffix(".jsonl")]
        ):
            raise ValueError("Owned data hash mismatch")
    splits = {s: read(evidence / f"{s}.jsonl") for s in ("train", "diagnostic")}
    E.audit(splits)
    if manifest["splits"] != {s: E.counts(rs) for s, rs in splits.items()}:
        raise ValueError("Wrong source counts")
    scanned = E.scan_rows(splits)
    if (
        report.get("candidates_sha256") != E.rows_hash(scanned)
        or report.get("scanned_records") != 6300
        or report.get("protected_index_sha256") != E.ITEM25_INDEX_SHA256
        or report.get("removed_records") != 0
        or report.get("retained_records") != 6300
        or report.get("direct_hits") != 0
        or report.get("invalid_empty_state_records") != 0
        or report.get("removed_families") != 0
        or report.get("source_level") is not True
        or report.get("scanner_sha256")
        != sha(Path(E.__file__).with_name("item33_full_suite_scan.py"))
    ):
        raise ValueError("Incomplete or unbound source scan")
    render = manifest["render_validation"]
    from jebadiah_prompt import PROMPT_SOURCE_SHA256

    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "results/runs/9b-chat-v1/adapter/prompt_contract.json"
        ).read_text()
    )
    if (
        render.get("truncated") != 0
        or render.get("max_seq_length") != 2048
        or render.get("prompt_budget") != 1984
        or render.get("padding_reserve") != 64
        or render.get("checked_renders") != 18900
        or render.get("tokenizer_sha256") != E.TOKENIZER_SHA256
        or render.get("prompt_source_sha256") != PROMPT_SOURCE_SHA256
        or render.get("chat_template_sha256") != contract["chat_template_sha256"]
        or set(render.get("max_prompt_tokens_by_type", {})) != {"choice", "noul"}
        or any(not 0 < n <= 1984 for n in render["max_prompt_tokens_by_type"].values())
    ):
        raise ValueError("Render checks required")
    return manifest, splits


def compose(base, evidence, output, licenses):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty")
    for filename, expected in {
        **A3_HASHES,
        "manifest.json": A3_MANIFEST_SHA256,
    }.items():
        if sha(base / filename) != expected:
            raise ValueError("Wrong pinned A3 input: " + filename)
    manifest, splits = validate_artifacts(evidence, licenses)
    original = read(base / "train.jsonl")
    calib = read(base / "calib.jsonl")
    if (
        sum(len(r["questions"]) for r in original) != TRAIN_QUESTIONS
        or sum(len(r["questions"]) for r in calib) != CALIB_QUESTIONS
    ):
        raise ValueError("Wrong A3 presentation counts")
    owned = splits["train"]
    diagnostic = splits["diagnostic"]
    for key in (lambda r: r["id"], lambda r: r["family_id"], state_key):
        if {key(r) for r in original + calib} & {key(r) for r in owned + diagnostic}:
            raise ValueError("A3/owned collision")
        if {key(r) for r in owned} & {key(r) for r in diagnostic}:
            raise ValueError("Owned diagnostic leakage")
    result, receipt = replace_slots(original, owned)
    for key in (lambda r: r["id"], lambda r: r["family_id"], state_key):
        if {key(r) for r in result} & {key(r) for r in calib + diagnostic}:
            raise ValueError("Train/heldout collision")
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate composed IDs")
    if sum(len(r["questions"]) for r in result if r["source"] == E.SOURCE) > CAP:
        raise ValueError("Source cap exceeded")
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "train.jsonl", result)
    shutil.copyfile(base / "calib.jsonl", output / "calib.jsonl")
    shutil.copyfile(evidence / "diagnostic.jsonl", output / "evidence-diagnostic.jsonl")
    write_json(output / "replacement-slots.json", receipt)
    report = {
        "name": "item47-rung1-a3",
        "private": True,
        "a3_revision": A3_REVISION,
        "a3_input_hashes": A3_HASHES,
        "a3_manifest_sha256": A3_MANIFEST_SHA256,
        "evidence_manifest_sha256": sha(evidence / "manifest.json"),
        "source_scan_sha256": sha(evidence / "overlap-scan-report.json"),
        "slots": {k: v for k, v in receipt.items() if k != "replacements"},
        "questions": TRAIN_QUESTIONS,
        "calibration_questions": CALIB_QUESTIONS,
        "calibration": "Original A3 bytes unchanged; owned diagnostic separate, no temperature fitting",
        "base_policy": "Existing A3 private exceptions unchanged; no shipping clearance",
        "files": {
            n: {"sha256": sha(output / n)}
            for n in (
                "train.jsonl",
                "calib.jsonl",
                "evidence-diagnostic.jsonl",
                "replacement-slots.json",
            )
        },
        "training_launched": False,
    }
    write_json(output / "manifest.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "evidence", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument(
        "--licenses",
        type=Path,
        default=Path(__file__).with_name("manifests") / "item47-source-licenses.json",
    )
    a = p.parse_args()
    print(json.dumps(compose(a.base, a.evidence, a.out, a.licenses), indent=2))
