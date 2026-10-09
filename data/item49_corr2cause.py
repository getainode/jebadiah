"""Apache-2.0 deterministic Corr2Cause converter. CPU only, no model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from itertools import combinations, permutations
import json
from pathlib import Path
import re

from item33_skill_data import digest, sha, write_json, write_rows, validate_source
from item33_full_suite_scan import scan, ITEM25_INDEX_SHA256
from item32_ablation_mix import read
from item36_skill_mix import A3_HASHES
from lint_data import lint_record

SOURCE = "corr2cause"
REVISION = "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"
SEED = "item49-rung2-20261009"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
POLICY = Path(__file__).with_name("manifests") / "item49-source-licenses.json"
PROMPT = "Does the hypothesis follow necessarily from the statistical relations in the premise? A possible relation alone is insufficient."
OPTIONS = {"valid": "Necessarily follows", "not_valid": "Does not necessarily follow"}


def split_input(text):
    if not isinstance(text, str) or not text.startswith("Premise: "):
        raise ValueError("Expected original Corr2Cause input")
    parts = text[9:].split("\nHypothesis: ")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Missing or ambiguous hypothesis")
    return parts


@lru_cache(maxsize=32768)
def skeleton_key(n, edges):
    """Canonical unlabeled undirected skeleton, coarser than a DAG or MEC.

    Grouping all collider orientations together deliberately excludes more than
    necessary. Every permutation of the same DAG stays in the same family.
    """
    pairs = list(combinations(range(n), 2))
    indices = {p: i for i, p in enumerate(pairs)}
    best = min(
        sum(1 << indices[tuple(sorted((perm[a], perm[b])))] for a, b in edges)
        for perm in permutations(range(n))
    )
    return f"corr2cause:skeleton:{n}:{best}"


@lru_cache(maxsize=32768)
def graph_family(premise):
    match = re.match(r"Suppose there is a closed system of ([2-6]) variables, (.+?)\. All ", premise)
    if not match:
        raise ValueError("Unknown original graph premise")
    n = int(match[1])
    variables = re.findall(r"\b[A-Z]\b", match[2])
    if len(set(variables)) != n or len(variables) != n:
        raise ValueError("Invalid graph variables")
    indices = {v: i for i, v in enumerate(variables)}
    # Under the upstream faithful closed-DAG construction, an edge is absent
    # iff some (possibly empty) separating set is stated for the pair.
    independent = re.findall(r"([A-Z]) (?:is independent of|and ([A-Z]) are independent given) ?([A-Z])?", premise)
    nonedges = set()
    for first, conditional_second, marginal_second in independent:
        second = conditional_second or marginal_second
        if first not in indices or second not in indices or first == second:
            raise ValueError("Invalid independence pair")
        nonedges.add(tuple(sorted((indices[first], indices[second]))))
    correlations = re.findall(r"([A-Z]) correlates with ([A-Z])\.", premise)
    covered = nonedges | {tuple(sorted((indices[a], indices[b]))) for a, b in correlations}
    pairs = set(combinations(range(n), 2))
    if covered != pairs:
        raise ValueError("Incomplete statistical relations")
    return skeleton_key(n, tuple(sorted(pairs - nonedges)))


def upstream_rows(path):
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            yield from csv.DictReader(handle)
    elif path.suffix == ".json":
        import ijson
        with path.open("rb") as handle:
            yield from ijson.items(handle, "item")
    else:
        raise ValueError("Unsupported data file")


def raw_scan_rows(raw, policy):
    # All released data files, all original and perturbation splits, no sample.
    # Do not train on perturbation or dev/test records.
    for filename in sorted(policy["sources"][SOURCE]["complete_files"]):
        for i, row in enumerate(upstream_rows(raw / filename)):
            texts = {k: v for k, v in row.items() if k in ("input", "premise", "hypothesis")}
            if not texts or not all(isinstance(v, str) and v for v in texts.values()):
                raise ValueError("Missing upstream text")
            yield {"id": f"item49:raw:{filename}:{i}", "family": SOURCE,
                   "state": texts, "questions": {}}


def validate_policy(raw, licenses=POLICY):
    policy = json.loads(licenses.read_text())
    source = policy["sources"][SOURCE]
    validate_source(source, policy["source_exclusions"])
    if (source["revision"] != REVISION or source["ancestry_verified"] is not True
            or source["generator_sha256"] != sha(Path(__file__))
            or source["generator_license"] != "Apache-2.0"
            or source["model_written_training_text"] is not False
            or source["model_redistribution_ok"] is not True
            or digest(source["notice_text"]) != source["license_sha256"]):
        raise ValueError("Unverified ancestry or converter")
    actual = {p.name for p in raw.iterdir() if p.suffix in (".csv", ".json")}
    if actual != set(source["complete_files"]):
        raise ValueError("Incomplete upstream release")
    for filename, info in source["complete_files"].items():
        if sha(raw / filename) != info["sha256"]:
            raise ValueError("Wrong upstream bytes: " + filename)
    if sha(raw / "README.md") != source["readme_sha256"]:
        raise ValueError("Wrong source card")
    return policy


def existing_families(base):
    for name, expected in A3_HASHES.items():
        if sha(base / name) != expected:
            raise ValueError("Wrong pinned A3 input: " + name)
    result = set()
    for split in ("train", "calib"):
        for row in read(base / f"{split}.jsonl"):
            if SOURCE not in json.dumps({k: row.get(k) for k in ("source", "subset", "source_family", "provenance")}).lower():
                continue
            result.add(graph_family(a3_premise(row["state"])))
    return result


def a3_premise(state):
    # Tasksource's paired-text format variants preserve the same premise.
    if isinstance(state, str):
        for prefix, separator in (("text_A: ", "\ntext_B: "),
                                  ("A: ", "\nB: "),
                                  ("Passage A:\n", "\n\nPassage B:\n")):
            if state.startswith(prefix) and separator in state:
                return state[len(prefix):].split(separator, 1)[0]
    raise ValueError("Unrecognized A3 Corr2Cause exposure")


def convert(row, index, kind):
    if kind not in ("choice", "noul") or row["label"] not in ("0", "1"):
        raise ValueError("Invalid binary label or type")
    premise, hypothesis = split_input(row["input"])
    family = graph_family(premise)
    if int(row["num_variables"]) != int(family.split(":")[-2]):
        raise ValueError("Graph size disagrees with upstream metadata")
    q = {"type": kind, "instructions": PROMPT}
    if kind == "choice":
        q["criteria"] = dict(OPTIONS)
    gold = row["label"] == "1"
    return {"id": f"item49:corr2cause:train:{index}", "set": "item49-rung2",
            "subset": SOURCE, "source": SOURCE, "source_family": SOURCE,
            "license": "MIT", "area": "knowledge", "skill": "causal-identification",
            "family": family, "family_id": family, "state": row["input"],
            "questions": {"decision": q},
            "label": {"decision": ("valid" if gold else "not_valid") if kind == "choice" else gold},
            "provenance": {"repo": "causalnlp/corr2cause", "revision": REVISION,
                           "original_split": "train", "row": index, "template": row["template"],
                           "num_variables": int(row["num_variables"])}}


def renderer(tokenizer):
    from transformers import AutoTokenizer
    from jebadiah_prompt import Renderer, PROMPT_SOURCE_SHA256
    contract = json.loads((Path(__file__).resolve().parents[1] / "results/runs/9b-chat-v1/adapter/prompt_contract.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
    if (sha(tokenizer / "tokenizer.json") != TOKENIZER_SHA256
            or PROMPT_SOURCE_SHA256 != contract["prompt_source_sha256"]
            or digest(tok.chat_template) != contract["chat_template_sha256"]):
        raise ValueError("Wrong pinned prompt/tokenizer contract")
    return tok, Renderer(tok, max_tokens=1984), contract


def render_one(row, tok, rend):
    q = row["questions"]["decision"]
    order = list(q["criteria"]) if q["type"] == "choice" else None
    out = rend.render(row["state"], q, order)
    n = len(tok.encode(out.prompt, add_special_tokens=False))
    return not out.truncated and n <= 1984 and len(set(out.cand_ids)) == 2, n


def select(raw, base, tok, rend, train_count=6000, diagnostic_count=300):
    if not 0 < train_count <= 6000 or not 0 < diagnostic_count <= 300:
        raise ValueError("Invalid candidate budget")
    excluded = existing_families(base)
    pools = defaultdict(list)
    seen = set()
    duplicate = excluded_rows = 0
    # Group every original train row before sampling; both question types use
    # the original binary validity label, never an invented 3-way NLI target.
    for i, row in enumerate(upstream_rows(raw / "train.csv")):
        premise, _ = split_input(row["input"])
        family = graph_family(premise)
        if family in excluded:
            excluded_rows += 1
            continue
        key = digest(row["input"])
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        kind = ("choice", "noul")[int(key[:8], 16) % 2]
        pools[family].append(convert(row, i, kind))
    families = sorted(pools, key=lambda f: digest(SEED + ":family:" + f))
    held = set(families[:max(2, len(families) // 10)])
    splits = {}; rejected = Counter()
    for split, wanted in (("train", train_count), ("diagnostic", diagnostic_count)):
        selected = []
        for kind in ("choice", "noul"):
            family_pools = [sorted((r for r in pools[f] if r["questions"]["decision"]["type"] == kind),
                                   key=lambda r: digest(SEED + ":row:" + r["id"]))
                            for f in families if (f in held) == (split == "diagnostic")]
            # Round robin across families prevents one large graph dominating.
            budget = wanted // 2 + (wanted % 2 if kind == "choice" else 0)
            cursors = [0] * len(family_pools)
            added = 0
            while added < budget:
                advanced = False
                for j, pool in enumerate(family_pools):
                    if cursors[j] >= len(pool):
                        continue
                    advanced = True
                    row = pool[cursors[j]]; cursors[j] += 1
                    valid, _ = render_one(row, tok, rend)
                    if not valid:
                        rejected[split] += 1
                        continue
                    selected.append(row); added += 1
                    if added == budget:
                        break
                if not advanced:
                    raise ValueError("Insufficient graph-disjoint renderable candidates")
        splits[split] = selected
    return splits, {"excluded_a3_families": sorted(excluded), "excluded_original_train_rows": excluded_rows,
                    "duplicate_original_train_inputs": duplicate, "available_unused_families": len(families),
                    "diagnostic_family_partition": sorted(held), "overflow_candidates_skipped": dict(rejected)}


def audit(splits):
    ids = set(); states = set(); families = {}
    for split, rows in splits.items():
        for r in rows:
            errors = []; lint_record(r, r["id"], errors)
            if errors:
                raise ValueError("; ".join(errors))
            if r["id"] in ids or digest(r["state"]) in states:
                raise ValueError("Duplicate candidate")
            ids.add(r["id"]); states.add(digest(r["state"]))
            if families.setdefault(r["family_id"], split) != split:
                raise ValueError("Graph-family leakage")
            p = r["provenance"]
            kind = r["questions"]["decision"]["type"]
            label = r["label"]["decision"]
            if (r["source"] != SOURCE or r["source_family"] != SOURCE or r["license"] != "MIT"
                    or r["area"] != "knowledge" or p["original_split"] != "train"
                    or p["revision"] != REVISION or p["repo"] != "causalnlp/corr2cause"
                    or r["family_id"] != graph_family(split_input(r["state"])[0])
                    or r["family"] != r["family_id"]
                    or kind not in ("choice", "noul")
                    or (kind == "choice" and (r["questions"]["decision"].get("criteria") != OPTIONS or label not in OPTIONS))
                    or (kind == "noul" and type(label) is not bool)
                    or r["questions"]["decision"]["instructions"] != PROMPT):
                raise ValueError("Invalid converted candidate")


def counts(rows):
    return {"questions": len(rows), "graph_families": len({r["family_id"] for r in rows}),
            "types": dict(Counter(r["questions"]["decision"]["type"] for r in rows)),
            "labels": dict(Counter(str(r["label"]["decision"]) for r in rows)),
            "templates": dict(Counter(r["provenance"]["template"] for r in rows)),
            "num_variables": dict(Counter(str(r["provenance"]["num_variables"]) for r in rows))}


def render_check(splits, tok, rend, contract):
    maximum = Counter(); checked = 0
    for rows in splits.values():
        for r in rows:
            q = r["questions"]["decision"]; kind = q["type"]
            orders = (list(OPTIONS), list(reversed(OPTIONS))) if kind == "choice" else (None,)
            for order in orders:
                out = rend.render(r["state"], q, order)
                n = len(tok.encode(out.prompt, add_special_tokens=False))
                if out.truncated or n > 1984 or len(set(out.cand_ids)) != 2:
                    raise ValueError("Render truncation or incomplete candidates")
                maximum[kind] = max(maximum[kind], n); checked += 1
    return {"checked_renders": checked, "max_prompt_tokens_by_type": dict(maximum),
            "max_seq_length": 2048, "prompt_budget": 1984, "padding_reserve": 64, "truncated": 0,
            "choice_orders": ["canonical", "reversed"], "noul_order": "true,false",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "prompt_source_sha256": contract["prompt_source_sha256"],
            "chat_template_sha256": contract["chat_template_sha256"]}


def check_index(index):
    # Lead-only: workers never invoke --index or this function.
    if str(index.resolve()) != "/Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl":
        raise ValueError("Protected scan must run at the exact Studio path")
    if sha(index) != ITEM25_INDEX_SHA256:
        raise ValueError("Wrong protected index")


def scan_phase(rows, output, index, phase):
    check_index(index)
    work = output / phase; work.mkdir(parents=True)
    candidate = work / "candidates.jsonl"
    write_rows(candidate, rows)
    (work / "protected.pkl").symlink_to(index.resolve())
    try:
        scan(work)
    finally:
        (work / "protected.pkl").unlink()
    report = json.loads((work / "scan-report.json").read_text())
    report.update(status="passed" if report["removed_records"] == 0 else "rejected",
                  source_level=True, protected_index_sha256=ITEM25_INDEX_SHA256,
                  scanner_sha256=sha(Path(__file__).with_name("item33_full_suite_scan.py")),
                  candidates_sha256=sha(candidate))
    write_json(work / "scan-report.json", report)
    if report["status"] != "passed":
        write_json(output / "overlap-scan-report.json", {"status": "rejected", "source_level": True, "failed_phase": phase, "report": report})
        raise ValueError("Entire upstream Corr2Cause source rejected, no row salvage")
    return report


def converted_scan_rows(splits):
    for rows in splits.values():
        for r in rows:
            yield {**r, "family": SOURCE}


def build(raw, base, output, tokenizer, licenses=POLICY, index=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty")
    policy = validate_policy(raw, licenses)
    # Admission of the complete upstream release precedes any selection in the
    # lead's scan path. The scan-free worker build is only a pending preview.
    raw_report = None
    if index is not None:
        raw_report = scan_phase(raw_scan_rows(raw, policy), output, index, "raw-source-scan")
    tok, rend, contract = renderer(tokenizer)
    splits, selection = select(raw, base, tok, rend)
    audit(splits)
    render = render_check(splits, tok, rend, contract)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "pending_studio_scan", "source_level": True,
              "protected_index_sha256": ITEM25_INDEX_SHA256}
    if index is not None:
        converted_report = scan_phase(converted_scan_rows(splits), output, index, "converted-source-scan")
    # A rejected source never receives train/diagnostic output files.
    for split, rows in splits.items():
        write_rows(output / f"{split}.jsonl", rows)
    (output / "NOTICE-Corr2Cause.txt").write_text(policy["sources"][SOURCE]["notice_text"])
    if index is not None:
        report = {"status": "passed", "source_level": True, "raw": raw_report, "converted": converted_report,
                  "protected_index_sha256": ITEM25_INDEX_SHA256,
                  "raw_files_sha256": digest(json.dumps(policy["sources"][SOURCE]["complete_files"], sort_keys=True)),
                  "converted_sha256": {s: sha(output / f"{s}.jsonl") for s in splits},
                  "scanned_records": raw_report["scanned_records"] + converted_report["scanned_records"]}
    write_json(output / "overlap-scan-report.json", report)
    manifest = {"name": "item49-rung2-corr2cause", "seed": SEED,
                "generator_sha256": sha(Path(__file__)), "license_manifest_sha256": sha(licenses),
                "source_revision": REVISION, "source_files": policy["sources"][SOURCE]["complete_files"],
                "selection": selection, "splits": {s: counts(rs) for s, rs in splits.items()},
                "render_validation": render, "overlap_scan": report,
                "files": {f"{s}.jsonl": {"sha256": sha(output / f"{s}.jsonl")} for s in splits},
                "training_launched": False, "model_calls": 0}
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "base", "out", "tokenizer"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--licenses", type=Path, default=POLICY)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.base, args.out, args.tokenizer, args.licenses, args.index), indent=2))
