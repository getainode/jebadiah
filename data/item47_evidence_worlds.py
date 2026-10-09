"""Apache-2.0 finite multi-document evidence source, executable rules only.

CPU data preparation, with no model, corpus, network or training calls. The
entire finite source is scanned on Studio before any composition is admitted.
"""

from __future__ import annotations
import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import random
from item33_skill_data import (
    canonical,
    digest,
    sha,
    write_json,
    write_rows,
    validate_source,
)
from lint_data import lint_record
from item33_full_suite_scan import scan, ITEM25_INDEX_SHA256

SOURCE = "item47-owned-multidoc-evidence"
SEED = "item47-rung1-20261009"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
INDEX_PATH = "/Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl"
# (subject variable, predicate, object variable, document). Variable 0 is fixed.
# These are complete rule/template families, not random row or vocabulary splits.
RULES = {
    "forward2": (2, [(0, "assigns", 1, 0)], 1),
    "forward3": (3, [(0, "assigns", 1, 0), (1, "routes to", 2, 1)], 2),
    "forward4": (
        4,
        [(0, "assigns", 1, 0), (1, "routes to", 2, 1), (2, "reports to", 3, 2)],
        3,
    ),
    "converge3": (
        3,
        [(0, "assigns", 1, 0), (0, "routes to", 2, 1), (2, "reports to", 1, 1)],
        1,
    ),
    "branch3": (
        3,
        [
            (0, "assigns", 1, 0),
            (1, "routes to", 2, 1),
            (1, "reports to", 3, 1),
            (3, "links to", 2, 1),
        ],
        2,
    ),
    "reverse2": (2, [(1, "assigns", 0, 0)], 1),
    "reverse3": (3, [(0, "assigns", 1, 0), (2, "routes to", 1, 1)], 2),
    "diamond4": (
        4,
        [
            (0, "assigns", 1, 0),
            (0, "routes to", 2, 1),
            (1, "reports to", 3, 2),
            (2, "links to", 3, 2),
        ],
        3,
    ),
    "cycle3": (
        3,
        [(0, "assigns", 1, 0), (1, "routes to", 2, 1), (2, "reports to", 0, 1)],
        2,
    ),
    "crosslink4": (
        4,
        [
            (0, "assigns", 1, 0),
            (2, "routes to", 1, 1),
            (2, "reports to", 3, 2),
            (3, "links to", 0, 2),
        ],
        3,
    ),
}
TRAIN_RULES = tuple(RULES)[:5]
DIAGNOSTIC_RULES = tuple(RULES)[5:]
STATUSES = ("supported", "refuted", "insufficient")
VALUES = ("ready", "paused", "sealed", "pending")
SCOPE = (
    "Use only these documents. Entity IDs and periods must match exactly. "
    "Relations apply only at the stated period. A recorded status has exactly one value "
    "for that entity and period. A different recorded value rules out an asserted value. "
    "Missing evidence proves neither a claim nor its negation. Resolve the referenced "
    "entity using every query condition; an unresolved query is insufficient evidence."
)
CRITERIA = {
    "supported": "The evidence proves the claim.",
    "refuted": "The evidence proves the opposite of the claim.",
    "insufficient": "The evidence proves neither the claim nor its opposite.",
}


def fact(subject, predicate, obj, period, document, fid):
    return dict(
        subject=subject,
        predicate=predicate,
        object=obj,
        period=period,
        document=document,
        fid=fid,
    )


def resolve(world):
    """Relational join, carrying the IDs of facts used as a proof witness."""
    query = world["query"]
    bindings = [({0: query["anchor"]}, [])]
    for a, pred, b, _doc in query["edges"]:
        following = []
        for env, trace in bindings:
            for f in world["facts"]:
                if f["predicate"] != pred or f["period"] != query["period"]:
                    continue
                extended = dict(env)
                valid = True
                for var, entity in ((a, f["subject"]), (b, f["object"])):
                    if var in extended and extended[var] != entity:
                        valid = False
                        break
                    extended[var] = entity
                if valid:
                    following.append((extended, trace + [f["fid"]]))
        bindings = following
    return bindings


def oracle(world):
    query = world["query"]
    bindings = resolve(world)
    entities = {env[query["terminal"]] for env, _ in bindings}
    if len(entities) > 1:
        raise ValueError("Ambiguous referenced entity")
    if not bindings:
        return {"status": "insufficient", "join": [], "terminal": [], "bindings": {}}
    entity = next(iter(entities))
    values = [
        f
        for f in world["facts"]
        if f["subject"] == entity
        and f["predicate"] == "status"
        and f["period"] == query["period"]
    ]
    if len({f["object"] for f in values}) > 1:
        raise ValueError("Inconsistent status evidence")
    env, trace = min(bindings, key=lambda p: canonical(p))
    status = "insufficient"
    if values:
        agrees = values[0]["object"] == query["value"]
        supported = agrees != query["negated"]
        status = "supported" if supported else "refuted"
    return {
        "status": status,
        "join": trace,
        "terminal": [f["fid"] for f in values],
        "bindings": {str(k): v for k, v in env.items()},
    }


def reference_oracle(world):
    """Independent exhaustive assignment oracle, not the join implementation."""
    q = world["query"]
    triples = {
        (f["subject"], f["predicate"], f["object"])
        for f in world["facts"]
        if f["period"] == q["period"]
    }
    variables = sorted({v for a, _, b, _ in q["edges"] for v in (a, b)} - {0})
    entities = sorted(
        {x for a, p, b in triples if p != "status" for x in (a, b)} | {q["anchor"]}
    )
    targets = set()
    for values in itertools.product(entities, repeat=len(variables)):
        env = {0: q["anchor"], **dict(zip(variables, values))}
        if all((env[a], p, env[b]) in triples for a, p, b, _ in q["edges"]):
            targets.add(env[q["terminal"]])
    if len(targets) > 1:
        raise ValueError("Ambiguous referenced entity")
    if not targets:
        return "insufficient"
    found = {b for a, p, b in triples if a == next(iter(targets)) and p == "status"}
    if len(found) > 1:
        raise ValueError("Inconsistent status evidence")
    if not found:
        return "insufficient"
    truth = (next(iter(found)) == q["value"]) != q["negated"]
    return "supported" if truth else "refuted"


def world_for(rule, number):
    count, edges, terminal = RULES[rule]
    rng = random.Random(digest(SEED + ":" + rule + ":" + str(number)))
    # Every world and all of its minimal flips have one stable group ID.
    group = "item47:" + rule + ":" + str(number)
    prefix = digest(group)[:8].upper()
    entities = {v: f"R{prefix}{v}" for v in range(4)}
    decoy = f"R{prefix}D"
    unrelated = f"R{prefix}E"
    year = 2030 + rng.randrange(60)
    month = rng.randrange(1, 12)
    period = f"{year}-{month:02d}"
    old = f"{year}-{month+1:02d}"
    desired, alternate = rng.sample(VALUES, 2)
    negated = bool(number % 2)
    query = dict(
        anchor=entities[0],
        edges=copy.deepcopy(edges),
        terminal=terminal,
        period=period,
        value=desired,
        negated=negated,
    )
    facts = []
    for i, (a, p, b, d) in enumerate(edges):
        facts.append(fact(entities[a], p, entities[b], period, d, f"join{i}"))
        # Near matches at the wrong time or with the wrong subject cannot join.
        facts.append(fact(entities[a], p, decoy, old, d, f"time{i}"))
        facts.append(fact(decoy, p, unrelated, period, d, f"entity{i}"))
    # A value known only for a different entity/time does not settle the claim.
    facts.append(fact(decoy, "status", desired, period, count - 1, "decoy-status"))
    facts.append(
        fact(entities[terminal], "status", desired, old, count - 1, "old-status")
    )
    evidence_value = alternate if negated else desired
    facts.append(
        fact(
            entities[terminal],
            "status",
            evidence_value,
            period,
            count - 1,
            "claim-status",
        )
    )
    rng.shuffle(facts)
    return dict(
        group_id=group,
        rule=rule,
        documents=count,
        facts=facts,
        query=query,
        alternate=alternate,
        missing_fid="claim-status" if number % 3 == 0 else f"join{number % len(edges)}",
    )


def variant(world, mode):
    changed = copy.deepcopy(world)
    terminal = next(f for f in changed["facts"] if f["fid"] == "claim-status")
    if mode == "refuted":
        terminal["object"] = (
            changed["query"]["value"]
            if changed["query"]["negated"]
            else changed["alternate"]
        )
    elif mode == "insufficient":
        # Delete one join premise or the terminal fact. Wrong-period facts remain.
        changed["facts"].remove(
            next(f for f in changed["facts"] if f["fid"] == changed["missing_fid"])
        )
    elif mode != "supported":
        raise ValueError("Unknown minimal flip")
    return changed


def state_for(world):
    q = world["query"]

    def name(var):
        return q["anchor"] if var == 0 else f"entity X{var}"

    conditions = "; ".join(f"{name(a)} {p} {name(b)}" for a, p, b, _ in q["edges"])
    claim = f"At {q['period']}, the referenced entity's status is {'not ' if q['negated'] else ''}{q['value']}."
    docs = []
    for d in range(world["documents"]):
        lines = []
        for f in world["facts"]:
            if f["document"] != d:
                continue
            if f["predicate"] == "status":
                sentence = f"At {f['period']}, {f['subject']} has status {f['object']}."
            else:
                sentence = (
                    f"At {f['period']}, {f['subject']} {f['predicate']} {f['object']}."
                )
            lines.append(sentence)
        docs.append({"document_id": f"D{d+1}", "text": " ".join(lines)})
    return {
        "scope": SCOPE,
        "documents": docs,
        "reference_query": f"At {q['period']}, satisfy all conditions: {conditions}. The referenced entity is X{q['terminal']}.",
        "claim": claim,
    }


def generate():
    splits = {"train": [], "diagnostic": []}
    for split, rules, size in (
        ("train", TRAIN_RULES, 200),
        ("diagnostic", DIAGNOSTIC_RULES, 10),
    ):
        for rule in rules:
            for number in range(size):
                base = world_for(rule, number)
                for mode in STATUSES:
                    world = variant(base, mode)
                    proof = oracle(world)
                    for kind in ("choice", "noul"):
                        key = base["group_id"] + ":" + mode + ":" + kind
                        question = {
                            "type": kind,
                            "instructions": (
                                "Assess the claim using only the supplied documents and every reference-query condition."
                                if kind == "choice"
                                else "Do the supplied documents prove the claim, using every reference-query condition?"
                            ),
                        }
                        if kind == "choice":
                            question["criteria"] = dict(CRITERIA)
                        splits[split].append(
                            {
                                "id": "item47:" + digest(key)[:24],
                                "set": "item47-rung1",
                                "subset": SOURCE,
                                "source": SOURCE,
                                "license": "Apache-2.0",
                                "area": "language",
                                "skill": "multidoc_evidence",
                                "family": f"item47:rule:{rule}",
                                "family_id": f"item47:rule:{rule}",
                                "group_id": base["group_id"],
                                "rule": rule,
                                "evidence_status": mode,
                                "world": world,
                                "proof": proof,
                                "state": state_for(world),
                                "questions": {"decision": question},
                                "label": {
                                    "decision": (
                                        mode
                                        if kind == "choice"
                                        else mode == "supported"
                                    )
                                },
                            }
                        )
    return splits


def audit(splits):
    ids = set()
    families = {}
    groups = {}
    state_splits = {}
    all_groups = {}
    for split, rows in splits.items():
        expected_rules = TRAIN_RULES if split == "train" else DIAGNOSTIC_RULES
        for r in rows:
            if (
                r["source"] != SOURCE
                or r["subset"] != SOURCE
                or r["license"] != "Apache-2.0"
                or r["area"] != "language"
            ):
                raise ValueError("Wrong source contract")
            if r["rule"] not in expected_rules:
                raise ValueError("Rule-family leakage")
            if (
                r["family_id"] != f"item47:rule:{r['rule']}"
                or r["family"] != r["family_id"]
            ):
                raise ValueError("Wrong rule family")
            errors = []
            lint_record(r, r["id"], errors, max_options=20)
            if errors:
                raise ValueError("; ".join(errors))
            if r["id"] in ids:
                raise ValueError("Duplicate ID")
            ids.add(r["id"])
            for key, table in (
                (r["family_id"], families),
                (r["group_id"], groups),
                (canonical(r["state"]), state_splits),
            ):
                if table.setdefault(key, split) != split:
                    raise ValueError("Train/diagnostic leakage")
            world = r["world"]
            if world["rule"] != r["rule"] or world["group_id"] != r["group_id"]:
                raise ValueError("World provenance mismatch")
            n, edges, terminal = RULES[r["rule"]]
            if (
                world["documents"] != n
                or [list(e) for e in world["query"]["edges"]]
                != [list(e) for e in edges]
                or world["query"]["terminal"] != terminal
            ):
                raise ValueError("Wrong rule program")
            if any(f["document"] not in range(n) for f in world["facts"]):
                raise ValueError("Wrong document index")
            if len({f["fid"] for f in world["facts"]}) != len(world["facts"]):
                raise ValueError("Duplicate fact IDs")
            if r["state"] != state_for(world):
                raise ValueError("Rendered state not bound to world")
            proof = oracle(world)
            if proof != r["proof"] or proof["status"] != r["evidence_status"]:
                raise ValueError("Wrong executable proof")
            q = r["questions"]["decision"]
            mode = proof["status"]
            expected_q = {
                "type": q["type"],
                "instructions": (
                    "Assess the claim using only the supplied documents and every reference-query condition."
                    if q["type"] == "choice"
                    else "Do the supplied documents prove the claim, using every reference-query condition?"
                ),
            }
            if q["type"] == "choice":
                expected_q["criteria"] = dict(CRITERIA)
            if q["type"] not in ("choice", "noul") or q != expected_q:
                raise ValueError("Wrong question contract")
            if r["label"]["decision"] != (
                mode if q["type"] == "choice" else mode == "supported"
            ):
                raise ValueError("Wrong label")
            all_groups.setdefault((split, r["group_id"]), {})[(mode, q["type"])] = r
    for group in all_groups.values():
        if set(group) != set(itertools.product(STATUSES, ("choice", "noul"))):
            raise ValueError("Incomplete minimal-flip group")
        base = group[("supported", "choice")]["world"]
        for mode in STATUSES:
            for kind in ("choice", "noul"):
                if group[(mode, kind)]["world"] != variant(base, mode):
                    raise ValueError("Nonminimal fact flip")
        if (
            len(
                {
                    f["document"]
                    for f in base["facts"]
                    if f["fid"] in oracle(base)["join"] + oracle(base)["terminal"]
                }
            )
            != base["documents"]
        ):
            raise ValueError("Proof does not join all documents")
    if len(splits["train"]) != 6000 or len(splits["diagnostic"]) != 300:
        raise ValueError("Wrong complete-source counts")
    expected = generate()
    if any(rows_hash(splits[s]) != rows_hash(expected[s]) for s in expected):
        raise ValueError("Data differs from finite generator")


def counts(rows):
    return {
        "questions": len(rows),
        "world_groups": len({r["group_id"] for r in rows}),
        "rules": dict(Counter(r["rule"] for r in rows)),
        "types": dict(Counter(r["questions"]["decision"]["type"] for r in rows)),
        "documents": dict(Counter(str(r["world"]["documents"]) for r in rows)),
        "evidence_status": dict(Counter(r["evidence_status"] for r in rows)),
        "negated_claims": sum(r["world"]["query"]["negated"] for r in rows),
    }


def render_check(splits, tokenizer_path):
    from transformers import AutoTokenizer
    from jebadiah_prompt import Renderer, PROMPT_SOURCE_SHA256

    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "results/runs/9b-chat-v1/adapter/prompt_contract.json"
        ).read_text()
    )
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if sha(tokenizer_path / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Wrong pinned tokenizer")
    if (
        PROMPT_SOURCE_SHA256 != contract["prompt_source_sha256"]
        or digest(tok.chat_template) != contract["chat_template_sha256"]
    ):
        raise ValueError("Wrong prompt contract")
    renderer = Renderer(tok, max_tokens=1984)
    maximum = {}
    checked = 0
    for rows in splits.values():
        for r in rows:
            q = r["questions"]["decision"]
            keys = list(q.get("criteria", {"true": None, "false": None}))
            shuffled = list(keys)
            random.Random(digest(r["id"])).shuffle(shuffled)
            for order in (keys, list(reversed(keys)), shuffled):
                rendered = renderer.render(r["state"], q, order)
                n = len(tok.encode(rendered.prompt, add_special_tokens=False))
                if rendered.truncated or n > 1984:
                    raise ValueError("Truncation/overflow: " + r["id"])
                if len(set(rendered.cand_ids)) != len(keys) or set(
                    rendered.keys
                ) != set(keys):
                    raise ValueError("Incomplete options")
                maximum[q["type"]] = max(maximum.get(q["type"], 0), n)
                checked += 1
    return {
        "checked_renders": checked,
        "max_prompt_tokens_by_type": maximum,
        "truncated": 0,
        "prompt_budget": 1984,
        "padding_reserve": 64,
        "max_seq_length": 2048,
        "order_checks": ["canonical", "reversed", "seeded_shuffle"],
        "tokenizer_sha256": TOKENIZER_SHA256,
        "prompt_source_sha256": PROMPT_SOURCE_SHA256,
        "chat_template_sha256": digest(tok.chat_template),
    }


def scan_rows(splits):
    return [{**r, "family": SOURCE} for rows in splits.values() for r in rows]


def rows_hash(rows):
    return digest(
        "".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in rows
        )
    )


def validate_policy(licenses):
    policy = json.loads(licenses.read_text())
    inherited = json.loads(
        Path(__file__)
        .with_name("manifests")
        .joinpath("item43-source-licenses.json")
        .read_text()
    )
    if (
        policy["source_exclusions"] != inherited["source_exclusions"]
        or policy.get("source_level_rejection") is not True
    ):
        raise ValueError("Full inherited source exclusions required")
    if set(policy["sources"]) != {SOURCE}:
        raise ValueError("Unexpected source manifest")
    source = policy["sources"][SOURCE]
    validate_source(source, policy["source_exclusions"])
    if source["generator_sha256"] != sha(Path(__file__)) or source[
        "revision"
    ] != "sha256:" + sha(Path(__file__)):
        raise ValueError("Generator hash mismatch")
    if source["license_sha256"] != sha(Path(__file__).resolve().parents[1] / "LICENSE"):
        raise ValueError("License hash mismatch")
    if (
        source.get("model_redistribution_ok") is not True
        or source.get("generation_seed") != SEED
    ):
        raise ValueError("Incomplete source freeze")
    if source["external_inputs"] != [] or source["generator_model"] is not None:
        raise ValueError("External/model text forbidden")
    return policy, source


def build(output, tokenizer, licenses, index=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty")
    if index is not None:
        if str(index.resolve()) != INDEX_PATH:
            raise ValueError("Scan must run on Studio at protected index location")
        if sha(index) != ITEM25_INDEX_SHA256:
            raise ValueError("Wrong protected index")
    policy, source = validate_policy(licenses)
    splits = generate()
    audit(splits)
    if source["complete_split_sha256"] != {
        s: rows_hash(rs) for s, rs in splits.items()
    }:
        raise ValueError("Wrong licensed data freeze")
    render = render_check(splits, tokenizer)
    if source["complete_split_counts"] != {s: len(rs) for s, rs in splits.items()}:
        raise ValueError("Wrong licensed source scope")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "pending_studio_scan",
        "source_level": True,
        "protected_index_sha256": ITEM25_INDEX_SHA256,
    }
    if index is not None:
        work = output / "source-scan"
        work.mkdir()
        write_rows(work / "candidates.jsonl", scan_rows(splits))
        (work / "protected.pkl").symlink_to(index.resolve())
        try:
            scan(work)
        finally:
            (work / "protected.pkl").unlink()
        report = json.loads((work / "scan-report.json").read_text())
        report.update(
            status="passed" if report["removed_records"] == 0 else "rejected",
            source_level=True,
            protected_index_sha256=ITEM25_INDEX_SHA256,
            scanner_sha256=sha(Path(__file__).with_name("item33_full_suite_scan.py")),
            candidates_sha256=sha(work / "candidates.jsonl"),
        )
        write_json(output / "overlap-scan-report.json", report)
        if report["status"] != "passed":
            raise ValueError("Entire owned source rejected")
    write_json(output / "overlap-scan-report.json", report)
    for s, rs in splits.items():
        write_rows(output / f"{s}.jsonl", rs)
    manifest = {
        "name": "item47-rung1-evidence",
        "seed": SEED,
        "generator_sha256": sha(Path(__file__)),
        "license_manifest_sha256": sha(licenses),
        "render_validation": render,
        "overlap_scan": report,
        "splits": {s: counts(rs) for s, rs in splits.items()},
        "files": {f"{s}.jsonl": {"sha256": sha(output / f"{s}.jsonl")} for s in splits},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--index", type=Path)
    p.add_argument(
        "--licenses",
        type=Path,
        default=Path(__file__).with_name("manifests") / "item47-source-licenses.json",
    )
    a = p.parse_args()
    print(json.dumps(build(a.out, a.tokenizer, a.licenses, a.index), indent=2))
