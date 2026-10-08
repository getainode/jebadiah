"""Extract pinned AINode definitions without importing AINode, then freeze 852 renders.

Maintainer utility only. Training and tests do not need the source checkout or legacy pool.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess

from ainode_prompt_verbatim import PROMPT_SOURCE_COMMIT, PROMPT_SOURCE_SHA256

NAMES = {
    "decide.py": "SYSTEM_PROMPT ANSWER_INSTRUCTION MAX_OPTIONS TOP_LOGPROBS BOOLEAN_OPTIONS DecideError option_label option_labels serialize_state build_messages".split(),
    "systemone.py": "CHOICE NOUL SCORE QUESTION_TYPES NOUL_OPTIONS MIN_SCORE_LEVELS MAX_SCORE_LEVELS MAX_CRITERIA Translated option_text criteria_pairs choice_options noul_options score_options translate_one translate_questions".split(),
}


def generate(source, pool, output):
    ns = {}
    exec("from __future__ import annotations\nimport json\nfrom typing import Any, NamedTuple, Optional", ns)
    source_hashes = {}
    for filename, names in NAMES.items():
        src = subprocess.check_output(["git", "-C", str(source), "show", PROMPT_SOURCE_COMMIT + ":ainode/api/" + filename], text=True)
        source_hashes[filename] = hashlib.sha256(src.encode()).hexdigest()
        selected = []
        for node in ast.parse(src).body:
            identifiers = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [node.target.id] if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else []
            if any(name in names for name in identifiers):
                selected.append(node)
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), ns)
    fixtures = []
    pool_hashes = {}
    for filename in ("train.jsonl", "calib.jsonl"):
        path = Path(pool) / filename
        pool_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        random.Random(0).shuffle(rows)
        for record in rows[:300]:
            for qid, question in record["questions"].items():
                translated = ns["translate_one"]("q", question)
                fixtures.append({"state": record["state"], "question": question,
                    "names": translated.names,
                    "messages": ns["build_messages"](ns["serialize_state"](record["state"]), None, translated.question, translated.options)})
    if len(fixtures) != 852:
        raise ValueError(f"Expected original 852 renders, got {len(fixtures)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in fixtures).encode()
    (output / "prompt_renders.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    (output / "prompt_fixture_contract.json").write_text(json.dumps({
        "prompt_source_commit": PROMPT_SOURCE_COMMIT, "prompt_source_sha256": PROMPT_SOURCE_SHA256,
        "source_files_sha256": source_hashes, "pool_files_sha256": pool_hashes,
        "renders": len(fixtures), "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "generation": "Pinned source AST extraction, seed 0, 300 train plus 300 calibration records",
    }, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--output", default=str(Path(__file__).parent / "fixtures"))
    generate(**vars(p.parse_args()))
