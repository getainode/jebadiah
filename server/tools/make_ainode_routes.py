"""Regenerate src/jebadiah_server/ainode_routes.py from an AINode checkout.

The standalone server answers /v1/systemone and /v1/decide with AINode's own validation and
answer shaping, copied here definition by definition (the source text of each one, unchanged),
so the wire behaves the same whether a request lands on a fleet or on this one process.
The prompt renderer is NOT copied from here: it comes from the model repository's
scripts/ainode_prompt_verbatim.py, which is what the model was trained on.

    python tools/make_ainode_routes.py /path/to/ainode
"""
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

WANT = {
    "ainode/api/decide.py": [
        "BOOLEAN_OPTIONS", "DEFAULT_SCORE_MIN", "DEFAULT_SCORE_MAX", "CHOICE", "NOUL", "SCORE",
        "CALIBRATION_RAW", "DecideError", "_score_options", "normalize_questions", "pick_answer",
        "calibration_mode", "QUESTION_KINDS", "TEMPERATURES_FILE", "read_temperatures",
    ],
    "ainode/api/systemone.py": [
        "Translated", "probability", "normalized_confidence", "answer_from_decision",
        "decide_questions",
    ],
}


def definitions(path: Path, names: list[str]) -> list[str]:
    text = path.read_text()
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in names:
            found[name] = ast.get_source_segment(text, node)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    return [found[n] for n in names]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "../ainode")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    blocks = []
    for rel, names in WANT.items():
        blocks += definitions(root / rel, names)
    body = "\n\n\n".join(blocks) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    header = f'''"""AINode's /v1/decide and /v1/systemone route semantics, copied VERBATIM from its source.

Do not edit by hand: regenerate with tools/make_ainode_routes.py.

source commit: {commit}
files: ainode/api/decide.py ({", ".join(WANT["ainode/api/decide.py"])})
       ainode/api/systemone.py ({", ".join(WANT["ainode/api/systemone.py"])})
source_sha256: {digest}  (sha256 of the copied definitions, in this order)

The prompt text itself is not rendered from here. It is rendered by the model repository's
scripts/ainode_prompt_verbatim.py, the copy the model was trained against; the tests prove the
two agree with AINode's live source.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple, Optional

from jebadiah_server.model_scripts import MAX_OPTIONS

AINODE_SOURCE_COMMIT = "{commit}"
AINODE_SOURCE_SHA256 = "{digest}"


'''
    out = Path(__file__).resolve().parent.parent / "src/jebadiah_server/ainode_routes.py"
    out.write_text(header + body)
    print(f"wrote {out} from {commit[:12]} sha {digest[:16]}")


if __name__ == "__main__":
    main()
