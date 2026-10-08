"""Check all 852 decision prompts against a local llama-server tokenizer.

Run before and after installing the optional chat profiles. Store full token arrays,
not just token counts. Requires the dependencies of clients/python and a cached
Jeb GGUF repository's tokenizer files, but never downloads or loads a model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clients/python/src"))

from jebadiah_decide._contract import CHAT_TEMPLATE_KWARGS, build_messages, serialize_state  # noqa: E402
from jebadiah_decide._contract.ainode_prompt_verbatim import (  # noqa: E402
    PROMPT_SOURCE_SHA256, translate_one,
)
from jebadiah_decide.backends import post_json  # noqa: E402
from jebadiah_decide.tokenizer import LightTokenizer  # noqa: E402


def check(tokenizer_dir: Path, server: str) -> dict:
    fixtures = ROOT / "train/fixtures"
    contract = json.loads((fixtures / "prompt_fixture_contract.json").read_text())
    payload = gzip.decompress((fixtures / "prompt_renders.jsonl.gz").read_bytes())
    assert hashlib.sha256(payload).hexdigest() == contract["uncompressed_sha256"]
    assert PROMPT_SOURCE_SHA256 == contract["prompt_source_sha256"]
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == contract["renders"] == 852
    tok = LightTokenizer(str(tokenizer_dir))
    token_arrays = []
    digest = hashlib.sha256()
    for i, row in enumerate(rows):
        translated = translate_one("q", row["question"])
        messages = build_messages(serialize_state(row["state"]), None,
                                  translated.question, translated.options)
        assert messages == row["messages"] and translated.names == row["names"], i
        golden = tok.apply_chat_template(row["messages"], **CHAT_TEMPLATE_KWARGS)
        prompt = tok.apply_chat_template(messages, **CHAT_TEMPLATE_KWARGS)
        assert prompt.encode("utf-8") == golden.encode("utf-8"), i
        ids = tok.encode(prompt, add_special_tokens=False)
        runtime_ids = post_json(server.rstrip("/") + "/tokenize", {
            "content": prompt, "add_special": False, "parse_special": True,
        }, timeout=30)["tokens"]
        assert ids == runtime_ids, f"runtime token IDs differ at fixture {i}"
        token_arrays.append(ids)
        # Length framing preserves boundaries between fixtures in the digest.
        digest.update(struct.pack("<I", len(ids)))
        digest.update(struct.pack(f"<{len(ids)}I", *ids))
    return {
        "renders": len(rows),
        "prompt_source_sha256": contract["prompt_source_sha256"],
        "template_sha256": hashlib.sha256((tokenizer_dir / "chat_template.jinja").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "token_ids_sha256": digest.hexdigest(),
        "token_ids": token_arrays,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-dir", required=True, type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--record", type=Path, help="write the baseline token arrays")
    ap.add_argument("--compare", type=Path, help="compare every token ID with the baseline")
    args = ap.parse_args()
    result = check(args.tokenizer_dir, args.server)
    if args.compare:
        baseline = args.compare.read_bytes()
        if args.compare.suffix == ".gz":
            baseline = gzip.decompress(baseline)
        assert result == json.loads(baseline), "decision prompt identity changed"
    if args.record:
        data = (json.dumps(result, separators=(",", ":")) + "\n").encode()
        args.record.write_bytes(gzip.compress(data, mtime=0) if args.record.suffix == ".gz" else data)
    print(f"PASS: {result['renders']} complete decision prompts match local llama-server token IDs")
    if args.compare:
        print("PASS: all token arrays and tokenizer/template hashes equal the baseline")
    print(f"Token ID SHA256: {result['token_ids_sha256']}")


if __name__ == "__main__":
    main()
