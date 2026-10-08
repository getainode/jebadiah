"""852 byte-identical question renders against independent pinned-source golden outputs."""
import gzip
import hashlib
import json
from pathlib import Path

import ainode_prompt_verbatim as renderer


def test_852_pinned_renders():
    root = Path(__file__).parent / "fixtures"
    contract = json.loads((root / "prompt_fixture_contract.json").read_text())
    payload = gzip.decompress((root / "prompt_renders.jsonl.gz").read_bytes())
    assert hashlib.sha256(payload).hexdigest() == contract["uncompressed_sha256"]
    assert renderer.PROMPT_SOURCE_COMMIT == contract["prompt_source_commit"]
    assert renderer.PROMPT_SOURCE_SHA256 == contract["prompt_source_sha256"]
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == contract["renders"] == 852
    for row in rows:
        t = renderer.translate_one("q", row["question"])
        actual = renderer.build_messages(renderer.serialize_state(row["state"]), None, t.question, t.options)
        assert t.names == row["names"]
        for got, expected in zip(actual, row["messages"], strict=True):
            assert got["role"].encode("utf-8") == expected["role"].encode("utf-8")
            assert got["content"].encode("utf-8") == expected["content"].encode("utf-8")


if __name__ == "__main__":
    test_852_pinned_renders()
    print("PASS: 852 pinned question renders match byte for byte; no AINode import")
