"""The prompt this server sends is the prompt the model was trained on.

Three links, each proved separately:
  1. the bundled scripts are byte-for-byte the model repository's scripts/ (the trainer's copies);
  2. the bundled renderer produces the same messages as AINode's live source (the method of
     train/test_prompt_identity.py), and the route semantics copied from AINode match it too;
  3. what the routes render is exactly what jebadiah_prompt.Renderer renders, through the
     tokenizer's own chat template with thinking off, as the model's prompt_contract.json pins.
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pytest

from conftest import TEST_MODEL, model_dir
from jebadiah_server import ainode_routes
from jebadiah_server.model_scripts import HERE, SCRIPT_FILES, prompt, verbatim

AINODE_SRC = Path(os.environ.get("AINODE_SRC", "../ainode"))
ROOT = Path(__file__).resolve().parent.parent

# Every shape the wire allows, plus the edge cases the renderer has opinions about.
REQUESTS = [
    json.loads((Path(HERE) / "example-request.json").read_text()),
    {"state": "plain text state, verbatim\nwith a newline", "questions": {
        "a": {"type": "choice", "instructions": "  padded instructions  ",
              "criteria": {" spaced key ": "desc\nwith  newline", "plain": None, "empty": ""}},
        "b": {"type": "noul", "instructions": "no criteria at all"},
        "c": {"type": "noul", "instructions": "one side", "criteria": {"false": "only false"}},
        "d": {"type": "score", "instructions": "list levels", "criteria": ["low", "mid", "high"]},
        "e": {"type": "score", "instructions": "object levels",
              "criteria": {"bad": "no good", "ok": None, "great": "very good"}},
    }},
    {"state": {"z": 1, "a": [1, 2, {"y": "é unicode", "b": None}], "n": 1.5}, "questions": {
        "wide": {"type": "choice", "instructions": "twenty options",
                 "criteria": {f"opt{i}": f"option number {i}" for i in range(20)}},
        "ignored": {"type": "noul", "instructions": "extra fields", "passingAnswer": True,
                    "label": "true"},
    }},
    {"state": None, "questions": {"x": {"type": "choice", "instructions": "no state",
                                        "criteria": {"yes": None, "no": None}}}},
]


def identity_rows():
    """REQUESTS plus, when JEB_IDENTITY_DATA names training jsonl files (colon separated), 300
    shuffled rows of each, like the trainer's own identity test."""
    rows = list(REQUESTS)
    for path in filter(None, os.environ.get("JEB_IDENTITY_DATA", "").split(":")):
        data = [json.loads(line) for line in open(path)]
        random.Random(0).shuffle(data)
        rows += data[:300]
    return rows


def test_bundled_scripts_are_the_model_repositorys():
    from huggingface_hub import hf_hub_download
    for repo in dict.fromkeys(["frontier-infra/jebadiah-9b-v2", TEST_MODEL]):
        for name in SCRIPT_FILES + ("example-request.json",):
            theirs = Path(hf_hub_download(repo, f"scripts/{name}")).read_bytes()
            ours = (Path(HERE) / name).read_bytes()
            assert ours == theirs, f"{name} differs from {repo}/scripts/{name}"


def test_verbatim_hash_matches_its_header():
    assert verbatim.PROMPT_SOURCE_SHA256 == json.loads(
        (Path(model_dir()) / "prompt_contract.json").read_text())["prompt_source_sha256"]


@pytest.mark.skipif(not (AINODE_SRC / "ainode/api/systemone.py").is_file(),
                    reason="no AINode checkout (set AINODE_SRC)")
def test_renderer_matches_ainode_live_source():
    sys.path.insert(0, str(AINODE_SRC))
    try:
        from ainode.api import decide as D, systemone as S
    finally:
        sys.path.remove(str(AINODE_SRC))
    n = 0
    for row in identity_rows():
        for qid, q in row["questions"].items():
            t_real = S.translate_one("q", q)
            t_copy = verbatim.translate_one("q", q)
            real = D.build_messages(D.serialize_state(row["state"]), None, t_real.question, t_real.options)
            copy = verbatim.build_messages(verbatim.serialize_state(row["state"]), None,
                                           t_copy.question, t_copy.options)
            assert real == copy and t_real.names == t_copy.names, (row.get("id"), qid)
            n += 1
        # /v1/decide with a shared instructions block goes through the same build_messages
        real = D.build_messages("s", "shared guidance", "q?", ["a", "b"])
        assert real == verbatim.build_messages("s", "shared guidance", "q?", ["a", "b"])
    assert n >= 10


@pytest.mark.skipif(not (AINODE_SRC / "ainode/api/decide.py").is_file(),
                    reason="no AINode checkout (set AINODE_SRC)")
def test_route_semantics_are_ainodes_source():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import make_ainode_routes as gen
    finally:
        sys.path.remove(str(ROOT / "tools"))
    blocks = []
    for rel, names in gen.WANT.items():
        blocks += gen.definitions(AINODE_SRC / rel, names)
    live = hashlib.sha256(("\n\n\n".join(blocks) + "\n").encode()).hexdigest()
    assert live == ainode_routes.AINODE_SOURCE_SHA256, (
        "AINode's route code changed since ainode_routes.py was generated; "
        "rerun tools/make_ainode_routes.py and review the diff")


def test_contract_checks_pass(engine):
    c = engine.contract
    assert c["ok"], c["problems"]
    assert c["model_contract"]["chat_template_kwargs"] == prompt.CHAT_TEMPLATE_KWARGS
    assert c["model_contract"]["chat_template_kwargs"]["enable_thinking"] is False
    assert c["label_tokens"] == c["model_contract"]["single_token_labels"]


def test_routes_render_the_training_prompt(engine, client):
    """The prompt the /v1/systemone route hands the model is Renderer.render's, byte for byte,
    and /v1/decide renders the same text for the same question when it has no instructions."""
    fresh = prompt.Renderer(engine.tokenizer, max_tokens=2048)
    for row in REQUESTS:
        engine.calls = []
        r = client.post("/v1/systemone", json=row)
        assert r.status_code == 200, r.text
        (items, _), = engine.calls
        for item, (qid, q) in zip(items, row["questions"].items()):
            want = fresh.render(row["state"], q)
            assert item.rendered.prompt == want.prompt, qid
            assert item.rendered.cand_ids == want.cand_ids, qid
            t = verbatim.translate_one(qid, q)
            dec = engine.render_decide(verbatim.serialize_state(row["state"]), None, t.question, t.options)
            assert dec.prompt == want.prompt and dec.cand_ids == want.cand_ids, qid
            # thinking off: the template closes an empty think block before the answer token
            assert want.prompt.endswith("<think>\n\n</think>\n\n"), want.prompt[-60:]
            assert "passingAnswer" not in want.prompt


def test_example_prompt_is_pinned(engine):
    """The model card's example, rendered: pinned so a template or renderer drift is loud."""
    row = REQUESTS[0]
    got = engine.render_systemone(row["state"], row["questions"]["route"]).prompt
    assert got == (
        "<|im_start|>system\nYou are a decision function. Answer with the single letter of the "
        "best option and nothing else.<|im_end|>\n<|im_start|>user\nSTATE:\n"
        '{"ticket":"Customer says the invoice total does not match the quote."}\n\n'
        "QUESTION: Which team should take this ticket?\n\nOPTIONS:\n"
        "A. billing: an invoice, a charge or a refund\nB. support: a product question\n"
        "C. sales: a quote or a renewal\n\nAnswer with the label of one option and nothing else."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
