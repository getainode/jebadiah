"""The /v1/systemone shim, the vendored prompt contract, and (when a tokenizer is on disk) the real
renderer against the golden rendering."""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from jebadiah_decide import Jeb
from jebadiah_decide._contract import CONTRACT_FILES

from conftest import ROUTE_TOP, TEMPS, URGENT_TOP, load

REQ = load("example-request.json")
GOLD = load("example-rendered.json")
HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def post(url, body, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def shim(renderer, runtime):
    def respond(path, b):
        prompt = b["prompt"]
        top = ROUTE_TOP if prompt == GOLD["route"]["prompt"] else URGENT_TOP
        return 200, {"prompt_eval_count": 110 if prompt == GOLD["route"]["prompt"] else 95,
                     "logprobs": [{"top_logprobs": [{"token": t, "logprob": v} for t, v in top]}]}
    rt = runtime(respond)
    jeb = Jeb("ollama", url=rt.url, model="jeb", renderer=renderer, temperature_table=TEMPS)
    from jebadiah_decide.serve import make_server
    httpd = make_server(jeb, "127.0.0.1", 0, api_key="secret")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_shim_answers_the_jev_wire(shim):
    code, out = post(shim + "/v1/systemone", {"model": "jev-latest", **REQ}, key="secret")
    assert code == 200
    assert out["answers"]["route"]["choice"] == "billing"
    assert out["model"] == "jev-latest"
    assert out["calibration"]["applied"] is True


def test_shim_raw_calibration_is_per_request(shim):
    _, raw = post(shim + "/v1/systemone", {**REQ, "calibration": "raw"}, key="secret")
    _, cal = post(shim + "/v1/systemone", REQ, key="secret")
    assert raw["calibration"]["applied"] is False and cal["calibration"]["applied"] is True
    assert raw["answers"]["route"]["probabilities"] != cal["answers"]["route"]["probabilities"]


def test_shim_auth_and_errors(shim):
    assert post(shim + "/v1/systemone", REQ)[0] == 401
    code, out = post(shim + "/v1/systemone", {"state": {}, "questions": {}}, key="secret")
    assert code == 422 and out["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("name", CONTRACT_FILES)
def test_contract_files_are_the_servers_copies(name):
    ours = os.path.join(REPO_ROOT, "clients", "python", "src", "jebadiah_decide", "_contract", name)
    theirs = os.path.join(REPO_ROOT, "server", "src", "jebadiah_server", "model_scripts", name)
    if not os.path.exists(theirs):
        pytest.skip("not inside the jebadiah repository")
    assert open(ours, "rb").read() == open(theirs, "rb").read()


def _tokenizer_dir():
    d = os.environ.get("JEBADIAH_TEST_TOKENIZER")
    return d if d and os.path.isdir(d) else None


def test_route_file_is_the_servers_copy_but_one_import():
    from jebadiah_decide._contract import ROUTE_FILE
    ours = open(os.path.join(REPO_ROOT, "clients", "python", "src", "jebadiah_decide", "_contract", ROUTE_FILE)).read().splitlines()
    theirs_path = os.path.join(REPO_ROOT, "server", "src", "jebadiah_server", ROUTE_FILE)
    if not os.path.exists(theirs_path):
        pytest.skip("not inside the jebadiah repository")
    theirs = open(theirs_path).read().splitlines()
    diff = [(a, b) for a, b in zip(ours, theirs) if a != b]
    assert len(ours) == len(theirs) and diff == [
        ("from ainode_prompt_verbatim import MAX_OPTIONS  # the one line changed from the server copy",
         "from jebadiah_server.model_scripts import MAX_OPTIONS")]


@pytest.mark.skipif(_tokenizer_dir() is None, reason="set JEBADIAH_TEST_TOKENIZER to a Jebadiah repo folder")
def test_real_renderer_matches_golden():
    from jebadiah_decide.client import _MessageRenderer
    from jebadiah_decide.tokenizer import LightTokenizer
    tok = LightTokenizer(_tokenizer_dir())
    r = _MessageRenderer(tok, 2048)
    for qid, q in REQ["questions"].items():
        rd = r.render(REQ["state"], q)
        assert rd.prompt == GOLD[qid]["prompt"]
        assert rd.cand_ids == GOLD[qid]["cand_ids"]
        assert len(tok.encode(rd.prompt)) == GOLD[qid]["n_tokens"]
        assert tok.apply_chat_template(rd.messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) == rd.prompt


@pytest.mark.skipif(_tokenizer_dir() is None, reason="set JEBADIAH_TEST_TOKENIZER to a Jebadiah repo folder")
def test_light_tokenizer_matches_transformers():
    transformers = pytest.importorskip("transformers")
    from jebadiah_decide._contract import CHAT_TEMPLATE_KWARGS, build_messages
    from jebadiah_decide.tokenizer import LightTokenizer
    hf = transformers.AutoTokenizer.from_pretrained(_tokenizer_dir())
    lt = LightTokenizer(_tokenizer_dir())
    for state in ['{"ticket":"x"}', "héllo wörld 日本語 🙂 \t tabs\n\nnew", "x" * 3000]:
        m = build_messages(state, "Shared rules.", "Q?", ["a", "b", "c"])
        a = hf.apply_chat_template(m, tokenize=False, **CHAT_TEMPLATE_KWARGS)
        assert a == lt.apply_chat_template(m, tokenize=False, **CHAT_TEMPLATE_KWARGS)
        assert hf.encode(a, add_special_tokens=False) == lt.encode(a)
