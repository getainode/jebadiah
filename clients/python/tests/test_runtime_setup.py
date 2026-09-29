"""What `jeb serve` and `jeb doctor` do before the first question: Ollama's auto-pull, LM Studio's
model lookup and its instructions, and AINode's /v1/decide shape end to end through the shim."""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from jebadiah_decide import Jeb, JebError
from jebadiah_decide.backends import LMStudio, Ollama

from conftest import ROUTE_TOP, TEMPS, URGENT_TOP, load

GOLD = load("example-rendered.json")


def test_ollama_pulls_a_missing_model(runtime):
    events = []

    def respond(path, b):
        if path == "/api/version":
            return 200, {"version": "0.34.4"}
        if path == "/api/tags":
            return 200, {"models": [{"name": "gpt-oss:20b"}]}
        if path == "/api/pull":
            lines = [{"status": "pulling", "total": 100, "completed": c} for c in (0, 50, 100)] + [{"status": "success"}]
            return 200, "\n".join(json.dumps(x) for x in lines).encode()
        return 404, {}
    rt = runtime(respond)
    info = Ollama(url=rt.url, model="hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0").prepare(events.append)
    pull = [r for r in rt.requests if r["path"] == "/api/pull"][0]["body"]
    assert pull == {"model": "hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0", "stream": True}
    assert any("Pulling" in e for e in events) and events[-1].strip() == "pulled"
    assert info["ollama"] == "0.34.4"


def test_ollama_present_model_is_not_pulled(runtime):
    rt = runtime(lambda path, b: (200, {"version": "x"}) if path == "/api/version" else
                 (200, {"models": [{"name": "hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0"}]}))
    Ollama(url=rt.url, model="hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0").prepare(lambda m: None)
    assert all(r["path"] != "/api/pull" for r in rt.requests)


def test_ollama_down_says_how_to_start_it():
    with pytest.raises(JebError, match="Ollama is not answering.*ollama serve"):
        Ollama(url="http://127.0.0.1:9", model="m").prepare(lambda m: None)


def test_lmstudio_finds_the_loaded_jeb(runtime):
    rt = runtime(lambda path, b: (200, {"data": [{"id": "qwen3-8b"}, {"id": "jebadiah-9b-v2"}]}))
    lm = LMStudio(url=rt.url)
    lm.prepare(lambda m: None)
    assert lm.model == "jebadiah-9b-v2"


def test_lmstudio_without_jeb_says_what_to_click(runtime):
    rt = runtime(lambda path, b: (200, {"data": [{"id": "qwen3-8b"}]}))
    with pytest.raises(JebError, match="Developer tab"):
        LMStudio(url=rt.url).prepare(lambda m: None)


def test_lmstudio_auth(runtime):
    rt = runtime(lambda path, b: (401, {"error": "unauthorized"}))
    with pytest.raises(JebError, match="wants an API key"):
        LMStudio(url=rt.url).prepare(lambda m: None)


@pytest.fixture
def shim(renderer, runtime):
    def respond(path, b):
        prompt = b["prompt"]
        route = "OPTIONS:\nA. billing" in prompt
        top = ROUTE_TOP if route else URGENT_TOP
        return 200, {"prompt_eval_count": None, "logprobs": [{"top_logprobs": [{"token": t, "logprob": v} for t, v in top]}]}
    rt = runtime(respond)

    class Tok:   # counts only; ids are placeholders
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text) // 4))

        def apply_chat_template(self, messages, tokenize=False, **kw):
            return "\n".join(m["content"] for m in messages)
    renderer.tok = Tok()
    renderer._label_ids = {chr(65 + i): 32 + i for i in range(26)}
    jeb = Jeb("ollama", url=rt.url, model="jeb", renderer=renderer, temperature_table=TEMPS)
    from jebadiah_decide.serve import make_server
    httpd = make_server(jeb, "127.0.0.1", 0, quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", rt
    httpd.shutdown()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_v1_decide_through_the_shim(shim):
    url, rt = shim
    code, out = post(url + "/v1/decide", {"state": {"ticket": "invoice wrong"}, "questions": {
        "route": {"question": "Which team?", "options": ["billing", "support", "sales"]},
        "urgent": {"question": "Is the customer blocked?", "type": "boolean"}}})
    assert code == 200, out
    d = out["decisions"]
    assert d["route"]["answer"] == "billing" and set(d["route"]["distribution"]) == {"billing", "support", "sales"}
    assert d["urgent"]["answer"] in ("yes", "no")
    assert abs(sum(d["route"]["distribution"].values()) - 1) < 1e-5
    assert out["calibration"]["applied"] is True
    body = rt.requests[0]["body"]
    assert body["raw"] is True and body["think"] is False


def test_v1_decide_errors_are_400(shim):
    url, _ = shim
    code, out = post(url + "/v1/decide", {"state": {}, "questions": {"q": {"question": "x"}}})
    assert code == 400 and "needs 'options' or a 'type'" in out["error"]["message"]
    code, out = post(url + "/v1/decide", {"state": {}, "questions": {"q": {"question": "x", "options": [str(i) for i in range(25)]}}})
    assert code == 400 and "25 options, but Ollama" in out["error"]["message"]


def test_health_and_cors(shim):
    url, _ = shim
    with urllib.request.urlopen(url + "/health") as r:
        h = json.loads(r.read())
        assert r.headers["Access-Control-Allow-Origin"] == "*"
    assert h["backend"] == "ollama" and h["max_options"] == 20
