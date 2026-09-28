"""Every HTTP backend against a fake runtime: the request it sends, and the answer it reads."""
from __future__ import annotations

import math

import pytest

from jebadiah_decide import Jeb, JebError
from jebadiah_decide.client import softmax

from conftest import ROUTE_TOP, TEMPS, URGENT_TOP, load

REQ = load("example-request.json")
GOLD = load("example-rendered.json")


def top_for(prompt):
    return ROUTE_TOP if prompt == GOLD["route"]["prompt"] else URGENT_TOP


def expected_route(temperature):
    lp = dict(ROUTE_TOP)
    return softmax([lp["A"], lp["B"], lp["C"]], temperature)


def check_answers(out, temperature=TEMPS["choice"]):
    route = out["answers"]["route"]
    want = expected_route(temperature)
    assert route["choice"] == "billing"
    for k, p in zip(["billing", "support", "sales"], want):
        assert math.isclose(route["probabilities"][k], round(p, 6), abs_tol=1e-6)
    lp = dict(URGENT_TOP)
    noul = softmax([lp["A"], lp["B"]], TEMPS["noul"])[0]
    assert math.isclose(out["answers"]["urgent"]["noul"], round(noul, 6), abs_tol=1e-6)


def test_example_numbers_match_the_card(renderer, runtime):
    """With Ollama's real top 20 and the 9B's temperatures, the route answer is the card's."""
    rt = runtime(lambda path, b: (200, {"prompt_eval_count": 110 if b["prompt"] == GOLD["route"]["prompt"] else 95,
                                        "logprobs": [{"top_logprobs": [{"token": t, "logprob": v} for t, v in top_for(b["prompt"])]}]}))
    out = Jeb("ollama", url=rt.url, model="m", renderer=renderer, temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])
    p = out["answers"]["route"]["probabilities"]
    assert (p["billing"], p["support"], p["sales"]) == pytest.approx((0.641142, 0.036095, 0.322763), abs=2e-5)


def test_ollama_request_and_readout(renderer, runtime):
    def respond(path, b):
        n = 110 if b["prompt"] == GOLD["route"]["prompt"] else 95
        return 200, {"prompt_eval_count": n,
                     "logprobs": [{"token": "A", "top_logprobs": [{"token": t, "logprob": v} for t, v in top_for(b["prompt"])]}]}
    rt = runtime(respond)
    out = Jeb("ollama", url=rt.url, model="hf.co/x:Q8_0", renderer=renderer, temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])
    check_answers(out)
    body = rt.requests[0]["body"]
    assert rt.requests[0]["path"] == "/api/generate"
    assert body["raw"] is True and body["think"] is False and body["stream"] is False
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["options"] == {"num_predict": 1, "temperature": 0}
    assert body["prompt"] == GOLD["route"]["prompt"]
    assert out["backend"] == "ollama" and out["warnings"] == []
    assert out["calibration"] == {"applied": True, "temperatures": TEMPS}


def test_prompt_token_mismatch_is_an_error(renderer, runtime):
    rt = runtime(lambda path, b: (200, {"prompt_eval_count": 7, "logprobs": [{"top_logprobs": [{"token": "A", "logprob": -1}]}]}))
    with pytest.raises(JebError, match="counted 7 prompt tokens"):
        Jeb("ollama", url=rt.url, model="m", renderer=renderer, temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])


def test_missing_label_gets_floor_and_a_warning(renderer, runtime):
    rt = runtime(lambda path, b: (200, {"prompt_eval_count": 110 if b["prompt"] == GOLD["route"]["prompt"] else 95,
                                        "logprobs": [{"top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "C", "logprob": -2.5}]}]}))
    out = Jeb("ollama", url=rt.url, model="m", renderer=renderer, temperatures=False).decide(REQ["state"], REQ["questions"])
    assert out["answers"]["route"]["probabilities"]["support"] == out["answers"]["route"]["probabilities"]["sales"]
    assert any("outside the ollama top 20" in w for w in out["warnings"])
    assert out["calibration"]["applied"] is False


def test_lmstudio_sends_messages_with_reasoning_off(renderer, runtime):
    def respond(path, b):
        prompt = b["messages"][1]["content"]
        n = 110 if prompt == GOLD["route"]["prompt"] else 95
        return 200, {"usage": {"prompt_tokens": n}, "choices": [{"logprobs": {"content": [
            {"token": "A", "top_logprobs": [{"token": t, "logprob": v} for t, v in top_for(prompt)]}]}}]}
    rt = runtime(respond)
    out = Jeb("lmstudio", url=rt.url, model="jebadiah-9b-v2", api_key="tok", renderer=renderer,
              temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])
    check_answers(out)
    r = rt.requests[0]
    assert r["path"] == "/v1/chat/completions" and r["auth"] == "Bearer tok"
    assert r["body"]["reasoning_effort"] == "none" and r["body"]["top_logprobs"] == 20 and r["body"]["max_tokens"] == 1


def test_llama_server_reads_by_token_id(renderer, runtime):
    ids = {"A": 32, "B": 33, "C": 34, "D": 35}

    def respond(path, b):
        if path == "/tokenize":
            return 200, {"tokens": list(range(110 if b["content"] == GOLD["route"]["prompt"] else 95))}
        top = [{"id": ids.get(t, 999), "token": t, "logprob": v} for t, v in top_for(b["prompt"]) if t in ids]
        return 200, {"completion_probabilities": [{"top_logprobs": top}]}
    rt = runtime(respond)
    out = Jeb("llama-server", url=rt.url, renderer=renderer, temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])
    check_answers(out)
    comp = [r for r in rt.requests if r["path"] == "/completion"][0]["body"]
    assert comp["n_probs"] == 1000 and comp["post_sampling_probs"] is False and comp["cache_prompt"] is False
    assert comp["n_predict"] == 1


def test_llama_server_tokenization_mismatch(renderer, runtime):
    rt = runtime(lambda path, b: (200, {"tokens": [1, 2, 3]}) if path == "/tokenize" else (200, {}))
    with pytest.raises(JebError, match="tokenized the prompt differently"):
        Jeb("llama-server", url=rt.url, renderer=renderer, temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])


def test_vllm_completions(renderer, runtime):
    def respond(path, b):
        n = 110 if b["prompt"] == GOLD["route"]["prompt"] else 95
        return 200, {"usage": {"prompt_tokens": n}, "choices": [{"text": "A", "logprobs": {"top_logprobs": [dict(top_for(b["prompt"]))]}}]}
    rt = runtime(respond)
    out = Jeb("vllm", url=rt.url + "/v1", model="frontier-infra/jebadiah-9b-v2", renderer=renderer,
              temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])
    check_answers(out)
    r = rt.requests[0]
    assert r["path"] == "/v1/completions" and r["body"]["logprobs"] == 20 and r["body"]["max_tokens"] == 1


def test_systemone_passthrough(runtime):
    answer = {"model": "frontier-infra/jebadiah-9b-v2", "answers": {"urgent": {"type": "noul", "noul": 0.16}},
              "calibration": {"applied": False, "temperatures": {}}}
    rt = runtime(lambda path, b: (200, answer))
    out = Jeb("ainode", url=rt.url + "/v1", model="frontier-infra/jebadiah-9b-v2", api_key="k",
              temperatures=False).decide(REQ["state"], REQ["questions"])
    assert out["answers"] == answer["answers"]
    r = rt.requests[0]
    assert r["path"] == "/v1/systemone" and r["auth"] == "Bearer k"
    assert r["body"] == {"state": REQ["state"], "questions": REQ["questions"],
                         "model": "frontier-infra/jebadiah-9b-v2", "calibration": "raw"}


def test_runtime_http_error_is_a_jeb_error(renderer, runtime):
    rt = runtime(lambda path, b: (400, {"error": "top_logprobs must be between 0 and 20"}))
    with pytest.raises(JebError, match="HTTP 400"):
        Jeb("ollama", url=rt.url, model="m", renderer=renderer, temperature_table=TEMPS).decide(REQ["state"], REQ["questions"])


def test_render_backends_need_a_repo():
    with pytest.raises(JebError, match="needs repo="):
        Jeb("ollama", model="m")
    with pytest.raises(JebError, match="needs --model"):
        Jeb("ollama", repo="x")
    with pytest.raises(JebError, match="unknown backend"):
        Jeb("nope")
