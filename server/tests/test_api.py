"""Response shape per type, the option cap, the all-or-nothing 503, calibration and auth."""
import math

import pytest
from fastapi.testclient import TestClient

from conftest import TEST_MODEL, fake_probs
from jebadiah_server.app import create_app

EXAMPLE = {"state": {"ticket": "Customer says the invoice total does not match the quote."},
           "questions": {
               "route": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "invoices", "support": "product", "sales": "quotes"}},
               "urgent": {"type": "noul", "instructions": "The customer is blocked."},
               "level": {"type": "score", "instructions": "How upset?",
                         "criteria": ["calm", "annoyed", "angry", "furious"]}}}


def post(client, body, path="/v1/systemone", **kw):
    return client.post(path, json=body, **kw)


# ------------------------------------------------------------------ shapes


def test_answer_shapes_per_type(client):
    r = post(client, EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"model", "answers", "usage", "latency_ms", "calibration"}
    assert body["model"] == TEST_MODEL
    a = body["answers"]
    assert list(a) == ["route", "urgent", "level"]

    p3 = fake_probs(3)
    choice = a["route"]
    assert set(choice) == {"type", "choice", "confidence", "probabilities"}
    assert choice["type"] == "choice" and choice["choice"] == "billing"
    assert list(choice["probabilities"]) == ["billing", "support", "sales"]
    assert choice["probabilities"]["billing"] == pytest.approx(p3[0], abs=1e-6)
    # the inferred chance-corrected confidence: (n * p_max - 1) / (n - 1)
    assert choice["confidence"] == pytest.approx((3 * round(p3[0], 6) - 1) / 2, abs=1e-6)

    noul = a["urgent"]
    assert noul == {"type": "noul", "noul": pytest.approx(fake_probs(2)[0], abs=1e-6)}

    score = a["level"]
    p4 = fake_probs(4)
    assert set(score) == {"type", "score", "confidence", "legend", "probabilities"}
    assert score["legend"] == {"0": "calm", "1": "annoyed", "2": "angry", "3": "furious"}
    assert list(score["probabilities"]) == ["0", "1", "2", "3"]
    assert score["score"] == pytest.approx(sum(i * p for i, p in enumerate(p4)), abs=1e-5)
    assert score["confidence"] == pytest.approx((4 * round(p4[0], 6) - 1) / 3, abs=1e-6)

    assert body["usage"]["output_tokens"] == 3 and body["usage"]["input_tokens"] > 100
    assert body["calibration"]["applied"] is True
    assert set(body["calibration"]["temperatures"]) == {"choice", "noul", "score"}


def test_questions_of_one_request_go_to_the_model_together(client, engine):
    post(client, EXAMPLE)
    assert len(engine.calls) == 1
    items, calibrated = engine.calls[0]
    assert [i.kind for i in items] == ["choice", "noul", "score"] and calibrated is True


def test_choice_answers_the_callers_key_and_ignores_extra_fields(client, engine):
    q = {"type": "choice", "instructions": "pick", "passingAnswer": "b",
         "criteria": {" spaced ": None, "b": "second"}}
    engine.override = lambda items: [[0.2, 0.8]]
    r = post(client, {"state": "s", "questions": {"q": q}})
    assert r.status_code == 200
    ans = r.json()["answers"]["q"]
    assert ans["choice"] == "b" and "passingAnswer" not in ans
    assert list(ans["probabilities"]) == ["spaced", "b"]


def test_calibration_raw_opt_out(client, engine):
    r = post(client, {**EXAMPLE, "calibration": "raw"})
    assert r.status_code == 200
    assert r.json()["calibration"]["applied"] is False
    assert r.json()["calibration"]["temperatures"]  # still says what was skipped
    assert engine.calls[0][1] is False
    assert post(client, {**EXAMPLE, "calibration": "fitted"}).status_code == 422


# -------------------------------------------------------------- refusals


def test_choice_cap_is_refused_not_truncated(client, engine):
    crit = {f"o{i}": None for i in range(21)}
    r = post(client, {"state": "s", "questions": {"wide": {"type": "choice", "instructions": "x",
                                                           "criteria": crit}}})
    assert r.status_code == 422
    assert "21 'criteria' options is more than the 20" in r.json()["error"]["message"]
    assert engine.calls == []
    crit.pop("o20")
    ok = post(client, {"state": "s", "questions": {"wide": {"type": "choice", "instructions": "x",
                                                            "criteria": crit}}})
    assert ok.status_code == 200 and len(ok.json()["answers"]["wide"]["probabilities"]) == 20


@pytest.mark.parametrize("questions,needle", [
    ({}, "'questions' must be a non-empty object"),
    ({"q": {"type": "maybe", "instructions": "x"}}, "'type' must be one of"),
    ({"q": {"type": "noul"}}, "non-empty 'instructions'"),
    ({"q": {"type": "choice", "instructions": "x", "criteria": {"only": None}}}, "at least 2"),
    ({"q": {"type": "noul", "instructions": "x", "criteria": {"yes": None}}}, "names only 'true' and 'false'"),
    ({"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}}, "2 to 10"),
    ({"q": {"type": "score", "instructions": "x", "criteria": ["a", "a"]}}, "repeats a level"),
])
def test_bad_shapes_are_422(client, questions, needle):
    r = post(client, {"state": "s", "questions": questions})
    assert r.status_code == 422 and needle in r.json()["error"]["message"], r.text
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_not_json_is_422(client):
    r = client.post("/v1/systemone", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_over_long_prompt_is_refused(engine):
    engine.max_prompt_tokens, before = 300, engine.max_prompt_tokens
    engine.renderer.max_tokens = 300
    try:
        c = TestClient(create_app(engine, api_key=""))
        r = post(c, {"state": "word " * 2000, "questions": {"q": EXAMPLE["questions"]["urgent"]}})
        assert r.status_code == 422 and "is refused, not cut" in r.json()["error"]["message"]
        assert engine.calls == []
    finally:
        engine.max_prompt_tokens = before
        engine.renderer.max_tokens = before


# ------------------------------------------------------------------- 503s


def test_non_finite_answer_is_503_never_partial(client, engine):
    engine.override = lambda items: [fake_probs(3), [math.nan, math.nan], fake_probs(4)]
    r = post(client, EXAMPLE)
    assert r.status_code == 503
    assert "urgent" in r.json()["error"]["message"] and "answers" not in r.json()


def test_engine_error_is_503(client, engine):
    def boom(items):
        raise RuntimeError("MPS backend out of memory")
    engine.override = boom
    r = post(client, EXAMPLE)
    assert r.status_code == 503 and "out of memory" in r.json()["error"]["message"]
    assert r.json()["error"]["type"] == "service_unavailable"


def test_not_ready_is_503(client, engine):
    engine.status = "loading"
    assert post(client, EXAMPLE).status_code == 503
    assert client.get("/health").status_code == 503
    assert client.get("/health").json()["status"] == "loading"


def test_unknown_model_is_503_and_aliases_answer(client):
    assert post(client, {**EXAMPLE, "model": "someone/else"}).status_code == 503
    for name in (TEST_MODEL, TEST_MODEL.split("/")[-1], "jebadiah"):
        assert post(client, {**EXAMPLE, "model": name}).status_code == 200, name


# ----------------------------------------------------------------- /v1/decide


def test_decide_shape(client, engine):
    body = {"state": {"t": 1}, "instructions": "Be strict.", "questions": {
        "team": {"question": "Which team?", "options": ["billing", "support", "sales"]},
        "esc": {"question": "Escalate?", "type": "boolean"},
        "sev": {"question": "Severity?", "type": "score", "min": 1, "max": 5}}}
    r = post(client, body, "/v1/decide")
    assert r.status_code == 200, r.text
    out = r.json()
    assert set(out) == {"model", "node", "latency_ms", "decisions", "usage", "calibration"}
    team = out["decisions"]["team"]
    assert set(team) == {"answer", "confidence", "distribution", "latency_ms"}
    assert team["answer"] == "billing" and team["confidence"] == team["distribution"]["billing"]
    assert list(out["decisions"]["esc"]["distribution"]) == ["yes", "no"]
    assert list(out["decisions"]["sev"]["distribution"]) == ["1", "2", "3", "4", "5"]
    assert [i.kind for i in engine.calls[0][0]] == ["choice", "noul", "score"]
    assert "Be strict." in engine.calls[0][0][0].rendered.prompt
    assert out["usage"]["completion_tokens"] == 3


def test_decide_bad_shape_is_400_and_label_cap_refused(client, engine):
    assert post(client, {"questions": {"q": {"question": "x"}}}, "/v1/decide").status_code == 400
    wide = {"q": {"question": "x", "options": [f"o{i}" for i in range(engine.max_options + 1)]}}
    r = post(client, {"questions": wide}, "/v1/decide")
    assert r.status_code == 400 and "single label tokens" in r.json()["error"]["message"]
    ok = {"q": {"question": "x", "options": [f"o{i}" for i in range(engine.max_options)]}}
    assert post(client, {"questions": ok}, "/v1/decide").status_code == 200


# ------------------------------------------------------------------- auth


def test_auth(engine):
    c = TestClient(create_app(engine, api_key="s3cret"))
    assert post(c, EXAMPLE).status_code == 401
    assert post(c, EXAMPLE, headers={"authorization": "Bearer nope"}).status_code == 401
    assert post(c, EXAMPLE, headers={"authorization": "s3cret"}).status_code == 401
    r = post(c, EXAMPLE, headers={"authorization": "Bearer s3cret"})
    assert r.status_code == 200
    assert c.post("/v1/decide", json={"questions": {}}).status_code == 401
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"authorization": "Bearer s3cret"}).status_code == 200
    for open_path in ("/", "/health", "/docs", "/openapi.json"):
        assert c.get(open_path).status_code == 200, open_path
    assert c.get("/health").json()["authentication"] is True


def test_auth_key_from_environment(engine, monkeypatch):
    monkeypatch.setenv("JEBADIAH_API_KEY", "from-env")
    c = TestClient(create_app(engine))
    assert post(c, EXAMPLE).status_code == 401
    assert post(c, EXAMPLE, headers={"authorization": "Bearer from-env"}).status_code == 200


def test_no_key_means_open(engine, monkeypatch):
    monkeypatch.delenv("JEBADIAH_API_KEY", raising=False)
    assert post(TestClient(create_app(engine)), EXAMPLE).status_code == 200


# ---------------------------------------------------------------- service


def test_models_health_docs_playground(client):
    m = client.get("/v1/models").json()["data"][0]
    assert m["id"] == TEST_MODEL and "jebadiah" in m["aliases"] and m["max_choice_criteria"] == 20
    h = client.get("/health").json()
    assert h["status"] == "ready" and h["prompt_contract"]["ok"]
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/systemone", "/v1/decide", "/v1/models", "/health"} <= set(paths)
    html = client.get("/").text
    assert "Made in Texas" in html and "/v1/systemone" in html
    # no external requests: nothing loads from another origin
    for needle in ("http://", "https://", "//cdn", "@import"):
        assert needle not in html, needle
