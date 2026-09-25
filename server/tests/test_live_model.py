"""Against the real weights. Opt in: JEB_LIVE=1 (JEB_TEST_MODEL picks the repo, default
frontier-infra/jebadiah-4b-v2). Loads the model on the auto device and checks the route's
numbers against the model repository's own Scorer asked one question at a time, the path
scripts/decide_standalone.py takes."""
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import TEST_MODEL
from jebadiah_server.app import create_app
from jebadiah_server.engine import Engine
from jebadiah_server.model_scripts import HERE

pytestmark = pytest.mark.skipif(os.environ.get("JEB_LIVE") != "1", reason="set JEB_LIVE=1")

REQUEST = {
    **json.loads((Path(HERE) / "example-request.json").read_text()),
}
REQUEST["questions"] = {
    **REQUEST["questions"],
    "risk": {"type": "score", "instructions": "How much money is at stake?",
             "criteria": ["none", "a little", "a lot"]},
    "long": {"type": "choice", "instructions": "Which word best describes the ticket?",
             "criteria": {w: None for w in ["billing", "bug", "outage", "praise", "spam", "legal"]}},
}


@pytest.fixture(scope="module")
def live():
    engine = Engine(model_id=TEST_MODEL, batch_size=8)
    engine.load()
    assert engine.ready, engine.error
    return engine


def test_route_matches_the_model_repos_scorer(live):
    from jebadiah_server.model_scripts import model_module
    jm = model_module()
    client = TestClient(create_app(live, api_key=""))
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    answers = r.json()["answers"]
    scorer = jm.Scorer(live._model, live.tokenizer, temperatures=live.temperatures,
                       device=live.device, renderer=live.renderer)
    for qid, q in REQUEST["questions"].items():
        keys, probs = scorer.score(REQUEST["state"], {qid: q}, batch_size=1)[qid]
        got = answers[qid]
        if q["type"] == "noul":
            assert got["noul"] == pytest.approx(probs[keys.index("true")], abs=1e-2), qid
        elif q["type"] == "choice":
            best = max(range(len(probs)), key=lambda i: probs[i])
            assert got["choice"] == keys[best], qid
            for k, p in zip(keys, probs):
                assert got["probabilities"][k] == pytest.approx(p, abs=1e-2), (qid, k)
        else:
            for i, p in enumerate(probs):
                assert got["probabilities"][str(i)] == pytest.approx(p, abs=1e-2), (qid, i)


def test_raw_differs_from_fitted(live):
    client = TestClient(create_app(live, api_key=""))
    fitted = client.post("/v1/systemone", json=REQUEST).json()
    raw = client.post("/v1/systemone", json={**REQUEST, "calibration": "raw"}).json()
    assert raw["calibration"]["applied"] is False and fitted["calibration"]["applied"] is True
    assert raw["answers"]["route"]["choice"] == fitted["answers"]["route"]["choice"]
    assert raw["answers"]["route"]["probabilities"] != fitted["answers"]["route"]["probabilities"]
