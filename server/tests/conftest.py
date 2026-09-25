"""Shared fixtures. The tokenizer, template and contract come from a real Jebadiah repository
(JEB_TEST_MODEL, default frontier-infra/jebadiah-4b-v2, weights not downloaded); the model
itself is replaced by a fake that returns fixed distributions, except in test_live_model.py."""
import math
import os

import pytest
from fastapi.testclient import TestClient

from jebadiah_server.app import create_app
from jebadiah_server.engine import Engine

TEST_MODEL = os.environ.get("JEB_TEST_MODEL", "frontier-infra/jebadiah-4b-v2")
NO_WEIGHTS = ["*.json", "*.jinja", "*.txt", "scripts/*", "LICENSE"]


def model_dir(repo: str = TEST_MODEL) -> str:
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=NO_WEIGHTS)


def fake_probs(n: int) -> list[float]:
    """A fixed, strictly decreasing distribution: option 0 wins."""
    w = [math.exp(-0.7 * i) for i in range(n)]
    s = sum(w)
    return [x / s for x in w]


class FakeEngine(Engine):
    """The real tokenizer, renderer and contract; a scorer that never touches a model."""

    def __init__(self, **kw):
        super().__init__(model_id=TEST_MODEL, **kw)
        self.model_dir = model_dir()
        self.load_tokenizer_only()
        self.status = "ready"
        self.calls = []
        self.override = None

    def score(self, items, calibrated):
        self.calls.append((items, calibrated))
        if self.override is not None:
            return self.override(items)
        return [fake_probs(len(i.rendered.cand_ids)) for i in items]


@pytest.fixture(scope="session")
def shared_engine():
    return FakeEngine()


@pytest.fixture
def engine(shared_engine):
    shared_engine.calls = []
    shared_engine.override = None
    shared_engine.status = "ready"
    shared_engine.error = None
    return shared_engine


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine, api_key=""))
