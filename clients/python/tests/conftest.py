"""Offline fixtures: a fake runtime (a real HTTP server on localhost that records requests and
answers from a canned function) and a fake renderer built from the golden rendering of
example-request.json, so no test needs a tokenizer, a model or the network."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")


def load(name):
    return json.load(open(os.path.join(DATA, name)))


@dataclass
class FakeRendered:
    prompt: str
    keys: list
    letters: list
    cand_ids: list
    truncated: bool = False
    messages: list = field(default_factory=list)


class FakeTok:
    """encode() gives the golden token count for a golden prompt (ids are placeholders)."""

    def __init__(self, counts):
        self.counts = counts

    def encode(self, text, add_special_tokens=False):
        return list(range(self.counts[text]))


class FakeRenderer:
    def __init__(self):
        self.golden = load("example-rendered.json")
        self.tok = FakeTok({g["prompt"]: g["n_tokens"] for g in self.golden.values()})
        self.by_instr = {}
        req = load("example-request.json")
        for qid, q in req["questions"].items():
            self.by_instr[q["instructions"]] = self.golden[qid]

    def render(self, state, q):
        g = self.by_instr[q["instructions"]]
        return FakeRendered(g["prompt"], g["keys"], g["letters"], g["cand_ids"],
                            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": g["prompt"]}])


@pytest.fixture
def renderer():
    return FakeRenderer()


class FakeRuntime:
    def __init__(self, respond):
        self.respond = respond
        self.requests = []
        rt = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rt.requests.append({"path": self.path, "body": None, "auth": self.headers.get("Authorization")})
                self._answer(*rt.respond(self.path, None))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rt.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                self._answer(*rt.respond(self.path, body))

            def _answer(self, code, out):
                data = (out if isinstance(out, bytes) else json.dumps(out).encode())
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def runtime():
    made = []

    def make(respond):
        rt = FakeRuntime(respond)
        made.append(rt)
        return rt

    yield make
    for rt in made:
        rt.close()


# Log probabilities Ollama 0.34.4 returned for example-request.json's route question on
# jebadiah-9b-v2-Q8_0 (2026-09-28), top 20; A/B/C are the labels.
ROUTE_TOP = [("A", -0.3989), ("C", -1.2131), ("B", -3.812), ("D", -4.9415), ("E", -7.5925), ("F", -8.6171),
             ("M", -9.1085), ("c", -9.4216), ("G", -9.4668), ("S", -9.5535), ("R", -9.6702), ("H", -9.7399),
             ("I", -9.7541), ("a", -9.859), ("V", -9.872), ("P", -9.8871), ("O", -9.9423), (" A", -9.9451),
             ("T", -10.0664), ("L", -10.0992)]
URGENT_TOP = [("B", -0.17), ("A", -1.84), ("C", -6.0)]
TEMPS = {"choice": 1.1863, "noul": 1.0903, "score": 1.2162}
