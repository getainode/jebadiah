"""Minimal loopback /v1/systemone server over the Jebadiah in-process engine.

Model loading, the MPS delta-rule fix, rendering, batching and the logit read are the
Decision Index engine (eval/decision-index/jebadiah_engine.py in this repository), which
imports the model's own published scripts/ from the Hub snapshot unchanged. The only change
here: that engine refused `score` questions (the Decision Index has none); the published
jebadiah_prompt.py renders and answers them, so this subclass lets them through. Every answer
carries the unrounded softmax, as the engine already did for choice.

  python systemone_shim.py --adapter frontier-infra/jebadiah-9b-v1 --adapter-revision <sha> \
      --base Qwen/Qwen3.5-9B-Base --base-revision <sha> --port 8790

Binds 127.0.0.1 only. One request at a time (the harness is serial anyway).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "decision-index"))
from jebadiah_engine import JebadiahEngine  # noqa: E402
from decision_index.engines import Unsupported  # noqa: E402


class JebadiahSystemOne(JebadiahEngine):
    def __call__(self, state, questions):
        renderer = self.scorer.renderer
        rendered, raw = {}, {}
        for qid, q in questions.items():
            if q["type"] not in ("choice", "noul", "score"):
                raise Unsupported(f"question type {q['type']!r}")
            q2, notes = self._translate(q)
            r = self.scorer.render(state, q2)
            if r.truncated:
                raise Unsupported(f"prompt exceeds the declared context of {self.context} tokens")
            n = len(self.tok.encode(r.prompt, add_special_tokens=False))
            rendered[qid] = (r, q2["type"], n)
            raw[qid] = {"labels": r.letters, "label_scheme": r.label_scheme, "prompt_tokens": n}
            if notes:
                raw[qid]["translation"] = notes
        probs = {}
        for qid in rendered:  # JevBench sends one question per request
            probs[qid] = self.scorer.score_rendered([(rendered[qid][0], rendered[qid][1])])[0]
        answers = {}
        for qid, q in questions.items():
            r = rendered[qid][0]
            a = self.answer_from_probs(q, r.keys, probs[qid])
            if q["type"] == "noul":
                a["noul"] = float(probs[qid][r.keys.index("true")])
            else:
                a["probabilities"] = dict(zip(r.keys, probs[qid]))
            answers[qid] = a
            raw[qid]["probabilities"] = dict(zip(r.keys, probs[qid]))
        usage = {"input_tokens": sum(v[2] for v in rendered.values()), "output_tokens": 0}
        return {"model": self.model_id, "answers": answers, "usage": usage}, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    t0 = time.time()
    eng = JebadiahSystemOne(adapter=a.adapter, adapter_revision=a.adapter_revision,
                            base=a.base, base_revision=a.base_revision)
    eng.model_id = a.name
    print(f"[shim] loaded {a.name} in {time.time() - t0:.1f}s", flush=True)
    print("[shim] provenance " + json.dumps(eng.provenance, sort_keys=True), flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": eng.model_id})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            t = time.perf_counter()
            try:
                resp, raw = eng(req["state"], req["questions"])
            except Unsupported as e:
                return self._send(422, {"error": str(e)})
            eng.synchronize()
            resp["runtime"] = {"engine_s": time.perf_counter() - t, "raw": raw}
            self._send(200, resp)

    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
