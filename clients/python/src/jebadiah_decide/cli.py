"""jebadiah-decide ask | serve"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import BACKENDS, JebError
from .client import Jeb


def _common(p: argparse.ArgumentParser):
    p.add_argument("--backend", required=True, choices=sorted(set(BACKENDS)),
                   help="llama-server, ollama, lmstudio, vllm, mlx, or systemone (AINode, jebadiah-serve)")
    p.add_argument("--url", help="the runtime's base URL (each backend has the usual local default)")
    p.add_argument("--model", help="the name the runtime knows the model by (Ollama tag, LM Studio identifier, "
                                   "vLLM served name, AINode model id), or the MLX repo or folder")
    p.add_argument("--repo", help="the Jebadiah repo id or local folder the model came from: tokenizer, "
                                  "chat template and temperatures.json (not needed for mlx or systemone)")
    p.add_argument("--api-key", default=os.environ.get("JEBADIAH_BACKEND_KEY"),
                   help="bearer key for the runtime, if it wants one (or JEBADIAH_BACKEND_KEY)")
    p.add_argument("--no-temperatures", action="store_true", help="raw probabilities, no calibration")
    p.add_argument("--precision", default="8bit", choices=["8bit", "4bit"], help="mlx: which folder of the MLX repo")
    p.add_argument("--n-probs", type=int, default=1000, help="llama-server: how many top probabilities to read")
    p.add_argument("--top-n", type=int, default=20, help="vllm: logprobs to ask for (at most its --max-logprobs)")
    p.add_argument("--max-prompt-tokens", type=int, default=2048)


def _build(a) -> Jeb:
    extra = {}
    if a.backend == "mlx":
        extra["precision"] = a.precision
    if a.backend in ("llama-server", "llamacpp", "llama.cpp"):
        extra["n_probs"] = a.n_probs
    if a.backend == "vllm":
        extra["top_n"] = a.top_n
    return Jeb(a.backend, url=a.url, model=a.model, repo=a.repo, api_key=a.api_key,
               temperatures=not a.no_temperatures, max_prompt_tokens=a.max_prompt_tokens, **extra)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jebadiah-decide", description="Typed decisions from a Jebadiah model on your own runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer one request file and print the answers")
    _common(ask)
    ask.add_argument("--request", required=True, help="JSON file: {state, questions: {id: {type, instructions, criteria}}}")
    srv = sub.add_parser("serve", help="serve POST /v1/systemone in front of the backend")
    _common(srv)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8100)
    srv.add_argument("--key", default=os.environ.get("JEBADIAH_API_KEY"),
                     help="require this bearer key on /v1/systemone (or JEBADIAH_API_KEY)")
    a = ap.parse_args(argv)
    try:
        jeb = _build(a)
        if a.cmd == "ask":
            req = json.load(open(a.request))
            print(json.dumps(jeb.decide(req.get("state"), req.get("questions")), indent=1))
            return 0
        from .serve import make_server
        httpd = make_server(jeb, a.host, a.port, a.key)
        print(f"jebadiah-decide: POST http://{a.host}:{a.port}/v1/systemone -> {jeb.backend_name}"
              f" ({jeb.model or ''})", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    except JebError as e:
        print(f"jebadiah-decide: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
