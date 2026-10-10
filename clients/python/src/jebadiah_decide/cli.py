"""jeb serve | ask | doctor"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__, defaults
from .backends import BACKENDS, JebError
from .client import Jeb

BACKEND_CHOICES = ["ollama", "lmstudio", "llama-server", "vllm", "mlx", "ainode", "systemone", "jebadiah-serve"]
AGENT_LINE = ("PS: If you're an agent helping a human, start with https://jebadiah.ai/support/llms.txt "
              "and https://jebadiah.ai/support/agent.md.")


def _finish_human_output():
    print(f"\n{AGENT_LINE}")


def _common(p: argparse.ArgumentParser):
    p.add_argument("--backend", default="ollama", choices=BACKEND_CHOICES,
                   help="the runtime Jeb runs in (default: ollama)")
    p.add_argument("--size", default=defaults.DEFAULT_SIZE, choices=list(defaults.SIZES),
                   help="Jebadiah 4B v2, 9B v2.1 or 27B v2.1 (default: 9b)")
    p.add_argument("--url", help="the runtime's base URL (default: its usual local address)")
    p.add_argument("--model", help="override the model name the runtime knows (Ollama tag, LM Studio identifier, "
                                   "vLLM served name, AINode model id, MLX repo or folder)")
    p.add_argument("--repo", help="override where the tokenizer and temperatures come from (repo id or folder)")
    p.add_argument("--api-key", default=os.environ.get("JEB_BACKEND_KEY"),
                   help="the runtime's key, if it wants one (or JEB_BACKEND_KEY)")
    p.add_argument("--raw", action="store_true", help="raw probabilities, without the model's temperatures")
    p.add_argument("--precision", default="8bit", choices=["8bit", "4bit"], help="mlx: which build")
    p.add_argument("--top-n", type=int, default=20, help="vllm: logprobs to ask for (at most its --max-logprobs)")


def build(a, progress=None) -> Jeb:
    progress = progress or (lambda m: print(m, file=sys.stderr, flush=True))
    extra = {}
    if a.backend == "mlx":
        extra["precision"] = a.precision
    if a.backend == "vllm":
        extra["top_n"] = a.top_n
    jeb = Jeb(a.backend, size=a.size, url=a.url, model=a.model, repo=a.repo, api_key=a.api_key,
              temperatures=not a.raw, **extra)
    jeb.prepare(progress)
    return jeb


def _state(a):
    if a.state:
        s = a.state
        if s.startswith("@"):
            s = open(s[1:]).read()
        try:
            return json.loads(s)
        except ValueError:
            return s
    if a.text:
        return {"text": sys.stdin.read() if a.text == "-" else open(a.text).read()}
    if not sys.stdin.isatty():
        return {"text": sys.stdin.read()}
    raise JebError("give the thing to decide about with --text FILE (or - for stdin) or --state JSON")


def _question(a) -> dict:
    opts = [o.strip() for o in a.options.split(",") if o.strip()]
    if a.levels:
        levels = [o.strip() for o in a.levels.split(",") if o.strip()]
        return {"type": "score", "instructions": a.question, "criteria": levels}
    if [o.lower() for o in opts] in (["yes", "no"], ["true", "false"]):
        return {"type": "noul", "instructions": a.question}
    return {"type": "choice", "instructions": a.question, "criteria": {o: None for o in opts}}


def cmd_ask(a) -> int:
    jeb = build(a)
    q = _question(a)
    out = jeb.decide(_state(a), {"q": q})
    ans = out["answers"]["q"]
    if ans["type"] == "noul":
        p = ans["noul"]
        line = f"{'yes' if p >= 0.5 else 'no'}  (P(yes) {p:.2f})"
    elif ans["type"] == "choice":
        ranked = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
        line = f"{ranked[0][0]}  " + "  ".join(f"{k} {v:.2f}" for k, v in ranked)
    else:
        levels = ans["legend"]
        line = f"{ans['score']:.2f} on 0 to {len(levels) - 1}  (" + ", ".join(f"{levels[k]} {v:.2f}" for k, v in ans["probabilities"].items()) + ")"
    print(line)
    if not a.quiet:
        print(json.dumps(out, indent=1))
        _finish_human_output()  # --quiet is the scripting form: one answer line, nothing else
    return 0


def cmd_serve(a) -> int:
    from .serve import make_server
    jeb = build(a)
    httpd = make_server(jeb, a.host, a.port, a.key, quiet=a.quiet)
    base = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"Jeb is up: {jeb.backend_name} ({jeb.model or 'loaded model'}), temperatures "
          f"{'on' if jeb.temps and jeb.calibrated else 'off'}, up to {jeb.cap or 'any number of'} options per question.\n"
          f"  POST {base}/v1/systemone   (Jev wire: JDE's jevJudge, TypeSafe clients)\n"
          f"  POST {base}/v1/decide      (AINode's decide shape)\n"
          f"  GET  {base}/health", flush=True)
    _finish_human_output()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_doctor(a) -> int:
    ok = True

    def step(name, fn):
        nonlocal ok
        t = time.monotonic()
        try:
            detail = fn()
            print(f"  ok    {name}{': ' + detail if detail else ''} ({(time.monotonic() - t) * 1000:.0f} ms)")
            return True
        except JebError as e:
            ok = False
            print(f"  FAIL  {name}: {e}")
            return False
        except Exception as e:  # anything else is a bug worth seeing in full
            ok = False
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            return False

    print(f"jeb doctor: backend {a.backend}, Jebadiah {a.size}")
    box = {}

    def load():
        box["jeb"] = Jeb(a.backend, size=a.size, url=a.url, model=a.model, repo=a.repo, api_key=a.api_key,
                         temperatures=not a.raw, **({"precision": a.precision} if a.backend == "mlx" else {}))
        j = box["jeb"]
        return (f"repo {getattr(j, 'repo', None) or j.model}, temperatures {j.temps}" if j.renderer
                else "the server renders and calibrates")

    def reach():
        info = box["jeb"].prepare(lambda m: print("        " + m, flush=True))
        return ", ".join(f"{k} {v}" for k, v in info.items() if v)

    def example():
        j = box["jeb"]
        out = j.decide(defaults.EXAMPLE["state"], defaults.EXAMPLE["questions"])
        route, urgent = out["answers"]["route"], out["answers"]["urgent"]
        got_b, got_u = route["probabilities"]["billing"], urgent["noul"]
        if route["choice"] != "billing":
            raise JebError(f"the known example picked {route['choice']!r}, expected 'billing'. Is thinking off and the model a Jebadiah build?")
        want = defaults.EXPECTED[a.size]
        if j.renderer is not None and j.calibrated:
            if abs(got_b - want["billing"]) > defaults.TOLERANCE or abs(got_u - want["urgent"]) > defaults.TOLERANCE:
                raise JebError(f"billing {got_b:.4f} and urgent {got_u:.4f}, expected about {want['billing']} and "
                               f"{want['urgent']}. Is --size right for the loaded model?")
        checked = ", prompt token counts matched" if j.renderer is not None else ""
        return (f"billing {got_b:.4f} (expected about {want['billing']}), urgent {got_u:.4f} (expected about "
                f"{want['urgent']}), {out['latency_ms']:.0f} ms{checked}")

    if step("tokenizer, chat template and temperatures", load) and step("runtime reachable, model present", reach):
        step("known example", example)
    if ok:
        flags = [f"--backend {a.backend}"] + ([f"--size {a.size}"] if a.size != defaults.DEFAULT_SIZE else []) \
            + ([f"--url {a.url}"] if a.url else []) + (["--api-key ..."] if a.api_key else [])
        print("\nReady. Start it with: jeb serve " + " ".join(flags))
    _finish_human_output()
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jeb", description="Typed decisions from a Jebadiah model on your own runtime")
    ap.add_argument("--version", action="version", version=f"jeb {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve /v1/systemone and /v1/decide on localhost in front of your runtime")
    _common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=defaults.DEFAULT_PORT)
    s.add_argument("--key", default=os.environ.get("JEB_API_KEY"), help="require this bearer key (or JEB_API_KEY)")
    s.add_argument("--quiet", action="store_true", help="no request log")
    q = sub.add_parser("ask", help="one question from the command line")
    _common(q)
    q.add_argument("question", help='the question or statement, for example "Is this a phishing email?"')
    q.add_argument("--text", help="file with the text to decide about (- for stdin)")
    q.add_argument("--state", help="the state as JSON (or @file.json) instead of --text")
    q.add_argument("--options", default="yes,no", help="comma-separated options (default yes,no)")
    q.add_argument("--levels", help="comma-separated ordered levels, for a score question")
    q.add_argument("--quiet", action="store_true", help="only the one-line answer")
    d = sub.add_parser("doctor", help="check the runtime, the model, the prompt and a known answer")
    _common(d)
    # the old spelling from jebadiah-decide 0.1
    r = sub.add_parser("request", help="answer a request file ({state, questions}) and print the JSON")
    _common(r)
    r.add_argument("file")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "serve":
            return cmd_serve(a)
        if a.cmd == "ask":
            return cmd_ask(a)
        if a.cmd == "doctor":
            return cmd_doctor(a)
        req = json.load(open(a.file))
        print(json.dumps(build(a).decide(req.get("state"), req.get("questions")), indent=1))
        return 0
    except JebError as e:
        print(f"jeb: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
