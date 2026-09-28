"""A small /v1/systemone server in front of any backend, standard library only.

It exists so a client written for the Jev wire (JDE's jevJudge, TypeSafe's SDK, a bench) can use a
Jeb running in Ollama, LM Studio, llama-server, vLLM or MLX without a second copy of the renderer.

    jebadiah-decide serve --backend ollama --model hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0 \
        --repo frontier-infra/jebadiah-9b-v2-GGUF --port 8100

Requests are answered one at a time (a lock around the model), which is what a local runtime
does anyway. Errors come back as {"error": {"message", "type"}}: 422 for a question it cannot read,
502 when the runtime failed.
"""
from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import JebError


def make_server(jeb, host: str = "127.0.0.1", port: int = 8100, api_key: str | None = None) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "jebadiah-decide"

        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, kind: str, message: str):
            self._send(code, {"error": {"message": message, "type": kind}})

        def log_message(self, fmt, *args):  # one short line per request, no bodies
            print(f"{self.address_string()} {self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok", "backend": jeb.backend_name, "model": jeb.model,
                                        "calibration": {"applied": bool(jeb.calibrated and jeb.temps),
                                                        "temperatures": jeb.temps}})
            self._error(404, "not_found", "GET /health or POST /v1/systemone")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._error(404, "not_found", "POST /v1/systemone")
            if api_key:
                given = self.headers.get("Authorization", "")
                if not hmac.compare_digest(given.encode(), ("Bearer " + api_key).encode()):
                    return self._error(401, "auth_error", "missing or wrong bearer key")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except ValueError as e:
                return self._error(422, "invalid_request_error", f"body is not valid JSON: {e}")
            raw = body.get("calibration") == "raw"
            try:
                with lock:
                    saved = jeb.calibrated
                    jeb.calibrated = saved and not raw
                    try:
                        out = jeb.decide(body.get("state"), body.get("questions"))
                    finally:
                        jeb.calibrated = saved
            except JebError as e:
                msg = str(e)
                bad_request = msg.startswith("question") or msg.startswith("'questions'")
                return self._error(422 if bad_request else 502,
                                   "invalid_request_error" if bad_request else "backend_error", msg)
            if body.get("model"):
                out["model"] = body["model"]
            self._send(200, out)

    return ThreadingHTTPServer((host, port), Handler)
