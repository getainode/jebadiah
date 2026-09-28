"""`jeb serve`: AINode's decision wire on localhost, in front of any runtime. Standard library only.

    POST /v1/systemone   the Jev shape (what JDE's jevJudge and TypeSafe's SDK send)
    POST /v1/decide      AINode's decide shape
    GET  /health, GET /v1/models

Requests are answered one at a time (a lock around the runtime), which is what a local runtime
does anyway. Errors come back as {"error": {"message", "type"}}: 422 (systemone) or 400 (decide)
for a request it cannot answer as asked, 502 when the runtime failed, 401 for a wrong key.
Browsers can call it from any origin (CORS is open), since it only listens on localhost unless
you pass --host.
"""
from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import JebError

CLIENT_ERRORS = ("question", "'questions'", "'state'", "'instructions'", "'calibration'", "every question")


def make_server(jeb, host: str = "127.0.0.1", port: int = 8100, api_key: str | None = None,
                quiet: bool = False) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "jeb"

        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _error(self, code: int, kind: str, message: str):
            self._send(code, {"error": {"message": message, "type": kind}})

        def log_message(self, fmt, *args):
            if not quiet:
                print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok", "backend": jeb.backend_name, "model": jeb.model,
                                        "calibration": {"applied": bool(jeb.calibrated and jeb.temps),
                                                        "temperatures": jeb.temps},
                                        "max_options": jeb.cap})
            if path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": jeb.model or "jebadiah", "object": "model",
                                                                     "owned_by": "frontier-infra"}]})
            self._error(404, "not_found", "POST /v1/systemone or /v1/decide, GET /health")

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path not in ("/v1/systemone", "/v1/decide"):
                return self._error(404, "not_found", "POST /v1/systemone or /v1/decide")
            if api_key:
                given = self.headers.get("Authorization", "")
                if not hmac.compare_digest(given.encode(), ("Bearer " + api_key).encode()):
                    return self._error(401, "auth_error", "missing or wrong bearer key")
            bad = 422 if path == "/v1/systemone" else 400
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except ValueError as e:
                return self._error(bad, "invalid_request_error", f"body is not valid JSON: {e}")
            try:
                with lock:
                    if path == "/v1/systemone":
                        cal = body.get("calibration")
                        if cal not in (None, "raw"):
                            raise JebError(f"'calibration' must be \"raw\" when given (got {cal!r})")
                        out = jeb.decide(body.get("state"), body.get("questions"),
                                         calibrated=jeb.calibrated and cal != "raw")
                    else:
                        out = jeb.decide_ainode(body)
            except JebError as e:
                msg = str(e)
                client = msg.startswith(CLIENT_ERRORS) or "options" in msg.split(":")[0]
                return self._error(bad if client else 502, "invalid_request_error" if client else "backend_error", msg)
            if body.get("model"):
                out["model"] = body["model"]
            self._send(200, out)

    return ThreadingHTTPServer((host, port), Handler)
