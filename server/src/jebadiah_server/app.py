"""The HTTP surface: POST /v1/systemone, POST /v1/decide, GET /v1/models, GET /health, the
playground at / and the OpenAPI docs at /docs.

Validation and answer shaping are AINode's own code (ainode_routes.py, copied verbatim), and the
prompt is the model repository's renderer, so a request answered here and the same request
answered by an AINode fleet go through the same text. What differs is the engine underneath:
AINode asks vLLM for one constrained label token and reads the top 20 logprobs; this server runs
the model itself and reads the label logits directly, every question of a request in batches.
"""
from __future__ import annotations

import hmac
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from jebadiah_server import __version__
from jebadiah_server import ainode_routes as routes
from jebadiah_server.engine import Engine, Item, finite
from jebadiah_server.model_scripts import (
    MAX_CRITERIA,
    serialize_state,
    translate_questions,
    verbatim,
)
from jebadiah_server.openapi_docs import (
    DECIDE_EXAMPLE,
    DECIDE_REQUEST_SCHEMA,
    DECIDE_RESPONSE_EXAMPLE,
    ERROR_RESPONSES,
    SYSTEMONE_EXAMPLE,
    SYSTEMONE_REQUEST_SCHEMA,
    SYSTEMONE_RESPONSE_EXAMPLE,
)

# Both copies raise their own DecideError: the verbatim renderer's (translate_questions,
# serialize_state) and the route module's (normalize_questions, calibration_mode).
BAD_REQUEST = (verbatim.DecideError, routes.DecideError)
API_KEY_ENV = "JEBADIAH_API_KEY"
OPEN_PATHS = {"/", "/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
PLAYGROUND = Path(__file__).with_name("playground.html")


def error(status: int, message: str, kind: str) -> JSONResponse:
    """The {"error": {"message", "type"}} body every AINode /v1 path answers with."""
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def unprocessable(message: str) -> JSONResponse:
    return error(422, message, "invalid_request_error")


def bad_request(message: str) -> JSONResponse:
    return error(400, message, "invalid_request_error")


def unavailable(message: str) -> JSONResponse:
    return error(503, message, "service_unavailable")


def short_name(model_id: str) -> str:
    return model_id.rstrip("/").rsplit("/", 1)[-1]


def decision_entry(options: list[str], probs: list[float], latency_ms: float) -> dict:
    """One /v1/decide decision from the model's probabilities over `options`.

    The same entry AINode's decision_from_payload builds from an engine's logprobs: the
    distribution keyed by option text and rounded to 6 places, the answer its argmax (ties to
    the option listed first), and the confidence the answer's own probability.
    """
    labels = [str(i) for i in range(len(options))]
    dist = {label: round(float(p), 6) for label, p in zip(labels, probs)}
    picked = routes.pick_answer(labels, dist, None)
    by_label = dict(zip(labels, options))
    return {"answer": by_label.get(picked),
            "confidence": dist.get(picked),
            "distribution": {by_label[label]: dist[label] for label in labels},
            "latency_ms": round(latency_ms, 1)}


def create_app(engine: Engine, api_key: Optional[str] = None,
               aliases: tuple[str, ...] = ("jebadiah",)) -> FastAPI:
    api_key = api_key if api_key is not None else (os.environ.get(API_KEY_ENV) or None)
    names = {engine.model_id, short_name(engine.model_id), *aliases}

    app = FastAPI(
        title="Jebadiah",
        version=__version__,
        summary="Typed decisions from a Jebadiah model: one forward pass per question, "
                "probabilities over the options you supply.",
        description=(
            "`POST /v1/systemone` takes the Jev wire shape (questions keyed by id, each "
            "`{type, instructions, criteria}`, types `choice`, `noul`, `score`). "
            "`POST /v1/decide` takes AINode's single-question shape. Both answer every "
            "question or none: a question that cannot be answered is a 503, never a partial "
            "200. Set `JEBADIAH_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`."),
        docs_url="/docs", redoc_url="/redoc")
    app.state.engine = engine

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if api_key and request.url.path not in OPEN_PATHS and request.method != "OPTIONS":
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {api_key}".encode()):
                resp = error(401, "missing or invalid API key: send Authorization: Bearer <key>",
                             "authentication_error")
                resp.headers["WWW-Authenticate"] = "Bearer"
                return resp
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["server-timing"] = f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
        return response

    def resolve_model(requested: Any) -> tuple[Optional[str], Optional[JSONResponse]]:
        if requested is None:
            return engine.model_id, None
        if not isinstance(requested, str) or not requested.strip():
            raise routes.DecideError("'model' must be a non-empty string when given")
        if requested.strip() not in names:
            return None, unavailable(f"this server is not serving '{requested.strip()}'; "
                                     f"it serves '{engine.model_id}'")
        return engine.model_id, None

    def calibration_block(mode: Optional[str]) -> dict:
        temps = engine.temperatures
        return {"applied": bool(temps) and mode != routes.CALIBRATION_RAW, "temperatures": temps}

    async def run(items: list[Item], calibrated: bool) -> tuple[Optional[list], Optional[str]]:
        try:
            return await run_in_threadpool(engine.score, items, calibrated), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def too_long(key: str, n: int) -> str:
        return (f"question '{key}': the prompt is {n} tokens, more than the "
                f"{engine.max_prompt_tokens} this server takes (--max-prompt-tokens). The state "
                "is refused, not cut")

    # -------------------------------------------------------------- /v1/systemone

    @app.post("/v1/systemone", tags=["decisions"], summary="Typed questions, Jev wire shape",
              openapi_extra={"requestBody": {"required": True, "content": {"application/json": {
                  "schema": SYSTEMONE_REQUEST_SCHEMA, "example": SYSTEMONE_EXAMPLE}}}},
              responses={200: {"description": "An answer for every question",
                               "content": {"application/json": {"example": SYSTEMONE_RESPONSE_EXAMPLE}}},
                         **ERROR_RESPONSES})
    async def systemone(request: Request):
        """Every question in `questions` is asked against the same `state` and answered in one
        batched pass. `choice` answers the criteria key, with the chance-corrected
        `confidence` `(n * p_max - 1) / (n - 1)` and `probabilities` per key; `noul` answers
        P(true); `score` answers the expected level index with a `legend`. The model's
        fitted temperatures are applied unless the body says `"calibration": "raw"`. More
        than 20 choice criteria is refused with a 422, never truncated."""
        started = time.monotonic()
        raw = await request.body()
        try:
            body = json.loads(raw or b"{}")
        except ValueError as exc:
            return unprocessable(f"body is not valid JSON: {exc}")
        if not isinstance(body, dict):
            return unprocessable("body must be a JSON object")
        try:
            model, refused = resolve_model(body.get("model"))
            translated = translate_questions(body.get("questions"))
            routes.decide_questions(translated)
            serialize_state(body.get("state"))
            calibration = routes.calibration_mode(body.get("calibration"))
        except BAD_REQUEST as exc:
            return unprocessable(str(exc))
        if refused is not None:
            return refused
        if not engine.ready:
            return unavailable(f"the model is {engine.status}" + (f": {engine.error}" if engine.error else ""))

        items: list[Item] = []
        for key in translated:
            rendered = engine.render_systemone(body.get("state"), body["questions"][key])
            if rendered.truncated:
                # the Renderer cuts an over-long state the way training did; the route refuses
                # instead, and counts the uncut prompt (build_messages with no instructions)
                full = engine.render_decide(serialize_state(body.get("state")), None,
                                            translated[key].question, translated[key].options)
                return unprocessable(too_long(key, engine.count_tokens(full.prompt)))
            items.append(Item(rendered, translated[key].kind, engine.count_tokens(rendered.prompt)))

        probs, failed = await run(items, calibration != routes.CALIBRATION_RAW)
        ms = (time.monotonic() - started) * 1000
        if probs is None:
            return unavailable(f"the model failed for '{model}': {failed}")

        answers: dict[str, dict] = {}
        failures: list[str] = []
        for (key, item), p in zip(translated.items(), probs):
            if len(p) != len(item.names) or not finite(p):
                failures.append(f"{key}: the model produced no finite distribution")
                continue
            answer = routes.answer_from_decision(item, decision_entry(item.options, p, ms))
            if answer is None:
                failures.append(f"{key}: the model named no option this route can read")
                continue
            answers[key] = answer
        if failures:
            return unavailable(f"the model failed for '{model}': " + "; ".join(failures[:5]))
        return {
            "model": model,
            "answers": answers,
            "usage": {"input_tokens": sum(i.tokens for i in items), "output_tokens": len(answers)},
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "calibration": calibration_block(calibration),
        }

    # ----------------------------------------------------------------- /v1/decide

    @app.post("/v1/decide", tags=["decisions"], summary="AINode's decide shape",
              openapi_extra={"requestBody": {"required": True, "content": {"application/json": {
                  "schema": DECIDE_REQUEST_SCHEMA, "example": DECIDE_EXAMPLE}}}},
              responses={200: {"description": "A decision for every question",
                               "content": {"application/json": {"example": DECIDE_RESPONSE_EXAMPLE}}},
                         **ERROR_RESPONSES})
    async def decide(request: Request):
        """AINode's `/v1/decide`: questions keyed by id, each `{question, options}` or
        `{question, type: "boolean"}` or `{question, type: "score", min, max}`, plus an
        optional shared `instructions` string that goes into the system message. Each
        decision carries `answer` (the option text), `confidence` (its probability) and the
        full `distribution`. Bad shapes are a 400. Options past the model's single-token
        labels (68 on the Qwen3.5 tokenizer) are refused. Note the model was trained on the
        `/v1/systemone` rendering, which never sends `instructions`."""
        started = time.monotonic()
        raw = await request.body()
        try:
            body = json.loads(raw or b"{}")
        except ValueError as exc:
            return bad_request(f"body is not valid JSON: {exc}")
        if not isinstance(body, dict):
            return bad_request("body must be a JSON object")
        try:
            model, refused = resolve_model(body.get("model"))
            questions = routes.normalize_questions(body.get("questions"))
            state = serialize_state(body.get("state"))
            instructions = body.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise routes.DecideError("'instructions' must be a string when given")
            calibration = routes.calibration_mode(body.get("calibration"))
        except BAD_REQUEST as exc:
            return bad_request(str(exc))
        if refused is not None:
            return refused
        if not engine.ready:
            return unavailable(f"the model is {engine.status}" + (f": {engine.error}" if engine.error else ""))

        items: list[Item] = []
        for key, spec in questions.items():
            if len(spec["options"]) > engine.max_options:
                return bad_request(
                    f"question '{key}': {len(spec['options'])} options is more than the "
                    f"{engine.max_options} this model reads as single label tokens")
            rendered = engine.render_decide(state, instructions, spec["question"], spec["options"])
            n = engine.count_tokens(rendered.prompt)
            if n > engine.max_prompt_tokens:
                return bad_request(too_long(key, n))
            items.append(Item(rendered, spec["kind"], n))

        probs, failed = await run(items, calibration != routes.CALIBRATION_RAW)
        ms = (time.monotonic() - started) * 1000
        if probs is None:
            return unavailable(f"the model failed for '{model}': {failed}")
        decisions: dict[str, dict] = {}
        failures: list[str] = []
        for (key, spec), p in zip(questions.items(), probs):
            if len(p) != len(spec["options"]) or not finite(p):
                failures.append(f"{key}: the model produced no finite distribution")
                continue
            decisions[key] = decision_entry(spec["options"], p, ms)
        if failures:
            return unavailable(f"the model failed for '{model}': " + "; ".join(failures[:5]))
        prompt_tokens = sum(i.tokens for i in items)
        return {
            "model": model,
            "node": platform.node() or None,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "decisions": decisions,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(decisions),
                      "calls": 1},
            "calibration": calibration_block(calibration),
        }

    # ------------------------------------------------------------ the rest

    @app.get("/v1/models", tags=["service"], summary="The model this server answers with")
    async def models():
        return {"object": "list", "data": [{
            "id": engine.model_id, "object": "model", "owned_by": "jebadiah",
            "revision": engine.revision, "aliases": sorted(names - {engine.model_id}),
            "ready": engine.ready, "max_choice_criteria": MAX_CRITERIA,
            "max_decide_options": engine.max_options or None,
        }]}

    @app.get("/health", tags=["service"], summary="Liveness and model status (no key needed)")
    async def health():
        body = {**engine.describe(), "authentication": bool(api_key), "version": __version__}
        return JSONResponse(body, status_code=200 if engine.ready else 503)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def playground():
        return HTMLResponse(PLAYGROUND.read_text())

    return app

