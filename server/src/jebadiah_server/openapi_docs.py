"""Request schemas and examples for /docs. Documentation only: the routes validate with
AINode's own code, so these never reject anything the routes would accept."""
import json
from pathlib import Path

SYSTEMONE_EXAMPLE = json.loads(
    (Path(__file__).parent / "model_scripts" / "example-request.json").read_text())

_text_or_null = {"anyOf": [{"type": "string"}, {"type": "null"}]}

SYSTEMONE_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["questions"],
    "properties": {
        "model": {"type": "string", "description": "Optional. The served repo id, its short "
                  "name, or an alias from GET /v1/models. Anything else is a 503."},
        "state": {"description": "Any JSON value. A string goes to the model verbatim; "
                  "anything else as compact JSON with sorted keys."},
        "questions": {
            "type": "object",
            "description": "Questions keyed by id. Only type, instructions and criteria are "
                           "read; other fields are ignored and never echoed.",
            "additionalProperties": {
                "type": "object",
                "required": ["type", "instructions"],
                "properties": {
                    "type": {"type": "string", "enum": ["choice", "noul", "score"]},
                    "instructions": {"type": "string"},
                    "criteria": {
                        "description": "choice: {key: description or null}, 2 to 20 keys. "
                                       "noul: optional {true, false} descriptions. score: an "
                                       "ordered list of 2 to 10 levels, or {level: description}.",
                        "anyOf": [{"type": "object", "additionalProperties": _text_or_null},
                                  {"type": "array", "items": {"type": "string"}}]},
                },
            },
        },
        "calibration": {"type": "string", "enum": ["raw"],
                        "description": "Send \"raw\" to skip the model's fitted temperatures."},
    },
}

SYSTEMONE_RESPONSE_EXAMPLE = {
    "model": "frontier-infra/jebadiah-9b-v2",
    "answers": {
        "route": {"type": "choice", "choice": "billing", "confidence": 0.93,
                  "probabilities": {"billing": 0.953333, "support": 0.03, "sales": 0.016667}},
        "urgent": {"type": "noul", "noul": 0.21},
        "quality": {"type": "score", "score": 2.7, "confidence": 0.5,
                    "legend": {"0": "poor", "1": "fair", "2": "good", "3": "excellent"},
                    "probabilities": {"0": 0.01, "1": 0.04, "2": 0.19, "3": 0.76}},
    },
    "usage": {"input_tokens": 312, "output_tokens": 3},
    "latency_ms": 180.4,
    "calibration": {"applied": True,
                    "temperatures": {"choice": 1.1863, "noul": 1.0903, "score": 1.2162}},
}

DECIDE_EXAMPLE = {
    "state": {"ticket": "Customer says the invoice total does not match the quote."},
    "questions": {
        "team": {"question": "Which team should take this ticket?",
                 "options": ["billing", "support", "sales"]},
        "escalate": {"question": "Should this be escalated?", "type": "boolean"},
        "severity": {"question": "How severe is this?", "type": "score", "min": 1, "max": 5},
    },
}

DECIDE_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["questions"],
    "properties": {
        "model": {"type": "string"},
        "state": {"description": "Any JSON value, serialized as for /v1/systemone."},
        "instructions": {"type": "string", "description": "Optional guidance appended to the "
                         "system message for every question."},
        "questions": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "type": {"type": "string", "enum": ["boolean", "score"],
                             "description": "boolean: options yes/no. score: the integers "
                                            "min..max (default 1..5). Ignored when options "
                                            "is given."},
                    "min": {"type": "integer"},
                    "max": {"type": "integer"},
                },
            },
        },
        "calibration": {"type": "string", "enum": ["raw"]},
    },
}

DECIDE_RESPONSE_EXAMPLE = {
    "model": "frontier-infra/jebadiah-9b-v2",
    "node": "gpu-box",
    "latency_ms": 175.2,
    "decisions": {
        "team": {"answer": "billing", "confidence": 0.95,
                 "distribution": {"billing": 0.95, "support": 0.03, "sales": 0.02},
                 "latency_ms": 170.0},
    },
    "usage": {"prompt_tokens": 260, "completion_tokens": 1, "calls": 1},
    "calibration": {"applied": True,
                    "temperatures": {"choice": 1.1863, "noul": 1.0903, "score": 1.2162}},
}

_err = {"application/json": {"example": {"error": {"message": "...", "type": "..."}}}}
ERROR_RESPONSES = {
    400: {"description": "/v1/decide: the request shape is wrong", "content": _err},
    401: {"description": "JEBADIAH_API_KEY is set and the bearer key is missing or wrong",
          "content": _err},
    422: {"description": "/v1/systemone: the request shape is wrong, or a choice has more "
                         "than 20 criteria, or the prompt is over the token limit",
          "content": _err},
    503: {"description": "The model is loading or failed, the model named is not served "
                         "here, or a question could not be answered. Never a partial 200.",
          "content": _err},
}
