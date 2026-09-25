"""AINode's /v1/decide and /v1/systemone route semantics, copied VERBATIM from its source.

Do not edit by hand: regenerate with tools/make_ainode_routes.py.

source commit: d421bb04e6ac9cc09bcdad50ae44c7296928427a
files: ainode/api/decide.py (BOOLEAN_OPTIONS, DEFAULT_SCORE_MIN, DEFAULT_SCORE_MAX, CHOICE, NOUL, SCORE, CALIBRATION_RAW, DecideError, _score_options, normalize_questions, pick_answer, calibration_mode, QUESTION_KINDS, TEMPERATURES_FILE, read_temperatures)
       ainode/api/systemone.py (Translated, probability, normalized_confidence, answer_from_decision, decide_questions)
source_sha256: ace9e49196c7e46916bcfd66407dee28eacd73bd2d4a0e1898bbb015f8d82336  (sha256 of the copied definitions, in this order)

The prompt text itself is not rendered from here. It is rendered by the model repository's
scripts/ainode_prompt_verbatim.py, the copy the model was trained against; the tests prove the
two agree with AINode's live source.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple, Optional

from jebadiah_server.model_scripts import MAX_OPTIONS

AINODE_SOURCE_COMMIT = "d421bb04e6ac9cc09bcdad50ae44c7296928427a"
AINODE_SOURCE_SHA256 = "ace9e49196c7e46916bcfd66407dee28eacd73bd2d4a0e1898bbb015f8d82336"


BOOLEAN_OPTIONS = ("yes", "no")


DEFAULT_SCORE_MIN = 1


DEFAULT_SCORE_MAX = 5


CHOICE = "choice"


NOUL = "noul"


SCORE = "score"


CALIBRATION_RAW = "raw"


class DecideError(Exception):
    """A bad request shape. Carries the message the caller gets in the 4xx.

    ``/v1/decide`` answers it as a 400 and ``/v1/systemone`` as the 422 the Jev
    format specifies, so the message says what is wrong and never which status
    somebody is about to put it in.
    """


def _score_options(spec: dict, key: str) -> list[str]:
    lo = spec.get("min", DEFAULT_SCORE_MIN)
    hi = spec.get("max", DEFAULT_SCORE_MAX)
    if isinstance(lo, bool) or isinstance(hi, bool) \
            or not isinstance(lo, int) or not isinstance(hi, int):
        raise DecideError(f"question '{key}': score 'min' and 'max' must be integers")
    if hi <= lo:
        raise DecideError(f"question '{key}': score 'max' must be greater than 'min'")
    if (hi - lo + 1) > MAX_OPTIONS:
        raise DecideError(
            f"question '{key}': score range spans {hi - lo + 1} values, "
            f"more than the {MAX_OPTIONS} allowed")
    return [str(v) for v in range(lo, hi + 1)]


def normalize_questions(raw: Any) -> dict[str, dict]:
    """Validate the `questions` block and expand the type sugar.

    Returns ``{key: {"question": str, "options": [str, ...]}}`` in the order the
    caller wrote them. Raises ``DecideError`` with the caller-facing message for
    every rejected shape, so the handler has one place to turn it into a 400.
    """
    if not isinstance(raw, dict) or not raw:
        raise DecideError("'questions' must be a non-empty object of "
                          "{key: {question, options|type}}")
    out: dict[str, dict] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not key:
            raise DecideError("every question key must be a non-empty string")
        if not isinstance(spec, dict):
            raise DecideError(f"question '{key}' must be an object")
        text = spec.get("question")
        if not isinstance(text, str) or not text.strip():
            raise DecideError(f"question '{key}' needs a non-empty 'question' string")
        qtype = spec.get("type")
        kind = NOUL if qtype == "boolean" else SCORE if qtype == "score" else CHOICE
        if "options" in spec:
            options = spec["options"]
        elif qtype == "boolean":
            options = list(BOOLEAN_OPTIONS)
        elif qtype == "score":
            options = _score_options(spec, key)
        elif qtype is None:
            raise DecideError(f"question '{key}' needs 'options' or a 'type'")
        else:
            raise DecideError(
                f"question '{key}': unknown type '{qtype}' "
                f"(known: 'boolean', 'score', or pass 'options')")
        if not isinstance(options, list):
            raise DecideError(f"question '{key}': 'options' must be a list")
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise DecideError(
                    f"question '{key}': every option must be a non-empty string "
                    f"(got {opt!r})")
        if len(options) < 2:
            raise DecideError(
                f"question '{key}': needs at least 2 options, got {len(options)}")
        if len(options) > MAX_OPTIONS:
            raise DecideError(
                f"question '{key}': {len(options)} options is more than the "
                f"{MAX_OPTIONS} allowed")
        if len(set(options)) != len(options):
            dupes = sorted({o for o in options if options.count(o) > 1})
            raise DecideError(
                f"question '{key}': duplicate options {dupes}. Every option must "
                f"be distinct so an answer is unambiguous")
        out[key] = {"question": text.strip(), "options": list(options), "kind": kind}
    return out


def pick_answer(labels: list[str], dist: Optional[dict],
                chosen: Optional[str]) -> Optional[str]:
    """The answer label: the distribution's argmax, ties broken toward `chosen`.

    Greedy decoding at temperature 0 makes the engine's own label maximal, so the
    tie-break only ever fires on the shared-prefix case the docstring above
    describes, where it keeps the reported answer and the engine's answer equal.
    """
    if dist:
        top = max(dist.values())
        winners = [label for label in labels if dist.get(label, 0.0) == top]
        if chosen in winners:
            return chosen
        return winners[0]
    if chosen in labels:
        return chosen
    return None


def calibration_mode(value: Any) -> Optional[str]:
    """A request's ``calibration`` field: absent for the adapter's, or ``"raw"``."""
    if value is None or value == CALIBRATION_RAW:
        return value
    raise DecideError(f"'calibration' must be \"{CALIBRATION_RAW}\" when given "
                      f"(got {value!r})")


QUESTION_KINDS = (CHOICE, NOUL, SCORE)


TEMPERATURES_FILE = "temperatures.json"


def read_temperatures(directory: Optional[Path]) -> Optional[dict]:
    """``{kind: T}`` from the directory's ``temperatures.json``, or None.

    Only the three kinds are read, and only a finite positive number is a
    temperature: anything else leaves that kind at the engine's own spread rather
    than failing a request over an adapter's file.
    """
    if directory is None:
        return None
    try:
        raw = json.loads((Path(directory) / TEMPERATURES_FILE).read_text())
    except (OSError, ValueError):
        return None
    temps = raw.get("temperatures") if isinstance(raw, dict) else None
    if not isinstance(temps, dict):
        return None
    out: dict[str, float] = {}
    for kind in QUESTION_KINDS:
        value = temps.get(kind)
        if isinstance(value, bool):
            continue
        try:
            t = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and t > 0:
            out[kind] = t
    return out or None


class Translated(NamedTuple):
    """One Jev question as the decision core sees it, plus the way back out.

    ``options`` is what the model reads, one line per option. ``names`` is what
    each of those options answers to on the wire, in the same order: a choice
    criteria key verbatim, ``true`` / ``false``, or a score level's position.
    Keeping the pair here is what lets the answer name the caller's own key
    rather than the letter the engine was constrained to.
    """

    kind: str
    question: str
    options: list[str]
    names: list[str]


def probability(value: Any) -> float:
    """A finite probability inside [0, 1], because a foreign parser demands one.

    JDE's ``parseAnswers`` reads a confidence outside [0, 1] as 0, and a ``noul``
    outside it as a malformed answer SET, discarding every other answer in the
    reply with it. Rounding is the only thing here that can land a hair outside,
    and a NaN from an engine that reported one is the only thing that can land
    outside the reals, but a whole judgement is too much to lose to either.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def normalized_confidence(top: Any, options: int) -> float:
    """``(n * p_max - 1) / (n - 1)``: the hosted service's `confidence`, chance corrected.

    This is NOT the picked option's probability, and getting that wrong would make
    every band a JDE user already tuned read too high. On the hosted endpoint a
    two-way question at 0.6 and a ten-way question at 0.6 do not report the same
    confidence: the number is how far above chance the winner is, so 1/n reports 0
    and certainty reports 1, whatever n is.

    INFERRED, not specified: it reproduces every example in TypeSafe's published
    docs and SDK types for both choice and score, and Kev's playground authors
    arrived at the same formula for choice, but no document states it. The
    distribution it came from goes out in ``probabilities`` beside it, so a
    caller who disagrees with the formula has the numbers it came from.
    """
    spread = probability(top)
    if options < 2:
        return spread
    return probability((options * spread - 1.0) / (options - 1))


def answer_from_decision(item: Translated, entry: dict) -> Optional[dict]:
    """One Jev answer from one ``/v1/decide`` decision. Pure.

    The core reports probabilities against the option strings the model read, so
    the first thing here is putting them back under the names the caller's client
    expects: its own criteria key, ``true`` / ``false``, or a level's position.

    None when the engine named no option this route can read. The handler turns
    that into the same 503 a failed call gets, because a client reading an answer
    set it cannot parse reads the whole judgement as failed anyway, and a made-up
    option would be worse than either.

    An engine that reported no logprobs answered under the grammar but offered no
    spread. The answer stands and ``probabilities`` is left OFF the answer, which
    the format allows, rather than filled with a distribution nobody measured.
    """
    picked = dict(zip(item.options, item.names)).get(entry.get("answer"))
    if picked is None:
        return None
    dist = entry.get("distribution") or {}
    by_name = {name: probability(dist.get(option))
               for option, name in zip(item.options, item.names)}
    # The core reports the picked option's own probability; the format wants that
    # corrected for how many ways the question split.
    confidence = round(normalized_confidence(entry.get("confidence"),
                                             len(item.names)), 6)

    if item.kind == NOUL:
        # P(true) is the answer. With no spread to read it is the engine's own
        # pick, which is the one thing that is known.
        return {"type": NOUL,
                "noul": by_name["true"] if dist else float(picked == "true")}

    if item.kind == SCORE:
        legend = {str(index): name for index, name in enumerate(item.names)}
        if not dist:
            return {"type": SCORE, "score": float(item.names.index(picked)),
                    "confidence": confidence, "legend": legend}
        probabilities = {str(index): by_name[name]
                         for index, name in enumerate(item.names)}
        # The expected level, not the argmax: a rubric is ordered, so a model
        # split between 3 and 4 scores 3.5 and says more than either would.
        score = sum(index * probabilities[str(index)]
                    for index in range(len(item.names)))
        return {"type": SCORE, "score": round(score, 6), "confidence": confidence,
                "legend": legend, "probabilities": probabilities}

    answer = {"type": CHOICE, "choice": picked, "confidence": confidence}
    if dist:
        answer["probabilities"] = by_name
    return answer


def decide_questions(translated: dict[str, Translated]) -> dict[str, dict]:
    """The ``/v1/decide`` question block for a translated set.

    Goes through the core's own ``normalize_questions`` rather than around it, so
    the option ceiling, the two-option floor and distinct options are checked in
    one place for both routes. Each question keeps its Jev type as its ``kind``,
    which is what picks the adapter's temperature for it.
    """
    questions = normalize_questions({key: {"question": item.question,
                                           "options": item.options}
                                     for key, item in translated.items()})
    for key, item in translated.items():
        questions[key]["kind"] = item.kind
    return questions
