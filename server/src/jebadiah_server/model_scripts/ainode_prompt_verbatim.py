"""AINode's prompt renderer, copied VERBATIM from the repo so training and serving render the
same bytes. Do not edit by hand: regenerate with train/make_verbatim.py.

source commit: e5c089386e0239c9eb270eeb490d181722b8da5b
files: ainode/api/decide.py (SYSTEM_PROMPT, ANSWER_INSTRUCTION, MAX_OPTIONS, TOP_LOGPROBS, BOOLEAN_OPTIONS, DecideError, option_label, option_labels, serialize_state, build_messages)
       ainode/api/systemone.py (CHOICE, NOUL, SCORE, QUESTION_TYPES, NOUL_OPTIONS, MIN_SCORE_LEVELS, MAX_SCORE_LEVELS, MAX_CRITERIA, Translated, option_text, criteria_pairs, choice_options, noul_options, score_options, translate_one, translate_questions)
prompt_source_sha256: d2660ebec28bd3f1704235bda88d24a397c1c62475e740519cb8ef2d08f25fdd  (sha256 of the copied definitions, in this order)

The served path is: systemone.translate_questions -> decide.normalize_questions (shape checks only)
-> decide.build_messages(serialize_state(state), None, question, options) -> the model's chat
template with add_generation_prompt=True and enable_thinking=False -> one label token.
"""
from __future__ import annotations

import json
from typing import Any, NamedTuple, Optional

PROMPT_SOURCE_COMMIT = "e5c089386e0239c9eb270eeb490d181722b8da5b"
PROMPT_SOURCE_SHA256 = "d2660ebec28bd3f1704235bda88d24a397c1c62475e740519cb8ef2d08f25fdd"

SYSTEM_PROMPT = ("You are a decision function. Answer with the single letter "
                 "of the best option and nothing else.")


ANSWER_INSTRUCTION = "Answer with the label of one option and nothing else."


MAX_OPTIONS = 255


TOP_LOGPROBS = 20


BOOLEAN_OPTIONS = ("yes", "no")


class DecideError(Exception):
    """A bad request shape. Carries the message the caller gets in the 4xx.

    ``/v1/decide`` answers it as a 400 and ``/v1/systemone`` as the 422 the Jev
    format specifies, so the message says what is wrong and never which status
    somebody is about to put it in.
    """


def option_label(index: int) -> str:
    """Zero-based option index to its letter label: A..Z, AA, AB, ... IU.

    Bijective base-26 (spreadsheet columns), so the scheme keeps going past Z
    without a separator and without ever colliding.
    """
    if index < 0:
        raise ValueError("option index cannot be negative")
    n = index + 1
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def option_labels(count: int) -> list[str]:
    """The labels for a question with `count` options, in option order."""
    return [option_label(i) for i in range(count)]


def serialize_state(state: Any) -> str:
    """The state as the model sees it: a string verbatim, anything else compact JSON."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, separators=(",", ":"), sort_keys=True,
                          ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise DecideError(f"'state' is not JSON-serializable: {exc}") from exc


def build_messages(state: str, instructions: Optional[str], question: str,
                   options: list[str]) -> list[dict]:
    """The chat messages for one question. Pure, so the tests can pin the text.

    The state comes BEFORE the question on purpose: every question in a request
    then shares a byte-identical prefix (system message plus state), so the
    engine's prefix cache prefills the shared part once no matter how many
    questions are asked against it.
    """
    system = SYSTEM_PROMPT
    extra = (instructions or "").strip()
    if extra:
        system = f"{SYSTEM_PROMPT}\n\n{extra}"
    lines = ["STATE:", state, "", f"QUESTION: {question}", "", "OPTIONS:"]
    lines += [f"{option_label(i)}. {opt}" for i, opt in enumerate(options)]
    lines += ["", ANSWER_INSTRUCTION]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


CHOICE = "choice"


NOUL = "noul"


SCORE = "score"


QUESTION_TYPES = (CHOICE, NOUL, SCORE)


NOUL_OPTIONS = ("true", "false")


MIN_SCORE_LEVELS = 2


MAX_SCORE_LEVELS = 10


MAX_CRITERIA = TOP_LOGPROBS


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


def option_text(name: str, description: Any) -> str:
    """One option line: the name the answer will carry, then what it means.

    The name comes first and VERBATIM because it is the string the caller's
    client compares against, and a model that has read it beside its description
    is choosing between meanings rather than between labels. The description is
    flattened to one line, because the prompt renders one option per line and a
    description with a newline in it would read as two options.
    """
    if isinstance(description, str) and description.strip():
        return f"{name}: {' '.join(description.split())}"
    return name


def criteria_pairs(key: str, criteria: Any, kind: str) -> list[tuple[str, Any]]:
    """The ``(name, description)`` pairs of an object ``criteria``, in order.

    Insertion order is the rubric order for a score, and JSON parsing preserves
    it, so nothing here sorts. A name is validated stripped and kept as written:
    the answer has to carry the caller's own key back, byte for byte, because the
    caller's code looks that key up.
    """
    if not isinstance(criteria, dict) or not criteria:
        raise DecideError(
            f"question '{key}': a {kind} question needs a non-empty 'criteria' "
            "object of {name: description}")
    pairs: list[tuple[str, Any]] = []
    for name, description in criteria.items():
        if not isinstance(name, str) or not name.strip():
            raise DecideError(f"question '{key}': every 'criteria' name must be a "
                              f"non-empty string (got {name!r})")
        if description is not None and not isinstance(description, str):
            raise DecideError(f"question '{key}': the 'criteria' description for "
                              f"'{name}' must be a string")
        pairs.append((name.strip(), description))
    return pairs


def choice_options(key: str, criteria: Any) -> tuple[list[str], list[str]]:
    """A choice question's options and the criteria keys they answer to."""
    pairs = criteria_pairs(key, criteria, "choice")
    if len(pairs) < 2:
        raise DecideError(f"question '{key}': a choice needs at least 2 'criteria' "
                          f"options, got {len(pairs)}")
    if len(pairs) > MAX_CRITERIA:
        raise DecideError(
            f"question '{key}': {len(pairs)} 'criteria' options is more than the "
            f"{MAX_CRITERIA} this node can report a probability for. One engine "
            f"call carries back the top {TOP_LOGPROBS} labels, so a wider option "
            "set would answer with a distribution missing its tail")
    return ([option_text(name, desc) for name, desc in pairs],
            [name for name, _ in pairs])


def noul_options(key: str, criteria: Any) -> tuple[list[str], list[str]]:
    """A noul's two options, always ``true`` then ``false``.

    The criteria block is optional here, and may describe one side only: some
    clients send both, some send neither, and a yes-or-no question is still
    answerable from its instructions alone. What a caller may not do is rename
    the sides, because the answer is P(true) and nothing else can stand in for
    it.
    """
    described: dict[str, Any] = {}
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise DecideError(f"question '{key}': 'criteria' must be an object of "
                              "{true: description, false: description}")
        for name, description in criteria.items():
            flat = name.strip() if isinstance(name, str) else name
            if flat not in NOUL_OPTIONS:
                raise DecideError(f"question '{key}': a noul's 'criteria' names only "
                                  f"'true' and 'false' (got {name!r})")
            if description is not None and not isinstance(description, str):
                raise DecideError(f"question '{key}': the 'criteria' description for "
                                  f"'{flat}' must be a string")
            described[flat] = description
    return ([option_text(name, described.get(name)) for name in NOUL_OPTIONS],
            list(NOUL_OPTIONS))


def score_options(key: str, criteria: Any) -> tuple[list[str], list[str]]:
    """A score's levels in rubric order: a list by position, an object by insertion.

    Both spellings are accepted because both are in the wild: the list form names
    the levels and nothing else, the object form names them and says what each
    one means. Either way position 0 is the first level the caller wrote, which
    is what the legend and the expected score are counted against.
    """
    if isinstance(criteria, list):
        names: list[str] = []
        for level in criteria:
            if not isinstance(level, str) or not level.strip():
                raise DecideError(f"question '{key}': every 'criteria' level must be "
                                  f"a non-empty string (got {level!r})")
            names.append(level.strip())
        options = list(names)
    elif isinstance(criteria, dict):
        pairs = criteria_pairs(key, criteria, "score")
        names = [name for name, _ in pairs]
        options = [option_text(name, desc) for name, desc in pairs]
    else:
        raise DecideError(
            f"question '{key}': a score question needs 'criteria', either an ordered "
            "list of levels or an object of {level: description}")
    if not MIN_SCORE_LEVELS <= len(names) <= MAX_SCORE_LEVELS:
        raise DecideError(
            f"question '{key}': a score's 'criteria' needs {MIN_SCORE_LEVELS} to "
            f"{MAX_SCORE_LEVELS} ordered levels, got {len(names)}")
    if len(set(names)) != len(names):
        raise DecideError(f"question '{key}': 'criteria' repeats a level name. Every "
                          "level must be distinct so a score names one of them")
    return options, names


def translate_one(key: str, spec: Any) -> Translated:
    """One question off the wire. Reads three fields and ignores the rest.

    ``type``, ``instructions`` and ``criteria`` are the whole question as far as
    this route is concerned, which is also exactly what JDE's
    ``questionsForWire`` sends. A field beyond them belongs to the caller's own
    code, so it is neither read nor echoed.
    """
    if not isinstance(spec, dict):
        raise DecideError(f"question '{key}' must be an object")
    kind = spec.get("type")
    if kind not in QUESTION_TYPES:
        raise DecideError(f"question '{key}': 'type' must be one of "
                          f"{', '.join(QUESTION_TYPES)} (got {kind!r})")
    instructions = spec.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise DecideError(f"question '{key}' needs a non-empty 'instructions' string")
    criteria = spec.get("criteria")
    if kind == NOUL:
        options, names = noul_options(key, criteria)
    elif kind == SCORE:
        options, names = score_options(key, criteria)
    else:
        options, names = choice_options(key, criteria)
    return Translated(kind, instructions.strip(), options, names)


def translate_questions(raw: Any) -> dict[str, Translated]:
    """Every question in the body, in the order the caller wrote them."""
    if not isinstance(raw, dict) or not raw:
        raise DecideError("'questions' must be a non-empty object of {id: question}")
    out: dict[str, Translated] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise DecideError("every question id must be a non-empty string")
        out[key] = translate_one(key, spec)
    return out

