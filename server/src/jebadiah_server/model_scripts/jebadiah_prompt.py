"""The Jebadiah prompt: AINode's own /v1/systemone -> /v1/decide rendering, then the base model's
chat template, then ONE label token. Train, eval and the local server all go through here, and
the served route (AINode) renders the same bytes because the renderer is AINode's, copied
verbatim (ainode_prompt_verbatim.py, hashed into the training config).

A question is the Jev wire shape {type, instructions, criteria}. AINode translates it to a list of
option lines with letter labels (A..Z, AA..) and the wire names each label answers to:
  choice -> the criteria keys, in criteria order (or the order the caller passes)
  noul   -> ["true", "false"]  (always this order, AINode's rule)
  score  -> the level texts in rubric order; we report them as level indices "0".."k-1"
The answer position is the token right after the generation prompt, and the candidate tokens are
the label strings themselves ("A", "B", ...), each one ordinary token at that boundary (checked
against the tokenizer at load time).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ainode_prompt_verbatim import (
    MAX_CRITERIA,
    PROMPT_SOURCE_COMMIT,
    PROMPT_SOURCE_SHA256,
    Translated,
    build_messages,
    criteria_pairs,
    option_label,
    option_text,
    serialize_state,
    translate_one,
)

# Keys that must never appear inside a question object sent to the model. The linter rejects
# them; the server refuses them.
LABEL_KEYS = {"label", "labels", "expected", "passingAnswer", "passing_answer", "answer", "answers",
              "target", "targets", "gold", "reference", "truth", "correct"}

CHAT_TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False, "thinking": False}


def flatten_text(v) -> str:
    """Instructions and descriptions must be strings on AINode's wire. Sources that ship an object
    or a list (Kev's {question, focus}) are flattened once, at conversion time, by joining the
    string values with a space. Never called on a state."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        return " ".join(s for s in (flatten_text(x) for x in v.values()) if s)
    if isinstance(v, list):
        return " ".join(s for s in (flatten_text(x) for x in v) if s)
    return json.dumps(v, ensure_ascii=False)


def wire_keys(q: dict, order: list | None = None) -> list[str]:
    """The answer keys of a question in the order the options are shown."""
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        keys = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
        if order is not None:
            assert sorted(order) == sorted(keys), "order must be a permutation of the option keys"
            keys = list(order)
        return keys
    if t == "noul":
        return ["true", "false"]
    if t == "score":
        return [str(i) for i in range(len(crit))]
    raise ValueError(f"unknown question type {t!r}")


def reorder_choice(q: dict, order: list) -> dict:
    """The same choice question with its criteria in `order` (AINode shows criteria order)."""
    crit = q["criteria"]
    if not isinstance(crit, dict):
        crit = {str(c): None for c in crit}
    return {**q, "criteria": {k: crit[k] for k in order}}


@dataclass
class Rendered:
    prompt: str            # the chat-templated text ending in the generation prompt
    keys: list[str]        # wire keys in display order
    letters: list[str]     # the label shown for each key, same order
    cand_ids: list[int]    # token id of each label at the answer boundary, same order
    truncated: bool        # the state was cut to fit the token budget
    label_scheme: str = "ainode"   # "ainode" (option_label, what /v1/decide emits) or "extended"


class Renderer:
    """Renders (state, question) exactly as AINode's /v1/systemone would, with this tokenizer's
    chat template, and knows the label token ids."""

    def __init__(self, tokenizer, max_tokens: int = 2048):
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self._label_ids: dict[str, int] = {}
        # every label AINode can emit up to 255 options must be one ordinary token at the boundary
        probe = self.render_messages("x", "q?", ["o1", "o2"])
        base = tokenizer.encode(probe, add_special_tokens=False)
        specials = set(tokenizer.all_special_ids)
        for i in range(255):
            lab = option_label(i)
            comb = tokenizer.encode(probe + lab, add_special_tokens=False)
            suffix = comb[len(base):]
            if comb[:len(base)] != base or len(suffix) != 1 or suffix[0] in specials:
                break
            self._label_ids[lab] = suffix[0]
        self.max_options = len(self._label_ids)
        if self.max_options < 26:
            raise ValueError(f"only {self.max_options} labels are single tokens in this tokenizer")
        # Past AINode's single-token range (68 labels on the Qwen3.5 tokenizer: A..Z, AA..AZ,
        # BA..BP) the served route cannot answer anyway (its cap is 20). For the LOCAL read of a
        # wider question (Jevals Banking77, 77 options) we fall back to an extended alphabet: the
        # same A..Z, then every two-letter uppercase string that is one ordinary token at the
        # boundary, in alphabetical order. Rendered.label_scheme says which alphabet was used.
        import string
        self._extended: list[tuple[str, int]] = [(L, i) for L, i in self._label_ids.items() if len(L) == 1]
        for a in string.ascii_uppercase:
            for b in string.ascii_uppercase:
                lab = a + b
                comb = tokenizer.encode(probe + lab, add_special_tokens=False)
                suffix = comb[len(base):]
                if comb[:len(base)] == base and len(suffix) == 1 and suffix[0] not in specials:
                    self._extended.append((lab, suffix[0]))
        self.max_options_extended = len(self._extended)

    def render_messages(self, state_text: str, question: str, options: list[str],
                        letters: list[str] | None = None) -> str:
        messages = build_messages(state_text, None, question, options)
        if letters is not None:
            # the extended alphabet: swap AINode's letters for ours in the option lines, nothing else
            user = messages[1]["content"]
            head, _, rest = user.partition("\nOPTIONS:\n")
            lines = rest.split("\n")
            for i in range(len(options)):
                assert lines[i].startswith(f"{option_label(i)}. ")
                lines[i] = f"{letters[i]}. " + lines[i][len(option_label(i)) + 2:]
            messages[1]["content"] = head + "\nOPTIONS:\n" + "\n".join(lines)
        return self.tok.apply_chat_template(messages, tokenize=False, **CHAT_TEMPLATE_KWARGS)

    def render(self, state, q: dict, order: list | None = None) -> Rendered:
        if q["type"] == "choice" and order is not None:
            q = reorder_choice(q, order)
        crit = q.get("criteria")
        if q["type"] == "choice" and isinstance(crit, dict) and len(crit) > MAX_CRITERIA:
            # AINode's route refuses this many options (its top-20 logprob read); the local read
            # renders them the same way and scores every label directly off the logits
            pairs = criteria_pairs("q", crit, "choice")
            t = Translated("choice", str(q.get("instructions", "")).strip(),
                           [option_text(n, d) for n, d in pairs], [n for n, _ in pairs])
        else:
            t = translate_one("q", q)
        keys = wire_keys(q, order)
        n_opt = len(t.names)
        scheme = "ainode"
        if n_opt <= self.max_options:
            letters = [option_label(i) for i in range(n_opt)]
            cand_ids = [self._label_ids[L] for L in letters]
            shown = None
        elif n_opt <= self.max_options_extended:
            letters = [L for L, _ in self._extended[:n_opt]]
            cand_ids = [i for _, i in self._extended[:n_opt]]
            shown = letters
            scheme = "extended"
        else:
            raise ValueError(f"{n_opt} options exceed the {self.max_options_extended} single-token labels")
        state_text = serialize_state(state)
        prompt = self.render_messages(state_text, t.question, t.options, shown)
        truncated = False
        n = len(self.tok.encode(prompt, add_special_tokens=False))
        if n > self.max_tokens:
            # cut the state from its end so the question, options and template survive intact
            overhead = n - len(self.tok.encode(state_text, add_special_tokens=False))
            keep = max(self.max_tokens - overhead - 4, 16)
            ids = self.tok.encode(state_text, add_special_tokens=False)[:keep]
            state_text = self.tok.decode(ids) + " [truncated]"
            prompt = self.render_messages(state_text, t.question, t.options, shown)
            truncated = True
        return Rendered(prompt, keys, letters, cand_ids, truncated, scheme)


def answer_from_probs(q: dict, keys: list[str], probs: list[float]) -> dict:
    """The Jev wire answer for one question from its probabilities over `keys` (display order).
    choice: {type, choice, confidence, probabilities}; noul: {type, noul}; score: {type, score,
    confidence, legend, probabilities}. Ties: choice -> the option listed first; score -> the
    lower level (the Jevals rule). Confidence is (p_max - 1/n) / (1 - 1/n), the chance-corrected
    top probability AINode's route also reports."""
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": round(float(probs[keys.index("true")]), 6)}
    best = 0
    for i, p in enumerate(probs):
        if p > probs[best]:
            best = i
    n = len(keys)
    p_max = float(probs[best])
    confidence = 1.0 if n == 1 else (p_max - 1.0 / n) / (1.0 - 1.0 / n)
    if t == "choice":
        return {"type": "choice", "choice": keys[best], "confidence": round(confidence, 6),
                "probabilities": {k: round(float(p), 6) for k, p in zip(keys, probs)}}
    expected = sum(int(k) * float(p) for k, p in zip(keys, probs))
    crit = q.get("criteria") or []
    legend = {str(i): (c if isinstance(c, str) else str(c)) for i, c in enumerate(crit)} if isinstance(crit, list) \
        else {str(i): k for i, k in enumerate(crit)}
    return {"type": "score", "score": round(expected, 6), "confidence": round(confidence, 6),
            "legend": legend, "probabilities": {k: round(float(p), 6) for k, p in zip(keys, probs)}}
