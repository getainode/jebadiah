"""`Jeb`: one request format (the Jev wire: a state plus typed questions) over every runtime.

    from jebadiah_decide import Jeb
    jeb = Jeb("ollama", model="hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0",
              repo="frontier-infra/jebadiah-9b-v2-GGUF")
    jeb.decide({"ticket": "..."}, {"route": {"type": "choice", "instructions": "...",
                                              "criteria": {"billing": "...", "support": "..."}}})

For a runtime that only returns log probabilities (llama-server, Ollama, LM Studio, vLLM) the prompt
is rendered here, exactly as AINode renders it, with the tokenizer and chat template from `repo`,
and the model's per-type temperatures from `repo`'s temperatures.json are applied. MLX runs in this
process. A /v1/systemone server (AINode, jebadiah-serve) does all of that itself.
"""
from __future__ import annotations

import json
import math
import os
import time

# transformers prints an advisory about a missing torch on import; only the tokenizer is used here
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from ._contract import CHAT_TEMPLATE_KWARGS, DecideError, Renderer, answer_from_probs, build_messages, option_label
from .backends import BACKENDS, MLX, JebError, SystemOne

REPO_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
              "special_tokens_map.json", "temperatures.json"]


class _MessageRenderer(Renderer):
    """The contract's Renderer, also keeping the chat messages it rendered, for a runtime that
    applies the template itself (LM Studio)."""

    def render_messages(self, state_text, question, options, letters=None):
        messages = build_messages(state_text, None, question, options)
        if letters is not None:
            user = messages[1]["content"]
            head, _, rest = user.partition("\nOPTIONS:\n")
            lines = rest.split("\n")
            for i in range(len(options)):
                lines[i] = f"{letters[i]}. " + lines[i][len(option_label(i)) + 2:]
            messages[1]["content"] = head + "\nOPTIONS:\n" + "\n".join(lines)
        self.last_messages = messages
        return self.tok.apply_chat_template(messages, tokenize=False, **CHAT_TEMPLATE_KWARGS)

    def render(self, state, q, order=None):
        rd = super().render(state, q, order)
        rd.messages = self.last_messages
        return rd


def resolve_repo(repo: str, revision: str | None = None) -> str:
    """A local folder as is; a Hub repo id downloaded to the cache (tokenizer, template and
    temperatures only, a few MB, never the weights)."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, allow_patterns=REPO_FILES)


def read_temperatures(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    return {k: float(v) for k, v in json.load(open(path))["temperatures"].items()}


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Softmax over the labels only, after dividing by the temperature. log p = logit minus a
    constant, so log probabilities and logits give the same result."""
    z = [x / temperature for x in values]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


class Jeb:
    def __init__(self, backend: str = "llama-server", *, url: str | None = None, model: str | None = None,
                 repo: str | None = None, api_key: str | None = None, temperatures: bool = True,
                 max_prompt_tokens: int = 2048, revision: str | None = None, timeout: float = 900,
                 renderer=None, temperature_table: dict | None = None, **backend_options):
        if backend not in BACKENDS:
            raise JebError(f"unknown backend {backend!r}; one of: {', '.join(sorted(set(BACKENDS)))}")
        cls = BACKENDS[backend]
        self.backend_name = cls.name
        self.model = model
        self.calibrated = temperatures
        opts = dict(url=url or cls.default_url, model=model, api_key=api_key, timeout=timeout, revision=revision,
                    **backend_options)
        self.backend = cls(**opts)
        self.renderer = None
        self.temps: dict = {}
        if isinstance(self.backend, SystemOne):
            return
        if isinstance(self.backend, MLX):
            tok = self.backend.tok
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            self.renderer = renderer or Renderer(tok, max_prompt_tokens)
            tpath = self.backend.temperatures_path()
        else:
            if renderer is None and not repo:
                raise JebError(f"the {cls.name} backend needs repo=, the Jebadiah repo (or folder) the loaded "
                               "model came from, for its tokenizer, chat template and temperatures.json")
            folder = resolve_repo(repo, revision) if repo else None
            if renderer is None:
                try:
                    from transformers import AutoTokenizer
                except ImportError:
                    raise JebError("rendering needs transformers (the tokenizer only, no torch): "
                                   "pip install 'jebadiah-decide[render]'") from None
                tok = AutoTokenizer.from_pretrained(folder)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                renderer = _MessageRenderer(tok, max_prompt_tokens)
            self.renderer = renderer
            tpath = os.path.join(folder, "temperatures.json") if folder else None
        if temperatures:
            self.temps = temperature_table if temperature_table is not None else read_temperatures(tpath)
            if not self.temps:
                raise JebError("no temperatures.json found for this model; pass temperatures=False for raw probabilities")

    def _encode(self, prompt: str) -> list[int] | None:
        tok = getattr(self.renderer, "tok", None)
        return None if tok is None else tok.encode(prompt, add_special_tokens=False)

    def decide(self, state, questions: dict) -> dict:
        """Answers for every question, in the /v1/systemone response shape, plus `warnings`."""
        t0 = time.monotonic()
        if not isinstance(questions, dict) or not questions:
            raise JebError("'questions' must be a non-empty object of {id: question}")
        if isinstance(self.backend, SystemOne):
            out = self.backend.ask(state, questions, self.calibrated)
            out.setdefault("warnings", [])
            return out
        answers, warnings, n_in = {}, [], 0
        for qid, q in questions.items():
            if not isinstance(q, dict):
                raise JebError(f"question '{qid}' must be an object")
            try:
                rd = self.renderer.render(state, q)
            except (DecideError, ValueError, KeyError, AssertionError) as e:
                raise JebError(f"question '{qid}': {e}") from None
            ids = self._encode(rd.prompt)
            n_local = len(ids) if ids is not None else 0
            n_in += n_local
            reading = self.backend.read(rd, n_local, ids)
            cap = getattr(self.backend, "top_max", None)
            if reading.missing:
                warnings.append(f"{qid}: {reading.missing} of {len(rd.letters)} labels were outside the "
                                f"{self.backend_name} top {cap}; their probabilities are upper bounds")
            if getattr(rd, "truncated", False):
                warnings.append(f"{qid}: the state was cut to fit the prompt budget")
            probs = softmax(reading.values, float(self.temps.get(q["type"], 1.0)) if self.calibrated else 1.0)
            answers[qid] = answer_from_probs(q, rd.keys, probs)
        return {"model": self.model or getattr(self.backend, "path", None), "answers": answers,
                "usage": {"input_tokens": n_in, "output_tokens": 0},
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                "calibration": {"applied": bool(self.calibrated and self.temps), "temperatures": self.temps},
                "backend": self.backend_name, "warnings": warnings}


def decide(state, questions: dict, backend: str = "llama-server", **options) -> dict:
    """One-shot helper: builds a Jeb and asks once. Keep a Jeb around for more than one call."""
    return Jeb(backend, **options).decide(state, questions)
