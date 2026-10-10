"""`Jeb`: typed decisions from a Jebadiah model on the runtime you already run.

    from jebadiah_decide import Jeb
    jeb = Jeb("ollama")            # Jebadiah 9B v2.1, hf.co/frontier-infra/jebadiah-9b-v2-1-GGUF:Q8_0
    jeb.prepare()                  # checks Ollama is up and pulls the model on first use
    jeb.decide({"ticket": "..."}, {"route": {"type": "choice", "instructions": "...",
                                              "criteria": {"billing": "...", "support": "..."}}})

For a runtime that only returns log probabilities (llama-server, Ollama, LM Studio, vLLM) the prompt
is rendered here, exactly as AINode renders it, with the tokenizer and chat template from the model's
repository (a few MB, cached; no torch), and the model's per-type temperatures are applied. MLX runs
in this process. A /v1/systemone server (AINode, jebadiah-serve) does all of that itself.

Two request shapes, both AINode's wire: `decide(state, questions)` is /v1/systemone (the Jev shape,
the one the model was trained on) and `decide_ainode(body)` is /v1/decide.
"""
from __future__ import annotations

import json
import math
import os
import time
from types import SimpleNamespace

from . import defaults
from ._contract import (DecideError, Renderer, answer_from_probs, build_messages, option_label, option_labels,
                        routes, serialize_state)
from ._contract import CHAT_TEMPLATE_KWARGS
from .backends import BACKENDS, MLX, JebError, SystemOne

REPO_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "temperatures.json"]
CAP_NAMES = {"ollama": "Ollama", "lmstudio": "LM Studio", "vllm": "vLLM"}


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
    """A local folder as is; a Hub repo id downloaded to the Hugging Face cache (tokenizer,
    chat template and temperatures only, never the weights)."""
    if os.path.isdir(repo):
        return repo
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    if revision is None:
        try:   # already cached: no network, no delay
            folder = snapshot_download(repo, allow_patterns=REPO_FILES, local_files_only=True)
            if all(os.path.exists(os.path.join(folder, f)) for f in ("tokenizer.json", "chat_template.jinja")):
                return folder
        except Exception:
            pass
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


def load_tokenizer(folder: str):
    from .tokenizer import LightTokenizer
    return LightTokenizer(folder)


class Jeb:
    def __init__(self, backend: str = "ollama", *, size: str = defaults.DEFAULT_SIZE, url: str | None = None,
                 model: str | None = None, repo: str | None = None, api_key: str | None = None,
                 temperatures: bool = True, max_prompt_tokens: int = 2048, revision: str | None = None,
                 timeout: float = 900, renderer=None, temperature_table: dict | None = None, **backend_options):
        if backend not in BACKENDS:
            raise JebError(f"unknown backend {backend!r}; one of: {', '.join(sorted(set(BACKENDS)))}")
        if size not in defaults.SIZES:
            raise JebError(f"unknown size {size!r}; one of: {', '.join(defaults.SIZES)}")
        cls = BACKENDS[backend]
        self.backend_name = cls.name
        self.size = size
        self.model = model or (defaults.mlx_repo(size, backend_options.get("precision", "8bit"))
                               if cls.name == "mlx" else defaults.default_model(cls.name, size))
        self.calibrated = temperatures
        self.max_prompt_tokens = max_prompt_tokens
        self.backend = cls(url=url or cls.default_url, model=self.model, api_key=api_key, timeout=timeout,
                           revision=revision, **backend_options)
        self.renderer = None
        self.temps: dict = {}
        if isinstance(self.backend, SystemOne):
            return
        if isinstance(self.backend, MLX):
            self.renderer = renderer or Renderer(self.backend.tok, max_prompt_tokens)
            tpath = self.backend.temperatures_path()
        else:
            self.repo = repo or defaults.default_repo(cls.name, size)
            need = renderer is None or (temperatures and temperature_table is None)
            folder = resolve_repo(self.repo, revision) if need else None
            self.renderer = renderer or _MessageRenderer(load_tokenizer(folder), max_prompt_tokens)
            tpath = os.path.join(folder, "temperatures.json") if folder else None
        if temperatures:
            self.temps = temperature_table if temperature_table is not None else read_temperatures(tpath)
            if not self.temps:
                raise JebError("no temperatures.json found for this model; pass temperatures=False for raw probabilities")

    @property
    def cap(self) -> int | None:
        return getattr(self.backend, "top_max", None)

    def prepare(self, progress=print) -> dict:
        """Check the runtime is up and the model is there; Ollama pulls it on first use, LM Studio
        is searched for a loaded Jebadiah. Raises JebError saying exactly what to do."""
        info = self.backend.prepare(progress)
        if self.backend_name == "lmstudio":
            self.model = self.backend.model
        return info

    def _check_cap(self, qid: str, n: int):
        if self.cap and n > self.cap:
            name = CAP_NAMES.get(self.backend_name, self.backend_name)
            raise JebError(f"question '{qid}': {n} options, but {name} returns only its top {self.cap} log "
                           f"probabilities, so Jeb can't read more than {self.cap} options there. Split the "
                           "question into narrower ones, or use --backend llama-server or mlx (no cap).")

    def _encode(self, prompt: str) -> list[int] | None:
        tok = getattr(self.renderer, "tok", None)
        return None if tok is None else tok.encode(prompt, add_special_tokens=False)

    def _read(self, rd, qtype: str, calibrated: bool):
        ids = self._encode(rd.prompt)
        n_local = len(ids) if ids is not None else 0
        reading = self.backend.read(rd, n_local, ids)
        self.last_missing = reading.missing
        probs = softmax(reading.values, float(self.temps.get(qtype, 1.0)) if calibrated else 1.0)
        return probs, n_local

    # ------------------------------------------------------------------ /v1/systemone

    def decide(self, state, questions: dict, calibrated: bool | None = None) -> dict:
        """Answers for every question, in the /v1/systemone response shape, plus `warnings`."""
        t0 = time.monotonic()
        calibrated = self.calibrated if calibrated is None else calibrated
        if not isinstance(questions, dict) or not questions:
            raise JebError("'questions' must be a non-empty object of {id: question}")
        if isinstance(self.backend, SystemOne):
            out = self.backend.ask(state, questions, calibrated)
            out.setdefault("warnings", [])
            return out
        answers, warnings, n_in = {}, [], 0
        for qid, q in questions.items():
            if not isinstance(q, dict):
                raise JebError(f"question '{qid}' must be an object")
            crit = q.get("criteria")
            if q.get("type") == "choice" and isinstance(crit, (dict, list)):
                self._check_cap(qid, len(crit))
            try:
                rd = self.renderer.render(state, q)
            except (DecideError, ValueError, KeyError, AssertionError, TypeError) as e:
                raise JebError(f"question '{qid}': {e}") from None
            self._check_cap(qid, len(rd.keys))
            probs, n = self._read(rd, q["type"], calibrated)
            n_in += n
            if self.last_missing:
                warnings.append(f"{qid}: {self.last_missing} of {len(rd.letters)} labels were outside the "
                                f"{self.backend_name} top {self.cap}; their probabilities are upper bounds")
            if getattr(rd, "truncated", False):
                warnings.append(f"{qid}: the state was cut to {self.max_prompt_tokens} prompt tokens")
            answers[qid] = answer_from_probs(q, rd.keys, probs)
        return {"model": self.model or getattr(self.backend, "path", None), "answers": answers,
                "usage": {"input_tokens": n_in, "output_tokens": 0},
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                "calibration": {"applied": bool(calibrated and self.temps), "temperatures": self.temps if calibrated else {}},
                "backend": self.backend_name, "warnings": warnings}

    # ------------------------------------------------------------------ /v1/decide

    def decide_ainode(self, body: dict) -> dict:
        """AINode's /v1/decide: {state, instructions?, questions: {id: {question, options} |
        {question, type: "boolean"} | {question, type: "score", min, max}}, calibration?}.
        Each decision is {answer, confidence, distribution, latency_ms}."""
        t0 = time.monotonic()
        if isinstance(self.backend, SystemOne):
            return self.backend.decide(body)
        try:
            questions = routes.normalize_questions(body.get("questions"))
            state = serialize_state(body.get("state"))
            instructions = body.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise routes.DecideError("'instructions' must be a string when given")
            calibrated = routes.calibration_mode(body.get("calibration")) != routes.CALIBRATION_RAW and self.calibrated
        except (routes.DecideError, DecideError) as e:
            raise JebError(str(e)) from None
        decisions, n_in = {}, 0
        tok = self.renderer.tok
        label_ids = self.renderer._label_ids
        for key, spec in questions.items():
            options = spec["options"]
            self._check_cap(key, len(options))
            if len(options) > len(label_ids):
                raise JebError(f"question '{key}': {len(options)} options is more than the {len(label_ids)} "
                               "this model reads as single label tokens")
            messages = build_messages(state, instructions, spec["question"], options)
            prompt = tok.apply_chat_template(messages, tokenize=False, **CHAT_TEMPLATE_KWARGS)
            letters = option_labels(len(options))
            rd = SimpleNamespace(prompt=prompt, letters=letters, cand_ids=[label_ids[L] for L in letters],
                                 messages=messages, keys=list(options), truncated=False)
            n_tokens = len(tok.encode(prompt, add_special_tokens=False))
            if n_tokens > 8192:
                raise JebError(f"question '{key}': the prompt is {n_tokens} tokens, more than 8192")
            q0 = time.monotonic()
            probs, n = self._read(rd, spec["kind"], calibrated)
            n_in += n
            labels = [str(i) for i in range(len(options))]
            dist = {label: round(float(p), 6) for label, p in zip(labels, probs)}
            picked = routes.pick_answer(labels, dist, None)
            decisions[key] = {"answer": options[int(picked)], "confidence": dist[picked],
                              "distribution": {options[i]: dist[labels[i]] for i in range(len(options))},
                              "latency_ms": round((time.monotonic() - q0) * 1000, 1)}
        return {"model": self.model or getattr(self.backend, "path", None),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1), "decisions": decisions,
                "usage": {"prompt_tokens": n_in, "completion_tokens": len(decisions), "calls": len(decisions)},
                "calibration": {"applied": bool(calibrated and self.temps), "temperatures": self.temps if calibrated else {}},
                "backend": self.backend_name}


def decide(state, questions: dict, backend: str = "ollama", **options) -> dict:
    """One-shot helper: builds a Jeb and asks once. Keep a Jeb around for more than one call."""
    return Jeb(backend, **options).decide(state, questions)
