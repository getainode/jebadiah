"""Loading a Jebadiah model and reading option probabilities off it, on one device.

Everything model-facing is the model repository's own code (model_scripts/): the tokenizer and
model loaders, the Renderer that turns (state, question) into the exact training prompt, and the
Scorer that reads the label-token logits at the answer position in fp32. This module picks the
device and dtype, checks the model's prompt contract against the renderer it is about to use,
and serializes access to the one model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from jebadiah_server import ainode_routes as routes
from jebadiah_server.model_scripts import (
    HERE as SCRIPTS_DIR,
    PROMPT_SOURCE_SHA256,
    SCRIPT_FILES,
    build_messages,
    model_module,
    option_labels,
    prompt,
)

log = logging.getLogger("jebadiah.engine")

DEFAULT_MODEL = "frontier-infra/jebadiah-9b-v2"
PROMPT_CONTRACT_FILE = "prompt_contract.json"


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str):
    import torch
    if requested == "auto":
        return torch.float32 if device == "cpu" else torch.bfloat16
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
            "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[requested]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContractError(RuntimeError):
    """The model says it was trained on a prompt this server does not render."""


@dataclass
class Item:
    """One question ready for the model: its rendered prompt and its temperature kind."""
    rendered: "prompt.Rendered"
    kind: str
    tokens: int


@dataclass
class Engine:
    model_id: str = DEFAULT_MODEL
    revision: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8
    max_prompt_tokens: int = 8192
    strict_contract: bool = True

    status: str = "starting"
    error: Optional[str] = None
    model_dir: Optional[str] = None
    tokenizer: object = None
    renderer: object = None
    temperatures: Optional[dict] = None
    contract: dict = field(default_factory=dict)
    load_seconds: Optional[float] = None
    _model: object = None
    _scorers: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def max_options(self) -> int:
        return self.renderer.max_options if self.renderer is not None else 0

    # ------------------------------------------------------------------ loading

    def resolve_dir(self) -> str:
        if os.path.isdir(self.model_id):
            return os.path.abspath(self.model_id)
        from huggingface_hub import snapshot_download
        return snapshot_download(self.model_id, revision=self.revision)

    def load_tokenizer_only(self) -> None:
        """Tokenizer, renderer and contract without the weights. The tests use this."""
        jm = model_module()
        self.model_dir = self.model_dir or self.resolve_dir()
        self.tokenizer = jm.load_tokenizer(self.model_dir)
        self.renderer = prompt.Renderer(self.tokenizer, max_tokens=self.max_prompt_tokens)
        self.temperatures = routes.read_temperatures(Path(self.model_dir))
        self.contract = self.check_contract()

    def load(self) -> None:
        started = time.monotonic()
        try:
            self.status = "loading"
            self.device = pick_device(self.device)
            if self.device.startswith("mps"):
                # the model card's own note: transformers 5.17's threaded weight loader
                # intermittently segfaults moving tensors to MPS
                os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
            torch_dtype = pick_dtype(self.dtype, self.device)
            self.dtype = str(torch_dtype).replace("torch.", "")
            self.load_tokenizer_only()
            jm = model_module()
            log.info("loading %s on %s as %s", self.model_dir, self.device, self.dtype)
            self._model = jm.load_base(self.model_dir, dtype=torch_dtype, device=self.device)
            self._scorers = {
                True: jm.Scorer(self._model, self.tokenizer, self.max_prompt_tokens,
                                self.temperatures or {}, self.device, self.renderer),
                False: jm.Scorer(self._model, self.tokenizer, self.max_prompt_tokens,
                                 {}, self.device, self.renderer),
            }
            self.load_seconds = round(time.monotonic() - started, 1)
            self.status = "ready"
            log.info("ready in %.1f s", self.load_seconds)
        except Exception as exc:  # the server stays up and says why in /health
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")

    def check_contract(self) -> dict:
        """What the model repository says it was trained on, against what this server renders.

        prompt_source_sha256 names the copy of AINode's renderer the training ran on, and the
        chat template hash pins the template the tokenizer will apply. A mismatch on either
        means the prompt is not the training prompt, which is refused at load unless
        strict_contract is off. The scripts/ copies beside the weights are compared too.
        """
        jm = model_module()
        directory = Path(self.model_dir)
        out: dict = {"prompt_source_sha256": PROMPT_SOURCE_SHA256,
                     "chat_template_sha256": jm.template_sha256(self.tokenizer),
                     "label_tokens": self.renderer.max_options}
        path = directory / PROMPT_CONTRACT_FILE
        problems = []
        if path.is_file():
            want = json.loads(path.read_text())
            out["model_contract"] = want
            if want.get("prompt_source_sha256") != PROMPT_SOURCE_SHA256:
                problems.append("prompt_source_sha256 differs from the bundled renderer")
            if want.get("chat_template_sha256") != out["chat_template_sha256"]:
                problems.append("chat_template_sha256 differs from the tokenizer's template")
            if want.get("single_token_labels") not in (None, self.renderer.max_options):
                problems.append("single_token_labels differs from this tokenizer")
            kwargs = want.get("chat_template_kwargs")
            if kwargs is not None and kwargs != prompt.CHAT_TEMPLATE_KWARGS:
                problems.append("chat_template_kwargs differ (thinking must be off)")
        else:
            problems.append(f"no {PROMPT_CONTRACT_FILE} in the model directory")
        scripts = {}
        for name in SCRIPT_FILES:
            theirs = directory / "scripts" / name
            if theirs.is_file():
                scripts[name] = sha256_file(theirs) == sha256_file(Path(SCRIPTS_DIR) / name)
        out["scripts_match"] = scripts
        if scripts and not all(scripts.values()):
            problems.append("the model repository's scripts/ differ from the bundled copies")
        out["problems"] = problems
        out["ok"] = not problems
        if problems:
            msg = "prompt contract: " + "; ".join(problems)
            if self.strict_contract and path.is_file():
                raise ContractError(msg + " (pass --allow-contract-mismatch to serve anyway)")
            log.warning(msg)
        return out

    # ---------------------------------------------------------------- rendering

    def render_systemone(self, state, spec: dict) -> "prompt.Rendered":
        """The training path, unchanged: Renderer.render(state, question)."""
        return self.renderer.render(state, spec)

    def render_decide(self, state_text: str, instructions: Optional[str], question: str,
                      options: list[str]) -> "prompt.Rendered":
        """AINode's /v1/decide prompt: build_messages with the shared instructions block, the
        same chat template and kwargs, the same label tokens."""
        messages = build_messages(state_text, instructions, question, options)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  **prompt.CHAT_TEMPLATE_KWARGS)
        letters = option_labels(len(options))
        cand = [self.renderer._label_ids[L] for L in letters]
        return prompt.Rendered(text, list(options), letters, cand, False)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------ scoring

    def score(self, items: list[Item], calibrated: bool) -> list[list[float]]:
        """Probabilities for every item, batched, one request at a time on the device."""
        scorer = self._scorers[calibrated]
        out: list[list[float]] = []
        with self._lock:
            # similar lengths together, so a batch pads as little as possible
            order = sorted(range(len(items)), key=lambda i: items[i].tokens)
            probs: dict[int, list[float]] = {}
            for start in range(0, len(order), self.batch_size):
                chunk = order[start:start + self.batch_size]
                got = scorer.score_rendered([(items[i].rendered, items[i].kind) for i in chunk])
                probs.update(zip(chunk, got))
            out = [probs[i] for i in range(len(items))]
            if self.device.startswith("mps"):
                import torch
                torch.mps.empty_cache()
        return out

    def describe(self) -> dict:
        return {"status": self.status, "error": self.error, "model": self.model_id,
                "revision": self.revision, "model_dir": self.model_dir, "device": self.device,
                "dtype": self.dtype, "batch_size": self.batch_size,
                "max_prompt_tokens": self.max_prompt_tokens, "load_seconds": self.load_seconds,
                "temperatures": self.temperatures, "prompt_contract": self.contract or None}


def finite(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(v) for v in values)
