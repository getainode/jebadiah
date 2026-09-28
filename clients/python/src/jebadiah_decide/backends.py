"""One reader per runtime. Each takes a rendered question and returns the log probability (or logit)
of every option label at the answer position, in display order.

Every HTTP reader sends the prompt exactly as rendered, with thinking off, asks for one token, and
checks that the runtime counted the same prompt tokens as the local tokenizer. A label the runtime
did not return (outside its top N) gets the smallest value it did return, which is an upper bound,
and is counted in `missing` so the caller can warn.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class JebError(Exception):
    """A request the runtime refused, an answer that could not be read, or a prompt the runtime
    rendered differently from the local template."""


@dataclass
class Reading:
    values: list[float]      # one per label, display order; log probabilities or logits
    missing: int = 0         # labels outside the runtime's top N (given the floor value)


def post_json(url: str, body: dict, api_key: str | None = None, timeout: float = 900) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise JebError(f"{url}: HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise JebError(f"{url}: {e.reason}") from None


def _by_text(top: list[tuple[str, float]], letters: list[str]) -> Reading:
    """Match labels by exact token text ("A", not " A"); the first occurrence wins."""
    lp: dict[str, float] = {}
    for token, value in top:
        lp.setdefault(token, value)
    if not lp:
        raise JebError("the runtime returned no log probabilities")
    floor = min(lp.values())
    return Reading([lp.get(L, floor) for L in letters], sum(1 for L in letters if L not in lp))


def _check_tokens(runtime: str, counted: int | None, local: int) -> None:
    if counted is not None and counted != local:
        raise JebError(f"{runtime} counted {counted} prompt tokens, the local template {local}. "
                       "Is thinking off, and is the loaded model a Jebadiah build with its own template?")


class LlamaServer:
    """llama.cpp's llama-server: raw text to /completion, labels read by token id from the
    pre-sampling top n_probs. No cap of its own, so any option count works."""
    name = "llama-server"
    default_url = "http://127.0.0.1:8080"
    needs_renderer = True

    def __init__(self, url: str, n_probs: int = 1000, api_key: str | None = None, check_tokens: bool = True,
                 timeout: float = 900, **_):
        self.url, self.n_probs, self.api_key, self.check, self.timeout = url.rstrip("/"), n_probs, api_key, check_tokens, timeout
        self.top_max = None

    def read(self, rd, n_local: int, local_ids: list[int] | None = None) -> Reading:
        if self.check and local_ids is not None:
            srv = post_json(self.url + "/tokenize", {"content": rd.prompt, "add_special": False, "parse_special": True},
                            self.api_key, self.timeout).get("tokens")
            if srv != local_ids:
                raise JebError("llama-server tokenized the prompt differently from the local tokenizer. "
                               "Is the GGUF a Jebadiah build, and the llama.cpp v0.5.0 or later?")
        r = post_json(self.url + "/completion", {
            "prompt": rd.prompt, "n_predict": 1, "n_probs": self.n_probs, "post_sampling_probs": False,
            "cache_prompt": False, "temperature": 0.0}, self.api_key, self.timeout)
        try:
            top = r["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise JebError("llama-server returned no completion_probabilities") from None
        lp = {t["id"]: t["logprob"] for t in top}
        floor = min(lp.values())
        return Reading([lp.get(c, floor) for c in rd.cand_ids], sum(1 for c in rd.cand_ids if c not in lp))


class Ollama:
    """Ollama: raw text to /api/generate with think off; top_logprobs is capped at 20 by Ollama."""
    name = "ollama"
    default_url = "http://127.0.0.1:11434"
    needs_renderer = True
    top_max = 20

    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: float = 900, **_):
        if not model:
            raise JebError("the ollama backend needs --model, the name Ollama knows the model by")
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout

    def read(self, rd, n_local: int, local_ids=None) -> Reading:
        r = post_json(self.url + "/api/generate", {
            "model": self.model, "prompt": rd.prompt, "raw": True, "stream": False, "think": False,
            "logprobs": True, "top_logprobs": self.top_max,
            "options": {"num_predict": 1, "temperature": 0}}, self.api_key, self.timeout)
        _check_tokens("Ollama", r.get("prompt_eval_count"), n_local)
        try:
            top = r["logprobs"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise JebError("Ollama returned no logprobs (logprobs need Ollama with PR #12899, late 2025 or newer)") from None
        return _by_text([(t["token"], t["logprob"]) for t in top], rd.letters)


class LMStudio:
    """LM Studio's local server: the chat MESSAGES to /v1/chat/completions with reasoning off, which
    LM Studio renders to the same text. Its raw /v1/completions returns no logprobs. Capped at 20."""
    name = "lmstudio"
    default_url = "http://127.0.0.1:1234"
    needs_renderer = True
    top_max = 20

    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: float = 900, **_):
        if not model:
            raise JebError("the lmstudio backend needs --model, the identifier LM Studio shows for the loaded model")
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout

    def read(self, rd, n_local: int, local_ids=None) -> Reading:
        r = post_json(self.url + "/v1/chat/completions", {
            "model": self.model, "messages": rd.messages, "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": self.top_max, "reasoning_effort": "none", "stream": False},
            self.api_key, self.timeout)
        _check_tokens("LM Studio", (r.get("usage") or {}).get("prompt_tokens"), n_local)
        try:
            top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise JebError("LM Studio returned no logprobs. Is reasoning off for this model?") from None
        return _by_text([(t["token"], t["logprob"]) for t in top], rd.letters)


class VLLM:
    """vLLM's OpenAI server (or any server with the same /v1/completions logprobs): raw text in,
    the top `top_n` log probabilities of the next token out. vLLM caps logprobs at its
    --max-logprobs (20 by default), so raise that flag to go past 20 options."""
    name = "vllm"
    default_url = "http://127.0.0.1:8000"
    needs_renderer = True

    def __init__(self, url: str, model: str, api_key: str | None = None, top_n: int = 20, timeout: float = 900, **_):
        if not model:
            raise JebError("the vllm backend needs --model, the served model name")
        self.url, self.model, self.api_key, self.top_max, self.timeout = url.rstrip("/"), model, api_key, top_n, timeout
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]

    def read(self, rd, n_local: int, local_ids=None) -> Reading:
        r = post_json(self.url + "/v1/completions", {
            "model": self.model, "prompt": rd.prompt, "max_tokens": 1, "temperature": 0,
            "logprobs": self.top_max}, self.api_key, self.timeout)
        _check_tokens("vLLM", (r.get("usage") or {}).get("prompt_tokens"), n_local)
        try:
            top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError):
            raise JebError("the server returned no logprobs") from None
        return _by_text(list(top.items()), rd.letters)


class MLX:
    """In-process MLX on Apple silicon: one forward pass, then the output-head rows of the label
    tokens dequantised and multiplied in fp32 (the same readout as scripts/decide_mlx.py). Returns
    logits, which give the same softmax over the labels. No option cap."""
    name = "mlx"
    needs_renderer = False
    default_url = None
    top_max = None

    def __init__(self, model: str, precision: str = "8bit", revision: str | None = None, **_):
        import os
        path = model
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            root = snapshot_download(model, revision=revision, allow_patterns=[f"{precision}/*", "temperatures.json"])
            path = os.path.join(root, precision)
        import mlx.core as mx  # noqa: F401  (fails early with a clear ImportError off Apple silicon)
        from mlx_lm import load
        self.path = path
        self.model, wrapper = load(path)
        self.tok = getattr(wrapper, "_tokenizer", wrapper)
        lm = getattr(self.model, "language_model", self.model)
        self.core = lm.model
        tied = getattr(lm.args, "tie_word_embeddings", False) or not hasattr(lm, "lm_head")
        self.head = self.core.embed_tokens if tied else lm.lm_head

    def temperatures_path(self) -> str | None:
        import os
        for p in (os.path.join(self.path, "temperatures.json"), os.path.join(self.path, "..", "temperatures.json")):
            if os.path.exists(p):
                return p
        return None

    def read(self, rd, n_local: int, local_ids=None) -> Reading:
        import mlx.core as mx
        ids = local_ids if local_ids is not None else self.tok.encode(rd.prompt, add_special_tokens=False)
        h = self.core(mx.array(ids)[None])[0, -1].astype(mx.float32)
        cand = mx.array(rd.cand_ids)
        m = self.head
        if hasattr(m, "scales"):
            w = mx.dequantize(m.weight[cand], m.scales[cand],
                              m.biases[cand] if getattr(m, "biases", None) is not None else None,
                              group_size=m.group_size, bits=m.bits, mode=getattr(m, "mode", "affine"))
        else:
            w = m.weight[cand]
        out = w.astype(mx.float32) @ h
        mx.eval(out)
        return Reading(out.tolist(), 0)


class SystemOne:
    """A server that already speaks the Jev wire: AINode's /v1/systemone, or jebadiah-serve from
    this repository. It renders and calibrates server-side, so nothing is loaded here."""
    name = "systemone"
    default_url = "http://127.0.0.1:8000"
    needs_renderer = False

    def __init__(self, url: str, model: str | None = None, api_key: str | None = None, timeout: float = 900, **_):
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout
        if self.url.endswith("/v1/systemone"):
            self.url = self.url[: -len("/v1/systemone")]
        elif self.url.endswith("/v1"):
            self.url = self.url[:-3]

    def ask(self, state, questions: dict, calibrated: bool = True) -> dict:
        body = {"state": state, "questions": questions}
        if self.model:
            body["model"] = self.model
        if not calibrated:
            body["calibration"] = "raw"
        return post_json(self.url + "/v1/systemone", body, self.api_key, self.timeout)


BACKENDS = {
    "llama-server": LlamaServer, "llamacpp": LlamaServer, "llama.cpp": LlamaServer,
    "ollama": Ollama,
    "lmstudio": LMStudio, "lm-studio": LMStudio,
    "vllm": VLLM,
    "mlx": MLX,
    "systemone": SystemOne, "ainode": SystemOne, "jebadiah-serve": SystemOne,
}
