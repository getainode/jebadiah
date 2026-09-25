"""Loading a base model (plus an optional adapter) and reading option probabilities off its logits.

Shared by the trainer, the evaluator and the /v1/systemone server: one code path from a
rendered prompt to a probability per option key. No generation anywhere.
"""
from __future__ import annotations

import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from jebadiah_prompt import Renderer, Rendered


def load_tokenizer(base: str, revision: str | None = None):
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def model_class(base: str, revision: str | None = None):
    cfg = AutoConfig.from_pretrained(base, revision=revision)
    if getattr(cfg, "model_type", "") == "qwen3_5":
        # The text-only class: the multimodal ConditionalGeneration class would load a vision
        # tower nothing here uses.
        from transformers import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM
    return AutoModelForCausalLM


def load_base(base: str, revision: str | None = None, attn_implementation: str = "sdpa",
              dtype=torch.bfloat16, device: str = "cuda"):
    cls = model_class(base, revision)
    model = cls.from_pretrained(base, revision=revision, dtype=dtype, device_map=device,
                                attn_implementation=attn_implementation)
    model.config.use_cache = False
    return model


def load_adapter(model, adapter_dir: str):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter_dir)


def core_of(model):
    """The CausalLM underneath a PEFT wrapper (LoRA layers stay injected in place)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# v1: the candidate-token logits are computed in fp32 from the last hidden state, W[cand].float()
# @ h.float(), the fix the v0 report asks for. The bf16 lm_head put two option tokens on the same
# bf16 logit often enough that ties decided 0.4 to 3 percent of picks between identical repeats.
# JEB_FP32_HEAD=0 restores the v0 read (bf16 lm_head, then cast).
FP32_CANDIDATE_LOGITS = os.environ.get("JEB_FP32_HEAD", "1") != "0"


def option_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  cand_ids: torch.Tensor) -> torch.Tensor:
    """Logits over each row's candidate tokens at its answer position (the last real token).
    cand_ids is [B, Kmax] padded with -1; padded slots come back as -inf. Only the answer
    position goes through the head, so memory does not scale with vocab x sequence."""
    core = core_of(model)
    hidden = core.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    last = attention_mask.sum(dim=1) - 1
    h_last = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    safe = cand_ids.clamp(min=0)
    if FP32_CANDIDATE_LOGITS:
        head = core.lm_head
        w = head.weight[safe].float()                       # [B, K, H], a few KB per row
        cand = torch.einsum("bkh,bh->bk", w, h_last.float())
        if getattr(head, "bias", None) is not None:
            cand = cand + head.bias[safe].float()
    else:
        cand = core.lm_head(h_last).float().gather(1, safe)
    return cand.masked_fill(cand_ids < 0, float("-inf"))


class Scorer:
    """Batches of (state, question) -> probabilities over the option keys, with an optional
    temperature per question type applied to the candidate logits before the softmax."""

    def __init__(self, model, tokenizer, max_tokens: int = 2048, temperatures: dict | None = None,
                 device: str = "cuda", renderer: Renderer | None = None):
        self.model = model
        self.tok = tokenizer
        self.renderer = renderer or Renderer(tokenizer, max_tokens)
        self.max_tokens = max_tokens
        self.temperatures = temperatures or {}
        self.device = device
        self.model.eval()

    @property
    def max_options(self) -> int:
        return self.renderer.max_options

    def render(self, state, q: dict, order: list | None = None) -> Rendered:
        return self.renderer.render(state, q, order)

    @torch.no_grad()
    def score_rendered(self, items: list[tuple[Rendered, str]]) -> list[list[float]]:
        """items: (rendered, question type). Returns one probability list per item, in the
        rendered key order, after that type's temperature (1.0 when unset)."""
        enc = self.tok([r.prompt for r, _ in items], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        kmax = max(len(r.cand_ids) for r, _ in items)
        cand = torch.full((len(items), kmax), -1, dtype=torch.long, device=self.device)
        for i, (r, _) in enumerate(items):
            cand[i, :len(r.cand_ids)] = torch.tensor(r.cand_ids, device=self.device)
        logits = option_logits(self.model, enc.input_ids, enc.attention_mask, cand)
        out = []
        for i, (r, qtype) in enumerate(items):
            t = float(self.temperatures.get(qtype, 1.0))
            p = torch.softmax(logits[i, :len(r.cand_ids)] / t, dim=-1)
            out.append(p.tolist())
        return out

    def score(self, state, questions: dict, orders: dict | None = None, batch_size: int = 8):
        """All questions of one request, each over the same state. Returns {qid: (keys, probs)}."""
        rendered = {qid: self.render(state, q, (orders or {}).get(qid)) for qid, q in questions.items()}
        qids = list(rendered)
        result = {}
        for i in range(0, len(qids), batch_size):
            chunk = qids[i:i + batch_size]
            probs = self.score_rendered([(rendered[qid], questions[qid]["type"]) for qid in chunk])
            for qid, p in zip(chunk, probs):
                result[qid] = (rendered[qid].keys, p)
        return result


def read_temperatures(adapter_dir: str | None) -> dict:
    if not adapter_dir:
        return {}
    path = os.path.join(adapter_dir, "temperatures.json")
    if os.path.exists(path):
        return {k: float(v) for k, v in json.load(open(path))["temperatures"].items()}
    return {}


def template_sha256(tokenizer) -> str:
    import hashlib
    return hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()
