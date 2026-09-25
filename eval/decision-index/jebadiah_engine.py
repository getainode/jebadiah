"""Jebadiah 9B v1 as a Decision Index engine.

The model is read with its own published inference code: the `scripts/` directory of
frontier-infra/jebadiah-9b-v1 at a pinned revision (jebadiah_model.py, jebadiah_prompt.py,
ainode_prompt_verbatim.py), imported from the Hub snapshot as-is. Nothing in the prompt, the
label alphabet, the logit read or the temperatures is changed here. What this file adds:

- mechanical schema translation (recorded per question in raw_output): an `instructions` or
  option description that is not a string is flattened with the model's own `flatten_text`
  (the rule its training conversion used), because AINode's wire requires strings;
- capacity refusals instead of truncation: the published Renderer cuts the state when the
  prompt is over its token budget, so the budget is set to the declared context and any
  rendered prompt that came back truncated is refused as `Unsupported`, never scored;
  a question with more options than the single-token label alphabet holds is refused too;
- on Apple MPS only, a faster exact triangular inverse inside transformers' reference Gated
  DeltaNet prefill (mps_delta.py; the CUDA path is untouched and uses flash-linear-attention);
- batching by padded token budget rather than by a fixed 8 questions, so a long prompt runs
  alone (probabilities do not depend on the batch);
- the answer shape the kit validates: choice -> {type, choice, probabilities, confidence},
  noul -> {type, noul = P(true)}, from the published `answer_from_probs`, with the probabilities
  left unrounded (the helper rounds to 6 decimals for display).

Run:  python -m decision_index run --engine jebadiah_engine:JebadiahEngine [--option device=mps]
"""
from __future__ import annotations

import hashlib
import importlib
import os
import platform
import sys

from decision_index.engines import Engine, Unsupported

ADAPTER_REPO = "frontier-infra/jebadiah-9b-v1"
ADAPTER_REVISION = "ea86ab5029feb152f6a1e490896dd2cac5e6ac59"
BASE_REPO = "Qwen/Qwen3.5-9B-Base"
BASE_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
SCRIPTS = ("ainode_prompt_verbatim.py", "jebadiah_prompt.py", "jebadiah_model.py")


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class JebadiahEngine(Engine):
    name = "jebadiah-9b-v1"

    def __init__(self, adapter=ADAPTER_REPO, adapter_revision=ADAPTER_REVISION, base=BASE_REPO,
                 base_revision=BASE_REVISION, device=None, dtype=None, attn="sdpa", temperatures=True,
                 max_tokens=None, token_budget=32768, max_batch=8, mps_fast_delta=True, **options):
        super().__init__(**options)
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if self.device == "mps":
            # transformers' threaded weight loader intermittently segfaults moving tensors to MPS (1 load in 6 here)
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        dtype = dtype or "bfloat16"
        kernel_note = None
        if self.device == "mps" and mps_fast_delta:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            kernel_note = importlib.import_module("mps_delta").install()
        adapter_dir = adapter if os.path.isdir(adapter) else snapshot_download(adapter, revision=adapter_revision)
        scripts = os.path.join(adapter_dir, "scripts")
        sys.path.insert(0, scripts)
        jm = importlib.import_module("jebadiah_model")
        jp = importlib.import_module("jebadiah_prompt")
        self.jp = jp
        self.answer_from_probs = jp.answer_from_probs
        self.flatten_text = jp.flatten_text

        cfg = AutoConfig.from_pretrained(base, revision=base_revision)
        window = getattr(getattr(cfg, "text_config", cfg), "max_position_embeddings")
        self.context = int(min(max_tokens or window, window))
        self.token_budget = max(int(token_budget), self.context) if self.device == "cuda" else int(token_budget)
        self.max_batch = int(max_batch)

        tok = jm.load_tokenizer(base, base_revision)
        model = jm.load_adapter(jm.load_base(base, base_revision, attn_implementation=attn,
                                             dtype=getattr(torch, dtype), device=self.device), adapter_dir)
        temps = jm.read_temperatures(adapter_dir) if temperatures else {}
        self.scorer = jm.Scorer(model, tok, max_tokens=self.context, temperatures=temps, device=self.device)
        self.tok = tok
        self.model_id = f"{adapter}@{adapter_revision[:8]}"

        import peft
        import transformers
        contract = {}
        cpath = os.path.join(adapter_dir, "prompt_contract.json")
        if os.path.exists(cpath):
            import json
            contract = json.load(open(cpath))
        self.provenance = {
            "kind": "LoRA",
            "adapter_repo": adapter, "adapter_revision": adapter_revision,
            "base_model": base, "base_revision": base_revision,
            "inference_code": {name: _sha256(os.path.join(scripts, name)) for name in SCRIPTS},
            "inference_code_source": f"https://huggingface.co/{ADAPTER_REPO}/tree/{adapter_revision}/scripts",
            "prompt_source_sha256": jp.PROMPT_SOURCE_SHA256,
            "prompt_source_commit": jp.PROMPT_SOURCE_COMMIT,
            "chat_template_sha256": jm.template_sha256(tok),
            "chat_template_sha256_expected": contract.get("chat_template_sha256"),
            "temperatures": temps,
            "fp32_candidate_logits": jm.FP32_CANDIDATE_LOGITS,
            "declared_context_tokens": self.context,
            "declared_max_options": self.scorer.renderer.max_options_extended,
            "ainode_label_alphabet": self.scorer.renderer.max_options,
            "device": self.device, "dtype": dtype, "attn": attn,
            "delta_rule_kernel": kernel_note or "transformers default (flash-linear-attention when installed, else the torch reference)",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "python": platform.python_version()},
        }

    latency = ("In-process wall time of one request: AINode rendering, tokenization and one label-token "
               "logit read per question (questions batched by padded token budget, no prefix cache), "
               "device-synchronized; excludes model loading.")

    def runtime(self):
        t = self.torch
        info = {"device": self.device, "torch": t.__version__, "platform": platform.platform(), "machine": platform.machine()}
        if self.device == "cuda":
            info["gpu"] = t.cuda.get_device_name(0)
        return info

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _translate(self, q):
        """Mechanical schema translation to AINode's string-only wire; returns (question, notes)."""
        notes = []
        out = dict(q)
        if not isinstance(q.get("instructions"), str):
            out["instructions"] = self.flatten_text(q.get("instructions"))
            notes.append("instructions_flattened")
        crit = q.get("criteria")
        if isinstance(crit, dict) and any(v is not None and not isinstance(v, str) for v in crit.values()):
            out["criteria"] = {k: (v if v is None or isinstance(v, str) else self.flatten_text(v)) for k, v in crit.items()}
            notes.append("descriptions_flattened")
        return out, notes

    def __call__(self, state, questions):
        renderer = self.scorer.renderer
        rendered, raw = {}, {}
        for qid, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"question type {q['type']!r} is not one Jebadiah answers")
            q2, notes = self._translate(q)
            try:
                r = self.scorer.render(state, q2)
            except ValueError as exc:
                if "single-token labels" in str(exc):
                    raise Unsupported(f"{len(q['criteria'])} options exceed the model's {renderer.max_options_extended} single-token labels") from exc
                raise
            if r.truncated:
                raise Unsupported(f"prompt exceeds the declared context of {self.context} tokens; refused, not truncated")
            n = len(self.tok.encode(r.prompt, add_special_tokens=False))
            rendered[qid] = (r, q2["type"], n)
            raw[qid] = {"labels": r.letters, "label_scheme": r.label_scheme, "prompt_tokens": n}
            if notes:
                raw[qid]["translation"] = notes
        # batches by padded token budget, longest first, so memory is bounded and a long prompt runs alone
        order = sorted(rendered, key=lambda k: -rendered[k][2])
        probs = {}
        i = 0
        while i < len(order):
            batch = [order[i]]
            while i + len(batch) < len(order) and len(batch) < self.max_batch and rendered[order[i]][2] * (len(batch) + 1) <= self.token_budget:
                batch.append(order[i + len(batch)])
            out = self.scorer.score_rendered([(rendered[k][0], rendered[k][1]) for k in batch])
            for k, p in zip(batch, out):
                probs[k] = p
            i += len(batch)
        answers = {}
        for qid, q in questions.items():
            r = rendered[qid][0]
            a = self.answer_from_probs(q, r.keys, probs[qid])
            if q["type"] == "choice":
                # the published helper rounds to 6 decimals for the route; the kit ranks ToolRet and
                # BRIGHT candidates by probability across requests, where rounding would create ties,
                # so the answer carries the unrounded softmax (same argmax, same values to 1e-6)
                a["probabilities"] = dict(zip(r.keys, probs[qid]))
            else:
                a["noul"] = float(probs[qid][r.keys.index("true")])
            answers[qid] = a
            raw[qid]["probabilities"] = dict(zip(r.keys, probs[qid]))
        usage = {"input_tokens": sum(v[2] for v in rendered.values())}
        return {"model": self.model_id, "answers": answers, "usage": usage}, raw
