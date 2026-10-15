"""Jebadiah 9B v1 as a Decision Index engine.

The model is read with its own published inference code: the `scripts/` directory of
frontier-infra/jebadiah-9b-v1 at a pinned revision (jebadiah_model.py, jebadiah_prompt.py,
ainode_prompt_verbatim.py), imported from the Hub snapshot as-is. Nothing in the prompt, the
label alphabet, the logit read or the temperatures is changed here. The optional inference policies are reverse_avg (choice-only geometric mean of two
order reads) and repeat_payload (duplicate the complete user message before chat templating).
Both are off by default. What this file adds:

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
import math
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


def _flag(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


# Shared-prefix cache and batching ported from the frozen ITEM 31 engine,
# results dataset revision 587f4f496b9cf1d83f0bc8e495379496f65afa2f.
class _PrefixState:
    """What a forward over the shared prefix leaves behind, in the interface Qwen3.5's layers call on
    transformers' cache: per full-attention layer the keys and values (after RoPE), per Gated DeltaNet
    layer the last conv_kernel-1 inputs of the causal conv1d and the recurrent state. Recording
    (batch None) returns every input unchanged, so the prefix forward is the plain forward; with a
    batch it serves B suffix rows that continue from that one prefix."""

    def __init__(self, n_layers, length, batch=None, source=None):
        self.length, self.batch = length, batch
        if source is None:
            self.layers = [_PrefixLayer() for _ in range(n_layers)]
        else:
            self.layers = [lay.expanded(batch) for lay in source.layers]

    def for_batch(self, batch):
        return _PrefixState(len(self.layers), self.length, batch, self)

    def has_previous_state(self, layer_idx=0, state_idx=0):
        return self.batch is not None

    def get_seq_length(self, layer_idx=0):
        return self.length if self.batch is not None else 0

    def update_conv_state(self, x, layer_idx, conv_kernel_size=4, **kwargs):
        lay = self.layers[layer_idx]
        if self.batch is None:
            lay.conv_states[0] = x[:, :, -(conv_kernel_size - 1):].clone()
            return x
        import torch
        return torch.cat([lay.conv_states[0], x], dim=-1)

    def update_recurrent_state(self, state, layer_idx, **kwargs):
        if self.batch is None:
            self.layers[layer_idx].recurrent_states[0] = state

    def update(self, key, value, layer_idx, *args, **kwargs):
        lay = self.layers[layer_idx]
        if self.batch is None:
            lay.keys, lay.values = key, value
            return key, value
        import torch
        b = key.shape[0]
        cat = torch.cat
        return cat([lay.keys.expand(b, -1, -1, -1), key], dim=2), cat([lay.values.expand(b, -1, -1, -1), value], dim=2)


class _PrefixLayer:
    def __init__(self):
        self.conv_states, self.recurrent_states, self.keys, self.values = [None], [None], None, None
        self.record_past = False

    def expanded(self, b):
        out = _PrefixLayer()
        out.keys, out.values = self.keys, self.values
        if self.conv_states[0] is not None:
            out.conv_states = [self.conv_states[0].expand(b, -1, -1).contiguous()]
            out.recurrent_states = [self.recurrent_states[0].expand(b, *self.recurrent_states[0].shape[1:]).contiguous()]
        return out


class JebadiahEngine(Engine):
    name = "jebadiah-9b-v1"

    def __init__(self, adapter=ADAPTER_REPO, adapter_revision=ADAPTER_REVISION, base=BASE_REPO,
                 base_revision=BASE_REVISION, device=None, dtype=None, attn="sdpa", temperatures=True,
                 max_tokens=None, token_budget=32768, max_batch=8, mps_fast_delta=True, reverse_avg=False, repeat_payload=False, reverse_avg_below=None,
                 model=None, revision=None, prefix_cache=False, batch_tokens=None, prefix_align=64,
                 pad_waste=0.15, cuda_alloc_conf=None, **options):
        super().__init__(**options)
        if cuda_alloc_conf:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(cuda_alloc_conf))
        self.reverse_avg = _flag(reverse_avg)
        self.repeat_payload = _flag(repeat_payload)
        # Decider-style gate: keep the forward read when its top probability is at least this value.
        # ponytail: both orders are still scored; skip the reverse read when confident only if latency matters.
        self.reverse_avg_below = None if reverse_avg_below in (None, "", "none") else float(reverse_avg_below)
        if self.reverse_avg_below is not None:
            self.reverse_avg = True
        self.prefix_cache = _flag(prefix_cache)
        self.batch_tokens = int(batch_tokens) if batch_tokens else None
        self.prefix_align = max(1, int(prefix_align))
        self.pad_waste = float(pad_waste)
        self.trace = None
        self.oom_retries = 0
        if model and not revision:
            raise ValueError("merged model requires an immutable revision")
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
        if model:
            adapter, adapter_revision = model, revision
        adapter_dir = adapter if os.path.isdir(adapter) else snapshot_download(adapter, revision=adapter_revision)
        scripts = os.path.join(adapter_dir, "scripts")
        sys.path.insert(0, scripts)
        jm = importlib.import_module("jebadiah_model")
        jp = importlib.import_module("jebadiah_prompt")
        self.jm = jm
        self.jp = jp
        self.answer_from_probs = jp.answer_from_probs
        self.flatten_text = jp.flatten_text

        cfg = AutoConfig.from_pretrained(adapter_dir) if model else AutoConfig.from_pretrained(base, revision=base_revision)
        window = getattr(getattr(cfg, "text_config", cfg), "max_position_embeddings")
        self.context = int(min(max_tokens or window, window))
        self.token_budget = max(int(token_budget), self.context) if self.device == "cuda" else int(token_budget)
        self.max_batch = int(max_batch)

        tok = jm.load_tokenizer(adapter_dir) if model else jm.load_tokenizer(base, base_revision)
        merged = bool(model)
        model = jm.load_base(adapter_dir, attn_implementation=attn, dtype=getattr(torch, dtype), device=self.device) if merged else jm.load_adapter(jm.load_base(base, base_revision, attn_implementation=attn,
                                             dtype=getattr(torch, dtype), device=self.device), adapter_dir)
        temps = jm.read_temperatures(adapter_dir) if temperatures else {}
        self.scorer = jm.Scorer(model, tok, max_tokens=self.context, temperatures=temps, device=self.device)
        if self.repeat_payload:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from inference_renderer import repeat_renderer
            self.scorer.renderer = repeat_renderer(jp, tok, self.context, repeat_payload=True)
        self.tok = tok
        self.model_id = f"{adapter}@{adapter_revision[:8]}"

        if not merged:
            import peft
        import transformers
        contract = {}
        cpath = os.path.join(adapter_dir, "prompt_contract.json")
        if os.path.exists(cpath):
            import json
            contract = json.load(open(cpath))
        if merged:
            if contract.get("chat_template_sha256") and jm.template_sha256(tok) != contract["chat_template_sha256"]:
                raise RuntimeError("chat template differs from the model's prompt_contract.json")
            if contract.get("prompt_source_sha256") and jp.PROMPT_SOURCE_SHA256 != contract["prompt_source_sha256"]:
                raise RuntimeError("prompt source differs from the model's prompt_contract.json")
            if contract.get("chat_template_kwargs") and dict(jp.CHAT_TEMPLATE_KWARGS) != contract["chat_template_kwargs"]:
                raise RuntimeError("chat template kwargs differ from the model's prompt_contract.json")
        self.provenance = {
            "kind": "merged" if merged else "LoRA",
            "adapter_repo": adapter, "adapter_revision": adapter_revision,
            "base_model": base, "base_revision": base_revision,
            "inference_code": {name: _sha256(os.path.join(scripts, name)) for name in SCRIPTS},
            "inference_code_source": f"https://huggingface.co/{adapter if merged else ADAPTER_REPO}/tree/{adapter_revision}/scripts",
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
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, **({} if merged else {"peft": peft.__version__}), "python": platform.python_version()},
        }
        if merged:
            self.provenance["model_repo"] = adapter
            self.provenance["model_revision"] = adapter_revision
        if self.reverse_avg or self.repeat_payload:
            self.provenance["inference_variants"] = {"reverse_avg": self.reverse_avg, "repeat_payload": self.repeat_payload,
                                                    "reverse_avg_below": self.reverse_avg_below}

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

    def _tick(self):
        if self.trace is None:
            return None
        import time
        self.synchronize()
        return time.perf_counter()

    def _tock(self, t0, kind, rows, padded, useful):
        if t0 is not None:
            import time
            self.synchronize()
            self.trace.append({"kind": kind, "rows": rows, "padded_tokens": padded, "useful_tokens": useful,
                               "seconds": time.perf_counter() - t0})

    def _pack(self, lengths, extra):
        """Batches over lengths sorted longest first, as (start, size): a row joins while the batch stays within
        max_batch rows, within the padded-token budget (rows x (longest + extra)) and within pad_waste of padding."""
        budget = self.batch_tokens or self.token_budget
        out, i = [], 0
        while i < len(lengths):
            w, n, used = lengths[i], 1, lengths[i]
            while (i + n < len(lengths) and n < self.max_batch and (w + extra) * (n + 1) <= budget
                   and w * (n + 1) - (used + lengths[i + n]) <= self.pad_waste * w * (n + 1)):
                used += lengths[i + n]
                n += 1
            out.append((i, n))
            i += n
        return out

    def _score_shared_prefix(self, rendered, token_ids):
        """Every question of a request renders the same system + state text first, so their token ids
        share a prefix. The prefix runs once; each question's suffix then runs from the keys, values,
        conv inputs and recurrent state the prefix left, batched by padded tokens. The cut is the longest
        common token prefix (at least one token left per question), rounded down to a multiple of
        prefix_align (64, the Gated DeltaNet chunk) so the suffix's chunks fall where they fall in the full
        forward. Returns None, meaning score the plain way, when the rounded prefix is empty."""
        torch = self.torch
        qids = list(rendered)
        seqs = [token_ids[k] for k in qids]
        cut = min(len(x) for x in seqs) - 1
        first = seqs[0]
        for x in seqs[1:]:
            j = 0
            while j < cut and x[j] == first[j]:
                j += 1
            cut = j
        cut -= cut % self.prefix_align
        if cut <= 0:
            return None
        core = self.jm.core_of(self.scorer.model)
        body = core.model
        dev = self.device
        n_layers = len(body.layers)
        with torch.no_grad():
            t0 = self._tick()
            state = _PrefixState(n_layers, cut)
            body(input_ids=torch.tensor([first[:cut]], device=dev),
                 attention_mask={"full_attention": None, "linear_attention": None},
                 position_ids=torch.arange(cut, device=dev)[None], past_key_values=state, use_cache=False)
            self._tock(t0, "prefix", 1, cut, cut)
            order = sorted(qids, key=lambda k: -len(token_ids[k]))
            kv_cost = cut // 16      # a row's share of the prefix keys and values it attends over, in token units
            probs = {}
            for i, n in self._pack([len(token_ids[k]) - cut for k in order], kv_cost):
                batch = order[i:i + n]
                width = len(token_ids[batch[0]]) - cut
                t0 = self._tick()
                b = len(batch)
                pad = self.tok.pad_token_id
                ids = torch.full((b, width), pad, dtype=torch.long)
                mask = torch.zeros((b, width), dtype=torch.long)
                for r, k in enumerate(batch):
                    suf = token_ids[k][cut:]
                    ids[r, :len(suf)] = torch.tensor(suf)
                    mask[r, :len(suf)] = 1
                ids, mask = ids.to(dev), mask.to(dev)
                # full attention: every suffix row sees the whole prefix, then itself causally, never padding
                causal = torch.ones((width, width), dtype=torch.bool, device=dev).tril()
                own = causal[None] & mask.bool()[:, None, :]
                full = torch.cat([torch.ones((b, width, cut), dtype=torch.bool, device=dev), own], dim=-1)[:, None]
                hidden = body(input_ids=ids,
                              attention_mask={"full_attention": full, "linear_attention": mask if bool((mask == 0).any()) else None},
                              position_ids=(cut + torch.arange(width, device=dev))[None].expand(b, width),
                              past_key_values=state.for_batch(b), use_cache=False).last_hidden_state
                last = mask.sum(dim=1) - 1
                h_last = hidden[torch.arange(b, device=dev), last]
                kmax = max(len(rendered[k][0].cand_ids) for k in batch)
                cand = torch.full((b, kmax), -1, dtype=torch.long, device=dev)
                for r, k in enumerate(batch):
                    cand[r, :len(rendered[k][0].cand_ids)] = torch.tensor(rendered[k][0].cand_ids, device=dev)
                logits = self._candidate_logits(core, h_last, cand)
                for r, k in enumerate(batch):
                    rr, qtype = rendered[k][0], rendered[k][1]
                    t = float(self.scorer.temperatures.get(qtype, 1.0))
                    probs[k] = torch.softmax(logits[r, :len(rr.cand_ids)] / t, dim=-1).tolist()
                self._tock(t0, "suffix", b, (width + kv_cost) * b, sum(len(token_ids[k]) - cut for k in batch))
        return probs

    def _candidate_logits(self, core, h_last, cand_ids):
        """The published option_logits' read after the forward, unchanged: fp32 W[cand] @ h."""
        torch = self.torch
        safe = cand_ids.clamp(min=0)
        if self.jm.FP32_CANDIDATE_LOGITS:
            head = core.lm_head
            w = head.weight[safe].float()
            cand = torch.einsum("bkh,bh->bk", w, h_last.float())
            if getattr(head, "bias", None) is not None:
                cand = cand + head.bias[safe].float()
        else:
            cand = core.lm_head(h_last).float().gather(1, safe)
        return cand.masked_fill(cand_ids < 0, float("-inf"))

    def _probs(self, rendered, token_ids):
        probs = self._score_shared_prefix(rendered, token_ids) if self.prefix_cache and len(rendered) > 1 else None
        if probs is None:
            # batches by padded token budget, longest first, so memory is bounded and a long prompt runs alone
            order = sorted(rendered, key=lambda k: -rendered[k][2])
            probs = {}
            if self.batch_tokens:
                batches = self._pack([rendered[k][2] for k in order], 0)
            else:
                batches, i = [], 0
                while i < len(order):
                    n = 1
                    while i + n < len(order) and n < self.max_batch and rendered[order[i]][2] * (n + 1) <= self.token_budget:
                        n += 1
                    batches.append((i, n))
                    i += n
            for i, n in batches:
                batch = order[i:i + n]
                t0 = self._tick()
                out = self.scorer.score_rendered([(rendered[k][0], rendered[k][1]) for k in batch])
                self._tock(t0, "full", len(batch), rendered[order[i]][2] * len(batch), sum(rendered[k][2] for k in batch))
                for k, p in zip(batch, out):
                    probs[k] = p
        return probs

    def __call__(self, state, questions):
        renderer = self.scorer.renderer
        rendered, raw, token_ids, reversed_ids = {}, {}, {}, {}
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
            ids = self.tok.encode(r.prompt, add_special_tokens=False)
            n = len(ids)
            token_ids[qid] = ids
            rendered[qid] = (r, q2["type"], n)
            raw[qid] = {"labels": r.letters, "label_scheme": r.label_scheme, "prompt_tokens": n}
            if notes:
                raw[qid]["translation"] = notes
            if self.repeat_payload:
                raw[qid]["repeat_payload"] = True
            if self.reverse_avg and q2["type"] == "choice":
                rev = dict(q2)
                crit = q2["criteria"]
                rev["criteria"] = dict(reversed(list(crit.items()))) if isinstance(crit, dict) else list(reversed(crit))
                rr = self.scorer.render(state, rev)
                if rr.truncated:
                    raise Unsupported(f"prompt exceeds the declared context of {self.context} tokens; refused, not truncated")
                # Tuple identifiers cannot collide with the caller's string question ids.
                rid = (qid, "reverse")
                reversed_ids[qid] = rid
                rids = self.tok.encode(rr.prompt, add_special_tokens=False)
                rendered[rid] = (rr, q2["type"], len(rids))
                token_ids[rid] = rids
        try:
            probs = self._probs(rendered, token_ids)
        except self.torch.OutOfMemoryError:
            # the box is shared, so free memory moves; retry this request one row per forward (same values to
            # bf16 batch noise) before letting the runner halt on a device error
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device == "mps":
                self.torch.mps.empty_cache()
            self.oom_retries += 1
            saved = (self.max_batch, self.batch_tokens)
            self.max_batch, self.batch_tokens = 1, 1
            try:
                probs = self._probs(rendered, token_ids)
            finally:
                self.max_batch, self.batch_tokens = saved
            for qid in raw:
                raw[qid]["oom_retry"] = True
        for qid, rid in reversed_ids.items():
            keys = rendered[qid][0].keys
            forward = dict(zip(keys, probs[qid]))
            reverse = dict(zip(rendered[rid][0].keys, probs[rid]))
            if self.reverse_avg_below is not None and max(forward.values()) >= self.reverse_avg_below:
                raw[qid]["reverse_avg_skipped"] = True
                raw[qid]["order_probabilities"] = {"forward": forward, "reverse": reverse}
                continue
            # Preserve zero support from underflow in either read: log(0) = -inf.
            logs = [(math.log(forward[k]) + math.log(reverse[k])) / 2
                    if forward[k] > 0 and reverse[k] > 0 else -math.inf for k in keys]
            top = max(logs)
            if top == -math.inf:
                raise ValueError("reverse_avg reads have no common nonzero probability support")
            weights = [math.exp(x - top) for x in logs]
            total = math.fsum(weights)
            probs[qid] = [x / total for x in weights]
            raw[qid]["reverse_avg"] = True
            raw[qid]["order_probabilities"] = {"forward": forward, "reverse": reverse}
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
