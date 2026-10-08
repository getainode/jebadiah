"""Train Jebadiah: a LoRA on a base causal LM with the Nimble objective.

For each (state, question) the prompt is rendered by jebadiah_prompt.build_prompt, ending in
"Answer:"; the loss is the cross-entropy of the right option key among the candidate keys'
next-token logits at that position. Nothing else is supervised, nothing is generated.

Config: a JSON file with AINode's TrainingConfig keys (base_model, dataset_path, output_dir,
method, num_epochs, batch_size, learning_rate, lora_rank, lora_alpha, max_seq_length,
gradient_accumulation_steps, warmup_steps, weight_decay, use_gradient_checkpointing,
attn_implementation, eval_steps) plus a `decide` block for what is specific to this objective.

Numerics rules carried from ainode/training/AGENTS.md: the tokenizer never pads (the collator
does), logging_nan_inf_filter stays False, a non-finite loss ends the run, and the saved adapter
is scanned for non-finite values before the run is called complete.

v1 additions (jebadiah-rent sweep): the candidate logits are read in fp32 (jebadiah_model), and
`decide.score_targets` = "source" (v0) or "ordinal" (score questions train towards a unimodal
kernel around the hard label, `decide.score_ordinal_adjacent` = the weight of a neighbouring
level relative to the label; the temperature fit uses the same target with --target train).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_pt_utils import LengthGroupedSampler

from jebadiah_model import FP32_CANDIDATE_LOGITS, Scorer, load_base, load_tokenizer, option_logits, template_sha256
from hub_checkpoints import HubCheckpoint, resolve_resume
from jebadiah_prompt import PROMPT_SOURCE_COMMIT, PROMPT_SOURCE_SHA256, Renderer, wire_keys


class NonFiniteLoss(RuntimeError):
    pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class DecideDataset(Dataset):
    """One example per (record, question): the record's state, that question, its label and, when
    the source gives one, its soft target distribution over the wire keys."""

    def __init__(self, records):
        self.items = []
        self.lengths = None
        for r in records:
            for qid, q in r["questions"].items():
                target = (r.get("target") or {}).get(qid)
                self.items.append((r["state"], q, r["label"][qid], r.get("id", ""), qid, target))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def measure_lengths(self, renderer):
        """Lengths of the actual contracted prompts, without consuming the shuffle RNG.

        Choice permutations can move a token boundary slightly, but keep the same state,
        question and options. Canonical order is sufficient for padding-efficient sampling.
        """
        self.lengths = [len(renderer.tok.encode(renderer.render(state, q).prompt,
                                              add_special_tokens=False))
                        for state, q, *_ in self.items]
        return self.lengths


def label_index(q: dict, label, keys: list[str]) -> int:
    t = q["type"]
    if t == "noul":
        return keys.index("true" if label else "false")
    if t == "score":
        return keys.index(str(int(label)))
    return keys.index(str(label))


def ordinal_target(n_levels: int, label: int, adjacent: float) -> list[float]:
    """A unimodal target over ordered levels: the label keeps the most mass, each level one step
    away gets `adjacent` of it, two steps away adjacent**4, and so on (a Gaussian in level
    distance, SORD-style soft ordinal labels). At adjacent=0.2 a middle label of five keeps 0.71
    and an edge label 0.83; a level two away gets 0.0016 x the label's weight."""
    if adjacent <= 0:
        return [1.0 if i == label else 0.0 for i in range(n_levels)]
    w = [adjacent ** ((i - label) ** 2) for i in range(n_levels)]
    s = sum(w)
    return [x / s for x in w]


def make_target(q: dict, label, keys: list[str], target: dict | None, score_targets: str,
                score_adjacent: float) -> list[float]:
    """The distribution the loss is trained towards, in the rendered key order.
    score_targets="source": the source's distribution when it gives one, else one-hot (v0).
    score_targets="ordinal": score questions use the ordinal kernel around the hard label and
    ignore the source distribution; choice and noul are unchanged."""
    li = label_index(q, label, keys)
    if q["type"] == "score" and score_targets == "ordinal":
        # keys are the level indices as strings in display order (always "0".."k-1" for score)
        levels = [int(k) for k in keys]
        dist = ordinal_target(len(levels), levels.index(int(label)), score_adjacent)
        return dist
    if target:
        return [float(target.get(k, 0.0)) for k in keys]
    return [1.0 if i == li else 0.0 for i in range(len(keys))]


class DecideCollator:
    """Renders through AINode's renderer, tokenizes and right-pads a batch; returns the candidate
    token ids, the index of the right key and the target distribution (one-hot for a hard label,
    the source's distribution for a soft one, or the ordinal kernel for score questions when
    configured). Choice options are shuffled per example when configured, so the adapter cannot
    learn a position prior."""

    def __init__(self, tokenizer, renderer: Renderer, shuffle_choice: bool, seed: int = 0,
                 score_targets: str = "source", score_adjacent: float = 0.2,
                 pad_to_multiple_of: int | None = None):
        self.tok = tokenizer
        self.renderer = renderer
        self.shuffle_choice = shuffle_choice
        self.rng = random  # Trainer checkpoints and restores the global RNG state
        self.truncated = 0
        if score_targets not in ("source", "ordinal"):
            raise ValueError(f"score_targets must be 'source' or 'ordinal', got {score_targets!r}")
        self.score_targets = score_targets
        self.score_adjacent = float(score_adjacent)
        if pad_to_multiple_of is not None and pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        prompts, cands, labels, targets = [], [], [], []
        for state, q, label, _, _, target in batch:
            order = None
            if self.shuffle_choice and q["type"] == "choice":
                order = wire_keys(q)
                self.rng.shuffle(order)
            r = self.renderer.render(state, q, order)
            self.truncated += int(r.truncated)
            prompts.append(r.prompt)
            cands.append(r.cand_ids)
            li = label_index(q, label, r.keys)
            labels.append(li)
            targets.append(make_target(q, label, r.keys, target, self.score_targets, self.score_adjacent))
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                       pad_to_multiple_of=self.pad_to_multiple_of)
        kmax = max(len(c) for c in cands)
        cand = torch.full((len(batch), kmax), -1, dtype=torch.long)
        tgt = torch.zeros((len(batch), kmax))
        for i, (c, tg) in enumerate(zip(cands, targets)):
            cand[i, :len(c)] = torch.tensor(c)
            tgt[i, :len(tg)] = torch.tensor(tg)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask,
                "cand_ids": cand, "label_idx": torch.tensor(labels), "target": tgt}


class DecideTrainer(Trainer):
    """Loss: cross-entropy of the target distribution against the softmax over the candidate
    label logits at the answer position (the Nimble objective; Kev's soft form when the source
    gives a distribution). Padded candidate slots are -inf and carry zero target."""

    def __init__(self, *args, checkpoint_min_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        if checkpoint_min_tokens is not None and checkpoint_min_tokens < 1:
            raise ValueError("checkpoint_min_tokens must be positive")
        self.checkpoint_min_tokens = checkpoint_min_tokens
        self.checkpoint_modules = [module for module in self.model.modules()
                                   if getattr(module, "gradient_checkpointing", False)]
        if checkpoint_min_tokens is not None and not self.checkpoint_modules:
            raise ValueError("Adaptive checkpointing requires enabled gradient checkpointing")

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self.args.train_sampling_strategy == "group_by_length":
            if getattr(dataset, "lengths", None) is None:
                raise ValueError("Length grouping requires measured rendered prompt lengths")
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=dataset.lengths,
            )
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.checkpoint_min_tokens is not None and model.training:
            # Each microbatch is backpropagated before the next forward. Short batches can
            # retain activations; long batches retain the original non-reentrant checkpoint
            # functions and RNG preservation. Do not re-register PEFT's input-grad hooks.
            enabled = inputs["input_ids"].shape[1] >= self.checkpoint_min_tokens
            for module in self.checkpoint_modules:
                module.gradient_checkpointing = enabled
        logits = option_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["cand_ids"])
        logp = torch.log_softmax(logits, dim=-1)
        target = inputs["target"].to(logp.device)
        loss = -(target * logp.masked_fill(target == 0, 0.0)).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise NonFiniteLoss("non-finite candidate-logit loss")
        return (loss, {"logits": logits}) if return_outputs else loss


class FiniteGuard(TrainerCallback):
    """A non-finite loss or grad norm ends the run; nothing is saved (AGENTS.md rule)."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        checks = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                  if parameter.grad is not None]
        if checks and not torch.stack(checks).all():
            raise NonFiniteLoss(f"non-finite gradient at step {state.global_step}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        for k in ("loss", "grad_norm"):
            v = (logs or {}).get(k)
            if v is not None and not math.isfinite(float(v)):
                raise NonFiniteLoss(f"{k}={v} at step {state.global_step}")


class CalibEval(TrainerCallback):
    """Accuracy and NLL on the held-out slice every eval_steps, through the same scorer the
    evaluator uses (no temperature). Logged into the trainer's history."""

    def __init__(self, records, tokenizer, renderer, every, batch_size=8, limit=None):
        self.ds = DecideDataset(records).items[:limit]
        self.tok = tokenizer
        self.renderer = renderer
        self.every = every
        self.batch_size = batch_size

    def run(self, model):
        scorer = Scorer(model, self.tok, self.renderer.max_tokens, renderer=self.renderer)
        by_type = {}
        for i in range(0, len(self.ds), self.batch_size):
            chunk = self.ds[i:i + self.batch_size]
            rendered = [(scorer.render(s, q), q["type"]) for s, q, _, _, _, _ in chunk]
            probs = scorer.score_rendered(rendered)
            for (s, q, label, _, _, _), (r, _), p in zip(chunk, rendered, probs):
                li = label_index(q, label, r.keys)
                acc = by_type.setdefault(q["type"], {"n": 0, "correct": 0, "nll": 0.0})
                acc["n"] += 1
                acc["correct"] += int(max(range(len(p)), key=lambda j: p[j]) == li)
                acc["nll"] += -math.log(max(p[li], 1e-12))
        out = {}
        n_all = sum(v["n"] for v in by_type.values())
        out["calib_accuracy"] = sum(v["correct"] for v in by_type.values()) / max(n_all, 1)
        out["calib_nll"] = sum(v["nll"] for v in by_type.values()) / max(n_all, 1)
        for t, v in by_type.items():
            out[f"calib_accuracy_{t}"] = v["correct"] / v["n"]
            out[f"calib_nll_{t}"] = v["nll"] / v["n"]
        model.train()
        return out

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every and state.global_step % self.every == 0:
            metrics = self.run(model)
            state.log_history.append({"step": state.global_step, **metrics})
            print(json.dumps({"step": state.global_step, **{k: round(v, 4) for k, v in metrics.items()}}), flush=True)


def scan_saved_weights(out_dir: str) -> None:
    from safetensors.torch import load_file
    for path in glob.glob(os.path.join(out_dir, "*.safetensors")):
        for name, t in load_file(path).items():
            if not torch.isfinite(t.float()).all():
                raise NonFiniteLoss(f"non-finite value in saved weight {name} ({path})")


def target_modules_for(model, requested: list[str]) -> list[str]:
    present = set()
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in requested:
            present.add(leaf)
    missing = [m for m in requested if m not in present]
    return sorted(present), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=-1, help="cut the run short (smoke tests)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    cfg["max_steps"] = args.max_steps
    if cfg.get("method", "lora") != "lora":
        raise ValueError("Only method=lora is supported; refusing a different training method")
    for key in ("lora_rank", "lora_alpha", "max_seq_length", "batch_size", "gradient_accumulation_steps", "save_steps"):
        if key in cfg and int(cfg[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("num_epochs", "learning_rate"):
        if key in cfg and (not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0):
            raise ValueError(f"{key} must be finite and positive")
    device = cfg.get("device", "cuda")
    dtype = getattr(torch, cfg.get("dtype", "bfloat16" if device == "cuda" else "float32"))
    dec = cfg.get("decide", {})
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = int(cfg.get("seed", 17))
    torch.manual_seed(seed)
    random.seed(seed)

    t0 = time.time()
    tok = load_tokenizer(cfg["base_model"], cfg.get("base_revision"))
    renderer = Renderer(tok, int(cfg.get("max_seq_length", 4096)))
    prompt_contract = {"prompt_source_commit": PROMPT_SOURCE_COMMIT, "prompt_source_sha256": PROMPT_SOURCE_SHA256,
                       "chat_template_sha256": template_sha256(tok), "single_token_labels": renderer.max_options,
                       "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": False, "thinking": False}}
    if cfg.get("prompt_source_sha256") and cfg["prompt_source_sha256"] != PROMPT_SOURCE_SHA256:
        raise SystemExit(f"config pins prompt_source_sha256 {cfg['prompt_source_sha256'][:12]} but the renderer is {PROMPT_SOURCE_SHA256[:12]}")
    for key in ("chat_template_sha256", "single_token_labels"):
        if key in cfg and cfg[key] != prompt_contract[key]:
            raise ValueError(f"Pinned prompt contract mismatch for {key}")
    model = load_base(cfg["base_model"], cfg.get("base_revision"),
                      attn_implementation=cfg.get("attn_implementation", "sdpa"), dtype=dtype, device=device)
    load_s = time.time() - t0

    from peft import LoraConfig, get_peft_model
    requested = dec.get("target_modules", "all-linear")
    if requested == "all-linear":
        targets, missing = "all-linear", []
    else:
        targets, missing = target_modules_for(model, requested)
    lcfg = LoraConfig(r=int(cfg.get("lora_rank", 64)), lora_alpha=int(cfg.get("lora_alpha", 128)),
                      lora_dropout=float(dec.get("lora_dropout", 0.05)), bias="none",
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if cfg.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora_leaves = sorted({n.rsplit(".", 2)[-2] if n.endswith("lora_A.default") else n.rsplit(".", 1)[-1]
                          for n, mod in model.named_modules() if hasattr(mod, "lora_A")})
    print(json.dumps({"target_modules": targets, "lora_modules": lora_leaves, "missing": missing,
                      "trainable_params": trainable, "total_params": total, "load_s": round(load_s, 1),
                      **prompt_contract}), flush=True)

    train_records = read_jsonl(cfg["dataset_path"])
    calib_records = read_jsonl(cfg["eval_dataset_path"]) if cfg.get("eval_dataset_path") else []
    train_ds = DecideDataset(train_records)
    if cfg.get("group_by_length", False):
        length_t0 = time.perf_counter()
        lengths = train_ds.measure_lengths(renderer)
        print(json.dumps({"length_setup_s": round(time.perf_counter() - length_t0, 2),
                          "prompt_tokens_mean": round(sum(lengths) / len(lengths), 1),
                          "prompt_tokens_max": max(lengths)}), flush=True)
    collator = DecideCollator(tok, renderer, shuffle_choice=bool(dec.get("shuffle_choice_options", True)), seed=seed,
                              score_targets=dec.get("score_targets", "source"),
                              score_adjacent=float(dec.get("score_ordinal_adjacent", 0.2)),
                              pad_to_multiple_of=cfg.get("pad_to_multiple_of"))

    targs = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=float(cfg.get("num_epochs", 1)),
        max_steps=args.max_steps,
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=device == "cuda" and dtype == torch.bfloat16,
        use_cpu=device == "cpu",
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
        train_sampling_strategy="group_by_length" if cfg.get("group_by_length", False) else "random",
        remove_unused_columns=False,
        gradient_checkpointing=False,  # enabled on the PEFT model above
        optim="adamw_torch",
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
    )
    callbacks = [FiniteGuard(), HubCheckpoint(cfg)]
    calib_callback = None
    if calib_records:
        calib_callback = CalibEval(calib_records, tok, renderer,
                                   every=int(cfg.get("eval_steps", 0)), limit=dec.get("calib_eval_limit"))
        callbacks.append(calib_callback)
    trainer = DecideTrainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator,
                            callbacks=callbacks, checkpoint_min_tokens=cfg.get("checkpoint_min_tokens"))

    trainer.model_accepts_loss_kwargs = False  # mean loss must be divided by accumulation
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    try:
        trainer.train(resume_from_checkpoint=resolve_resume(cfg, targs.output_dir))
    except NonFiniteLoss as e:
        print(f"AINODE_ERROR:NAN_LOSS {e}", flush=True)
        sys.exit(3)
    train_s = time.time() - t1
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None

    final = calib_callback.run(model) if calib_callback else {}
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    scan_saved_weights(adapter_dir)
    json.dump(prompt_contract, open(os.path.join(adapter_dir, "prompt_contract.json"), "w"), indent=1)
    summary = {
        "config": cfg, "target_modules": targets, "lora_modules": lora_leaves, "trainable_params": trainable,
        **prompt_contract,
        "train_examples": len(train_ds), "calib_examples": len(DecideDataset(calib_records)),
        "truncated_prompts": collator.truncated, "steps": trainer.state.global_step,
        "wall_clock_s": round(train_s, 1), "load_s": round(load_s, 1),
        "peak_memory_allocated_gb": round(peak_gb, 2) if peak_gb is not None else None,
        "final_calib": final, "hardware": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "torch": torch.__version__, "adapter_dir": adapter_dir,
        "fp32_candidate_logits": FP32_CANDIDATE_LOGITS,
        "score_targets": collator.score_targets, "score_ordinal_adjacent": collator.score_adjacent,
    }
    json.dump(trainer.state.log_history, open(os.path.join(out_dir, "log_history.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out_dir, "train_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1), flush=True)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
