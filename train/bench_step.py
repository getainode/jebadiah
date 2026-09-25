"""Micro-benchmark: seconds per optimizer step and peak memory for a few batch shapes, with and
without gradient checkpointing, on this box. Uses real training examples."""
import json, sys, time, torch
from peft import LoraConfig, get_peft_model
from jebadiah_model import load_base, load_tokenizer, option_logits
from jebadiah_prompt import Renderer
from train_jebadiah import DecideCollator, DecideDataset, read_jsonl, target_modules_for

cfg = json.load(open(sys.argv[1]))
tok = load_tokenizer(cfg["base_model"], cfg.get("base_revision"))
renderer = Renderer(tok, 2048)
model = load_base(cfg["base_model"], cfg.get("base_revision"), attn_implementation=cfg.get("attn_implementation", "sdpa"))
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
print("trainable", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
import random
ds = DecideDataset(read_jsonl(cfg["dataset_path"]))
random.Random(0).shuffle(ds.items)
coll = DecideCollator(tok, renderer, shuffle_choice=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
lens = []
for i in range(0, 600, 1):
    b = coll([ds[i]]); lens.append(int(b["attention_mask"].sum()))
lens.sort(); print("token lengths of 600 examples: p50", lens[300], "p90", lens[540], "p99", lens[594], "max", lens[-1], "mean", sum(lens)/len(lens), flush=True)
# put the longest examples first so the peak is measured on them
ds.items.sort(key=lambda it: -len(tok.encode(renderer.render(it[0], it[1]).prompt, add_special_tokens=False)))
model.train()  # checkpointing only applies in train mode
for ckpt, bss in ((False, (1, 2)), (True, (2, 4))):
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
    for bs in bss:
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            times = []
            for step in range(7):
                batch = coll([ds[(step * bs + j) % 40] for j in range(bs)])
                batch = {k: v.cuda() for k, v in batch.items()}
                torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = option_logits(model, batch["input_ids"], batch["attention_mask"], batch["cand_ids"])
                    loss = -(batch["target"] * torch.log_softmax(logits, -1).masked_fill(batch["target"] == 0, 0.0)).sum(-1).mean()
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); times.append(time.time() - t)
            print(json.dumps({"checkpointing": ckpt, "batch": bs, "step_s": [round(x, 2) for x in times], "step_s_after_warmup": round(sum(times[2:]) / 5, 2),
                              "examples_per_s": round(bs * 5 / sum(times[2:]), 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                              "loss": round(loss.item(), 3), "tokens": int(batch["attention_mask"].sum())}), flush=True)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"checkpointing": ckpt, "batch": bs, "oom": True}), flush=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
print("BENCH_DONE", flush=True)
