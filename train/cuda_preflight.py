"""Fail before loading large weights if bf16 or Gated DeltaNet backward is broken."""
import json
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from peft import LoraConfig, get_peft_model
from jebadiah_model import option_logits


def tiny_config():
    return Qwen3_5TextConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=4, layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=4096, use_cache=False)


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A working bf16 CUDA device is required")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # import must succeed, no silent fallback
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (a @ a).float().square().mean().backward()
    assert torch.isfinite(a.grad).all()
    model = Qwen3_5ForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    leaves = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if hasattr(module, "lora_A")}
    required = {"in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"}
    if not required <= leaves:
        raise RuntimeError(f"DeltaNet LoRA projections missing: {required - leaves}")
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    for step in range(2):
        ids = torch.randint(0, 128, (1, 512), device="cuda")
        logits = option_logits(model, ids, torch.ones_like(ids), torch.tensor([[30, 31]], device="cuda"))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0], device="cuda"))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    print(json.dumps({"cuda_preflight": "passed", "gpu": torch.cuda.get_device_name(),
                      "capability": torch.cuda.get_device_capability(), "torch": torch.__version__,
                      "lora_leaves": sorted(leaves), "steps": 2}), flush=True)


if __name__ == "__main__":
    main()
