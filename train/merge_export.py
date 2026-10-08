"""CPU shardwise fp32 LoRA merge, bf16 storage, preserving the original Hub layout."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def merge(base, adapter, output):
    base, adapter, output = map(Path, (base, adapter, output))
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    for unsupported in ("fan_in_fan_out", "use_dora", "modules_to_save", "rank_pattern", "alpha_pattern", "lora_bias"):
        if cfg.get(unsupported):
            raise ValueError(f"Unsupported merge setting: {unsupported}")
    weights = load_file(adapter / "adapter_model.safetensors", device="cpu")
    pairs = {}
    for key in weights:
        if not key.startswith("base_model.model.") or not key.endswith(".lora_A.weight"):
            continue
        stem = key[len("base_model.model."):-len(".lora_A.weight")]
        pairs[stem + ".weight"] = (weights[key], weights[key.replace(".lora_A.", ".lora_B.")])
    if not pairs or len(weights) != 2 * len(pairs):
        raise ValueError("Every saved adapter tensor must belong to exactly one LoRA pair")
    index_path = base / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists() else ["model.safetensors"]
    output.mkdir(parents=True, exist_ok=True)
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    matched = set()
    for shard in shards:
        tensors = {}
        with safe_open(base / shard, framework="pt", device="cpu") as handle:
            meta = handle.metadata()
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                # Text-only Qwen CausalLM adapters map back to multimodal checkpoint language weights.
                adapter_key = key.replace("model.language_model.", "model.", 1)
                if adapter_key in pairs:
                    if adapter_key in matched:
                        raise ValueError(f"Duplicate target {adapter_key}")
                    a, b = pairs[adapter_key]
                    delta = b.float() @ a.float()
                    if delta.shape != tensor.shape:
                        raise ValueError(f"LoRA shape mismatch for {key}")
                    tensor = tensor.float() + scale * delta
                    matched.add(adapter_key)
                if tensor.is_floating_point():
                    tensor = tensor.to(torch.bfloat16)
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite merged tensor {key}")
                tensors[key] = tensor.contiguous()
        save_file(tensors, output / shard, metadata=meta or {"format": "pt"})
        del tensors
    missing = set(pairs) - matched
    if missing:
        raise ValueError(f"Unmatched LoRA targets: {sorted(missing)}")
    for path in base.iterdir():
        if path.is_file() and (path.suffix in (".json", ".txt", ".model", ".jinja") or path.name.startswith("LICENSE")):
            shutil.copyfile(path, output / path.name)
    shutil.copyfile(adapter / "prompt_contract.json", output / "prompt_contract.json")
    scripts = output / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("ainode_prompt_verbatim.py", "jebadiah_prompt.py", "jebadiah_model.py"):
        shutil.copyfile(Path(__file__).parent / name, scripts / name)
    report = {"merged_pairs": len(matched), "rank": cfg["r"], "alpha": cfg["lora_alpha"], "storage_dtype": "bfloat16"}
    (output / "merge_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.adapter, args.output)))
