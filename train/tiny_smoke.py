"""Build a deterministic local Qwen3.5 tiny model and 50 oracle-labeled training rows."""
import argparse
import json
import string
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM
from cuda_preflight import tiny_config
from ainode_prompt_verbatim import option_label
from v21_pipeline import sha256


def prepare(root):
    root = Path(root)
    base, data = root / "base", root / "data"
    base.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    labels = [option_label(i) for i in range(255)] + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    words = ["[UNK]", "[PAD]", "[EOS]"] + sorted(set(labels)) + ["true", "false", "yes", "no", "zero", "one", "two"]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ':\\n' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:\\n' }}{% endif %}"
    tok.save_pretrained(base)
    torch.manual_seed(17)
    config = tiny_config()
    config.vocab_size = len(tok)
    config.pad_token_id = tok.pad_token_id
    config.eos_token_id = tok.eos_token_id
    Qwen3_5ForCausalLM(config).save_pretrained(base)
    manifest = {"files": {}}
    for split, count in (("train", 50), ("calib", 12)):
        rows = []
        for i in range(count):
            kind = ("choice", "noul", "score")[i % 3]
            q = {"type": kind, "instructions": "Is the integer even?"}
            if kind == "choice":
                q["criteria"] = {"yes": "Even", "no": "Odd"}
                label = "yes" if i % 2 == 0 else "no"
            elif kind == "noul":
                label = i % 2 == 0
            else:
                q = {"type": kind, "instructions": "Choose integer remainder modulo three.", "criteria": ["zero", "one", "two"]}
                label = i % 3
            rows.append({"id": f"{split}-{i}", "family_id": f"{split}-family-{i}",
                         "state": {"integer": i + (1002 if split == "calib" else 0)},
                         "questions": {"q": q}, "label": {"q": label}})
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["files"][path.name] = {"sha256": sha256(path), "rows": count, "questions": count}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"TINY_READY base={base} data={data} train=50 calib=12")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    prepare(p.parse_args().root)
