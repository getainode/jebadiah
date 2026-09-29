"""The model's tokenizer and chat template from a Jebadiah repository's small files, without
transformers or torch: `tokenizers` for tokenizer.json and jinja2 for chat_template.jinja, set up
the way transformers renders chat templates. tests/test_tokenizer.py holds it to transformers'
output when both are installed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime


def _raise(message):
    raise ValueError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class LightTokenizer:
    def __init__(self, folder: str):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        cfg_path = os.path.join(folder, "tokenizer_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

        def text(v):
            return v.get("content") if isinstance(v, dict) else v

        self.eos_token = text(cfg.get("eos_token"))
        self.bos_token = text(cfg.get("bos_token"))
        self.pad_token = text(cfg.get("pad_token")) or self.eos_token
        specials = {t for t in (self.eos_token, self.bos_token, self.pad_token, text(cfg.get("unk_token"))) if t}
        specials.update(text(t) for t in cfg.get("additional_special_tokens") or [])
        for v in (cfg.get("added_tokens_decoder") or {}).values():
            if v.get("special"):
                specials.add(v["content"])
        self.all_special_ids = sorted(i for i in (self._tok.token_to_id(t) for t in specials) if i is not None)

        tpl_path = os.path.join(folder, "chat_template.jinja")
        template = open(tpl_path).read() if os.path.exists(tpl_path) else cfg.get("chat_template")
        if not template:
            raise ValueError(f"{folder} has no chat_template.jinja")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(template)

    @property
    def pad_token_id(self):
        return self._tok.token_to_id(self.pad_token) if self.pad_token else None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False, **kwargs):
        text = self._template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                     bos_token=self.bos_token or "", eos_token=self.eos_token or "", **kwargs)
        return self.encode(text) if tokenize else text
