"""Compare adapter and bf16 merged picks on up to 260 held-out calibration questions."""
import argparse
import gc
import json
from pathlib import Path

import torch

from jebadiah_model import Scorer, load_base, load_adapter, load_tokenizer


def verify(base, adapter, merged, calib, device="cuda", max_tokens=4096):
    rows = [json.loads(line) for line in Path(calib).read_text(encoding="utf-8").split("\n") if line.strip()]
    items = [(r["state"], q) for r in rows for q in r["questions"].values()][:260]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    def score(path, adapter_path=None):
        model = load_base(str(path), dtype=dtype, device=device)
        if adapter_path:
            model = load_adapter(model, str(adapter_path))
        scorer = Scorer(model, load_tokenizer(str(path)), max_tokens, device=device)
        results = []
        for state, question in items:
            rendered = scorer.render(state, question)
            results.append(scorer.score_rendered([(rendered, question["type"])])[0])
        del scorer, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return results
    before, after = score(base, adapter), score(merged)
    if not items:
        raise ValueError("No merge verification questions")
    picks = sum(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
                for a, b in zip(before, after))
    report = {"questions": len(items), "identical_picks": picks, "agreement": picks / len(items),
              "max_probability_delta": max(abs(x-y) for a, b in zip(before, after) for x, y in zip(a, b))}
    (Path(merged) / "merge_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["agreement"] < 0.99:
        raise ValueError(f"Merge agreement below 99 percent: {report}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("base", "adapter", "merged", "calib"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=4096)
    print(json.dumps(verify(**vars(p.parse_args()))))
