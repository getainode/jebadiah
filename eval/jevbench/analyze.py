"""Public-item numbers for one run, computed the way JevBench computes them.

  python3 analyze.py runs/<label> [price_in_per_m]

- accuracy, Brier (multi-class sum), ECE (top label, 10 bins): jevbench.summarize.metric,
  unchanged, over all 231 and per public file (easy / original / hard);
- hard-tier ECE and probability fidelity (TVD to gold_probs on the public probability items),
  combined into the v1.3 Calibration formula (composite_v13.calibration); this is the public
  half only, the official axis also uses 109 held-out hard items and the 308 sealed ones;
- latency p50/p95 raw and with the board's own-server adjustment (x2 + 0.15 s), and the
  Speed axis that would give (composite_v13.speed, endpoint kind "gpu");
- a cost estimate by the board's rule (measured input tokens x the base model's list price).
Writes <run>/public-metrics.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "repo"
sys.path.insert(0, str(REPO))
from jevbench import composite_v13 as c13  # noqa: E402
from jevbench.metrics import percentile  # noqa: E402
from jevbench.summarize import metric  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

FILES = {"easy": "easy.jsonl", "original": "original.jsonl", "hard": "hard.jsonl"}


def main(run, price=None):
    run = Path(run)
    recs = [json.loads(line) for line in open(run / "results.jsonl") if line.strip()]
    by_file = {k: load_jsonl(str(REPO / "datasets/public" / f)) for k, f in FILES.items()}
    tasks = [t for ts in by_file.values() for t in ts]
    out = {"all": metric(tasks, recs)}
    for k, ts in by_file.items():
        out[k] = metric(ts, recs)
    rid = {r["task_id"]: r for r in recs}
    hard = by_file["hard"]
    prob_items = [t for t in hard if t.provenance.get("gold_probs")]
    tvds = [c13.tvd(rid[t.id]["probs"] or {}, t.provenance["gold_probs"], t.labels)
            if rid.get(t.id, {}).get("probs") else 1.0 for t in prob_items]
    mean_tvd = sum(tvds) / len(tvds)
    ece_hard = out["hard"]["ece"]["ece"]
    fam = {}
    for t in hard:
        f = fam.setdefault(t.family, [0, 0])
        f[0] += 1
        f[1] += bool(rid.get(t.id, {}).get("correct"))
    lat = [r["latency_s"] for r in recs]
    p50, p95 = percentile(lat, .5), percentile(lat, .95)
    toks = [r["usage"].get("input_tokens", 0) for r in recs]
    mean_in = sum(toks) / len(toks)
    res = {
        "n": len(recs),
        "correct": out["all"]["n_correct"],
        "public_accuracy": out["all"]["accuracy"],
        "tiers_public": {k: {"n": out[k]["n_scorable"], "correct": out[k]["n_correct"], "accuracy": out[k]["accuracy"]}
                         for k in FILES},
        "valid": out["all"]["n_valid"],
        "renormalized": out["all"]["n_renormalized"],
        "brier_mean_all": out["all"]["brier_mean"],
        "ece_all": out["all"]["ece"]["ece"],
        "brier_mean_hard": out["hard"]["brier_mean"],
        "ece_hard": ece_hard,
        "probability_items": len(prob_items),
        "mean_tvd_gold_probs": mean_tvd,
        "calibration_v13_formula_public_hard_only": c13.calibration(ece_hard, mean_tvd),
        "hard_by_family": {k: f"{v[1]}/{v[0]}" for k, v in sorted(fam.items())},
        "ordinal_mae": out["all"]["ordinal_mae"],
        "latency_raw_s": {"p50": p50, "p95": p95, "max": max(lat), "sum": sum(lat)},
        "latency_adjusted_s": {"p50": c13.adjusted_latency(p50, "gpu"), "p95": c13.adjusted_latency(p95, "gpu")},
        "speed_axis_if_scored_like_board": c13.speed(p50, p95, "gpu"),
        "mean_input_tokens": mean_in,
    }
    if price:
        usd = mean_in * 1000 * price / 1e6
        res["cost_estimate"] = {"price_in_per_m": price, "usd_per_1000": usd, "cost_axis": c13.cost(usd)}
    (run / "public-metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
