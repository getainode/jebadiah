# 10. JevBench standing

**Question.** Where does Jebadiah stand on JevBench (fstandhartinger/jevbench, v1.4.2), a public benchmark for
one-pass typed decisions, next to the models on its board?

**Setup.** The 231 public items (48 easy, 72 standard, 111 hard) through JevBench's own unchanged `typesafe`
adapter and harness (`eval/jevbench/run_bench.sh`), one request per question, against a loopback `/v1/systemone`
shim (`eval/jevbench/shim/systemone_shim.py`) over the Decision Index engine, which imports each model's published
inference scripts unchanged and applies its temperatures. Metrics are computed with the benchmark's own functions
(`eval/jevbench/analyze.py`). Everything ran on an Apple M3 Ultra (PyTorch MPS, bf16). JevBench was never used for
training, calibration or model selection; `eval/jevbench/overlap_check.py` found no shared 8-word span between the
v1 training pool and the public items.

| system | public acc | easy | standard | hard | ECE hard | source |
|---|---:|---|---|---|---:|---|
| **Jebadiah 27B** (`27b-chat-v1`) | 0.866 | 48/48 | 70/72 | 82/111 | 0.113 | our run, standalone server |
| Jebadiah 9B v1 | 0.818 | 48/48 | 70/72 | 71/111 | 0.127 | our run |
| **Jebadiah 9B v2** (`9b-chat-v1`) | 0.818 | 48/48 | 71/72 | 70/111 | 0.052 | our run |
| Jebadiah 9B v3 (unpublished) | 0.814 | 48/48 | 69/72 | 71/111 | 0.071 | our run |
| Jebadiah 4B v3-c20f (unpublished) | 0.792 | 48/48 | 71/72 | 64/111 | 0.071 | our run |
| Jebadiah 4B v1 | 0.762 | 48/48 | 72/72 | 56/111 | 0.087 | our run |
| **Jebadiah 4B v2** (`4b-chat-v1`) | 0.758 | 48/48 | 70/72 | 57/111 | 0.088 | our run |
| JPT-4B | 0.879 | 48/48 | 68/72 | 87/111 | | self-reported by its author |
| Jev 1.13.0 | 0.866 | 48/48 | 71/72 | 81/111 | | board |
| Winnow-12B Q8 | 0.857 | 48/48 | 69/72 | 81/111 | | board |
| JPT-9B | 0.857 | | | | | self-reported by its author |
| JevK5 v0.2.0 | 0.853 | | | | | board |
| decider-4b v2 | 0.835 | | | | | board |
| decider-35b-a3b | 0.831 | | | | | board |
| Hopper | 0.823 | | | | | board |

**The 27B (2026-09-26).** Run the same way except for the server: `frontier-infra/jebadiah-27b` at revision
1c0d794f (which carries the refit score temperature, 0.76) served by this repository's standalone server
(`server/`, `jebadiah-serve --model <local copy> --device mps --alias jebadiah-27b-local`, bf16, 56 GB), which answers
`POST /v1/systemone` itself, so no shim was needed. The weights were a local copy whose 18 safetensors files and
tokenizer matched the hub's sha256 at that revision; the served model name the records carry was a local path and
is replaced there by `frontier-infra/jebadiah-27b@1c0d794f`. 231 of 231 answered and valid, none renormalized, same
dataset hash. Of its 29 hard misses, 21 are the planted surface answer; temporal and numeric 4 of 15. On the 111 hard
items it gets 18 right that 9B v2 misses and misses 6 that 9B v2 gets. Latency on MPS is p50 1.9 s, p95 25.7 s
(1,528 s for all 231), which is a Mac bf16 number and not a serving claim.

**Where we stand, candidly.** On the public items Jebadiah 9B (v1 or v2, 0.818) is below every model on that list.
The easy and standard tiers are at the ceiling; the whole gap is the hard tier (70 or 71 of 111 against Jev's 81 and
JPT-4B's 87). Of 9B v1's 40 hard misses, 34 are the item's planted surface answer, above all on temporal and numeric
items (4 of 15 right for 9B v1, 2 of 15 for 9B v2). The chat checkpoint did not close the gap; it made the 9B much
better calibrated where it is wrong. These are self-run public numbers. The board's real ranking also uses 308
sealed items that only the maintainer runs, where every top-ten row drops 48 to 57 points; nothing here is tuned
toward JevBench, but nobody can say how we would fare there until it is run.

**Records.** `results/jevbench/<label>/` (`public-metrics.json`, the harness `summary.json` and `manifest.json`,
and per-item `results.jsonl`). The board rows are copied from the benchmark's own published results at v1.4.2 and
from the JPT authors' own reports.
