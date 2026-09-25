# 08. The chat checkpoint (what became v2)

**Question.** Every Jebadiah before this trained on the Qwen3.5 Base checkpoints. The models leading the Decision
Index start from chat checkpoints. With everything else in the v1 recipe fixed (same `data-v1` pool, same
hyperparameters, same prompt contract, same temperature target), what does the chat checkpoint with thinking off do?

| run | base | headline | Jevals PubMedQA | Jevals Banking77 | Jevals HelpSteer2 | Nimble 324 | Kev transfer | Nimble 13 macro | human yes/no pooled 6 | DSj HelpSteer2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4b-v1` | Qwen3.5-4B-Base | 70.34 | 86.7 | 68.3 | 37.0 | 71.9 | 82.2 | 75.9 | 83.2 | 9.2 |
| `4b-chat-v1` | Qwen3.5-4B | **72.49** | 88.7 | 70.0 | 40.0 | 77.2 | 83.2 | 75.9 | 83.2 | 9.3 |
| `9b-v1` | Qwen3.5-9B-Base | 73.29 | 89.7 | 70.0 | 40.3 | 78.7 | 84.0 | 77.0 | 84.6 | 11.9 |
| `9b-chat-v1` | Qwen3.5-9B | **73.93** | 90.3 | 70.7 | 40.7 | 81.2 | 83.8 | 77.0 | 84.2 | 10.4 |

**What we learned.**

- The checkpoint was the biggest single lever we found. On the 4B it is worth 2.15 headline points with no new data,
  more than any data change we tried, and it removes 4B v1's Nimble 324 regression (77.2 against 71.9).
- On the 9B the gain is 0.64 points, most of it Nimble 324 (81.2 against 78.7). The other components move less than a
  point, Kev transfer and the human yes/no pool are slightly down, and the HelpSteer2 Decision Score is 10.4 against
  11.9. No record here carries a significance test, so we claim the 4B gain and call the 9B gain small.
- Training is also faster on the chat checkpoints: 67 against 99 minutes for the 4B, 90 against 116 for the 9B
  (A100 80 GB PCIe for the chat runs, H100 PCIe for v1, so this is not a like-for-like timing).
- On JevBench the 9B chat model answers exactly as many public items as 9B v1 (189 of 231) and is much better
  calibrated on the hard tier (ECE 0.052 against 0.127). The 4B chat model is one item behind 4B v1 (see 10).

These two runs were merged into the base weights and published as `frontier-infra/jebadiah-4b-v2` and
`frontier-infra/jebadiah-9b-v2`. A merge check on 260 fixed questions per model gave the same pick as the run's own
eval record on 257 (9B) and 259 (4B); every difference was a near tie.

**Records.** `results/runs/{4b-chat-v1,9b-chat-v1}/`, `results/jevbench/jebadiah-{4b,9b}-chat-v1-local/`. The nonce
pass on the two chat runs has not been run yet.
