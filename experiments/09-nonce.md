# 09. Nonce robustness

**Question.** Is a decision stable when irrelevant noise enters the state? A good decision model should not change
its answer because a request id changed.

**Setup.** `eval/nonce_eval.py` plants a random UUID in the state (and, as a second variant, in the instructions),
three passes per question, on the four head sets (Jevals PubMedQA, Banking77, HelpSteer2 and Nimble 324), and
compares every pass with the clean read. `scripts/nonce_all.sh` runs it over every finished adapter and both
untrained bases.

| model | Banking77 pick agreement | Banking77 accuracy clean / with nonce | PubMedQA pick agreement |
|---|---:|---|---:|
| Qwen3.5-4B-Base, untrained | 0.873 | 0.543 / 0.487 | 0.978 |
| 4B v0 | 0.961 | 0.687 / 0.676 | 0.984 |
| 4B v1 | 0.964 | 0.683 / 0.684 | 0.982 |
| Qwen3.5-9B-Base, untrained | 0.936 | 0.697 / 0.670 | 0.979 |
| 9B v0 | 0.982 | 0.710 / 0.709 | 0.987 |
| 9B v1 | 0.966 | 0.700 / 0.688 | 0.978 |

**What we learned.** Fine-tuning made the model more stable, not less. On choice and yes/no the trained models keep
96 to 100 percent of their picks, and the untrained 4B base loses about 6 accuracy points on Banking77 under a nonce
where the trained 4B loses none. Score questions flip their top level more often (up to about 10 percent of
HelpSteer2 items for v1) while moving their probabilities little: a calibrated score model sits between adjacent
levels, so a consumer should read the expected level rather than the argmax. The full table for every run is in
`results/SUMMARY.md`.

**Records.** `results/nonce/<run>/<set>.json` (summary blocks; the per-question rows are not committed).
