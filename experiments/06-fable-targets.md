# 06. New targets for the synthetic pool from the best teacher

**Question.** If label noise from the open teachers is what held v2 back, do the same synthetic states with Fable 5.1
labels train a better model?

**Setup.** `fable-targets/fable_targets.py` sent each of the 3,253 synthetic states (all of its questions in one
call) to Fable 5.1 and cached the returned distributions. `fable-targets/finalize_fable_targets.py` wrote the re-targeted
records, the per-family comparison `fable-targets/COMPARE.md`, and two pools on the same states as `data-dose-c20`:
`data-v3-c20f` (the synthetic choice targets swapped to Fable's) and `data-v3-c20f-n3k` (the same plus up to 3,000
Fable-labelled synthetic yes/no questions). Fable agreed with the v2 teachers' argmax on 78.8 percent of choice
questions and 86.7 percent of yes/no questions.

| run | headline | Jevals HelpSteer2 | Nimble 324 | Kev transfer | human yes/no pooled 6 | JevBench public |
|---|---:|---:|---:|---:|---:|---:|
| `4b-c20` (teacher-average labels) | 71.56 | 37.3 | 76.5 | 82.3 | 83.0 | |
| `4b-v3-c20f` (Fable labels) | 71.25 | 39.7 | 74.4 | 83.5 | 83.1 | 0.792 |
| `4b-v3-c20f-n3k` (+ Fable yes/no) | 69.86 | 34.0 | 75.0 | 82.3 | 82.7 | |

**What we learned.** Better labels did not move our headline (71.25 against 71.56, within run-to-run noise), but the
Fable-labelled 4B is the best 4B Base checkpoint we have on JevBench's public items (0.792 against 0.762 for 4B v1;
see 10). Synthetic yes/no still hurt, even from the best teacher: adding it cost 1.4 headline points and did not help
the human yes/no sets.

**Records.** `results/runs/{4b-v3-c20f,4b-v3-c20f-n3k}/`, `results/nonce/{4b-v3-c20f,4b-v3-c20f-n3k}/`,
`fable-targets/COMPARE.md`. The Fable answers file is not committed; the re-targeted pools' sha256 are in
`data/manifests/pools.json`.
