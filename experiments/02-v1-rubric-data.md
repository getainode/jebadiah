# 02. v1: human rubric data and the ordinal score fix

**Question.** Does adding human-rated score data (HelpSteer2 train and SummEval, under the eval exclusions) on top
of the score fix make the score type honest on held-out human rubrics?

**Pool.** `data-v1`: 11,013 records, 15,813 questions (5,704 choice, 3,024 noul, 7,085 score, 44.8 percent score).
Built by `data/convert_data.py` as committed; sha256 in `data/manifests/pools.json`.

| run | headline | Jevals HelpSteer2 acc | DSj HelpSteer2 | Nimble 324 | human yes/no pooled 6 |
|---|---:|---:|---:|---:|---:|
| `9b-v0` | 72.53 | 40.3 | -21.4 | 76.9 | 82.8 |
| `9b-v1` (published as 9B v1) | 73.29 | 40.3 | 11.9 | 78.7 | 84.6 |
| `4b-v0` | 70.94 | 36.3 | -20.2 | 77.5 | 81.2 |
| `4b-v1` (published as 4B v1) | 70.34 | 37.0 | 9.2 | 71.9 | 83.2 |
| `4b-v1-lr2e-4` | 69.68 | 38.7 | 3.5 | 76.5 | 81.2 |

**What we learned.**

- The 9B went from a loss to the top of our score measurements: DSj on HelpSteer2 from -21.4 to 11.9, ECE there
  from about 0.39 to 0.045. Rubric accuracy itself did not move (40.3 both times). The model became honest about
  not knowing helpfulness, not better at it.
- The 4B could not absorb 45 percent score questions for free: Nimble 324 fell 5.6 points against 4B v0. Capacity
  is real at 4B.
- Learning rate 2e-4 lost to 1e-4 overall. 1e-4 stays.

**Records.** `results/runs/{9b-v1,4b-v1,4b-v1-lr2e-4}/`, `results/nonce/{9b-v1,4b-v1}/`.
