# 04. How much synthetic choice (the dose experiment)

**Question.** v2 mixed synthetic choice and synthetic yes/no. Does synthetic choice alone, at a moderate share,
keep the choice gains without the loss on human yes/no?

**Pools.** Built by `data/dose/make_dose_pool.py` from the v1 pool and the synthetic records (whole states, no
synthetic noul): `data-dose-c20` adds 3,984 synthetic choice questions (20.1 percent of 19,797), `data-dose-c33`
adds all 7,858 (33.2 percent of 23,671).

| run | headline | Jevals Banking77 | Nimble 324 | Kev transfer | human yes/no pooled 6 |
|---|---:|---:|---:|---:|---:|
| `4b-v1` | 70.34 | 68.3 | 71.9 | 82.2 | 83.2 |
| `4b-c20` | 71.56 | 70.7 | 76.5 | 82.3 | 83.0 |
| `4b-c33` | 70.04 | 68.7 | 74.1 | 79.5 | 82.3 |

**What we learned.** A fifth of synthetic choice was the best 4B Base result we had at that point (+1.2 headline) and
left the human yes/no sets flat. A third was too much. c20 became the base of every later Base-checkpoint pool.

**Records.** `results/runs/{4b-c20,4b-c33}/`, `results/nonce/{4b-c20,4b-c33}/`.
