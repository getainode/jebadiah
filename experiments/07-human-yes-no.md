# 07. Human yes/no data instead of synthetic yes/no

**Question.** Synthetic yes/no kept hurting the six human-labelled yes/no eval sets. Does public human-labelled
yes/no data help instead, and does a small slice of our own human labels add anything?

**Data.** `data/human-noul/convert_human_noul.py` converts four licensed public sources into 1,000 balanced records
each: CondaQA (Apache-2.0), ContractNLI (CC BY 4.0), ShARC (license stated only as a CC BY-SA 3.0 tag, so no run we report trains on it) and LexGLUE
UNFAIR-ToS (CC BY 4.0), with an
overlap check against every reported test set. `human-noul-alt.jsonl` (CondaQA, ContractNLI, UNFAIR-ToS) is the one
the v3 pools use. `data/human/prolific-r1.jsonl` holds 54 yes/no questions on synthetic states labelled by
Prolific readers in round 1 (target = the mean of 2 or 3 raters, only families where readers agreed at least 70
percent of the time; no rater identifiers). `data/compose_pools.py` builds `data-v3-c20fh-alt`, `data-v3-full` and
`data-v3-full-s` byte for byte.

| run | pool | headline | Jevals PubMedQA | Jevals HelpSteer2 | Nimble 324 | human yes/no pooled 6 | DSj HelpSteer2 |
|---|---|---:|---:|---:|---:|---:|---:|
| `4b-v3-c20f` | Fable c20 | 71.25 | 86.3 | 39.7 | 74.4 | 83.1 | 2.7 |
| `4b-v3-c20fh-alt` | + 3,000 public human yes/no | 71.06 | 85.7 | 38.3 | 74.7 | 82.8 | 3.5 |
| `4b-v3-full` | + 54 Prolific | 71.14 | 87.0 | 35.3 | 77.8 | 83.7 | -6.2 |
| `9b-v1` | data-v1 | 73.29 | 89.7 | 40.3 | 78.7 | 84.6 | 11.9 |
| `9b-v3` | data-v3-full | 72.61 | 90.0 | 34.0 | 80.2 | 83.5 | -4.6 |
| `9b-v3s` | data-v3-full, score sets doubled | 71.83 | 88.0 | 38.3 | 75.3 | 83.3 | 4.7 |

**What we learned.** Human yes/no data did not lift the human yes/no eval sets either: the pooled number moves within
about half a point of 4B v1 (83.2) in every run. The full v3 recipe on the 9B gained on PubMedQA and Nimble 324 but
lost 6 points of HelpSteer2 accuracy and went below zero on its Decision Score, because the score share fell.
Doubling the score sets won back most of HelpSteer2 and gave up Nimble 324. None of these beat 9B v1 on the headline,
and none was published. On JevBench, 9B v3 matches 9B v1 on accuracy with much better hard-tier calibration (see 10).
`4b-v3-full` trained on a DGX Spark (GB10) rather than a rented box, which is why its training time is 577 minutes.

**Records.** `results/runs/{4b-v3-c20fh-alt,4b-v3-full,9b-v3,9b-v3s}/`, `results/nonce/{4b-v3-c20fh-alt,9b-v3}/`,
`data/human-noul/human-noul-manifest.json`, `data/human-noul/pools-manifest.json`.
