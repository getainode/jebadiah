# 05. Teacher bake-off: whose yes/no line is closest to people's

**Question.** The v2 labels came from two open teachers on our own hardware. On questions with a human answer key,
how far off are they, and would a frontier model or a panel label better?

**Setup.** `bakeoff/build_questions.py` draws 1,000 human-labelled yes/no items (250 each from CondaQA,
ContractNLI, ShARC and UNFAIR-ToS, balanced 125 true / 125 false, from the converter in `data/human-noul/`) plus 400
Measuring Hate Speech items (150 clear no, 150 clear yes, 100 close calls). Every teacher got the same 136 batches of
10 (`bakeoff/harness.py`, seed 20260924) and returned a calibrated P(true) per item. Frontier teachers ran through
their vendors' command-line tools on subscription logins, with every API key variable removed from the child
environment; the two open teachers ran through AINode's `/v1/decide`. `bakeoff/analyze.py` writes the tables in
`bakeoff/metrics.md`.

| teacher | accuracy on 1,000 human items | mean abs per-source gap in true rate (pp) | ECE |
|---|---:|---:|---:|
| Fable 5.1 | 89.1 | 4.7 | 0.042 |
| Opus 5.5 | 88.0 | 5.0 | 0.051 |
| GPT-6 astra | 88.5 | 5.1 | 0.080 |
| Grok 4.7 | 82.4 | 9.2 | 0.074 |
| Qwen3.8 27B (v2 teacher) | 80.0 | 13.4 | 0.091 |
| DeepSeek V4 Flash (v2 teacher) | 73.6 | 21.2 | 0.225 |
| v2 pool target (mean of the two) | 76.2 | 18.0 | |

**What we learned.**

- Any of the three strongest frontier teachers beats the v2 pool target by 12 to 13 points on the human key. The
  pool looked balanced overall only because its per-source errors cancel (+34.8 pp over-flagging on UNFAIR-ToS,
  -23.2 pp on CondaQA).
- A panel of the best three is not more accurate than Fable alone (88.6 against 89.1); it is a little better
  calibrated.
- On the 60 close-call hate-speech items every teacher over-flags by 18 to 35 points and none beats a coin on which
  side of 0.5 the annotators landed. That part needs people, not a better teacher.
- The first pass through the vendors' APIs cost $10.91 in total; the harness pass used subscriptions and cost nothing.

**Records.** `bakeoff/metrics.md`. The per-item answer files are not committed (they are teacher outputs on
third-party text); `analyze.py` rebuilds every table from them.
