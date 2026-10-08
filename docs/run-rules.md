# Rules for every Jeb training run

Standing rule since 2026-10-08. Jeb has to get better, never worse.

Why: v2.1 run 1 changed three things at once (a source-level contamination scrub, a new synthetic data mix and
a new recipe: rank 64, 4,096 tokens, 7,500 steps). It scored 38.12 against v2's 44.67 on the frozen proxy, and
the run could not say which change hurt.

1. **The baseline to beat is 9B v2: 44.67** on the frozen 10% Decision Index proxy
   (`eval/decision-index/proxy-0.3-10pct/`, manifest SHA256 `74d81622...3405`).
2. **One change per run**, measured against the current best recipe on that proxy, with the paired group
   bootstrap interval (`bootstrap.py`).
3. **Keep a change only if it beats the baseline beyond the noise** (the 95% interval of the difference is above
   zero). Otherwise drop it.
4. **Combine changes only after each has proven itself alone**, then measure the combination as its own run.
5. **No full 175k run and no 27B run** until a combined recipe beats v2 on the proxy.
6. **Before each run, write one line in the run log below**: what it tests, and which result means keep or drop.

Constraints are not experiments. The source-level contamination rule and the license manifest always apply to
anything we ship; a run that measures their cost is still a single-change run.

## Run log

| Run | Change vs | Tests | Keep if | Result |
|---|---|---|---|---|
| v2 | published | baseline | | 44.67 |
| v2.1 r1 | v2 | three changes at once (breaks rule 2) | | 38.12, dropped |
| A1 | v2 | the scrub alone: v2 recipe on v2 data minus BoolQ, DBpedia14, MNLI, SummEval and 42 overlapping HelpSteer2 records | the cost of the required scrub; becomes the clean floor | pending |
| A3 | A1 | new v2.1 sources, capped at 10% each and 50% total | interval vs A1 above zero | pending |
