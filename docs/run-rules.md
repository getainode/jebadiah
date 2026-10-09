# Rules for every Jeb training run

Standing rule since 2026-10-08. Jeb has to get better, never worse.

Why: v2.1 run 1 changed three things at once (a source-level contamination scrub, a new synthetic data mix and
a new recipe: rank 64, 4,096 tokens, 7,500 steps). It scored 38.12 against v2's 44.67 on the frozen proxy, and
the run could not say which change hurt.

1. **The baseline to beat is 9B v2: 44.67** on the frozen 10% Decision Index proxy
   (`eval/decision-index/proxy-0.3-10pct/`, manifest SHA256 `74d81622...3405`).
   The current accepted private recipe is **A3: 46.97**; new rungs must beat A3
   under rule 2, while v2 remains the comparison floor.
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
| A1 | v2 | the scrub alone: v2 recipe on v2 data minus BoolQ, DBpedia14, MNLI, SummEval and 42 overlapping HelpSteer2 records | the cost of the required scrub; becomes the clean floor | 45.25; vs v2 +0.57, CI [-0.80, +1.88]; private diagnostic floor |
| A3 | A1 | new v2.1 sources, capped at 10% each and 50% total | interval vs A1 above zero | 46.97; vs A1 +1.73, CI [+0.32, +3.23]; vs v2 +2.30, CI [+1.04, +3.60]; accepted best recipe |
| N3 | A3 (46.97) | rank 16 to 64, alpha 32 to 128, fixed alpha/rank 2; identical data and recipe | frozen-proxy paired 95% interval vs A3 above zero | 42.29; vs A3 -4.68, CI [-6.25, -3.12]; vs v2 -2.38, CI [-4.19, -0.60]; drop; estimated $3.2351 |
| R3 | A3 (46.97) | add deterministically regenerated item33 4,050 skill train questions; 450 diagnostic holdout questions; preserve original calibration and recipe | frozen-proxy paired 95% interval vs A3 above zero | 46.91; vs A3 -0.06, CI [-1.26, +1.08]; vs v2 +2.24, CI [+0.74, +3.73]; drop; initial merge gate failed (0.08355), lead msg_c4892b870ed6 allowed private proxy; unchanged recovery gate passed (0.02915); estimated $3.9585 including failed stage and recovery |
| R3 export recovery | final R3 checkpoint step 3155 | unchanged fp32 merge then bf16 cast, zero additional optimizer steps; identify shifted question | preserve original 0.05 gate and record private-only exception | completed, zero additional optimizer steps; gate passed at max shift 0.02915, 259/260 picks, zero confident flips; $0.5943 included in R3 spend |

Item36 total estimated compute: **$7.1935**, including both proxies and the
failed R3 stage/export recovery. All five timeouts sum to **$19.9375** at
$2.75/hour, below the shared $20 cap. See [full item36 results](item36-results.md)
and [verification repeatability follow-up](item36-merge-followup.md). Models
stay private and are never submitted; A3 remains the best recipe.

## ITEM 34 proposed ladder

Research proposals only, not launched runs. [Evidence and ranked gaps](item34-nimble-v3-research.md)
come from public Nimble configs and Decision Index scores; v3's actual data and optimizer recipe
are undisclosed. Establish A1's compliant floor first. Test each rung independently against the
current accepted compliant recipe, keeping source exclusions, prompt, seed, question presentations,
training steps and all other settings fixed. A later combination is its own measured run.

For every rung, **keep only if the frozen-proxy paired 95% difference interval is above zero and
the candidate proxy score exceeds 44.67**; otherwise drop. Additional checks below also apply.
No full 175k or 27B run is authorized by this proposal.

| Rung | One change and what it tests | Additional keep/drop check | Estimated RTX PRO 6000 training + proxy cost |
|---|---|---|---|
| N1 | Replace 10% of training slots with independently authored, verified minimal factual-flip pairs, matching domain/type counts; tests evidence sensitivity and compositional decisions | Keep only with improved correctness on a separate owned pair holdout; otherwise drop | 1.5 to 3 h, $4.13 to $8.25 |
| N2 | Replace 10% of choice slots with our own deterministic 32/64/128/255-option taxonomy tasks with hard distractors/no-match; tests wide-choice and OOS competition | Keep only if the paired retrieval-area 95% interval is also above zero; otherwise drop | 2 to 4 h, $5.50 to $11.00 |
| N3 | Rank 16 to 64 at fixed alpha/rank=2 (alpha 32 to 128); tests adapter capacity | Standard proxy rule above | 2 to 3 h, $5.50 to $8.25 |
| N4 | LoRA dropout 0.05 to 0.0; tests deterministic adapter updates | Standard proxy rule above; record train/holdout divergence | 1.5 to 3 h, $4.13 to $8.25 |
| N5 | Peak LR 1e-4 to 5e-5, retaining cosine schedule/warmup/epochs; tests conservative updates | Standard proxy rule above | 1.5 to 3 h, $4.13 to $8.25 |

Costs assume a fixed pilot of at most 15,000 question presentations, about 4.18 questions/s from
[ITEM 27](item27-training-speed.md), plus setup/merge/uploads and the ITEM 31 proxy's 18.7-minute
evaluation. Rate: $2.75/h. These are estimates, not measured runtimes; cap jobs at the upper bound
and inspect throughput early. Data authoring/review is local; optional paid teacher generation is
outside these estimates and needs a separate decision. If an accepted recipe already has a rung's
setting, skip that rung rather than claim a new experiment.

Nimble v3 and its run code/results are CC BY-NC 4.0. Learn from public facts only: never train on
its outputs or weights, initialize from its adapter, copy its restricted data, or import its run code
into commercial Jeb. Independently implement experiments with our permitted data and the existing
source-first contamination and license manifests.
