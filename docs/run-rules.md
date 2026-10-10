# Rules for every Jeb training run

Standing rule since 2026-10-08. Jeb has to get better, never worse.

Why: v2.1 run 1 changed three things at once (a source-level contamination scrub, a new synthetic data mix and
a new recipe: rank 64, 4,096 tokens, 7,500 steps). It scored 38.12 against v2's 44.67 on the frozen proxy, and
the run could not say which change hurt.

1. **The baseline to beat is 9B v2: 44.67** on the frozen 10% Decision Index proxy
   (`eval/decision-index/proxy-0.3-10pct/`, manifest SHA256 `74d81622...3405`).
   The 9B keep baseline is now the **two-seed A3 average: 46.82**
   (seeds 17 and 18: 46.97 and 46.66). New rungs must beat that average
   under rule 2, while v2 remains the comparison floor. Paired bootstrap draws
   compare the candidate score with the mean of both A3 scores on the same draws.
2. **One change per run**, measured against the current best recipe on that proxy, with the paired group
   bootstrap interval (`bootstrap.py`).
3. **Keep a change only if it beats the baseline beyond the noise** (the 95% interval of the difference is above
   zero). Otherwise drop it.
4. **Combine changes only after each has proven itself alone**, then measure the combination as its own run.
5. **No full 175k run and no 27B run** until a combined recipe beats v2 on the proxy.
6. **Before each run, write one line in the run log below**: what it tests, and which result means keep or drop.

Constraints are not experiments. The source-level contamination rule and the license manifest always apply to
anything we ship; a run that measures their cost is still a single-change run.

ITEM 52: same-seed training reproduces bit-identical served BF16 weights and
all proxy answers. Changing seed 17 to 18 moves the proxy -0.31, Tools -5.44,
and CLINC -2.46 macro-F1 points. Adopted review policy treats Tools deltas within
about 6 points as seed noise and CLINC losses beyond the observed 2.5-point
spread as data regressions. These are operational allowances from two controls,
not estimated variance or causal proof; HF-assigned driver/kernel differences
are recorded in the report.

## Run log

| Run | Change vs | Tests | Keep if | Result |
|---|---|---|---|---|
| v2 | published | baseline | | 44.67 |
| v2.1 r1 | v2 | three changes at once (breaks rule 2) | | 38.12, dropped |
| A1 | v2 | the scrub alone: v2 recipe on v2 data minus BoolQ, DBpedia14, MNLI, SummEval and 42 overlapping HelpSteer2 records | the cost of the required scrub; becomes the clean floor | 45.25; vs v2 +0.57, CI [-0.80, +1.88]; private diagnostic floor |
| A3 | A1 | new v2.1 sources, capped at 10% each and 50% total | interval vs A1 above zero | 46.97; vs A1 +1.73, CI [+0.32, +3.23]; vs v2 +2.30, CI [+1.04, +3.60]; accepted best recipe |
| ITEM 47 rung 1 data preparation | A3 (46.97) | replace 1,000 matched language choice/noul slots with owned 2 to 4 document proof joins and minimal fact flips; total 21,190 presentations and original 512-question calibration unchanged | frozen-proxy paired whole-group bootstrap 95% interval vs A3 entirely above zero, score above 44.67, and separate held-out rule-family diagnostic improves; otherwise drop | CPU-only data PR; 6,000 train candidates plus 300 diagnostic, render checks passed; full-suite Studio scan and final composition pending; no training launched |
| ITEM 49 rung 2 data preparation | A3 (46.97) | replace 1,000 matched knowledge choice/noul slots with unused original-train Corr2Cause graph families; aggregate source cap 2,119, total 21,190 presentations and original 512-question calibration unchanged | frozen-proxy paired whole-group bootstrap 95% interval vs A3 entirely above zero, score above 44.67, and separate graph-disjoint causal diagnostic improves; otherwise drop | Studio CPU-only data PR; 6,000 train candidates plus 300 diagnostic, 9,450 renders without truncation; full-release and converted-text Studio scans and final composition pending; no training launched |
| Rung 2b (item51) | A3 (46.97) | append the exact 1,000 scanned rung2 Corr2Cause questions to unchanged A3 train; 22,190 presentations and 2,774 optimizer steps, original calibration bytes and all recipe settings unchanged | frozen-proxy paired whole-group bootstrap 95% interval vs A3 entirely above zero and score above 44.67; inspect Tools, five areas, CLadder, CLINC and causal diagnostic vs A3 and rung2 | 46.68; vs A3 -0.30, CI [-1.78, +0.99]; vs rung2 +0.72, CI [-0.95, +2.35]; drop; Tools -4.53; CLadder 63.20%; CLINC 58.56%; estimated $3.9050 |
| A3 replicate 1 (ITEM 52 control) | A3 (46.97) | zero changes: retrain byte-identical a3/ at bb4ce7b with original runtime commit 86a8203, same parent, seed 17, recipe and flags; measure repeatability | control only, no candidate keep decision; report paired 95% interval vs A3 and recompare existing rungs against replicate; merge gate 0.05 unchanged | 46.97; delta 0.00, CI [0.00, 0.00]; all served BF16 shards, temperatures and 11,079 scored answers reproduce A3; Tools/CLINC unchanged; rungs vs replicate: 1 -0.88 [-2.21, +0.38], 2 -1.02 [-2.49, +0.28], 2b -0.30 [-1.78, +0.99]; estimated $3.1656; seed-18 extension pending |
| A3 seed 18 (ITEM 52 control) | A3 seed 17 (46.97) | seed 17 to 18 only; same byte-identical a3/ data, parent, original runtime and recipe; measure A3 seed spread after seed-17 exported weights reproduced exactly | control only, no candidate keep decision; report paired interval, Tools/CLINC and rung comparisons against the two-seed spread; merge gate 0.05 unchanged | 46.66; vs A3 -0.31 [-1.77, +1.06]; Tools -5.44 [-10.70, -1.85], CLINC -2.46 macro-F1 points; rungs vs seed18: 1 -0.57 [-1.56, +0.61], 2 -0.71 [-2.34, +0.90], 2b +0.01 [-1.22, +1.23]; Tools seed sensitivity observed, larger rung CLINC losses persist; driver/kernel differences recorded; estimated $3.1717 under separately approved $8; private control, no candidate promotion |
| N2 data preparation | A3 (46.97) | replace floor(10% of choice slots) with item43 owned 32/64/128/255-option taxonomy match/no-match tasks; all other recipe fields and calibration unchanged | overall and Retrieval paired 95% intervals vs A3 above zero, score above 44.67 | data-only PR; CPU render checks passed; full-suite Studio scan pending; no training launched |
| ITEM 53 rung 3 data preparation | A3 (46.97) | unused SpaceNLI spatial patterns; independently prepare 1000 matched language-choice replacements at 21190 presentations and 1000 additions at 22190; original 512 calibration unchanged | paired whole-group frozen-proxy 95% interval vs A3 wholly above zero, score above 44.67, separate pattern-disjoint diagnostic improves; inspect all five areas | CPU data preparation only; 1200 train candidates and 300 diagnostic; full-source Studio scan gates both final compositions; no training launched |

| N3 | A3 (46.97) | rank 16 to 64, alpha 32 to 128, fixed alpha/rank 2; identical data and recipe | frozen-proxy paired 95% interval vs A3 above zero | 42.29; vs A3 -4.68, CI [-6.25, -3.12]; vs v2 -2.38, CI [-4.19, -0.60]; drop; estimated $3.2351 |
| R3 | A3 (46.97) | add deterministically regenerated item33 4,050 skill train questions; 450 diagnostic holdout questions; preserve original calibration and recipe | frozen-proxy paired 95% interval vs A3 above zero | 46.91; vs A3 -0.06, CI [-1.26, +1.08]; vs v2 +2.24, CI [+0.74, +3.73]; drop; initial merge gate failed (0.08355), lead msg_c4892b870ed6 allowed private proxy; unchanged recovery gate passed (0.02915); estimated $3.9585 including failed stage and recovery |
| R3 export recovery | final R3 checkpoint step 3155 | unchanged fp32 merge then bf16 cast, zero additional optimizer steps; identify shifted question | preserve original 0.05 gate and record private-only exception | completed, zero additional optimizer steps; gate passed at max shift 0.02915, 259/260 picks, zero confident flips; $0.5943 included in R3 spend |

| G1 | A3 (46.97) | add only item33 grounding train questions; retain grounding diagnostic holdout and unchanged original calibration | frozen-proxy paired 95% interval vs A3 above zero | 46.70; vs A3 -0.28, CI [-1.65, +1.07]; vs v2 +2.02, CI [+0.64, +3.41]; drop |
| D0 | A3 (46.97) | LoRA dropout 0.05 to 0.0; identical data and other recipe fields | frozen-proxy paired 95% interval vs A3 above zero | 46.57; vs A3 -0.40, CI [-1.64, +0.80]; vs v2 +1.90, CI [+0.30, +3.55]; drop |
| L5 | A3 (46.97) | peak LR 1e-4 to 5e-5; cosine schedule and warmup unchanged | frozen-proxy paired 95% interval vs A3 above zero | 47.03; vs A3 +0.06, CI [-1.34, +1.24]; vs v2 +2.36, CI [+0.81, +3.88]; drop |

Item36 total estimated compute: **$7.1935**, including both proxies and the
failed R3 stage/export recovery. All five timeouts sum to **$19.9375** at
$2.75/hour, below the shared $20 cap. See [full item36 results](item36-results.md)
and [verification repeatability follow-up](item36-merge-followup.md). Models
stay private and are never submitted; A3 remains the best recipe.

| N2 (item44) | A3 (46.97) | replace 1,170 of 11,708 choice slots with scanned owned wide-choice questions; unchanged A3 recipe, calibration and temperatures | overall and Retrieval frozen-proxy paired 95% intervals vs A3 above zero, score above 44.67 | 42.68, -4.29 [-5.72, -2.78] vs A3 (Retrieval -14.09, Tools -5.27): dropped. Merge gate 259/260, max shift 0.020. Proxy run by the lead after the worker pane died (2026-10-09). Private. |
| ITEM 39 27B A3 | published 27B, same chat parent revision | transfer accepted A3 mix and recipe to 27B; batch 1 with accumulation 8 preserves effective batch 8 for 96 GB fit | frozen-proxy paired 95% difference interval vs published 27B above zero, unchanged 0.05 merge gate passes | 56.62 vs published 55.08; +1.54, CI [+0.38, +2.84]; retain private; merge shift 0.00847, 260/260 picks, zero confident flips; estimated $10.98625 |

| ITEM 48 rung 1 | A3 (46.97) | replace 1,000 matched language choice/noul slots with scanned owned evidence-world questions at item47 revision 7b6b15f95cc9902e1ebd783656a3bd590350875d; all other A3 recipe fields, calibration and temperatures unchanged | frozen-proxy paired whole-group bootstrap 95% interval vs A3 entirely above zero, score above 44.67, and independent held-out evidence diagnostic macro accuracy improves; unchanged 0.05 merge gate | 46.09 vs A3 46.97; -0.88, CI [-2.21, +0.38]: drop. Evidence macro 54.67% to 99.67%; Tools -4.24, CI [-9.42, -0.94]. Initial 0.05 merge gate passed, max shift 0.01048, 259/260 picks, zero confident flips; no recovery. Estimated $4.24722 including diagnostic report retry. Private; [full results](item48-rung1-results.md) |

| ITEM 50 rung 2 | A3 (46.97) | replace only 1,000 matched knowledge slots with item49 original-train Corr2Cause, pinned item47 data 2242e85fae6f8aff3fc3cea3fa95869d99132522; 21,190 presentations, unchanged calibration and A3 recipe | frozen-proxy paired whole-group bootstrap 95% interval vs A3 entirely above zero, score above 44.67, and graph-disjoint causal diagnostic balanced accuracy improves; otherwise drop | 45.95; vs A3 -1.02, CI [-2.49, +0.28]; drop; causal balanced accuracy 51.36% to 80.42%, ordinary accuracy 90.00% to 89.67%; initial 0.05 merge gate passed (0.01594), no export recovery; estimated $4.3565 including one diagnostic persistence retry; private only |

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

| ITEM 56 outside-data PROBE B, seeds 17 and 18 | Two-seed A3 mean 46.82 | Jason option B, Desk #646: untouched A3 plus exact scanned Corr2Cause, SpaceNLI, MASSIVE and DeepMind math additions; original A3 runtime and one epoch; exposure increases | Mean of probe seeds beats unrounded mean of A3 seeds on identical paired whole-group bootstrap draws with 95% interval entirely above zero, and mean CLINC macro-F1 loses at most 2.5 points; retain 0.05 merge gate | Prelaunch: pending scanned math revision; probe only, split into single rungs only if it wins; no model promotion |
