# ITEM 48: owned evidence replacement against A3

**Drop rung 1.** The frozen proxy scores **46.09 vs A3 46.97**, difference **-0.88**, paired whole-group bootstrap 95% interval **[-2.21, +0.38]**. It exceeds the v2 floor of 44.67 and improves the independent evidence diagnostic from **54.67% to 99.67%**, but the interval is not entirely positive. The local diagnostic win cannot override the predeclared overall rule. A3 remains the accepted recipe.

The one change replaces 1,000 matched language slots, 500 choice and 500 Boolean, with the scanned item47 owned evidence source. Total exposure remains 21,190 question presentations. Private dataset `frontier-infra/jebadiah-data-v2-1-item47` is pinned at `7b6b15f95cc9902e1ebd783656a3bd590350875d`, folder `rung1/`. The Studio scan covers all 6,300 source records with zero direct or remaining hits. Original A3 calibration bytes and its 512 questions are unchanged. The 300 diagnostic questions never train the model or fit temperatures.

The parent is `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Rank 16, alpha 32, dropout 0.05, length 2,048, LR 1e-4, one epoch, seed 17, effective batch 8, ordinal targets, item27 speed flags, and autocast off match A3. The shared trainer, renderer, objective, and verifier are unchanged. Temperatures are copied byte for byte from `frontier-infra/jebadiah-9b-v2-1-a3@9e69926a007dd636e82d33485dcd48f9751c4248`: choice 1.0467, Boolean 1.0284, score 0.6805.

The pre-run line in [the run log](run-rules.md) was committed at `345e80213fe813f9cb388a44946e4ee959d3e1b7` before launch. Keep required a wholly positive overall interval, score above 44.67, and improved held-out evidence macro accuracy. The initial CUDA merge gate passed unchanged: max probability shift **0.010478854**, **259/260 identical picks**, and **zero confident flips**. No export recovery or extra optimizer steps were needed. The final checkpoint is **2,649 steps, epoch 1.0**.

## Frozen proxy

All **11,079 requests succeeded**, with zero whole-case failures and no rows outside the frozen manifest. Both result files match the frozen run IDs and payload hashes. The unchanged official kit recomputes scores for each of 2,000 whole-group resamples, seed 20261008, stratified by benchmark/domain/track. The interval describes variation conditional on this proxy, not full-suite or training-seed uncertainty.

All five areas below use index skill points. Language improves, while the other four areas regress; Tools has a wholly negative interval.

| Area | A3 | Rung 1 | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| Knowledge | 31.64 | 30.57 | -1.07 | [-3.73, +1.85] |
| Language | 50.32 | 52.77 | +2.45 | [+0.02, +4.91] |
| Retrieval | 50.29 | 48.66 | -1.62 | [-3.34, +0.22] |
| Tools | 67.73 | 63.48 | -4.24 | [-9.42, -0.94] |
| Arts | 33.40 | 32.01 | -1.39 | [-3.76, +0.91] |

The six requested benchmarks use their named raw metric, expressed as percent; differences are percentage points. HoVer/RAGTruth gains are small, while CLINC falls substantially. These individual observations do not replace the overall gate.

| Benchmark | Metric | Requests | A3 | Rung 1 | Difference |
|---|---|---:|---:|---:|---:|
| HoVer claim verification | accuracy | 401 | 67.58 | 68.33 | +0.75 |
| RAGTruth response-level hallucination | F1 on hallucinated class | 270 | 67.03 | 67.76 | +0.73 |
| CLadder | accuracy | 500 | 65.80 | 64.00 | -1.80 |
| ANLI | macro-F1 | 320 | 60.74 | 61.92 | +1.18 |
| CLINC150+OOS | macro-F1 | 550 | 76.82 | 61.53 | -15.28 |
| POP909-CL | accuracy | 200 | 9.50 | 12.00 | +2.50 |

## Independent owned diagnostic

Both models score the same 300 held-out questions with full menus through `eval_jebadiah.run_set`, batch 8, one repeat, and unchanged A3 temperatures. All prompts fit without truncation. Macro accuracy across the five held-out rule families rises **45.00 percentage points**, from 164/300 to 299/300 correct. The diagnostic holds out executable graph families, with shared vocabulary and finite rendering grammar; the result demonstrates transfer within that controlled source, while the overall proxy does not improve.

| Metric | A3 | Rung 1 |
|---|---:|---:|
| Macro rule accuracy | 54.67% | 99.67% |
| Three-way accuracy, n=150 | 41.33% | 100.00% |
| Boolean accuracy, n=150 | 68.00% | 99.33% |
| Boolean support precision | 75.00% | 100.00% |
| Boolean support recall | 6.00% | 98.00% |

Boolean support counts are TP/FP/FN **3/1/47** for A3 and **49/0/1** for rung 1.

| Rule / documents / polarity | Questions | A3 accuracy | Rung 1 accuracy |
|---|---:|---:|---:|
| rule:reverse2 | 60 | 58.33% | 100.00% |
| rule:reverse3 | 60 | 55.00% | 98.33% |
| rule:diamond4 | 60 | 55.00% | 100.00% |
| rule:cycle3 | 60 | 56.67% | 100.00% |
| rule:crosslink4 | 60 | 48.33% | 100.00% |
| documents:2 | 60 | 58.33% | 100.00% |
| documents:3 | 120 | 55.83% | 99.17% |
| documents:4 | 120 | 51.67% | 100.00% |
| polarity:False | 150 | 56.00% | 99.33% |
| polarity:True | 150 | 53.33% | 100.00% |

Complete groups require all three supported/refuted/insufficient variants correct, or all six answers for both types together. Pair correctness requires both named variants correct. There are 50 held-out worlds; the grouped checks preserve their related questions.

| Complete group | A3 correct / 50 | Rung 1 correct / 50 |
|---|---:|---:|
| choice | 1 | 50 |
| noul | 2 | 49 |
| both_types | 1 | 49 |

| Variant pair | A3 correct / 50 | Rung 1 correct / 50 |
|---|---:|---:|
| choice:supported/refuted | 2 | 50 |
| choice:supported/insufficient | 5 | 50 |
| choice:refuted/insufficient | 9 | 50 |
| noul:supported/refuted | 2 | 49 |
| noul:supported/insufficient | 3 | 49 |
| noul:refuted/insufficient | 49 | 50 |

## Jobs, cost, and verification

All jobs used RTX PRO 6000 at the planning rate of $2.75/hour. The initial allocation ceiling was $11.6875: 130 minutes training, 35 proxy, 45 paired diagnostic, 45 one export recovery. After training finished and recovery proved unnecessary, the diagnostic reporting retry fit within released allocation. Estimated total is **$4.24722**, including the failed diagnostic job, below the **$12 cap**. This is a runtime-based estimate, not an invoice.

| Job | Stage / outcome | Running seconds | Estimated USD |
|---|---|---:|---:|
| [6ac96b33095c57808930b2a7](https://huggingface.co/jobs/jbrashear/6ac96b33095c57808930b2a7) | Training and initial export, completed | 3124 | 2.38639 |
| [6ac97782095c57808930b7a7](https://huggingface.co/jobs/jbrashear/6ac97782095c57808930b7a7) | Frozen proxy, completed | 1123 | 0.85785 |
| [6ac97783095c57808930b7a9](https://huggingface.co/jobs/jbrashear/6ac97783095c57808930b7a9) | First diagnostic, report serialization failed | 658 | 0.50264 |
| [6ac97a70095c57808930b8aa](https://huggingface.co/jobs/jbrashear/6ac97a70095c57808930b8aa) | Paired diagnostic retry, completed | 655 | 0.50035 |

The first diagnostic completed both model calls but failed to serialize the evaluator's internal tuple-keyed `q_of` lookup. The reporting wrapper now stores only timing metadata and persists each model's partial report. The retry uses the same questions, settings, scoring, temperatures, and immutable models. No training or candidate selection changed. The failure log remains in the private checkpoint repo. Successful diagnostic code is pinned at `0a620db3c20aee8c7ba40db34f96e35eabc6d405`.

Atlas checks passed: every input manifest hash and question count, unchanged calibration and area/type histogram, live checkpoint recipe and final epoch/step, 852 byte-identical prompt renders, A3 score reproduction, frozen proxy code hashes, complete proxy ID/payload validation, diagnostic oracle/group bookkeeping, full report JSON round-trip, compilation, and whitespace checks. No GitHub-hosted CI ran; this repository has no automatic PR checks.

Private model `frontier-infra/jebadiah-9b-v2-1-rung1` is pinned at `e21876cc0a5b7c38b601bc3abe820131a1c1d080`; private checkpoints are in `frontier-infra/jebadiah-9b-v2-1-rung1-checkpoints`. Raw inference evidence stays private. Aggregate receipts are in [results/runs/item48](../results/runs/item48). No model was made public or submitted. The rung is dropped as a recipe; private artifacts remain for audit. Existing A3 source-policy exceptions still block shipment.

Cleanup removed the complete 155 MiB `/data/orca/scratch/item48` directory: dataset, suite, kit, environment, proof copies, downloaded inference results, logs, and monitor scripts. No model weights were downloaded onto Atlas. HF Jobs scratch is ephemeral. Lead review and merge of the results PR remain; the worker does not merge.
