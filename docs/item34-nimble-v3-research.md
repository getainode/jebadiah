# ITEM 34: what the public Nimble v3 evidence actually proves

Research only, 2026-10-08. No training, inference model calls, GPU jobs, or paid data generation were launched. Web retrieval used TinyFish. No Nimble weights, predictions, or training rows were downloaded or incorporated into Jeb.

**The public record verifies a larger adapter and a different inference contract, but does not reveal a reproducible v3 training recipe or identify the cause of the gain.** Nimble v2 to v3 improves the edition 0.3 public score from **39.90 to 57.19**, a **17.29-point** gain. Relative to the task-supplied Jeb 9B v2 public **44.09**, the lead is **13.10 points**. Both Nimble contracts pin the same chat base as Jeb v2, `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The board incorrectly labels Nimble v2 as `Qwen3.5-9B-Base`; its published adapter and schema name the chat checkpoint. This is a configuration fact, not an independent audit of training execution.

## 1. Verified recipe and inference facts

| Fact | What is public | Evidence |
|---|---|---|
| v3 adapter capacity | LoRA rank **64**, alpha **128**, dropout **0.0**, bias none, no saved auxiliary modules. Standard alpha/rank ratio 2; DoRA, RSLoRA and QALoRA disabled. | [v3 adapter config](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3/blob/8e927b9b4afdbb14479fac10a7364d1a695be208/adapter_config.json) |
| v2 capacity and base | Rank **16**, alpha **32**, dropout **0.05**; chat base and the same pinned revision above. Thus capacity quadruples while alpha/rank stays 2. | [v2 adapter](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/blob/4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe/adapter_config.json), [v2 schema](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/blob/4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe/schema_config.json) |
| Target modules | Both name `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`. This covers attention and MLP projections, including linear attention. It is not a newly added classification head. | Same adapter configs above |
| v3 base and codebook | Schema task `schema_candidate_classification_v2`, same pinned chat base, **255** distinct ordinary single-token uppercase codes. A through Z, then valid vocabulary codes such as AA, AB; codes and token IDs are saved and checked at the answer boundary. Both prompt-source hashes are pinned. | [v3 schema](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3/blob/8e927b9b4afdbb14479fac10a7364d1a695be208/schema_config.json), [extended schema](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/code/nimble/evaluation/extended_schema.py) |
| Prompt | System instruction selects the best fitting code using facts in context, treats context as data, and forbids reasoning/explanation. User content is JSON with `context` and the **whole request schema**, followed by `Requested field: <JSON field name>`. Each choice carries code, value and optional description. Qwen chat template, generation prompt enabled, thinking disabled. For more than 26 choices, wording changes from one-letter to short codes. | [parallel schema](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/code/nimble/evaluation/parallel_schema.py), extended schema above |
| Probability read | One forward read at the final answer position. Gather only allowed candidate token logits, mask padding, cast selected logits to float64 and softmax across candidates at **T=1**, with argmax prediction. No sampling, reasoning generation, Jev calls, or learned calibration head in this engine. Boolean order is false then true; `noul` returns the true probability. Score questions become enums of rubric indices and the engine returns their choice distribution. | [run engine](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/code/nimble/evaluation/decision_index_engine.py) |
| Context and execution | Engine default and saved run **32,768 tokens**, no truncation; over-limit prompts refused. Shared-prefix cache enabled for multi-field requests with prefix at least 256 tokens; otherwise padded full-prompt batches. BF16 base/autocast, SDPA; full-prompt OOM batches split recursively. **This is an inference limit, not evidence of a 32k training length.** | Run engine above, [environment](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/runs/bespoke-nimble-9b-v3/environment.json) |
| Published run | Card/run report **56.88 on edition 0.2.1**, kit commit `87d4650`. Environment says 16 processes on 8 H100 80GB GPUs, two processes per GPU, hash-sharded requests and merged results. Torch 2.8.0+cu128; README requires transformers 5.17.0, PEFT 0.21.0, FLA 0.5.2, causal-conv1d 1.7.0. Adapter SHA256 `070d80ac...6bc4`. Status reports 150,317 completed, 150,305 ok and 12 unsupported. These counts belong to that run, not the current 0.3 public panel. | [run README](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/README.md), environment above, [status](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index/blob/654b124d3508a7da0d849405744bde80c4bb1a1b/runs/bespoke-nimble-9b-v3/status.json) |
| Published v2 training contract | 4,926 rows, 2,418 groups, 160 families, 324 validation rows; type counts 2,280 choice / 1,312 noul / 1,334 score. Components: original 2,826, contrastive addition 1,000, longform pilot 100, decision-skills-local 1,000. LR **5e-5**, microbatch 2, accumulation 4, linear schedule, warmup 185, max steps 1,848 but stop after **one epoch**, max length 8,192; reconstructed maximum train prompt 4,448. Seed 17, weight decay 0. The step cap is not proof that all 1,848 updates ran. | v2 schema above |
| Earlier open recipe, not established v3 recipe | Original release: **2,676 training / 324 held-out examples**, hard-label candidate cross-entropy, rank 16, LR 5e-5, effective batch 8, seed 17, one epoch, BF16, 2,048-token budget; retained three-epoch linear schedule stopped after the selected epoch. README says it did **not distill from Jev**. The separate current training guide describes a 2,826-row retained/local set, not the published v3 corpus. | [Nimble README, methodology](https://github.com/bespokelabsai/nimble#methodology), [training guide](https://github.com/bespokelabsai/nimble/blob/main/docs/NIMBLE_TRAINING.md) |
| Contrastive curation | Two evidence sentences jointly determine a fact; edit at most eight words in one sentence to flip one relevant fact and the label. Separate checks verify facts, consistency and no answer hints; deleting either sentence must make the fact unknown. Python derives labels from checked rules; keep only fully passing pairs with different labels. Keep pair and source family in one split, cache requests for offline replay. Deletion cases are checks, not false-labeled training rows. Original labels are synthetic, without human review. | [README curation](https://github.com/bespokelabsai/nimble#contrastive-data-curation), [curation guide](https://github.com/bespokelabsai/nimble/blob/main/docs/TRAINING_EVAL_CURATION.md) |
| Teacher/generator details | Current curation docs specify Terra low effort for new training drafts, Terra medium for separate checks, Sol for fresh evaluation, reusing previously audited Sol rules. Luna/Sonnet alternatives and content-addressed caches are documented. These are generic/historical pipeline settings; **none is tied publicly to v3's actual training run**. Curator is Bespoke's bulk inference/data tooling, not proof that a particular v3 dataset was produced with it. | Curation guide above, [Curator](https://github.com/bespokelabsai/curator) |
| Calibration | Original release fit T=2.179 on 300 examples and checked on a separate 300, minimizing log loss; keep if held-out log loss fell without worse Brier. Repository transfers 2.179078721266035 to v2, explicitly not an independent v2 fit. v3 evaluation engine applies no temperature fit. | [README temperature section](https://github.com/bespokelabsai/nimble#probability-temperature), run engine above |

**Not disclosed for v3:** data identities, row/token counts, synthetic/human mix, loss, LR/schedule, batch size, epochs/steps, training length, teacher models, data ablations, benchmark-overlap audit, or training logs. The model file listing contains only card, adapter config/weights and schema config. Do not transfer v2 or original-release settings onto v3. Capacity, dropout, context and codebook differences are observable together; none has a public isolated ablation.

Cross-checks: Bespoke's [launch post](https://www.linkedin.com/posts/smaheswaran_introducing-bespoke-nimble-an-open-data-activity-7506767544402350081-znfQ) describes roughly 90% on its own narrow evaluation and explicitly cautions about other benchmarks. The [Decision Index news page](https://multimodalart-jev-decision-index.static.hf.space/news.html) says open scorers still softmax candidate logits and do not expose Jev's RLCD algorithm. TinyFish extracted its maintainer caveats but not the JS `ITEMS` news grid, even via raw/resolve URLs; individual feed entries were not independently inspected. The [maintainer methodology](https://multimodalart-jev-decision-index.static.hf.space/data/methodology.json) documents coverage/exclusions; [board notes](https://multimodalart-jev-decision-index.static.hf.space/data/index.json) say 0.3 rebuilds GSM8K, retires ForecastBench, moves WinoGrande to Knowledge, and excludes SGD/RouterBench from the index. The old v3 run's GSM8K 82.37 raw is therefore not its current 0.3 GSM8K 58.98.

[BenchLM v3](https://benchlm.ai/models/decision-index-0-3-bespoke-nimble-9b-v3) shows **54.7 full**, Knowledge 35.2, Language 56.3, Retrieval 53.6, Tools 76.8 and Arts 46.3: these are combined public/private values, not the **57.19 public** or public-only area scores below. BenchLM lists six display-only external rows and no site-wide overall rank; its page supplies no additional training recipe. [BenchLM v2](https://benchlm.ai/models/decision-index-0-3-bespoke-nimble-9b-v2) is likewise secondary evidence. The unrelated [PACT paper](https://arxiv.org/html/2609.35865v1) studies the earlier open recipe; it is not a Bespoke v3 training disclosure and does not establish a raw-accuracy gain over that recipe.

## 2. Where the lead is

Sources: [v03.json](https://multimodalart-jev-decision-index.static.hf.space/data/v03.json) for overall public/full scores and [index.json](https://multimodalart-jev-decision-index.static.hf.space/data/index.json) for public benchmark `skill` and area `skill`. Both fetched snapshots say `generated_utc=2026-10-07T18:25:58Z`. Jeb 9B v2 is absent from both retrieved bundles, including a cache-busted fetch, so its **44.09 public is task-supplied, not independently recovered here**. Its benchmark/area columns come from ITEM 31's frozen 10% proxy, 11,079 rows, baseline checkpoint `db21f2af4fa564afdc9dbc67255d5eb4ea35919d`; [proxy method](../eval/decision-index/proxy-0.3-10pct/README.md), [source report](https://huggingface.co/datasets/jbrashear/jebadiah-9b-v2-1-index-results/tree/main/analysis/item31-proxy-0.3-10pct). Proxy overall **44.67** is not public **44.09**. Nimble and Jeb 27B columns use the full public panel. These are descriptive, unpaired gaps, not significance estimates; home has only nine proxy cases and GPQA twenty.

| Model | 0.3 public | 0.3 full with private parts |
|---|---:|---:|
| Nimble v3 | 57.19 | 54.67 |
| Nimble v2 | 39.90 | 40.95 |
| Jeb 9B v2 | 44.09, task-supplied | unavailable here |
| Jeb 27B | 55.11 | 55.63 |

Jeb 27B uses a different base, `Qwen/Qwen3.8-27B`, so it is a scale comparator. Nimble v3 beats it by **2.08 public** but trails it by **0.96 full**. Board calibration ECE is **12.29 points** for Nimble v3 versus **2.40** for v2 and **1.40** for Jeb 27B. v3's mean confidence is 87.26% with 74.97% correctness in the calibration sample. This is stronger discrimination with overconfidence, not evidence that better calibration drove the gain. Positive scalar temperature preserves candidate argmax and the binary 0.5 boundary, though it can change expected-score ranking.

Public chance-corrected area scores, out of 100:

| Area | Jeb 9B proxy | Nimble v3 | Gap vs proxy | Nimble v2 | Jeb 27B |
|---|---:|---:|---:|---:|---:|
| Knowledge & Reasoning | 30.87 | 40.96 | +10.09 | 30.48 | 40.78 |
| Language Understanding | 46.61 | 60.68 | +14.07 | 44.31 | 59.66 |
| Retrieval & Classification | 45.10 | 55.48 | +10.38 | 34.93 | 53.89 |
| Tools & Automation | 67.99 | 84.24 | +16.25 | 59.21 | 78.13 |
| Arts & Human Taste | 31.90 | 44.05 | +12.15 | 27.56 | 40.73 |

Ranked by Nimble v3 minus Jeb 9B v2 **chance-corrected benchmark skill**, all out of 100. The numerical [companion CSV](../results/research/item34-public-gaps.csv) also includes native raw scores and the two other full-panel gaps. Benchmarks have different metric/chance baselines and area weights, so a skill-point gap is not a whole-index-point contribution.

| ID | Benchmark | Jeb 9B proxy | Nimble v3 | Gap | Nimble v2 | Jeb 27B |
|---|---|---:|---:|---:|---:|---:|
| 9 | Home appliance simulator | 33.33 | 98.86 | +65.53 | 2.27 | 72.73 |
| 22 | POP909-CL | 8.55 | 73.90 | +65.35 | 12.95 | 21.04 |
| 61 | HoVer claim verification | 31.68 | 69.76 | +38.08 | 25.90 | 40.06 |
| 44 | CLadder | 21.60 | 50.04 | +28.44 | 33.84 | 45.44 |
| 12 | ANLI | 29.25 | 55.47 | +26.22 | 52.35 | 57.28 |
| 41 | VAST | 36.06 | 59.68 | +23.62 | 8.70 | 45.77 |
| 40 | iSarcasmEval | 17.09 | 40.17 | +23.08 | 40.47 | 44.44 |
| 58 | BBH fixed-option tasks | 55.62 | 77.52 | +21.90 | 52.80 | 67.68 |
| 5 | CLINC150+OOS | 66.42 | 88.24 | +21.82 | 30.42 | 88.21 |
| 59 | RAGTruth response-level hallucination | 30.11 | 50.74 | +20.63 | 27.58 | 54.84 |
| 30 | GSM8K | 31.48 | 50.76 | +19.28 | 32.25 | 40.70 |
| 28 | WinoGrande | 57.48 | 72.84 | +15.36 | 51.22 | 68.74 |
| 37 | Amazon ESCI | 32.68 | 47.65 | +14.97 | 31.68 | 42.77 |
| 4 | BANKING77 | 68.65 | 83.44 | +14.79 | 68.99 | 78.44 |
| 62 | When2Call MCQ | 67.67 | 82.40 | +14.73 | 63.56 | 71.77 |
| 43 | CRUXEval | 29.51 | 44.05 | +14.54 | 30.69 | 56.86 |
| 38 | ACOS | 23.51 | 37.09 | +13.58 | 20.68 | 18.22 |
| 57 | MMLU-Pro | 45.02 | 55.07 | +10.05 | 45.70 | 58.06 |
| 25 | GPQA Diamond | 20.00 | 28.57 | +8.57 | 25.17 | 30.61 |
| 23 | cfcolor | 18.61 | 26.41 | +7.80 | 6.77 | 30.09 |
| 42 | NLI4CT | 58.02 | 65.78 | +7.76 | 53.95 | 67.24 |
| 1 | BFCL | 89.69 | 96.57 | +6.88 | 88.60 | 96.73 |
| 39 | FinEntity | 79.23 | 85.84 | +6.61 | 49.21 | 84.16 |
| 64 | New Yorker caption matching | 45.75 | 51.94 | +6.19 | 44.60 | 65.44 |
| 2 | ToolRet | 57.69 | 63.30 | +5.61 | 51.44 | 60.81 |
| 20 | BPoMP | 75.99 | 81.42 | +5.43 | 72.48 | 87.79 |
| 29 | HellaSwag | 81.69 | 85.01 | +3.32 | 79.99 | 91.43 |
| 11 | ContractNLI | 60.97 | 62.52 | +1.55 | 57.08 | 67.64 |
| 31 | ChessBench | 5.21 | 6.75 | +1.54 | 5.92 | 7.38 |
| 33 | SATA-Bench | 23.23 | 24.65 | +1.42 | 22.50 | 29.37 |
| 45 | HLE | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 |
| 21 | Humicroedit | 25.48 | 23.29 | -2.19 | 15.07 | 23.52 |
| 3 | API-Bank | 84.01 | 78.73 | -5.28 | 80.13 | 83.75 |
| 50 | Habermas Machine | 17.02 | 7.34 | -9.68 | 13.49 | 16.52 |
| 56 | PhishNChips phishing decisions | 18.00 | 7.90 | -10.10 | 31.20 | 21.40 |
| 32 | MuSR | 45.61 | 34.47 | -11.14 | 30.65 | 39.54 |
| 36 | BRIGHT | 44.35 | 29.02 | -15.33 | 18.71 | 42.88 |

The v2-to-v3 gains are biggest on home **+96.59**, POP909 **+60.95**, CLINC **+57.82**, VAST **+50.98**, HoVer **+43.86** and FinEntity **+36.63** skill points. Regressions include phishing **-23.30**, Habermas **-6.15**, API-Bank **-1.40** and English sarcasm **-0.30**. Thus it is not uniformly better.

Against Jeb 27B, v3's biggest leads are POP909 **+52.86**, HoVer **+29.70**, home **+26.13**, ACOS **+18.87**, VAST **+13.91**, When2Call **+10.63**, GSM8K **+10.06** and BBH **+9.84**. It loses BRIGHT **-13.86**, captions/phishing each **-13.50**, and CRUXEval **-12.81**. POP909's striking gain merits an independent licensed music/rule probe and training-overlap evidence before treating it as general ability; no public evidence here establishes contamination or rules it out.

## 3. Five proposed single-change ladder rungs

The proposals are in [run-rules.md](run-rules.md#item-34-proposed-ladder). First establish A1's compliant floor, then compare each rung to the current accepted compliant recipe with the same source exclusions, prompt bytes, seed, question count, training steps and frozen proxy. Preserve the 95% paired whole-group bootstrap keep rule and an absolute proxy score above 44.67. Combine only independently kept rungs. These are hypotheses, not claims to reproduce Nimble, and no rung was launched.

1. **Contrastive pairs:** replace 10% of training slots with independently authored, certificate-checked minimal factual flips from our permitted training families, matching the old domain/type histogram. Tests evidence sensitivity for home, HoVer, causal/NLI tasks without adding reasoning targets. Keep only on the proxy rule plus improvement on a separate owned contrast-pair holdout; otherwise drop.
2. **Wide-choice exposure:** replace 10% of choice slots with our own deterministic 32/64/128/255-option taxonomy routing tasks, including hard distractors and no-match. Tests full-codebook competition and OOS discrimination suggested by CLINC/BANKING gains. Keep on the proxy rule plus a positive paired retrieval-area interval; otherwise drop.
3. **Rank 64:** change rank 16 to 64 while preserving alpha/rank=2, thus alpha 32 to 128; hold dropout, modules, data, length and optimizer fixed. Tests capacity, the strongest directly verified training-side difference. The prior v2.1 failure changed multiple factors and does not isolate rank. Keep on the proxy rule; otherwise drop.
4. **Dropout zero:** change only LoRA dropout 0.05 to 0.0. Tests whether deterministic adapter learning helps small curated data; monitor train/held-out divergence. Keep on the proxy rule; otherwise drop.
5. **LR 5e-5:** halve Jeb v2's 1e-4 peak LR; retain its cosine schedule, warmup, epochs and everything else. Tests conservative updates; 5e-5 is verified for Nimble v2/original, **not v3**. Keep on the proxy rule; otherwise drop.

Cost basis: [ITEM 27](item27-training-speed.md) measured roughly **4.18 questions/s** sustained including uploads on RTX PRO 6000. A fixed pilot of at most 15,000 question presentations needs roughly **1.0 GPU hour** of optimizer work; ITEM 31's 11,079-row proxy took **18.7 minutes** (~$0.86) including job setup. Budget setup/merge/uploads and slower wide-option prompts explicitly. Estimated training plus proxy: contrasts, dropout and LR **1.5 to 3 h, $4.13 to $8.25 each**; rank **2 to 3 h, $5.50 to $8.25**; wide-choice **2 to 4 h, $5.50 to $11.00** at **$2.75/h**. These are estimates, not measured pilot costs; no 7.25 questions/s speedup is assumed. Set job timeout caps to the upper bound, inspect throughput early, and never expand to 175k or 27B unless rule 5 is met. Data work is local deterministic authoring and review; optional paid teacher generation has a separate unknown cost and is not included or authorized.

Calibration can be fitted on a separate licensed non-benchmark holdout from saved Jeb logits on CPU, but is not a proposed accuracy rung: scalar T cannot fix most discrete-metric gaps. Do not adopt Nimble's schema renderer without a separate train/serve contract migration; Jeb uses AINode's pinned prompt and already reads the full candidate codebook locally.

**License boundary:** [v3 model](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) and [run repository](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index) are **CC BY-NC 4.0**; Qwen base is separately Apache 2.0. Learn from public facts and independently implement experiments. Never train on Nimble outputs or weights, use its adapters as initialization, copy its restricted training examples, or import its run code into commercial Jeb. No synthetic generator may be prompted with its predictions as targets. All training examples retain Jeb's source-first contamination and license manifest controls.

Validation: checked the pinned configs against the retrieved main files, joined all 37 proxy benchmarks by numeric ID, verified every gap arithmetically, and checked public/full separation against both board bundles. Numerical CSV contains aggregate metrics only. No new evaluation was run; no causality, significance, v3 training provenance, or feed-grid inspection is claimed. Lead review/merge and any separately approved experiments remain.
