# T2b local tranche

The new tranche has **256,965 clean training questions** from **156 tasks across 82 upstream datasets**, plus **3,000 heldout questions** from five entire source families. With byte-identical A3 training, the local composition has **278,155 training presentations** and the original **512 calibration questions**. No training or GPU work was performed.

**The A3 composition gate is blocked.** The selected new tranche and holdout have zero direct full-suite hits, but untouched A3 has five direct hits against the fully indexed frozen suite: four math-QA training records and one HelpSteer2 calibration record. Whole-family removal would affect 102 math-QA and 2,531 HelpSteer2 questions across the original splits. Original A3 remains unchanged. A master decision on the conflicting zero-hit and unchanged-A3 requirements is needed before the proposed screen.

Publication is pending clarification of `LOCAL ONLY / no posting anywhere` versus the requested private upload and PR. Intended dataset: `frontier-infra/jebadiah-data-t2b`, private; published revision: **none**. The local branch is `fable/t2b-tranche`; no remote PR has been created. All downloads, caches, private suite payloads, indices and outputs are on `/Volumes/PRO-G40`.

## Inputs and scope

Both required collections were **not in A3 as collections**. Some of their underlying tasks were already in A3; deduplication checks the pinned A3 train and calibration states.

| Required collection | Pinned revision | Treatment |
|---|---|---|
| tasksource/tasksource-instruct | `4cdba593d93bf1e32c0fd1731d84c6780136fe51` | Complete 7,282,255-row train release audited; 3,886,906 rows marked commercial; zero retained after typed-source preference and clearance checks |
| tasksource/tasksource-jev-typed-decisions | `f8ff7e81ed820eeac1ad7afd63449fe99c8a76c3` | Full text train release and all nine Tasksource append shards, including the 2026-10-12 update; direct decision variants only |
| unchanged A3 | `bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed` | Original train, calibration and manifest verified against the item51 SHA256 pins |
| frozen suite | `e57106b5e0698e74bd1a88b3b4c19b94a0dc8328` | Private `jbrashear/decision-index-suite-0.3/selected-rows.jsonl.gz`; 124,971 selected records |

Typed-source preference prevents duplicate task counting between the two required releases. Instruct's uncovered entries include renamed civil-comments and Dynasent projections, token-label sequences, upstream-excluded quality views, and classification tasks without confirmed source terms. For example, pinned SIGA-nli and Crowdflower cards say unknown, and the pinned probability-words-NLI card supplies no licence. These are dropped even when a row's best-effort `license_use` classification says commercial. No inferred licence or generated menu is used to fill the gap. The manifest preserves the `license_use` field and the commercial-only filter for both collections.

The tranche is 43,035 questions below the 300,000 target. Counts distinguish tasks from upstream datasets: 156 trained tasks meet the 100-task breadth aim, while 82 upstream datasets fall short of 100 distinct upstreams. Strict gates were retained rather than duplicating rows or relaxing terms. Cleared legal candidates MAUD, CUAD and ACORD were rejected by the full-source scan, leaving no legal additions. Finance includes SentFin; financial-news sentiment is a holdout. Other retained sources cover NLI, sentiment/emotion, toxicity, prompt injection, medical decisions, commonsense, causal reasoning, intent/topic classification and multilingual tasks. Every count and licence is listed below.

## Caps, conversion and quality

The per-source **per-task cap is 5,000 questions**, chosen at the upper end of the requested range to maximize clean exposure. One task never exceeds that cap. Upstream dataset families govern overlap rejection and holdouts; they can contain multiple independently capped tasks. No global duplicate sampling or upsampling is used. A3 is an immutable prefix and its original calibration bytes are preserved.

The adapter in `data/convert_data.py` preserves the published state, question and option order. Choice criteria use numeric wire keys mapped to original option text; noul uses the source's true probability; score retains its ordered rubric. Record-level labels and soft target distributions remain outside `questions`. Noncommercial, unknown, mixed unconfirmed terms, non-direct variants, malformed decisions, tied option golds and more than 20 options are excluded. Generated responses, generated distractors, inseparable synthetic augmentation and known model-translated sources are excluded from additions. Nimble v3 contributes no inputs, outputs or weights. Existing accepted A3 provenance is recorded separately in the licence manifest.

Deduplication uses normalized state hashes within and across sources, including original A3 train and calibration. Selection is deterministic by row-ID hash and then task round robin. The raw-source gate removed another 7,940 converted candidates, including 5,000 previously selected EXAMS questions. Quality selection dropped 61,057 duplicate states and 1,644 over-budget renders; 95,443 candidates exceeded the task or holdout caps. State-based deduplication conservatively removes repeated contexts even when another task asks a different question about the same context.

The frozen Qwen tokenizer is checked by SHA256. Canonical, reversed and seeded-shuffled choice orders use the existing `jebadiah_prompt.Renderer`. The prompt budget is 1,984 tokens plus the original 64-token padding reserve, keeping the A3 2,048-token limit. Initial validation checked 741,937 renders. The raw-gated rebuild reused 259,965 byte-bound verified records and made 1,644 additional checks; retained rows have zero truncation. Maximum observed tokens by type: choice 1,952, noul 389, score 298. The tranche contains 229,664 choice, 15,731 noul and 11,570 score questions.

The implementation reuses `convert_data.wire_question`, `compose_pools.cat`, `pool_manifest.describe`, `lint_data` and the unchanged `item33_full_suite_scan.scan`, following the immutable-prefix item5x composition pattern. It adds a source adapter and build stages, without model calls or an alternative scanner.

## Whole-family and full-suite gates

The exact source policy from the frozen A3 manifest implements the same 22-family panel exclusion, including aliases, parents and derivatives. It is applied to task names and recorded source ancestry before source sampling. The wider A3 substring blacklist is preserved, including its conservative exclusion of lex_glue. The scanner's own removed-family count of 22 is a separate outcome from A3's 22-family panel exclusion.

The original item25 index was verified at `cf54ade9013c05db74f4c70925287382708de965fbf3cba62145ee821adad0ff`. Its public payloads did not fully protect the downloaded frozen selection: the exact-state/leaf coverage check missed 10,139 ForecastBench binary records and three Home-Appliance states. Additional frozen ToolRet, BRIGHT and ACOS leaves also supplied new protected spans. The builder supplements that original index with every selected state and leaf using item33's exact-state, normalized-leaf, 13-word-span and five-gram containment functions. It preserves the original index's public additions and rebuilt GSM8K protection.

Frozen selected gzip SHA256: `1f92d9723b24ca5483474bf324d0ed8e5d542770f89694a579244c45a59cb7fe`.
Supplemented index SHA256: `b92290742245132a59839fa565ecc70d691d26033533cb48eb3da702aebd9851`.
The private index and suite contents are excluded from git.

| Scan stage | Records | Direct hits | Whole families removed | Records removed | Remaining direct hits |
|---|---:|---:|---:|---:|---:|
| Raw source rows, including discarded formats and malformed rows | 637,018 | 2,741 plus 1 invalid empty state | 24 | 113,123 | 0 |
| All eligible converted source candidates, before sampling | 503,586 | 2,352 | 22 | 77,537 | 0 |
| Selected tranche plus heldout | 259,965 | 0 | 0 | 0 | 0 |
| Unchanged A3 train plus calibration, checked separately | 16,902 | 5 | 2 | 2,633 if scrubbed | 5 in preserved A3 |

Direct source hits were all 13-word-span matches. The raw scan additionally found one invalid empty state in EXAMS, which the existing scanner rejects at family level. DocRED had raw protected-text matches; neither whole family is retained. A single hit rejects the entire upstream dataset family across its tasks. This removes all siblings, including rows without direct matches. Scanner checks do not establish semantic independence or address base-model pretraining overlap. The new-tranche zero-hit result must not be described as a zero-hit result for the unchanged A3 composition.


The untouched A3 manifest also has exact upstream-name exposure to 10 families rejected by the new raw-source scan, totaling 1,022 questions, including the 102 math-QA questions already counted above. The other families are `AndyChiang/dgen`, `GGLab/PARADISE`, `jaredfern/codah`, `mteb/amazon_counterfactual`, `propsegment/nli`, `tasksource/arct`, `tasksource/equate`, `unigram/FOL-nli`, `webis/args_me`. Together with HelpSteer2, those exact-name matches imply at least 3,553 legacy questions across 11 source families need disposition under a global whole-family rule. This is a lower bound because aliases and different upstream revisions require review; it is not an approved A3 scrub.

## Heldout families

Each family contributes 600 questions. All tasks and source aliases within each family are absent from the tranche and pinned A3 train/calibration. No random row split substitutes for the family holdout.

| Heldout task | Whole upstream family | Questions |
|---|---|---:|
| SDOH-NLI | tasksource/SDOH-NLI | 600 |
| circa | tasksource/circa | 600 |
| multilingual/x-stance | michiel/xstance; original strombergnlp/x-stance | 600 |
| poem_sentiment | google-research-datasets/poem_sentiment | 600 |
| twitter-financial-news-sentiment | zeroshot/twitter-financial-news-sentiment | 600 |

GitHub issue similarity was an initial holdout candidate, but full-source overlap rejected it. Circa replaces it and has no A3 exposure. The heldout-family check set is independent of the private board and is not the original calibration split.

## Source counts and licences

Every retained task has `license_use == commercial`, an explicitly permitted licence, pinned source provenance and a separately retrieved licence card. The JSON manifest records the data revision, card revision, card URL and SHA256; a later pinned card is identified when the older data revision had no README. Unknown terms and custom mixed terms are excluded. A3 source entries are copied from its original manifest and are not new clearance claims.

| Task | Upstream family | Licence | Train | Heldout |
|---|---|---|---:|---:|
| HatemojiBuild | HannahRoseKirk/HatemojiBuild | cc-by-4.0 | 2364 | 0 |
| MSciNLI | sadat2307/MSciNLI | cc-by-sa-4.0 | 3979 | 0 |
| PARARULE-Plus | qbao775/PARARULE-Plus | mit | 3185 | 0 |
| SDOH-NLI | tasksource/SDOH-NLI | cc-by-4.0 | 0 | 600 |
| SpaceNLI | tasksource/SpaceNLI | mit | 3939 | 0 |
| Touche23-ValueEval | Touche23-ValueEval | cc-by-4.0, CC BY 4.0 (DPI) | 1474 | 0 |
| acceptability-prediction/binary_votes | tasksource/acceptability-prediction | apache-2.0 | 2184 | 0 |
| acceptability-prediction/rating_votes | tasksource/acceptability-prediction | apache-2.0 | 2136 | 0 |
| amazon_polarity/amazon_polarity | fancyzhx/amazon_polarity | apache-2.0 | 3444 | 0 |
| babi_nli/basic-coreference | tasksource/babi_nli | bsd | 233 | 0 |
| babi_nli/basic-deduction | tasksource/babi_nli | bsd | 235 | 0 |
| babi_nli/basic-induction | tasksource/babi_nli | bsd | 244 | 0 |
| babi_nli/compound-coreference | tasksource/babi_nli | bsd | 244 | 0 |
| babi_nli/conjunction | tasksource/babi_nli | bsd | 241 | 0 |
| babi_nli/counting | tasksource/babi_nli | bsd | 250 | 0 |
| babi_nli/indefinite-knowledge | tasksource/babi_nli | bsd | 242 | 0 |
| babi_nli/lists-sets | tasksource/babi_nli | bsd | 233 | 0 |
| babi_nli/path-finding | tasksource/babi_nli | bsd | 243 | 0 |
| babi_nli/positional-reasoning | tasksource/babi_nli | bsd | 211 | 0 |
| babi_nli/simple-negation | tasksource/babi_nli | bsd | 241 | 0 |
| babi_nli/single-supporting-fact | tasksource/babi_nli | bsd | 245 | 0 |
| babi_nli/size-reasoning | tasksource/babi_nli | bsd | 220 | 0 |
| babi_nli/three-arg-relations | tasksource/babi_nli | bsd | 240 | 0 |
| babi_nli/three-supporting-facts | tasksource/babi_nli | bsd | 236 | 0 |
| babi_nli/time-reasoning | tasksource/babi_nli | bsd | 234 | 0 |
| babi_nli/two-arg-relations | tasksource/babi_nli | bsd | 238 | 0 |
| babi_nli/two-supporting-facts | tasksource/babi_nli | bsd | 244 | 0 |
| babi_nli/yes-no-questions | tasksource/babi_nli | bsd | 240 | 0 |
| balanced-copa | pkavumba/balanced-copa | cc-by-4.0, BSD 2-Clause License (DPI) | 793 | 0 |
| blog_authorship_corpus/age | tasksource/blog_authorship_corpus | apache-2.0 | 1537 | 0 |
| causal-news | thagen/CNCv2 | cc0-1.0 | 1000 | 0 |
| causal-reasoning/transitivity | ludwigw/causal-reasoning-benchmarks | mit | 5000 | 0 |
| circa | google-research-datasets/circa | cc-by-4.0 | 0 | 600 |
| civil_comments/identity_attack_share | google/civil_comments | cc0-1.0 | 5000 | 0 |
| civil_comments/insult_share | google/civil_comments | cc0-1.0 | 3547 | 0 |
| clcd-english | tasksource/clcd-english | apache-2.0 | 2893 | 0 |
| cloth | AndyChiang/cloth | mit | 5000 | 0 |
| commonsense_qa | tau/commonsense_qa | mit | 5000 | 0 |
| commonsense_qa_2.0 | tasksource/commonsense_qa_2.0 | cc-by-4.0 | 3036 | 0 |
| corr2cause | tasksource/corr2cause | mit | 5000 | 0 |
| cycic_classification | tasksource/cycic_classification | apache-2.0 | 2918 | 0 |
| cycic_multiplechoice | tasksource/cycic_multiplechoice | apache-2.0 | 5000 | 0 |
| defeasible-nli/atomic | tasksource/defeasible-nli | apache-2.0, MIT License (DPI) | 1314 | 0 |
| defeasible-nli/social | tasksource/defeasible-nli | apache-2.0 | 1318 | 0 |
| edos | tasksource/edos | cc0-1.0 | 1000 | 0 |
| ethics/commonsense | hendrycks/ethics | mit | 896 | 0 |
| ethics/deontology | hendrycks/ethics | mit | 914 | 0 |
| ethics/justice | hendrycks/ethics | mit | 918 | 0 |
| ethics/virtue | hendrycks/ethics | mit | 917 | 0 |
| feasibilityQA | tasksource/feasibilityQA | mit | 2801 | 0 |
| fig-qa | nightingal3/fig-qa | mit | 5000 | 0 |
| google_wellformed_query | tasksource/google_wellformed_query | cc-by-sa-4.0 | 5000 | 0 |
| hate_speech18 | tasksource/hate_speech18 | cc-by-sa-3.0, CC BY-SA 3.0 (DPI) | 3052 | 0 |
| hatexplain/majority | Hate-speech-CNERG/hatexplain | cc-by-4.0 | 1000 | 0 |
| headline_cause/en_simple | IlyaGusev/headline_cause | cc0-1.0 | 3513 | 0 |
| health_fact | marcov/health_fact_promptsource | mit | 3374 | 0 |
| hlgd | tasksource/hlgd | apache-2.0, Apache License 2.0 (DPI) | 3090 | 0 |
| hope_edi/english | hope_edi/english | cc-by-4.0 | 3925 | 0 |
| jigsaw_toxicity | tasksource/jigsaw_toxicity | apache-2.0 | 3411 | 0 |
| logical-entailment | tasksource/logical-entailment | apache-2.0 | 3252 | 0 |
| lonli | tasksource/lonli | mit | 4024 | 0 |
| magpie/idiomatic | idioms-collaboration/magpie | mit, cc-by-4.0 (original source) | 1000 | 0 |
| mind2web/action | tasksource/mind2web-dom | cc-by-4.0 | 4 | 0 |
| mind2web/dom-element | tasksource/mind2web-dom | cc-by-4.0 | 5 | 0 |
| mindgames | sileod/mindgames | apache-2.0, Apache License 2.0 (DPI) | 3030 | 0 |
| monotonicity-entailment | tasksource/monotonicity-entailment | apache-2.0 | 2958 | 0 |
| moral_stories/full | LabHC/moral_stories | mit, MIT License (DPI) | 5000 | 0 |
| multilingual/AfriSenti-twitter-sentiment/amh | mteb/AfriSentiClassification | cc-by-4.0 | 413 | 0 |
| multilingual/AfriSenti-twitter-sentiment/arq | mteb/AfriSentiClassification | cc-by-4.0 | 413 | 0 |
| multilingual/AfriSenti-twitter-sentiment/ary | mteb/AfriSentiClassification | cc-by-4.0 | 411 | 0 |
| multilingual/AfriSenti-twitter-sentiment/hau | mteb/AfriSentiClassification | cc-by-4.0 | 412 | 0 |
| multilingual/AfriSenti-twitter-sentiment/ibo | mteb/AfriSentiClassification | cc-by-4.0 | 417 | 0 |
| multilingual/AfriSenti-twitter-sentiment/kin | mteb/AfriSentiClassification | cc-by-4.0 | 411 | 0 |
| multilingual/AfriSenti-twitter-sentiment/pcm | mteb/AfriSentiClassification | cc-by-4.0 | 420 | 0 |
| multilingual/AfriSenti-twitter-sentiment/por | mteb/AfriSentiClassification | cc-by-4.0 | 413 | 0 |
| multilingual/AfriSenti-twitter-sentiment/swa | mteb/AfriSentiClassification | cc-by-4.0 | 400 | 0 |
| multilingual/AfriSenti-twitter-sentiment/tso | mteb/AfriSentiClassification | cc-by-4.0 | 410 | 0 |
| multilingual/AfriSenti-twitter-sentiment/twi | mteb/AfriSentiClassification | cc-by-4.0 | 398 | 0 |
| multilingual/AfriSenti-twitter-sentiment/yor | mteb/AfriSentiClassification | cc-by-4.0 | 408 | 0 |
| multilingual/MLMA_hate_speech | nedjmaou/MLMA_hate_speech | mit | 2280 | 0 |
| multilingual/NusaX-senti/ace | mteb/NusaX-senti | cc-by-sa-4.0 | 362 | 0 |
| multilingual/NusaX-senti/ban | mteb/NusaX-senti | cc-by-sa-4.0 | 361 | 0 |
| multilingual/NusaX-senti/bbc | mteb/NusaX-senti | cc-by-sa-4.0 | 361 | 0 |
| multilingual/NusaX-senti/bjn | mteb/NusaX-senti | cc-by-sa-4.0 | 359 | 0 |
| multilingual/NusaX-senti/bug | mteb/NusaX-senti | cc-by-sa-4.0 | 360 | 0 |
| multilingual/NusaX-senti/eng | mteb/NusaX-senti | cc-by-sa-4.0 | 355 | 0 |
| multilingual/NusaX-senti/ind | mteb/NusaX-senti | cc-by-sa-4.0 | 358 | 0 |
| multilingual/NusaX-senti/jav | mteb/NusaX-senti | cc-by-sa-4.0 | 358 | 0 |
| multilingual/NusaX-senti/mad | mteb/NusaX-senti | cc-by-sa-4.0 | 358 | 0 |
| multilingual/NusaX-senti/min | mteb/NusaX-senti | cc-by-sa-4.0 | 357 | 0 |
| multilingual/NusaX-senti/nij | mteb/NusaX-senti | cc-by-sa-4.0 | 359 | 0 |
| multilingual/NusaX-senti/sun | mteb/NusaX-senti | cc-by-sa-4.0 | 358 | 0 |
| multilingual/disrpt/fas.rst.prstc.rels | multilingual-discourse-hub/disrpt | apache-2.0 | 577 | 0 |
| multilingual/disrpt/fra.sdrt.annodis.rels | multilingual-discourse-hub/disrpt | apache-2.0 | 548 | 0 |
| multilingual/klue/nli | klue/klue | cc-by-sa-4.0 | 1529 | 0 |
| multilingual/klue/sts | klue/klue | cc-by-sa-4.0 | 1532 | 0 |
| multilingual/klue/ynat | klue/klue | cc-by-sa-4.0 | 1518 | 0 |
| multilingual/miam | multilingual/miam | cc-by-sa-4.0 | 4320 | 0 |
| multilingual/multilingual-sentiments/all | tasksource/multilingual-sentiments | apache-2.0 | 4283 | 0 |
| multilingual/offenseval_dravidian/kannada | community-datasets/offenseval_dravidian | cc-by-4.0 | 1671 | 0 |
| multilingual/offenseval_dravidian/malayalam | community-datasets/offenseval_dravidian | cc-by-4.0 | 1636 | 0 |
| multilingual/offenseval_dravidian/tamil | community-datasets/offenseval_dravidian | cc-by-4.0 | 1693 | 0 |
| multilingual/tweet_sentiment_multilingual | mteb/tweet_sentiment_multilingual | cc-by-3.0 | 4365 | 0 |
| multilingual/x-stance | michiel/xstance | mit | 0 | 600 |
| multilingual/xcopa/et | cambridgeltl/xcopa | cc-by-4.0 | 44 | 0 |
| multilingual/xcopa/ht | cambridgeltl/xcopa | cc-by-4.0 | 46 | 0 |
| multilingual/xcopa/id | cambridgeltl/xcopa | cc-by-4.0 | 48 | 0 |
| multilingual/xcopa/it | cambridgeltl/xcopa | cc-by-4.0 | 44 | 0 |
| multilingual/xcopa/qu | cambridgeltl/xcopa | cc-by-4.0 | 44 | 0 |
| multilingual/xcopa/sw | cambridgeltl/xcopa | cc-by-4.0 | 44 | 0 |
| multilingual/xcopa/ta | cambridgeltl/xcopa | cc-by-4.0 | 48 | 0 |
| multilingual/xcopa/th | cambridgeltl/xcopa | cc-by-4.0 | 48 | 0 |
| multilingual/xcopa/tr | cambridgeltl/xcopa | cc-by-4.0 | 45 | 0 |
| multilingual/xcopa/vi | cambridgeltl/xcopa | cc-by-4.0 | 45 | 0 |
| multilingual/xcopa/zh | cambridgeltl/xcopa | cc-by-4.0 | 45 | 0 |
| multilingual/xstory_cloze/ar | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/en | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/es | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/eu | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/hi | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/id | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/my | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/ru | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/sw | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/te | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multilingual/xstory_cloze/zh | juletxara/xstory_cloze | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 324 | 0 |
| multiwoz22/categorical-slot | pfb30/multi_woz_v22 | apache-2.0 | 1000 | 0 |
| multiwoz22/dialogue-act | pfb30/multi_woz_v22 | apache-2.0 | 999 | 0 |
| nan-nli | joey234/nan-nli | cc-by-sa-4.0 | 232 | 0 |
| naturallogic | tasksource/naturallogic | apache-2.0 | 3654 | 0 |
| numer_sense | tasksource/numer_sense | mit | 4408 | 0 |
| onestop_qa | malmaud/onestop_qa | cc-by-sa-4.0, CC BY-SA 4.0 (DPI) | 1299 | 0 |
| patent-phrase-similarity | tasksource/patent-phrase-similarity | cc-by-4.0 | 4632 | 0 |
| perspectrum/stance | tasksource/perspectrum | cc-by-sa-4.0, cc-by-sa-4.0 (original source) | 1000 | 0 |
| poem_sentiment | google-research-datasets/poem_sentiment | cc-by-4.0 | 0 | 600 |
| prompt-injection-dataset | S-Labs/prompt-injection-dataset | mit | 3088 | 0 |
| prompt-injections | deepset/prompt-injections | apache-2.0 | 388 | 0 |
| prost | corypaik/prost | apache-2.0, Apache License 2.0 (DPI) | 5000 | 0 |
| proto_qa/proto_qa | community-datasets/proto_qa | cc-by-4.0 | 5000 | 0 |
| qasc | allenai/qasc | cc-by-4.0 | 5000 | 0 |
| quac/answerability | allenai/quac | mit | 985 | 0 |
| quartz | allenai/quartz | cc-by-4.0 | 2581 | 0 |
| ruletaker | tasksource/ruletaker | apache-2.0, Apache License 2.0 (DPI) | 3239 | 0 |
| scinli | tasksource/scinli | apache-2.0, CC BY-SA 4.0 (DPI) | 3962 | 0 |
| scone | tasksource/scone | cc0-1.0 | 2938 | 0 |
| sentfin | tasksource/sentfin | mit | 1000 | 0 |
| shell-safety-v2 | tomngdev/shell-safety-v2 | mit | 4138 | 0 |
| silicone/maptask | tasksource/silicone | cc-by-sa-4.0, CC BY 4.0 (DPI) | 1099 | 0 |
| snips_built_in_intents | sonos-nlu-benchmark/snips_built_in_intents | cc0-1.0, CC0 1.0 (DPI) | 197 | 0 |
| spartqa-mchoice | tasksource/spartqa-mchoice | mit | 5000 | 0 |
| spartqa-yn | tasksource/spartqa-yn | apache-2.0 | 5000 | 0 |
| spider/column | tasksource/spider | cc-by-sa-4.0 | 470 | 0 |
| stepgame | tasksource/stepgame | mit | 4548 | 0 |
| streusle/verb | tasksource/streusle | cc-by-sa-4.0 | 1000 | 0 |
| subjectivity | tasksource/subjectivity | mit | 625 | 0 |
| temporal-nli | tasksource/temporal-nli | apache-2.0 | 4053 | 0 |
| tracie | tasksource/tracie | apache-2.0 | 952 | 0 |
| twitter-financial-news-sentiment | zeroshot/twitter-financial-news-sentiment | mit | 0 | 600 |
| ukpconvarg/convincingness | tasksource/ukpconvarg | cc-by-4.0, cc-by-4.0 (original source) | 12 | 0 |
| utilitarianism | hendrycks/ethics | mit | 1558 | 0 |
| wnut_17/wnut_17 | flaitenberger/wnut_17 | cc-by-4.0 | 5000 | 0 |

## Local reproduction and validation

Create the external venv with `data/manifests/t2b-cpu-requirements.txt`. Fetch only the pinned text shards and source metadata listed by the build-summary input hashes; retain their paths under `research/`. Licence card files and the fetch receipt are external inputs, with their immutable URLs and hashes in the licence manifest. Do not fetch model weights. Set the cache and temporary paths before any download or package installation:

```sh
export HF_HOME=/Volumes/PRO-G40/hf-home
export TMPDIR=/Volumes/PRO-G40/scratch/tmp
export UV_CACHE_DIR=/Volumes/PRO-G40/uv-cache
export PIP_CACHE_DIR=/Volumes/PRO-G40/pip-cache
export PYTHONPYCACHEPREFIX=/Volumes/PRO-G40/scratch/tmp/pycache
export TOKENIZERS_PARALLELISM=false
```

Run `data/t2b_build.py` stages in order: `prepare`, `index`, `raw-scan`, `scan`, `finish`, `verify`, each with `--research /Volumes/PRO-G40/scratch/lead-ainode/t2b/research` and `--output /Volumes/PRO-G40/scratch/lead-ainode/t2b/build-final`. The default original index path is on PRO-G40. Verification exits 1 for the observed A3 conflict and persists both independent scan receipts. Do not consume the blocked `train.jsonl` preview as a cleared training set.

Outputs are local at `/Volumes/PRO-G40/scratch/lead-ainode/t2b/build-final/`: `tranche.jsonl`, `heldout.jsonl`, blocked combined `train.jsonl`, unchanged `calib.jsonl`, build summary, licence manifest and scan receipts. The PR-ready changes contain code, requirements, manifests and this report only. Planned upload artifacts are the clean new tranche and heldout, after the publication instruction is clarified.

`data/lint_data.py --train` passes on all four local data files. Six source-conversion and licence/deduplication checks pass. The full selected-data scan independently checked all 259,965 new/heldout records with zero hits. File SHA256 values, counts by type, caps, quality removals and both A3 hashes are in `data/manifests/t2b-build-summary.json`; source terms and evidence are in `data/manifests/t2b-source-licenses.json`. The requested 9B screen line is in the run log, with no training launched.
