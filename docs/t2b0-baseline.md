# T2b-0 clean A3 baseline

T2b-0 removes all eleven approved upstream families from pinned A3, with no additions. It retains 17,793 training questions in 13,225 records and 356 calibration questions in 124 records. All retained rows preserve their original bytes and order. This becomes the clean baseline without a candidate keep decision. Compare the two-seed proxy mean with A3's 46.82 and report the independent heldout-family score.

The original A3 runtime is `86a8203d792463536eb009a4d8da6c8681ffdd99`, with base `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Rank 16, alpha 32, learning rate 0.0001, one epoch, effective batch 8, length grouping, padding, objective, optimizer, schedule, evaluation flags and merge gate remain unchanged. Seeds are 17 and 18. Temperature fitting uses the original command on the scrubbed calibration. Calibration evaluation retains its original limit of 358, so all 356 remaining questions are evaluated.

## Whole-family removal

Identity resolution checks row identity, subset metadata and recorded upstream parents. All five HelpSteer2 rating views belong to the same family. The math-QA mapping covers tasksource and allenai aliases; PropSegmEnt covers its original sihaochen parent. HelpSteer and HelpSteer3 are separate retained sources. No extra parent exposure was found beyond the 3,553 questions in the evidence.

| Family | Training questions removed | Calibration questions removed |
|---|---:|---:|
| tasksource/math_qa | 102 | 0 |
| nvidia/HelpSteer2 | 2375 | 156 |
| AndyChiang/dgen | 102 | 0 |
| GGLab/PARADISE | 102 | 0 |
| jaredfern/codah | 102 | 0 |
| mteb/amazon_counterfactual | 103 | 0 |
| propsegment/nli | 102 | 0 |
| tasksource/arct | 102 | 0 |
| tasksource/equate | 102 | 0 |
| unigram/FOL-nli | 103 | 0 |
| webis/args_me | 102 | 0 |

The supplemented full-suite index has SHA256 `b92290742245132a59839fa565ecc70d691d26033533cb48eb3da702aebd9851`. The unchanged repository scanner checked all 13,349 retained records and found zero direct hits or invalid empty states. Both splits pass `lint_data.py --train`. Five ancestry and manifest-contract regression checks cover aliases, parents, rating siblings and false substring matches; two launcher checks preserve the recipe and proxy commands.

## Private inputs and evaluation

The existing private dataset `frontier-infra/jebadiah-data-t2b` contains the new `t2b0/` folder at revision `28f80a3c1430c27297e0944978bfda9310298623`. Runtime additions are pinned in that private folder, allowing jobs to run without publishing a source branch. The unchanged root `heldout.jsonl` has SHA256 `000451d29bb99b67b9b3925fb1cef5859f723cc6f5fd7c01850ec15a59fc472b`.

Both seeds score all 3,000 heldout questions, 600 each from SDOH-NLI, circa, x-stance, poem sentiment and financial-news sentiment. Scoring uses the final unmerged adapter on the pinned base, the training evaluator's Scorer, canonical option menus, batch 8 and untempered candidate argmax. It fits no temperatures on the heldout. Report ordinary accuracy per family and the equal-family macro mean. Full answers persist only to private checkpoint repositories.

## Jobs and cost

The first two jobs failed before model loading or any optimizer step because the scrub manifest used records where the original pipeline requires rows. The metadata contract is corrected and checked by executing the unchanged original pipeline validation block against the real data. Their measured compute costs were USD 0.4400 and USD 0.4423, total USD 0.8823. Row bytes and scan results are unchanged.

Each replacement seed reserves 84 minutes for training, calibration, merge, persistence and heldout scoring, then 35 minutes for the original frozen-proxy path. These four timeout ceilings sum to 238 minutes, USD 10.9083 at USD 2.75/h. Including both failed setup jobs, the entire task is bounded by USD 11.7906, below the hard USD 12 cap. No automatic retry is authorized. Every job ID and measured running-time cost will be recorded in the run log.

Failed setup jobs were `6acd1fa7dd8177968b42b736` (seed 17) and `6acd1fafde32f9b0f431e385` (seed 18). Replacement training jobs are `6acd22f1de32f9b0f431ec45` (seed 17) and `6acd22fadd8177968b42c0a5` (seed 18). Scores are pending. Private model destinations are `frontier-infra/jebadiah-9b-t2b0-s17` and `frontier-infra/jebadiah-9b-t2b0-s18`, with separate private checkpoint repositories. Only code, aggregate manifests and documentation are included in the authorized PR. Data, models, outputs and private suite contents remain private.
