# Item 37: G1, D0 and L5 against A3

**Frozen proxy, not a leaderboard score.**

PR: https://github.com/getainode/jebadiah/pull/19

[Durable private report and evidence](https://huggingface.co/datasets/jbrashear/jebadiah-9b-v2-1-index-results/blob/4587e420c1797b86842714a53b727637db75ccc5/analysis/item37-ladder/item37-report.md).

11,079 requests in 10,380 complete groups on the unchanged frozen manifest
`74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`.
Each comparison uses 2,000 paired complete-group draws stratified by benchmark/domain/track,
seed 20261008, recomputing official metrics per draw. Index and area deltas use
unrounded bootstrap calculations; benchmark tables use official scorer precision.

| Run | Proxy | Delta vs A3 | Paired 95% CI vs A3 | Delta vs v2 | Paired 95% CI vs v2 | Decision |
|---|---:|---:|---|---:|---|---|
| G1 | 46.70 | -0.28 | [-1.65, +1.07] | +2.02 | [+0.64, +3.41] | drop |
| D0 | 46.57 | -0.40 | [-1.64, +0.80] | +1.90 | [+0.30, +3.55] | drop |
| L5 | 47.03 | +0.06 | [-1.34, +1.24] | +2.36 | [+0.81, +3.88] | drop |

A3 is 46.9746844133; v2 is 44.6740036323.
**Drop all three changes and retain A3.** L5's +0.06 point estimate is within noise.
All three beat v2 beyond this proxy's paired noise, but none clears the stronger A3 gate.
Keep only a change whose paired interval versus A3 has a positive lower bound.
Evaluate any combination as a separate rung; these individual runs do not prove a combination.

## Five scored areas

| Area | A3 | v2 | G1 | G1 delta vs A3 / v2 | D0 | D0 delta vs A3 / v2 | L5 | L5 delta vs A3 / v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Knowledge & Reasoning | 31.64 | 30.87 | 33.74 | +2.10 / +2.87 | 30.20 | -1.44 / -0.67 | 32.27 | +0.63 / +1.41 |
| Language Understanding | 50.32 | 46.61 | 53.17 | +2.85 / +6.56 | 52.49 | +2.16 / +5.88 | 51.69 | +1.37 / +5.08 |
| Retrieval & Classification | 50.29 | 45.10 | 45.81 | -4.48 / +0.70 | 50.74 | +0.45 / +5.64 | 49.36 | -0.93 / +4.26 |
| Tools & Automation | 67.73 | 67.99 | 65.67 | -2.06 / -2.32 | 64.82 | -2.91 / -3.17 | 66.07 | -1.66 / -1.92 |
| Arts & Human Taste | 33.40 | 31.90 | 30.58 | -2.82 / -1.32 | 31.92 | -1.48 / +0.02 | 33.69 | +0.29 / +1.79 |

## Item 31 worst five

Raw and chance-corrected skill are out of 100. Differences are skill points.

| Benchmark | A3 raw / skill | v2 raw / skill | G1 raw / skill | G1 delta vs A3 / v2 | D0 raw / skill | D0 delta vs A3 / v2 | L5 raw / skill | L5 delta vs A3 / v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BANKING77 | 69.16 / 68.76 | 69.05 / 68.65 | 67.99 / 67.58 | -1.18 / -1.07 | 69.90 / 69.51 | +0.75 / +0.86 | 71.93 / 71.57 | +2.81 / +2.92 |
| When2Call MCQ | 68.94 / 58.59 | 75.75 / 67.67 | 68.12 / 57.49 | -1.10 / -10.18 | 68.94 / 58.59 | +0.00 / -9.08 | 73.57 / 64.76 | +6.17 / -2.91 |
| RAGTruth response-level hallucination | 67.03 / 31.64 | 66.29 / 30.11 | 71.22 / 40.33 | +8.69 / +10.22 | 75.49 / 49.18 | +17.54 / +19.07 | 70.21 / 38.23 | +6.59 / +8.12 |
| API-Bank | 86.27 / 86.01 | 84.31 / 84.01 | 90.20 / 90.01 | +4.00 / +6.00 | 86.27 / 86.01 | +0.00 / +2.00 | 86.27 / 86.01 | +0.00 / +2.00 |
| CLINC150+OOS | 76.82 / 76.68 | 66.62 / 66.42 | 74.61 / 74.46 | -2.22 / +8.04 | 79.44 / 79.32 | +2.64 / +12.90 | 72.99 / 72.83 | -3.85 / +6.41 |

G1 increases RAGTruth point skill by 8.69 and API-Bank by 4.00, while its
overall index falls 0.28 and retrieval area falls 4.48 versus A3. D0 improves
RAGTruth by 17.54, yet its overall index falls 0.40. L5 improves When2Call
by 6.17, but its overall +0.06 is insufficient under the keep rule.
Benchmark shifts here are descriptive point estimates.

## Training, merge and controls

G1 adds only 1,350 item33 grounding training questions, with a separate 150-question
grounding holdout. The original A3 calibration remains byte-identical at 512 questions.
D0 changes only LoRA dropout 0.05 to 0.0. L5 changes only peak LR 1e-4 to 5e-5.
All retain seed 17, one full epoch, rank/alpha 16/32, 2,048 tokens, batch 8,
cosine schedule, warmup 30, ordinal targets and adjacent mass 0.2, item27 speed
flags, and backbone autocast off. CUDA preflight passes on RTX PRO 6000.
All initial exports pass the unchanged 0.05 merge gate with zero confident flips.
Maximum shifts are G1 0.00929, D0 0.01536 and L5 0.01574. No recovery or
private merge-gate exception was needed.

G1 training count is 22,540 versus A3 21,190. Its full epoch adds optimizer updates,
so content and added exposure are not causally separated. Grounding holdout never
replaces the original temperature-fitting calibration.

Training loss uses the ordinal objective; held-out NLL uses hard gold labels,
so their difference is not a calibrated generalization-gap estimate. Early and
final holdout values are retained below to document training/holdout behavior.

Actual training proof and unchanged merge gate:

```json
{
  "g1": {
    "checkpoint_revision": "2f343260a2707e19087e232b1d0b353ec0d517f8",
    "final_epoch": 1.0,
    "optimizer_steps": 2818,
    "train_questions": 22540,
    "original_calibration_questions": 512,
    "training_seconds": 2299.1,
    "peak_memory_allocated_gb": 64.81,
    "truncated_prompts": 36,
    "actual_config_differences_vs_a3": {
      "dataset_sha256": [
        "46fe2a743535a2e841d6b5aa5df479209c9e0c14f04ca1c987570acddd69e2ee",
        "56bac2c84f31ff62db4b168463178e3849ac6118f62cce6ac5b2b7aff09414b2"
      ]
    },
    "first_logged_calib": {
      "calib_accuracy": 0.6703910614525139,
      "calib_accuracy_choice": 0.6484375,
      "calib_accuracy_noul": 0.7708333333333334,
      "calib_accuracy_score": 0.6194029850746269,
      "calib_nll": 0.7513838942221946,
      "calib_nll_choice": 0.7591487044722128,
      "calib_nll_noul": 0.5685224366525383,
      "calib_nll_score": 0.8749719853765578,
      "step": 200
    },
    "final_calib": {
      "calib_accuracy": 0.8575418994413407,
      "calib_nll": 0.4299844862591111,
      "calib_accuracy_choice": 0.875,
      "calib_nll_choice": 0.42612377218712266,
      "calib_accuracy_noul": 0.8333333333333334,
      "calib_nll_noul": 0.3244437428894327,
      "calib_accuracy_score": 0.8582089552238806,
      "calib_nll_score": 0.5092836113688399
    },
    "last_logged_training_loss": 0.7109721660614013,
    "initial_merge_verification": {
      "questions": 260,
      "identical_picks": 259,
      "agreement": 0.9961538461538462,
      "max_probability_delta": 0.009287834167480469,
      "clear_margin_flips": 0,
      "margin_threshold": 0.05,
      "max_delta_allowed": 0.05
    },
    "optimizer_steps_added_by_recovery": 0
  },
  "d0": {
    "checkpoint_revision": "b7c3bdd01217122c66cc0f2ba1ab8e2f442544e9",
    "final_epoch": 1.0,
    "optimizer_steps": 2649,
    "train_questions": 21190,
    "original_calibration_questions": 512,
    "training_seconds": 2098.9,
    "peak_memory_allocated_gb": 60.08,
    "truncated_prompts": 36,
    "actual_config_differences_vs_a3": {
      "decide.lora_dropout": [
        0.05,
        0.0
      ]
    },
    "first_logged_calib": {
      "calib_accuracy": 0.6871508379888268,
      "calib_accuracy_choice": 0.7421875,
      "calib_accuracy_noul": 0.78125,
      "calib_accuracy_score": 0.5671641791044776,
      "calib_nll": 0.744392209682377,
      "calib_nll_choice": 0.661875610551827,
      "calib_nll_noul": 0.502579096178872,
      "calib_nll_score": 0.9964532812125776,
      "step": 200
    },
    "final_calib": {
      "calib_accuracy": 0.840782122905028,
      "calib_nll": 0.44570556117288623,
      "calib_accuracy_choice": 0.8359375,
      "calib_nll_choice": 0.4534762547909875,
      "calib_accuracy_noul": 0.8541666666666666,
      "calib_nll_noul": 0.32109558174138203,
      "calib_accuracy_score": 0.835820895522388,
      "calib_nll_score": 0.5275556301453299
    },
    "last_logged_training_loss": 0.5869179248809815,
    "initial_merge_verification": {
      "questions": 260,
      "identical_picks": 260,
      "agreement": 1.0,
      "max_probability_delta": 0.015362590551376343,
      "clear_margin_flips": 0,
      "margin_threshold": 0.05,
      "max_delta_allowed": 0.05
    },
    "optimizer_steps_added_by_recovery": 0
  },
  "l5": {
    "checkpoint_revision": "870c1efbd8326f9254af9ed553cb832dc6996486",
    "final_epoch": 1.0,
    "optimizer_steps": 2649,
    "train_questions": 21190,
    "original_calibration_questions": 512,
    "training_seconds": 2191.1,
    "peak_memory_allocated_gb": 64.81,
    "truncated_prompts": 36,
    "actual_config_differences_vs_a3": {
      "learning_rate": [
        0.0001,
        5e-05
      ]
    },
    "first_logged_calib": {
      "calib_accuracy": 0.7625698324022346,
      "calib_accuracy_choice": 0.84375,
      "calib_accuracy_noul": 0.8020833333333334,
      "calib_accuracy_score": 0.6567164179104478,
      "calib_nll": 0.5996970041036076,
      "calib_nll_choice": 0.51423002727751,
      "calib_nll_noul": 0.43264061655504465,
      "calib_nll_score": 0.8010192894648206,
      "step": 200
    },
    "final_calib": {
      "calib_accuracy": 0.8435754189944135,
      "calib_nll": 0.4433608866542153,
      "calib_accuracy_choice": 0.8515625,
      "calib_nll_choice": 0.4389425999638878,
      "calib_accuracy_noul": 0.84375,
      "calib_nll_noul": 0.31952516378475754,
      "calib_accuracy_score": 0.835820895522388,
      "calib_nll_score": 0.536299469429065
    },
    "last_logged_training_loss": 0.6872276306152344,
    "initial_merge_verification": {
      "questions": 260,
      "identical_picks": 259,
      "agreement": 0.9961538461538462,
      "max_probability_delta": 0.01574254035949707,
      "clear_margin_flips": 0,
      "margin_threshold": 0.05,
      "max_delta_allowed": 0.05
    },
    "optimizer_steps_added_by_recovery": 0
  }
}
```

## Coverage


- G1: 11079/11079 requests, statuses {'ok': 11079}, 0 missing, 0 whole-case failures, 0 outside-manifest rows.
- D0: 11079/11079 requests, statuses {'ok': 11079}, 0 missing, 0 whole-case failures, 0 outside-manifest rows.
- L5: 11079/11079 requests, statuses {'ok': 11079}, 0 missing, 0 whole-case failures, 0 outside-manifest rows.

## Immutable artifacts

```json
{
  "training_code": "ebd3dfb3777e607a0d55d14823e238b6a44bb3ac",
  "data_revision": "98356116e9fdd9db8de8759bfab9eac5e83f0c03",
  "training": {
    "g1": "6ac845b0fee2c900701721fb",
    "d0": "6ac845b0095c578089300c3f",
    "l5": "6ac845b0fee2c900701721fd"
  },
  "proxy": {
    "g1": "6ac85215095c57808930114a",
    "d0": "6ac85216095c57808930114c",
    "l5": "6ac85216fee2c900701728b5"
  },
  "recovery": {},
  "models": {
    "g1": "a1e0623dfbb26276298097e628ea127291c798a8",
    "d0": "e3b1a9413149a2207db1059f3757d81d4379f368",
    "l5": "37bc756f3e375cc9caabfd536671931192c85665"
  },
  "results": {
    "g1": {
      "revision": "9e1c93b58b2f16128e43df87caf130fa3cd23fb0",
      "path": "runs/jebadiah-9b-v2-1-g1-a1e0623d-proxy-0.3-10pct/results.jsonl.gz"
    },
    "d0": {
      "revision": "9e1c93b58b2f16128e43df87caf130fa3cd23fb0",
      "path": "runs/jebadiah-9b-v2-1-d0-e3b1a941-proxy-0.3-10pct/results.jsonl.gz"
    },
    "l5": {
      "revision": "9e1c93b58b2f16128e43df87caf130fa3cd23fb0",
      "path": "runs/jebadiah-9b-v2-1-l5-37bc756f-proxy-0.3-10pct/results.jsonl.gz"
    }
  },
  "baselines": {
    "a3": {
      "model": "frontier-infra/jebadiah-9b-v2-1-a3",
      "model_revision": "9e69926a007dd636e82d33485dcd48f9751c4248",
      "results_revision": "870541621d1b76efe0c869133d94ef7a87690a7c"
    },
    "v2": {
      "model": "frontier-infra/jebadiah-9b-v2",
      "model_revision": "db21f2af4fa564afdc9dbc67255d5eb4ea35919d",
      "results_revision": "907795806437388db31e947a5b72b8cfc6860ded"
    }
  },
  "datasets": {
    "a3": {
      "repo": "frontier-infra/jebadiah-data-v2-1-item32",
      "revision": "bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed"
    },
    "g1": {
      "repo": "frontier-infra/jebadiah-data-v2-1-item37",
      "revision": "98356116e9fdd9db8de8759bfab9eac5e83f0c03"
    }
  },
  "proxy_code": "587f4f496b9cf1d83f0bc8e495379496f65afa2f",
  "suite_revision": "e57106b5e0698e74bd1a88b3b4c19b94a0dc8328",
  "frozen_manifest_sha256": "74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405"
}
```

## Spend

| Job | Stage | Runtime seconds | Estimated USD |
|---|---|---:|---:|
| [g1-training](https://huggingface.co/jobs/jbrashear/6ac845b0fee2c900701721fb) | COMPLETED | 3137 | 2.3963 |
| [d0-training](https://huggingface.co/jobs/jbrashear/6ac845b0095c578089300c3f) | COMPLETED | 2937 | 2.2435 |
| [l5-training](https://huggingface.co/jobs/jbrashear/6ac845b0fee2c900701721fd) | COMPLETED | 3030 | 2.3146 |
| [g1-proxy](https://huggingface.co/jobs/jbrashear/6ac85215095c57808930114a) | COMPLETED | 1122 | 0.8571 |
| [d0-proxy](https://huggingface.co/jobs/jbrashear/6ac85216095c57808930114c) | COMPLETED | 1122 | 0.8571 |
| [l5-proxy](https://huggingface.co/jobs/jbrashear/6ac85216fee2c900701728b5) | COMPLETED | 1123 | 0.8578 |

Estimated compute total: **$9.5265**, below the shared $25 cap.
Estimates use reported running durations and $2.75/hour; final billing can differ.
Initial timeout allocation was three 140-minute training jobs plus three 35-minute
proxies: $24.0625. Any failed stages and export recovery are included in measured spend.
No teacher inference or full-suite run was launched.

## Run log

Keep/drop rows were committed in ebd3dfb before the training jobs launched.

| Run | Change vs | Tests | Keep if | Result |
|---|---|---|---|---|
| G1 | A3 (46.97) | add only item33 grounding train questions; retain grounding diagnostic holdout and unchanged original calibration | frozen-proxy paired 95% interval vs A3 above zero | 46.70; vs A3 -0.28, CI [-1.65, +1.07]; vs v2 +2.02, CI [+0.64, +3.41]; drop |
| D0 | A3 (46.97) | LoRA dropout 0.05 to 0.0; identical data and other recipe fields | frozen-proxy paired 95% interval vs A3 above zero | 46.57; vs A3 -0.40, CI [-1.64, +0.80]; vs v2 +1.90, CI [+0.30, +3.55]; drop |
| L5 | A3 (46.97) | peak LR 1e-4 to 5e-5; cosine schedule and warmup unchanged | frozen-proxy paired 95% interval vs A3 above zero | 47.03; vs A3 +0.06, CI [-1.34, +1.24]; vs v2 +2.36, CI [+0.81, +3.88]; drop |

## Limits, validation and cleanup

Intervals are conditional on this frozen proxy, not full-suite sampling uncertainty
or variation across training seeds. All models remain private and are never submitted.
Inherited A3 diagnostic source-policy exceptions remain and block shipping.

Local hardware: 24 trainer tests and 20 data tests pass; regenerated item33 train,
calibration, manifest and scan hashes match; composition and isolated configuration
checks pass; every split lints; compilation and whitespace checks pass. The unchanged
scorer reproduces both pinned baselines with complete frozen-proxy coverage.
No GitHub-hosted CI was run. Review and merge remain with the lead.

The item37 PRO-G40 scratch directory is removed after durable private evidence upload,
including regenerated skill data and source-scan copies, G1 release data, downloaded
suite and baseline results, small model/checkpoint metadata, and local Hub cache.
No full model weights were downloaded to the Mac. Shared protected index is retained.
