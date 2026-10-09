# Item 36: N3 and R3 against A3

**Frozen proxy, not a leaderboard score.**

PR: https://github.com/getainode/jebadiah/pull/18

[Durable private report and evidence](https://huggingface.co/datasets/jbrashear/jebadiah-9b-v2-1-index-results/blob/d04fe988f3386e5be0b2c96b6bff9ee49a7d832c/analysis/item36-ladder/item36-report.md).

11,079 requests in 10,380 complete groups on the unchanged frozen manifest
`74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`.
Each comparison uses 2,000 paired complete-group draws stratified by benchmark/domain/track,
seed 20261008, recomputing official metrics per draw. Deltas use unrounded scores.

| Run | Proxy | Delta vs A3 | Paired 95% CI vs A3 | Delta vs v2 | Paired 95% CI vs v2 | Decision |
|---|---:|---:|---|---:|---|---|
| N3 | 42.29 | -4.68 | [-6.25, -3.12] | -2.38 | [-4.19, -0.60] | drop |
| R3 | 46.91 | -0.06 | [-1.26, +1.08] | +2.24 | [+0.74, +3.73] | drop |

A3 is 46.9746844133; v2 is 44.6740036323. The keep criterion is a positive
lower bound of the paired interval versus A3. **Drop both changes and retain A3.**
No combined rung is evaluated. N3 is worse than both A3 and v2 beyond this
proxy's paired noise. R3 is statistically compatible with A3, improving the
RAGTruth point skill by 14.31 but worsening When2Call by 4.72, BANKING77 by
1.78 and CLINC150+OOS by 5.22 points versus A3. Its positive interval versus
v2 does not meet the keep rule against the stronger A3 comparator.

## Five scored areas

| Area | A3 | N3 | N3 delta vs A3 / v2 | R3 | R3 delta vs A3 / v2 |
|---|---:|---:|---:|---:|---:|
| Knowledge & Reasoning | 31.64 | 30.21 | -1.43 / -0.66 | 33.18 | +1.54 / +2.31 |
| Language Understanding | 50.32 | 46.01 | -4.31 / -0.59 | 50.88 | +0.56 / +4.27 |
| Retrieval & Classification | 50.29 | 47.36 | -2.93 / +2.26 | 48.74 | -1.55 / +3.64 |
| Tools & Automation | 67.73 | 57.30 | -10.43 / -10.69 | 66.67 | -1.05 / -1.31 |
| Arts & Human Taste | 33.40 | 26.33 | -7.07 / -5.57 | 32.38 | -1.02 / +0.48 |

## Item 31 worst five

Raw and chance-corrected skill are out of 100. Skill differences are versus A3 and v2.

| Benchmark | A3 raw / skill | N3 raw / skill | N3 skill delta vs A3 / v2 | R3 raw / skill | R3 skill delta vs A3 / v2 |
|---|---:|---:|---:|---:|---:|
| BANKING77 | 69.16 / 68.76 | 68.82 / 68.42 | -0.34 / -0.23 | 67.40 / 66.98 | -1.78 / -1.67 |
| When2Call MCQ | 68.94 / 58.59 | 51.23 / 34.97 | -23.62 / -32.70 | 65.40 / 53.87 | -4.72 / -13.80 |
| RAGTruth response-level hallucination | 67.03 / 31.64 | 71.74 / 41.41 | +9.77 / +11.30 | 73.93 / 45.95 | +14.31 / +15.84 |
| API-Bank | 86.27 / 86.01 | 72.55 / 72.02 | -13.99 / -11.99 | 86.27 / 86.01 | +0.00 / +2.00 |
| CLINC150+OOS | 76.82 / 76.68 | 68.60 / 68.41 | -8.27 / +1.99 | 71.63 / 71.46 | -5.22 / +5.04 |

## Coverage, training and controls

- N3: 11079/11079 requests, statuses {'ok': 11079}, 0 missing, 0 whole-case failures, 0 outside-manifest rows.
- R3: 11079/11079 requests, statuses {'ok': 11079}, 0 missing, 0 whole-case failures, 0 outside-manifest rows.

Both start from the same pinned Qwen3.5-9B base, seed 17, one full epoch,
2,048 tokens, batch 8, LR 1e-4, cosine schedule, warmup 30, dropout 0.05,
ordinal score targets with adjacent mass 0.2, and item27 grouping/padding/checkpoint flags.
Backbone autocast is off. N3 only changes rank/alpha from 16/32 to 64/128.
R3 only adds 4,050 deterministically regenerated skill training questions.
Original A3 calibration stays byte-identical; 450 skill questions remain diagnostic holdout.

Item33 train/calibration/manifest/scan hashes exactly match PR17, with zero full-source
scanner overlaps. Cross-source ID/family/normalized-state checks pass. R3 has 25,240
training questions. Both inherit A3 private diagnostic contamination exceptions and
remain ineligible for shipping without a separately measured compliant floor.

Training and merge proof (full provenance and logs remain in private evidence):

```json
{
  "n3": {
    "final_epoch": 1.0,
    "optimizer_steps": 2649,
    "train_questions": 21190,
    "original_calibration_questions": 512,
    "training_seconds": 2288.1,
    "peak_memory_allocated_gb": 66.54,
    "actual_config_differences_vs_a3": {
      "lora_rank": [
        16,
        64
      ],
      "lora_alpha": [
        32,
        128
      ]
    },
    "merge_verification": {
      "questions": 260,
      "identical_picks": 260,
      "agreement": 1.0,
      "max_probability_delta": 0.007547736167907715,
      "clear_margin_flips": 0,
      "margin_threshold": 0.05,
      "max_delta_allowed": 0.05
    },
    "extra_optimizer_steps_during_recovery": 0
  },
  "r3": {
    "final_epoch": 1.0,
    "optimizer_steps": 3155,
    "train_questions": 25240,
    "original_calibration_questions": 512,
    "training_seconds": 2499.7,
    "peak_memory_allocated_gb": 64.81,
    "actual_config_differences_vs_a3": {
      "dataset_sha256": [
        "46fe2a743535a2e841d6b5aa5df479209c9e0c14f04ca1c987570acddd69e2ee",
        "86ddd8bd210ae7cefd795ee5093a2f4e98b0d9c5cdf0296d448b335ca3d97a9d"
      ]
    },
    "merge_verification": {
      "questions": 260,
      "identical_picks": 259,
      "agreement": 0.9961538461538462,
      "max_probability_delta": 0.02915129065513611,
      "clear_margin_flips": 0,
      "margin_threshold": 0.05,
      "max_delta_allowed": 0.05
    },
    "extra_optimizer_steps_during_recovery": 0,
    "initial_merge_failure": {
      "questions": 260,
      "identical_picks": 259,
      "agreement": 0.9961538461538462,
      "max_probability_delta": 0.08355367183685303,
      "clear_margin_flips": 0,
      "margin_threshold": 0.05,
      "max_delta_allowed": 0.05
    }
  }
}
```

## R3 merge history and repeatability

The initial full-epoch R3 job failed the unchanged 0.05 gate at probability shift
0.0835537, with 259/260 picks and zero confident flips. Lead message
`msg_c4892b870ed6` authorized private scoring. Export-only recovery from the
saved step-3155 checkpoint added zero optimizer steps and passed the same gate
at 0.0291513. The original failure was not reproduced and no merge algorithm changed.

Two independent Mac CPU fp32 exports match each other and all four published
shard hashes. CPU checks of the recovered highest-shift invoice213/discrepancy_severity
and flipped security44/urgency cases give shifts 0.00272 and 0.00297. Their
adapter/merged CPU predictions repeat bit-identically with seeds 17 and 29.
Calibration selection is fixed and eval disables dropout; a seed is not an established
CUDA fix. The initial failed artifact hashes and per-question probabilities were not
retained, so export nondeterminism and its original peak identity are not proven.
CUDA verification/inference variance remains a follow-up, with no extra GPU job launched.
See docs/item36-merge-followup.md in the PR and private repeatability evidence.

## Private immutable artifacts

```json
{
  "results": "870541621d1b76efe0c869133d94ef7a87690a7c",
  "a3_data": "bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed",
  "r3_data": "7430dae8b94337f3f9561ca98ecf7c6fc63dbecf",
  "v2_results": "907795806437388db31e947a5b72b8cfc6860ded",
  "r3_final_checkpoint": "8578df33929a7a50d3111df0d47afec1e8803c9d",
  "n3_results": {
    "repository": "jbrashear/jebadiah-9b-v2-1-index-results",
    "revision": "a374a3d0ec8b7c8fe011cbce9fb97d038c6711c5",
    "path": "runs/jebadiah-9b-v2-1-n3-fd54b835-proxy-0.3-10pct/results.jsonl.gz"
  },
  "n3_model": {
    "repository": "frontier-infra/jebadiah-9b-v2-1-n3",
    "revision": "fd54b8359c1d71d44439b10737ff3d7de7cd3b28"
  },
  "training_code": "66ae7af481eddadcf1afae8d505e32f53f1c507d",
  "recovery_code": "3483a58c1af5a08d618c681af3ca3799ba581a86",
  "proxy_code": "587f4f496b9cf1d83f0bc8e495379496f65afa2f",
  "a3_model": {
    "repository": "frontier-infra/jebadiah-9b-v2-1-a3",
    "revision": "9e69926a007dd636e82d33485dcd48f9751c4248"
  },
  "frozen_manifest_sha256": "74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405",
  "r3_results": {
    "repository": "jbrashear/jebadiah-9b-v2-1-index-results",
    "revision": "519cc4a0bf85b18eee20a5a2aeaf7591c5eafbcc",
    "path": "runs/jebadiah-9b-v2-1-r3-2f05d67e-proxy-0.3-10pct/results.jsonl.gz"
  },
  "r3_model": {
    "repository": "frontier-infra/jebadiah-9b-v2-1-r3",
    "revision": "2f05d67e5672775602bcf701156ddc5137e45381"
  }
}
```

## Spend

| Job | Stage | Runtime seconds | Estimated USD |
|---|---|---:|---:|
| [n3](https://huggingface.co/jobs/jbrashear/6ac82b48095c5780892fff24) | COMPLETED | 3112 | 2.3772 |
| [r3](https://huggingface.co/jobs/jbrashear/6ac82b48fee2c900701710ee) | ERROR | 3280 | 2.5056 |
| [n3-proxy](https://huggingface.co/jobs/jbrashear/6ac83816fee2c90070171812) | COMPLETED | 1123 | 0.8578 |
| [r3-export](https://huggingface.co/jobs/jbrashear/6ac8394efee2c900701718fc) | COMPLETED | 778 | 0.5943 |
| [r3-proxy](https://huggingface.co/jobs/jbrashear/6ac83d75fee2c90070171c11) | COMPLETED | 1124 | 0.8586 |

Estimated total compute: **$7.1935**, under the shared $20 cap.
Estimates use $2.75/hour and reported running durations; final billing can differ.
The failed R3 stage and export-only recovery are included. All five job timeouts
sum to 435 minutes, a $19.9375 allocation ceiling below the $20 cap. No teacher
inference or full-suite run was launched.

## Limits and verification

Intervals are conditional on this frozen proxy and describe evaluation variation,
not full-suite uncertainty or variation across training seeds. Adding R3 data also
adds full-epoch optimizer updates, as in A3. No causally separate claim is made about
content versus exposure. These private models are never submitted.

Local hardware: 24 tests and 117 subtests pass, composition and configuration checks
pass, compilation and whitespace checks pass, and the unchanged scorer reproduces
both pinned baselines. No GitHub-hosted CI was run. Final review and merge remain
with the lead. Local item36 scratch downloads and generated data are deleted after
durable private evidence upload, including the pinned base and both locally generated
R3 merged-model copies. The durable shared protected index is retained.
