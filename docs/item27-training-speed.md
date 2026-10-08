# Jeb v2.1 training throughput on RTX PRO 6000

The original 1.37 questions/s measurement described early steps, rather than sustained throughput. Read-only logs from the untouched production job `jbrashear/6ac7b1b0df2184ac91ad1e02` reached step 5,030 after 2:40:17, about 4.18 questions/s including periodic checkpoint uploads. Independent 150-step production-Trainer benchmarks measured 4.317 and 4.318 questions/s after warmup. An epoch estimate based on the warmed baseline is about 11.3 hours of optimizer work, plus setup, evaluation, uploads, merge and calibration.

The measurements use `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, dataset `frontier-infra/jebadiah-data-v2-1@a8e69f2dcab2ead8b164259d0c69102c5cc50e41`, rank 64 all-linear LoRA, dropout 0.05, 4,096-token prompt budget, seed 17, bf16 base weights, SDPA and eight questions per optimizer step. All tests used new private checkpoint repositories and no resume. The production run was never modified.

The final paired warmed comparison measured **4.262 to 7.254 questions/s, a 1.70x speedup**. The requested 3 to 5x sustained gain was not established. Grouping and adaptive checkpointing alone reached 6.299 questions/s with substantially more memory headroom.

## What costs time

1. **The decision backbone bypasses CUDA autocast.** `option_logits` calls `core.model` directly, while Accelerate wraps the outer PEFT model's `forward`. Thus setting Trainer `bf16=True` leaves PEFT's fp32 LoRA projection operations outside that CUDA autocast wrapper. The optional fix scopes bf16 autocast to the backbone, retains fp32 LoRA master weights and gradients, and explicitly disables autocast around the fp32 candidate head. In the step-22 traces, the grouped checkpointed recipe still issued 1,948 fp32 SIMT GEMM kernels; the autocast recipe issued zero. This confirms that the intended mixed-precision projection path actually executes. Casting inputs with `.float()` alone does not protect `einsum` from an outer autocast context.
2. **Short forwards pay many launches, and checkpointing repeats them.** An optimizer step with two accumulated microbatches invokes 96 FLA forwards and 48 FLA backwards across the 24 DeltaNet layers. A profiled cold-shape step contained 62,872 CUDA kernel launches, including 10,418 forward and 8,457 backward L2-normalization launches. These inflated counts include FLA autotuning, rather than a torch reference loop. After grouping and rounding, the corresponding tuned trace contained 16,319 launches, with 96 forward and 48 backward L2-normalization launches. The traces have different prompt lengths and tuning state, so their durations are diagnostic evidence rather than a paired performance ratio.
3. **Random batches multiply token work.** In 8,192 sampled questions the mean prompt length is 330, median 192, p90 732, p99 1,576, and maximum 3,895. Random microbatch 4 wastes 51.3% of token positions; grouping windows of 400 reduces that to 3.6% in the same length sample. At microbatch 8 the corresponding figures are 63.7% and 8.7%. Padding to 64 adds some positions but reduces shape variation. In the production benchmark, grouped and rounded batch 8 used 18.4% padding versus 51.5% for random batch 4.
4. **Checkpointing cannot simply be disabled.** Full-dataset grouped batch 8 without checkpointing exhausted the 96 GB device on its first, longest batch. The allocator reported 91.51 GiB allocated and 93.54 GiB total use before another 1.45 GiB allocation failed. Short prompts leave headroom, which the optional length threshold uses; long batches retain non-reentrant checkpointing and its RNG preservation.

## Kernels and other hypotheses

Live inspection of the Transformers wrapper selected `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` and `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn` on capability `(12, 0)`. The trace contains `ChunkGatedDeltaRuleFunction` and its backward, plus FLA CUDA kernels. FLA 0.5.2, Torch 2.11.0+cu128 and Triton 3.7.1 execute forward and backward on this Blackwell GPU. A torch DeltaNet fallback is ruled out for these runs. Preflight now checks the selected wrapper implementation and fails if it is the torch reference, in addition to running backward.

The convolution kernels accounted for 0.025 seconds of device activity in the profiled cold step, while other work and tuning dominated. This is consistent with causal-conv1d installation giving little sustained improvement. Profiling overhead entries such as `Command Buffer Full` are not GPU compute totals and must not be added to kernel durations.

Rendering and tokenizing 8,192 questions took 8.52 seconds on the rented host. Production collation took 1.4 to 1.9 seconds across 1,200 questions. It cannot explain the baseline slowdown. The dataset already flattens each record into questions, and the collator forwards a microbatch of questions together. Concatenating separate question prompts into a single causal sequence would change attention and DeltaNet recurrence across questions, so this change uses independent padded rows. It preserves the prompt, option order rules, supervision and output schema.

Periodic uploads affect wall time, but ordinary warmed optimizer steps were already slow between uploads. Benchmark timing measures optimizer-step work including finite guards and gradient clipping, excludes staging, profiler export and checkpoint upload, and discards the first 50 of 150 steps. Every reported primary comparison uses the production Trainer and an fp32 candidate head. Each variant measures 800 questions across the final 100 steps. The two grouped variants consume identical questions in identical order; random baseline sampling differs, with 385,698 real tokens over 1,200 questions versus 417,167 for grouping.

## Measurements

The committed [measurements](../results/runs/item27-speed/measurements.json) preserve step times and recipes. Raw profiler traces remain in the corresponding private checkpoint repositories as `item27-step.trace.json`.

| Run and recipe | Warmed questions/s | Peak allocated GB | Gain over warmed baseline |
| --- | ---: | ---: | ---: |
| r03 random batch 4, accumulation 2, checkpoint all | 4.317 | 31.32 | 1.00x |
| r03 batch 8, accumulation 1, grouping, pad 64, checkpoint all | 5.660 | 44.23 | 1.31x |
| r04 same batching, checkpoint at padded length >=512 | 6.299 | 66.56 | 1.46x |
| r05 fresh paired random baseline | 4.262 | 31.28 | 1.00x |
| r05 grouping, pad 64, checkpoint >=1024, backbone autocast | 7.254 | 100.38 | 1.70x |

The r05 peak is 93.49 GiB allocated on a device reporting 94.97 GiB capacity, so this is a short-run fit with little reserve. It is not a comfortable full-epoch memory recommendation. The r04 threshold of 512 retains more headroom; using that threshold with autocast is conservative but its combined throughput was not measured separately.

Five one-hour GPU caps bounded the maximum to $13.75. Completed GPU runtime totals approximately $5.53 at $2.75/h, before any provider billing rounding. No jobs were launched after r05.

## Numerical validation

The [150-step loss curves](../results/runs/item27-speed/loss-curves.png) show mean candidate loss 0.87875 for r04 and 0.89576 for r05, a +0.01701 shift. Individual 10-step block differences range from -0.1251 to +0.1710. Trainer/dropout seed 17 and grouped question order match, but the scratch harness initialized LoRA before Trainer seeded it, so cross-job comparisons also include initialization variation. These curves do not isolate the precision change or prove full-epoch quality. The archived [r05 harness](../results/runs/item27-speed/benchmark-r05.py) records exactly what ran.

On the M3 Ultra CPU, the r04 adapter and bf16 merge preserved 64/64 calibration picks, had zero clear-margin flips, and maximum probability shift 0.00543. The inference check loads one model at a time under the shared model lock, using fp32 arithmetic on the stored base/merged weights. This is a limited calibration probe rather than a full validation suite.

## Fresh-run controls

`--group-by-length` measures canonical rendered lengths once and shuffles sorted windows through Transformers' sampler. Choice permutations still happen in the collator. `--pad-to-multiple-of 64` changes masked padding only. `--checkpoint-min-tokens` retains activations for shorter batches and checkpoints longer ones without repeatedly registering input-gradient hooks. `--backbone-autocast` enables CUDA mixed precision for direct backbone calls. All new controls are opt-in, recorded in provenance, and checked by resume validation. A changed recipe requires a fresh output directory, a new checkpoint repository and `--resume none`.

The batching/checkpoint recipe and autocast fix are separate commits, so either can be reviewed and reverted independently. No GitHub-hosted workflow was added. The local suite verifies token/target preservation, fp32 candidate logits under outer autocast, effective-batch equivalence, exact grouped optimizer/RNG resume, and checkpoint switching with dropout.

## H100 and H200 economics

Brev searches on 2026-10-08 used case-insensitive filters and price sorting. The cheapest single H100 was Scaleway through Shadeform at $3.96/h; Nebius H100 SXM was $5.40/h. Nebius H200 SXM was $6.48/h. Disk charges can add to these prices. RTX PRO 6000 in these HF Jobs costs $2.75/h.

H100 at $3.96/h must sustain more than 1.44 times the selected RTX recipe's throughput to be cheaper per question. H200 at $6.48/h needs more than 2.36 times its throughput. Neither was benchmarked here. Once the host/precision issues are fixed, Hopper's memory bandwidth may help, but a cheaper-per-question claim needs a matched run. Changing GPU alone does not remove the bypassed autocast or per-forward host work.

## Launch pattern

For the batching/checkpoint recipe, use a new `JEB_ROOT` and new private destinations with:

```bash
JEB_ROOT=/workspace/jeb-item27-new-run bash scripts/v21.sh \
  --sizes 9b \
  --dataset-id frontier-infra/jebadiah-data-v2-1 \
  --dataset-revision a8e69f2dcab2ead8b164259d0c69102c5cc50e41 \
  --model-repo 'frontier-infra/jebadiah-{size}-item27-new-run' \
  --checkpoint-repo 'frontier-infra/jebadiah-{size}-item27-new-run-checkpoints' \
  --microbatch 8 --accumulation 1 --group-by-length \
  --pad-to-multiple-of 64 --checkpoint-min-tokens 512 \
  --save-steps 250 --resume none
```

Choose fresh names for every recipe. Adding `--backbone-autocast` changes projection numerics while keeping the candidate head fp32; its short-run numerical validation is recorded with the measurements. Memory thresholds are specific to this 9B/rank-64 recipe and do not constitute a 27B fit claim.
