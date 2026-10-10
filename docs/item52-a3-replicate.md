# ITEM 52: A3 repeatability and seed-18 controls

The exact seed-17 repeat scores 46.97, with paired difference +0.00 [+0.00, +0.00] against original A3. All four served BF16 weight shards match original A3 SHA256s, refitted temperature values match, and all 11,079 scored answer maps match exactly. It loses neither Tools nor CLINC. The adapter checkpoint file SHA256 differs, while the merged served weights are identical.

The approved seed-18 control scores 46.66, with paired difference -0.31 [-1.77, +1.06] against A3 seed 17. Seed 17 to 18 is the only intended recipe change. The original seed-17 result is reproducible under the same seed; the different-seed control measures the observed spread under the recorded HF host assignments.

| Model | Proxy | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|---:|
| Original A3, seed 17 | 46.97 | 31.64 | 50.32 | 50.29 | 67.73 | 33.40 |
| A3 repeat, seed 17 | 46.97 | 31.64 | 50.32 | 50.29 | 67.73 | 33.40 |
| A3, seed 18 | 46.66 | 30.97 | 53.23 | 50.84 | 62.29 | 33.33 |
| Rung 1 | 46.09 | 30.57 | 52.77 | 48.66 | 63.48 | 32.01 |
| Rung 2 | 45.95 | 30.49 | 51.25 | 50.52 | 61.71 | 34.28 |
| Rung 2b | 46.68 | 31.53 | 56.04 | 46.23 | 63.19 | 32.31 |

All area values are official chance-corrected skill points. Deltas and intervals use unrounded scores.

| Rung | Delta vs seed-17 repeat, paired 95% interval | Delta vs seed-18 A3, paired 95% interval | In observed two-seed headline range? |
|---|---:|---:|---|
| Rung 1 | -0.88 [-2.21, +0.38] | -0.57 [-1.56, +0.61] | below |
| Rung 2 | -1.02 [-2.49, +0.28] | -0.71 [-2.34, +0.90] | below |
| Rung 2b | -0.30 [-1.78, +0.99] | +0.01 [-1.22, +1.23] | within |

The observed A3 two-seed range is [46.66, 46.97], width 0.31 points. It is a descriptive range from two seeds, not a training-variance confidence interval. The seed-17 repeat is not a third independent seed. None of the reported rung comparisons beats either measured A3 seed beyond paired proxy noise; the lead subsequently adopted their average, 46.82, as the 9B keep baseline.

| Area | Seed 18 minus seed 17 | Paired 95% interval |
|---|---:|---|
| Knowledge | -0.66 | [-3.74, +2.20] |
| Language | +2.91 | [+0.24, +5.39] |
| Retrieval | +0.55 | [-1.68, +2.86] |
| Tools | -5.44 | [-10.70, -1.85] |
| Arts | -0.07 | [-2.70, +2.40] |

| Model | CLINC macro-F1 / skill | CLadder accuracy / skill | POP909 cluster macro accuracy / skill |
|---|---:|---:|---:|
| Original A3, seed 17 | 76.82 / 76.68 | 65.80 / 31.60 | 9.79 / 9.08 |
| A3 repeat, seed 17 | 76.82 / 76.68 | 65.80 / 31.60 | 9.79 / 9.08 |
| A3, seed 18 | 74.35 / 74.20 | 64.60 / 29.20 | 8.20 / 7.48 |
| Rung 1 | 61.53 / 61.30 | 64.00 / 28.00 | 12.43 / 11.75 |
| Rung 2 | 66.82 / 66.62 | 64.40 / 28.80 | 8.73 / 8.02 |
| Rung 2b | 58.56 / 58.31 | 63.20 / 26.40 | 6.35 / 5.62 |

Tools illustrates why the overall score hides seed sensitivity: seed 18 reproduces a -5.44-point Tools loss without any data change. Rung 1 and rung 2b Tools scores are inside the observed two-seed Tools range [62.29, 67.73]; rung 2 is slightly below it at 61.71. All three Tools paired intervals against seed 18 cross zero. This is consistent with a material seed/order component to the original Tools losses, but two seeds and different assigned drivers/kernels do not establish its cause or variance.

| Rung | Tools delta vs seed 18, paired 95% interval | CLINC macro-F1 delta vs seed 17 | CLINC macro-F1 delta vs seed 18 |
|---|---:|---:|---:|
| Rung 1 | +1.20 [-0.43, +3.06] | -15.28 | -12.82 |
| Rung 2 | -0.58 [-6.70, +5.66] | -10.00 | -7.54 |
| Rung 2b | +0.90 [-0.64, +2.54] | -18.26 | -15.80 |

All rungs remain below both observed A3 seeds on CLINC, whose control range is [74.35, 76.82] macro-F1 percent. Their CLINC losses exceed this observed spread: -12.82, -7.54 and -15.80 percentage points against seed 18. The two-seed result therefore weakens attribution of Tools loss to the added data, while it does not explain away the much larger CLINC losses. The lead adopted the two-seed average as the subsequent 9B keep baseline, with about 6 Tools points and 2.5 CLINC macro-F1 points as operational review allowances. Two runs do not establish causal effects.

Seed 18's Tools delta is -5.44 skill points and its CLINC delta is -2.46 macro-F1 percentage points. CLINC150+OOS has 550 requests, CLadder 500, and POP909 200. POP909 uses the official cluster macro accuracy for its index raw value. Both controls have all 11,079 proxy requests successful and zero whole-case failures.

All intervals use the unchanged official scorer and 2,000 paired complete catalog/group draws within benchmark/domain/track strata, seed 20261008. They are conditional on the frozen proxy and do not include training-seed uncertainty. The manifest ID-list SHA256 is 74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405; suite revision is e57106b5e0698e74bd1a88b3b4c19b94a0dc8328. Existing rung results were reused with no new GPU inference.

Both controls use byte-identical a3/ data from frontier-infra/jebadiah-data-v2-1-item32 at bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed, Qwen3.5-9B parent c202236235762e1c871ad0ccb60c8ee5ba337b9a, and original runtime commit 86a8203d792463536eb009a4d8da6c8681ffdd99. Rank 16 / alpha 32, dropout 0.05, max length 2048, LR 1e-4, one epoch, ordinal targets, item27 speed flags and autocast off remain fixed. The seed-18 runtime guard changes exactly one byte in the seed literal; checkpoints prove the sole recipe difference is seed 18. Both completed exactly 2,649 optimizer steps and one epoch. Each control was logged and committed before launch; the lead approved the additional $8 in msg_76e46223b4db. No causal diagnostic was run.

| Environment | Seed-17 repeat | Seed 18 |
|---|---|---|
| Python | 3.12.11 | 3.12.11 |
| torch | 2.11.0+cu128 | 2.11.0+cu128 |
| CUDA runtime | 12.8 | 12.8 |
| cuDNN numeric version | 91900 | 91900 |
| NVIDIA driver | 580.178.04 | 580.159.03 |
| Host platform | Linux-6.12.100-125.179.amzn2023.x86_64-x86_64-with-glibc2.35 | Linux-6.12.94-123.192.amzn2023.x86_64-x86_64-with-glibc2.35 |
| transformers | 5.17.0 | 5.17.0 |
| peft | 0.21.0 | 0.21.0 |
| accelerate | 1.15.0 | 1.15.0 |
| triton | 3.6.0 | 3.6.0 |
| flash-linear-attention | 0.5.2 | 0.5.2 |
| fla-core | 0.5.2 | 0.5.2 |
| causal-conv1d | 1.7.0 | 1.7.0 |
| tokenizers | 0.23.2 | 0.23.2 |
| safetensors | 0.8.0 | 0.8.0 |

Both training images use pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel, with the original dependency setup and nvcc 12.8.93. Final Triton is 3.6.0 after the setup initially requests 3.7.1 and then installs causal-conv1d; it matches the original requirements lock. Full installed inventories and freezes are retained. The original job retained Python 3.12.11 and torch 2.11.0+cu128, matching the repeat; its final installed freeze, driver and immutable image digest were not retained, so exact historical host/add-on versions cannot be asserted. HF assigned different drivers and host kernels to the two controls, so the observed seed spread includes those recorded host assignments. Proxy image/runner and inference code hashes match the original run, with torch 2.8.0+cu128, transformers 5.17.0 and Python 3.11.13; the repeat proxy kernel is 6.12.100 versus original 6.12.103.

Private models are frontier-infra/jebadiah-9b-v2-1-a3-rep1 at f1e2f66d3e2dce25596a9fc1b9d13fbfa72003da and frontier-infra/jebadiah-9b-v2-1-a3-seed18 at 2899e9d4c785ba0f845245c66c9c4e740efeeab9, with private -checkpoints repositories. Both unchanged merge gates require maximum probability shift <= 0.05 and zero confident flips. Seed-17 gate shift is 0.01013342; seed-18 is 0.01132429. Temperatures were refitted with the original A3 command in both controls.

| Job | ID | Running seconds | Estimated USD |
|---|---|---:|---:|
| Seed-17 training/export | 6ac99854095c57808930c4bf | 3018 | 2.3054 |
| Seed-17 proxy | 6ac9a47c095c57808930cd1f | 1126 | 0.8601 |
| Seed-18 training/export | 6ac9a555095c57808930cd95 | 3028 | 2.3131 |
| Seed-18 proxy | 6ac9b17c095c57808930d46f | 1124 | 0.8586 |

At $2.75/hour, seed-17 estimated compute is $3.1656 under its $8 cap, seed-18 is $3.1717 under the additional $8 cap, total $6.3372. Each control's 135-minute training and 35-minute proxy timeout allocation ceiling was $7.7917. These are duration-based compute estimates, not billing invoices. Results and provenance are retained in PR 29 and private results dataset jbrashear/jebadiah-9b-v2-1-index-results under analysis/item52-controls. All local scratch downloads, suite/data copies and analysis working files in /Volumes/PRO-G40/scratch/item52 are deleted after persistence. No merged model was downloaded to the Mac. Lead review/merge remain pending; no GitHub-hosted CI was run.
