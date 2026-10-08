# Jeb v2.1 rented-box trainer

The lead launches training. This change rents no hardware and starts no production training.

From a checkout on Ubuntu with an NVIDIA driver, supply `HF_TOKEN` from the approved Bitwarden write field through the environment, then run:

```bash
JEB_ROOT=/workspace/jeb-v21 bash scripts/v21.sh \
  --dataset-id frontier-infra/jebadiah-data-v2-1 \
  --dataset-revision <immutable-dataset-commit> \
  --model-repo 'frontier-infra/jebadiah-{size}-v2-1' \
  --checkpoint-repo 'frontier-infra/jebadiah-{size}-v2-1-checkpoints' \
  --save-steps 100
```

Both sizes run sequentially, 9B first. `scripts/setup.sh --v21` is an alias. The same command works inside an HF Jobs RTX PRO 6000 container with a writable `/workspace`; Job creation and spend approval belong to the lead. `--sizes 9b` or `--sizes 27b` selects one run. Repository flags require `{size}` when selecting both sizes. An existing public destination is refused before any upload; all newly created model and checkpoint repositories are private.

Defaults: LoRA only, rank 64, alpha 128, dropout 0.05, all-linear including Gated DeltaNet projections, maximum prompt length 4,096, one epoch, LR 1e-4, microbatch 1 and accumulation 8, effective batch 8. Override with `--rank`, `--alpha`, `--max-seq-length`, `--epochs`, `--lr`, `--microbatch`, `--accumulation`, and `--save-steps`. Unsupported training methods fail explicitly. The last-position fp32 candidate-logit objective, ordinal score supervision, soft choice/noul targets, and label alphabet are preserved. State truncation cannot silently exceed the sequence cap; an oversized question/rubric fails.

Setup installs Python 3.12, Torch 2.11.0+cu128, and the resolved CUDA dependency set. Ubuntu images without Python 3.12 use pinned uv 0.8.4. Torch's metadata pins Triton 3.6.0; we deliberately install Triton 3.7.1 without dependency resolution, preserving the historical DeltaNet backward workaround. Consequently `pip check` reports that known metadata conflict. Setup does not silently substitute latest packages. A bf16 CUDA matmul and two-step Qwen3.5 LoRA forward/backward smoke must pass before large-model training. FLA import is mandatory; causal-conv1d is not required, and its reference convolution can be slower. This exact CUDA combination has not been executed in this PR on a rented GPU.

The two starting chat checkpoints are immutable:

- `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Live tokenizer validation matched both saved prompt contracts: 68 ordinary single-token labels and 588 extended labels; the 9B/27B template hashes are `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` and `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`. The production trainer enforces those contracts before weight loading. Both checkpoints are downloaded before training starts. `--setup-only` installs and stages both checkpoints and the dataset, then stops. Budget at least 250 GB of writable disk and substantial host RAM for CPU shardwise merge. Downloaded weights total about 75 GB, with another roughly 75 GB for both exports, plus environments, optimizer checkpoints, and cache overhead.

## Dataset contract

The Hub dataset must be private. Its commit is resolved once and recorded in `data_provenance.json`; pass an immutable revision for a reproducible launch/relaunch. Default files are `train.jsonl`, `calib.jsonl`, and `manifest.json`; override with `--train-file`, `--calib-file`, and `--manifest-file`.

Each JSONL record has `id`, `family_id`, `state`, `questions`, and `label`, plus optional per-question `target` distributions. Both splits must contain choice, noul, and score questions. Family IDs and record IDs must be disjoint across splits. Invalid labels, supervision inside questions, invalid target distributions, empty splits, or manifest mismatch fail before loading weights. The trainer never reads `holdout.jsonl`.

```json
{
  "files": {
    "train.jsonl": {"sha256": "...", "rows": 100, "questions": 150},
    "calib.jsonl": {"sha256": "...", "rows": 20, "questions": 30}
  }
}
```

## Recovery and export

Every N optimizer steps, Trainer saves adapter weights, optimizer, scheduler, trainer state, and RNG state. A synchronous atomic private Hub commit includes the whole checkpoint and a completion marker. An upload failure stops training; it cannot masquerade as a successful remote save. The last two complete checkpoints remain visible in both local storage and the repository's current tree; Hub history can retain earlier blobs. At rank 64, the 27B adapter plus optimizer is about 5.6 GB per checkpoint, so upload bandwidth affects wall time.

Rerun the same command after interruption. `--resume auto` prefers a complete local checkpoint, then the latest complete remote one. On a replacement box, `--resume hub` explicitly selects remote recovery. Model identity, data hashes, LoRA settings, optimizer recipe, epoch/step target, and precision must match. Recovery allows changed snapshot and output paths. Option-order randomness uses Trainer's saved RNG state. A recipe change needs a new output directory and checkpoint repository.

After training, CPU shardwise merge computes `W + (alpha/r) BA` in fp32 and stores floating weights in bf16, preserving the original multimodal checkpoint names/config. Every adapter pair must match exactly once. Adapter versus merged decisions are compared on up to 260 calibration questions, requiring at least 99% pick agreement. Calibration then runs on the merged model: choice/noul fit training targets, score fits hard labels, and both fit variants are recorded.

The private model repository contains the original weight layout and tokenizer, `temperatures.json`, `prompt_contract.json`, `training_provenance.json`, merge evidence, and:

```text
scripts/
  ainode_prompt_verbatim.py
  jebadiah_prompt.py
  jebadiah_model.py
```

This matches the merged-model inference payload used by `frontier-infra/jebadiah-9b-v2`. No AINode install is needed. The vendored renderer is proved against 852 independent frozen message renders, regenerated by AST extraction from pinned AINode commit `e5c089386e0239c9eb270eeb490d181722b8da5b` using the exact hash-verified legacy train/calibration samples. The source checkout is needed only by `train/make_prompt_fixtures.py`, not by training or tests. These fixtures are regression evidence, not v2.1 training data; their source/license provenance remains the v1 dataset manifests and converter records under `data/manifests/`.

## Memory and runtime expectations

At rank 64, the 27B adapter has approximately 466.9 million parameters, four times the published rank-16 count. PEFT's fp32 adapter weights, gradients, and two Adam moments cost about 7.47 GB in total. The frozen bf16 starting weights account for roughly 54 to 56 GB. With microbatch 1, non-reentrant gradient checkpointing, SDPA, and 4,096-token examples, budget roughly **70 to 85 GB allocated**, with allocator reserves, kernels, and driver memory possibly pushing occupancy toward **90 GB**. A 96 GB RTX PRO 6000 is plausible, but this is an estimate, not a measured fit. Effective batch 8 does not allocate eight simultaneous sequences because accumulation is sequential.

The readiness plan's RTX PRO 6000 rank-16 estimates were 4 to 7 hours for 9B training and 9 to 14 hours for 27B training. Rank 64, longer examples, microbatch 1, and checkpoint uploads change throughput. Allow a preliminary 6 to 12 hours for 9B and 15 to 30 hours for 27B, plus staging, merge, and calibration; replace those ranges after representative warmed steps on the actual box. The $350 cap is an operator limit, not a rental or budget enforcement feature in this script.

## Studio verification

On the Studio, use Python 3.12 with Torch and the core dependencies:

```bash
python train/tiny_smoke.py --root /tmp/item26-smoke
OMP_NUM_THREADS=2 python train/v21_pipeline.py \
  --root /tmp/item26-smoke/run --sizes 9b \
  --base /tmp/item26-smoke/base --data-dir /tmp/item26-smoke/data \
  --device cpu --no-upload --microbatch 2 --accumulation 2 --save-steps 3
python -m pytest -q train/test_prompt_identity.py train/test_v21.py
```

The live CPU run trained all 50 oracle-labeled rows for 13 optimizer steps, using rank 64/alpha 128 and all 12 projection leaves including DeltaNet, then merged to bf16, preserved 12/12 calibration picks, and fitted all three temperatures. Maximum observed probability difference after merge was about 0.0002865. No model quality claim follows from this tiny test.

A second live run downloaded the synthetic splits from private `frontier-infra/item26-v21-smoke-data`, saved checkpoints and uploaded the merged payload privately to `frontier-infra/item26-v21-smoke`. Recovery from `frontier-infra/item26-v21-smoke-checkpoints` into a fresh directory with a different base-model path restored step 13 and produced the identical final adapter SHA256 `0721f43de8fad3b598dbce3997d112ce6d02e53859dacfbb4cc7821060a37b7b`. Credentials were read only from Bitwarden into child-process environments and never written to a token file. The unit suite also checks exact interrupted optimizer/RNG continuation, accumulation equivalence, private-repository refusal, complete merge coverage, sequence limits, and method validation. All checks ran on local hardware; no GitHub-hosted workflow was added.

The final entry-script run also retained only checkpoints 12 and 13 in the private repository current tree. A live recovery from the earlier checkpoint 12 trained the remaining step and reproduced that exact adapter hash, so recovery was tested with unfinished work as well as completed runs. The current unit suite has 13 passing tests.
