# Item 36 follow-up: merge-verification repeatability

R3's first full-epoch job failed the unchanged merge gate: maximum probability
shift 0.0835537, 259/260 identical picks, zero confident flips, limit 0.05.
The lead authorized private scoring and an export-only recovery from the saved
final step-3155 checkpoint. That recovery used the unchanged algorithm and
passed the same gate: maximum shift 0.0291513, 259/260 picks, zero confident
flips. No optimizer updates were added and no extra GPU diagnostic job was run.
Both outcomes remain in the private evidence and model metadata.

## Verified on our hardware

The existing `train/merge_export.py` already computes `B.float() @ A.float()`,
adds the delta to the fp32 base tensor, and casts the result to bf16 storage.
Two independent Mac Studio CPU exports have identical SHA256 hashes for all
four merged shards. All four also match the published recovery's shard hashes.
This proves repeatable CPU export in these checks; it does not establish that
both GPU verification runs used byte-identical failed-job artifacts.

Calibration is the first 260 questions in the unchanged A3 calibration file,
with deterministic file/dictionary order and no random subsampling.
`Scorer.__init__` calls `model.eval()`, and PEFT's non-trainable adapter load
also calls eval. Active LoRA dropout is therefore excluded by the executed
code path. The verification script does not set seeds or deterministic CUDA
algorithm/workspace controls.

Recovery diagnostics identify the highest-shift case as
`typed-decisions-train:tr_invoice_processing_000213/discrepancy_severity`, a
score question, with CUDA shift 0.0291513 and unchanged pick. Its one flipped
case is `typed-decisions-train:tr_security_incidents_000044/urgency`, also a
score question: CUDA shift 0.0141943 and original margin 0.0085878, below the
confident-flip threshold. Local CPU bf16-backbone/fp32-LoRA tests of those two
cases give shifts 0.0027238 and 0.0029749 respectively. These are targeted CPU
checks, not a complete 260-question CPU gate test. Both adapter and merged CPU predictions are bit-identical when those cases
are repeated with seeds 17 and 29. This excludes seed variation for these CPU
checks, not for untested CUDA behavior. An MPS load crashed before inference;
CPU checks completed successfully.

## What remains unresolved

The first failed job did not preserve merged-shard hashes, the final adapter
hash, or per-question probabilities. Its maximum-shift question cannot be
identified retrospectively from its aggregate report. The recovery's question
IDs are not proof of the original peak's identity. The initial 0.08355 result
was not reproduced; do not claim that a new fp32 algorithm fixed it, or that
export nondeterminism has been established.

Evidence narrows the problem to verification/inference numerical behavior or
an unrecorded difference in the failed artifact. CUDA fused DeltaNet/Triton,
GEMM/kernel selection, and fresh-process behavior remain candidates. A fixed
seed is not an established CUDA fix; CPU fp32 export is already repeatable.
The exact source needs a separately authorized CUDA repeatability investigation.

## Implementation-ready follow-up

Preserve adapter and merged-shard SHA256 hashes, rendered prompt hashes and
option order, package/driver/kernel provenance, and per-question before/after
probabilities on both pass and failure. Compare identical weights in the same
process and fresh processes with fixed and varied seeds. Then test deterministic
PyTorch/CUBLAS settings and isolate the fused DeltaNet path without changing
weights, calibration or the 0.05 gate. Accept the fix only after identical
artifacts produce stable verification results and the source of the earlier
variance is measured. This item records that work; it does not launch it.
