# ITEM 51: additive Corr2Cause against A3

The only change is appending the exact 1,000 already-scanned Corr2Cause questions selected by rung2/replacement-slots.json, removing no A3 rows or questions. Every A3 train byte is preserved in order as the new train prefix, and the 512-question calibration file is byte-identical. Every appended raw JSONL row is hash-bound to rung2/train.jsonl at 2242e85fae6f8aff3fc3cea3fa95869d99132522. No new source rows or scan were introduced. Both existing Studio scans passed: 0/1,035,117 release records and 0/6,300 converted records. NOTICE-Corr2Cause.txt is preserved.

Private data: frontier-infra/jebadiah-data-v2-1-item47 at c9fcf494319527d12888e0f5854980c11a51b91d, folder rung2b/. The mix has 22,190 presentations, 1,102 aggregate Corr2Cause presentations (below the 2,219 cap), and 2,774 optimizer steps for one epoch with effective batch 8. The diagnostic is the original separate 300-question, 12-family causal holdout.

All A3 settings remain fixed: Qwen3.5-9B parent c202236235762e1c871ad0ccb60c8ee5ba337b9a, rank 16, alpha 32, dropout 0.05, 2,048 tokens, LR 1e-4, one epoch, seed 17, ordinal score targets, item27 speed flags, autocast off, and unchanged temperatures from A3 9e69926a007dd636e82d33485dcd48f9751c4248. The shared trainer, renderer and verifier are unchanged. The larger step count follows from the additive data.

The pre-launch run log keeps only if the frozen-proxy paired whole-group bootstrap 95% interval vs A3 (46.97) is entirely above zero and candidate score exceeds 44.67. Report all five areas with Tools emphasized, CLadder and CLINC, and causal diagnostic accuracy, balanced accuracy, valid precision/recall and strata against A3 and rung2. Diagnostic metrics are descriptive for this experiment. The merge gate remains maximum shift 0.05 and zero confident flips. An export-only recovery adds no training steps.

Private model/checkpoints: frontier-infra/jebadiah-9b-v2-1-rung2b and -checkpoints, never public. RTX PRO 6000 costs $2.75/hour. Hard timeouts: training 130 minutes, proxy 35 minutes, diagnostic 45 minutes, optional recovery 45 minutes; total allocation ceiling $11.6875. Live results and cleanup pending. Local hardware validation only; no GitHub-hosted CI.

Training job [6ac98474095c57808930bb39](https://huggingface.co/jobs/jbrashear/6ac98474095c57808930bb39) launched using code d7398c8 after the pre-launch log commit. [PR 28](https://github.com/getainode/jebadiah/pull/28) remains open for lead review.

Training/export completed in 3,279 seconds, with exactly 2,774 optimizer steps and epoch 1.0. The unchanged gate passed: 260 questions, 259 identical picks, maximum probability shift 0.0114752, zero confident flips. A3 recipe fields and temperature bytes match exactly; the notice SHA256 is unchanged. Private model revision is ec9b0cafd70a28cdce502d563f5e7d80a2ac12ab. Frozen proxy job 6ac9919d095c57808930c117 and three-model diagnostic job 6ac9919d095c57808930c116 are running concurrently.
