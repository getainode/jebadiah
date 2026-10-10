# ITEM 51: drop additive Corr2Cause rung 2b

**Drop rung 2b; retain A3.** The frozen proxy scores **46.68 vs A3 46.97**, difference **-0.30**, paired whole-group bootstrap 95% interval **[-1.78, +0.99]**. All 11,079 requests succeeded, with zero whole-case failures or rows outside the manifest. The score exceeds 44.67, but the interval fails the predeclared requirement to be entirely above zero.

Against replacement rung 2 (45.95), the difference is **+0.72**, interval **[-0.95, +2.35]**. This comparison is inconclusive. Tools loss persists with every A3 question retained: **-4.53** points versus A3, interval **[-9.67, -1.27]**. The comparison also includes the 125 extra optimizer steps that follow from additive data.

Both intervals use the unchanged official kit, 2,000 complete-group resamples within benchmark/domain/track strata, seed 20261008, and eight Studio CPU workers per comparison. Every frozen run ID and payload hash matches. Manifest SHA256 is `74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`. These intervals cover resampling variation within the fixed proxy for this one training seed.

All five areas use chance-adjusted skill points. Area intervals are descriptive; the overall interval determines keep/drop.

| Area | A3 | Rung 2 | Rung 2b | Difference vs A3 | Paired 95% interval vs A3 |
|---|---:|---:|---:|---:|---|
| Tools | 67.73 | 61.71 | 63.19 | -4.53 | [-9.67, -1.27] |
| Knowledge | 31.64 | 30.49 | 31.53 | -0.11 | [-3.06, +2.59] |
| Language | 50.32 | 51.25 | 56.04 | +5.72 | [+3.24, +8.52] |
| Retrieval | 50.29 | 50.52 | 46.23 | -4.06 | [-5.93, -1.98] |
| Arts | 33.40 | 34.28 | 32.31 | -1.09 | [-3.90, +1.60] |

Requested benchmark and Tools detail follows. Values are native percentages; CLINC uses macro-F1, ToolRet uses nDCG@10, and the remaining rows use their listed accuracy metric.

| Benchmark | Requests | Metric | A3 | Rung 2 | Rung 2b |
|---|---:|---|---:|---:|---:|
| BFCL | 170 | case exact accuracy | 96.47 | 94.71 | 94.12 |
| ToolRet | 85 | nDCG@10 | 62.27 | 60.71 | 62.85 |
| API-Bank | 51 | accuracy | 86.27 | 86.27 | 82.35 |
| CLINC150+OOS | 550 | macro-F1 | 76.82 | 66.82 | 58.56 |
| Home appliance simulator | 9 | case exact accuracy | 33.33 | 22.22 | 22.22 |
| CLadder | 500 | accuracy | 65.80 | 64.40 | 63.20 |
| When2Call MCQ | 367 | accuracy | 68.94 | 56.40 | 64.85 |

The graph-disjoint causal diagnostic has 300 questions, 29 necessarily-valid labels, and 12 families excluded from training and calibration. All three passes scored 300 questions with zero truncations, using unchanged A3 temperatures and full menus. The fresh A3 and rung 2 metrics exactly reproduce item 50. Only timing from the auxiliary report is serialized, and every completed pass is saved in the private checkpoint repository.

| Causal diagnostic metric | A3 | Rung 2 | Rung 2b |
|---|---:|---:|---:|
| Accuracy | 90.00% | 89.67% | 91.00% |
| Balanced Accuracy | 51.36% | 80.42% | 67.31% |
| Valid Precision | 33.33% | 47.62% | 55.00% |
| Valid Recall | 3.45% | 68.97% | 37.93% |

Rung 2b has TP/FP/FN/TN = 11/9/18/262 and 273 correct answers. The majority-label accuracy floor is 90.33%. Balanced accuracy improves by 15.95 percentage points versus A3 and falls by 13.12 versus rung 2. All type, relation-template, variable-count and graph-family counts and metrics are retained in [the diagnostic summary](../results/runs/item51/causal-diagnostic-summary.json); predictions remain private. This diagnostic improvement does not override the overall keep rule.

The only experimental change appends the exact 1,000 previously scanned Corr2Cause questions selected by rung2/replacement-slots.json. Every A3 train byte remains in order as the new train prefix, and all 512 calibration-question bytes are unchanged. Every added raw JSONL row is individually hash-bound to the pinned scanned rung2 train set; no new source rows or source scan were introduced. Existing scans passed with 0/1,035,117 release records and 0/6,300 converted records. Aggregate Corr2Cause exposure is 1,102, below the 2,219-question cap. NOTICE-Corr2Cause.txt is preserved in the dataset and both private model repositories.

Private data is `frontier-infra/jebadiah-data-v2-1-item47@c9fcf494319527d12888e0f5854980c11a51b91d`, folder `rung2b/`. Inputs are unchanged A3 `frontier-infra/jebadiah-data-v2-1-item32@bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed`, folder `a3/`, and scanned rung 2 at `2242e85fae6f8aff3fc3cea3fa95869d99132522`. The [data manifest](../results/runs/item51/data-manifest.json) records every file hash.

All A3 settings remain fixed: Qwen3.5-9B parent c202236235762e1c871ad0ccb60c8ee5ba337b9a, rank 16, alpha 32, dropout 0.05, length 2,048, LR 1e-4, one epoch, seed 17, ordinal score targets, effective batch eight, item27 speed flags, autocast off and unchanged temperatures from A3 9e69926a007dd636e82d33485dcd48f9751c4248. The shared trainer, renderer and verifier were unchanged. The pre-launch log was committed at d7398c8b113cdceb00d6070819447240e9a66c32 before job submission. Runtime provenance has no recipe differences from A3 apart from train identity and paths.

The final checkpoint proves 22,190 presentations, 2,774 optimizer steps and epoch 1.0. Training took 2,479.7 seconds with 64.81 GB peak allocated GPU memory and fp32 candidate logits. It reports the same 36 inherited truncated presentations as item 50; the added-source render checks and all diagnostic passes have zero truncations. The unchanged merge gate passed at maximum probability shift 0.0114752054, zero confident flips and 259/260 identical picks. No recovery or extra optimizer steps occurred. A3 temperature bytes have SHA256 48424adbce6d97faa73dc7a6201df939e4d0119d97b90c4ceadfd3e1578c741b, and the notice SHA256 is 5a2e4417d27849586ad9eefdf63f81b45100cec83baadc8dfbeb49054d367d38.

Private merged model is `frontier-infra/jebadiah-9b-v2-1-rung2b@ec9b0cafd70a28cdce502d563f5e7d80a2ac12ab`; checkpoints remain in its private `-checkpoints` repository. All evaluations loaded that immutable model successfully. A3 source-policy exceptions and private scope remain unchanged. No model was made public or submitted.

| Job | ID | Outcome | Running seconds | Estimated cost |
|---|---|---|---:|---:|
| training | [6ac98474095c57808930bb39](https://huggingface.co/jobs/jbrashear/6ac98474095c57808930bb39) | COMPLETED | 3279 | $2.5048 |
| proxy | [6ac9919d095c57808930c117](https://huggingface.co/jobs/jbrashear/6ac9919d095c57808930c117) | COMPLETED | 1123 | $0.8578 |
| diagnostic | [6ac9919d095c57808930c116](https://huggingface.co/jobs/jbrashear/6ac9919d095c57808930c116) | COMPLETED | 710 | $0.5424 |

**Total estimated compute: $3.9050, below the $12 cap.** RTX PRO 6000 pricing is $2.75/hour; estimates use running seconds. Initial timeout ceilings were 130 minutes training, 35 proxy, 45 diagnostic and 45 for one possible recovery, maximum $11.6875. There were no retries or recovery. Training/export took 54.65 minutes and proxy 18.72, matching item 50's measured pattern.

Paired scoring artifacts are private under `jbrashear/jebadiah-9b-v2-1-index-results@b9aadd6cf73e6317afc7491b2e0796426fb9ed50`, folder `analysis/item51-rung2b/`. [The receipt](../data/manifests/item51-results-receipt.json) records input revisions, hashes, five areas, benchmarks, diagnostics, job states and costs. The source kit is pinned at 587f4f496b9cf1d83f0bc8e495379496f65afa2f and suite at e57106b5e0698e74bd1a88b3b4c19b94a0dc8328.

Local Studio validation passed all three builder tests, including real pinned inputs and rejection of altered A3 bytes, script compilation, per-row scanned-set hashes, unchanged calibration and recipe checks, diagnostic oracle/majority metrics and actual baseline metric reproduction. No GitHub-hosted CI was used. [PR 28](https://github.com/getainode/jebadiah/pull/28) remains open for the lead; the worker does not merge it.

Rebuild with a fresh item51 scratch directory and the existing authenticated HF session:

```sh
TMPDIR=$(~/bin/pro-g40-scratch item51) || exit 1
export TMPDIR
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item51-a3"
hf download frontier-infra/jebadiah-data-v2-1-item47 --repo-type dataset --revision 2242e85fae6f8aff3fc3cea3fa95869d99132522 --include 'rung2/*' --local-dir "$TMPDIR/item51-scanned"
python3 data/item51_rung2b_mix.py --base "$TMPDIR/item51-a3/a3" --rung2 "$TMPDIR/item51-scanned/rung2" --out "$TMPDIR/item51-rung2b"
ITEM51_INPUT_ROOT="$TMPDIR" PYTHONPATH=data python3 -m unittest discover -s data -p test_item51_rung2b_mix.py -v
```

Cleanup deleted the entire `/Volumes/PRO-G40/scratch/item51` directory (248.5 MiB), including input datasets, rebuilt mix, frozen suite and kit, prediction downloads, environments, reports, logs and monitor scripts. No model weights were downloaded onto Studio. HF Jobs scratch is ephemeral. [Cleanup proof](../results/runs/item51/cleanup.json) records all deleted entries. Lead review and merge of PR 28 remain.
