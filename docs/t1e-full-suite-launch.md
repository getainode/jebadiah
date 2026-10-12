# T1e full-suite launch

Approved by the master on 2026-10-12 14:39, Desk #855. Private evaluation only.

- Results: `frontier-infra/jebadiah-t1e-decision-index-results`, created private.
- Suite: `jbrashear/decision-index-suite-0.3` at `e57106b5e0698e74bd1a88b3b4c19b94a0dc8328`, edition 0.3, all 140,620 requests.
- Kit and resumable wrapper: `jbrashear/jebadiah-9b-v2-1-index-results` code at `587f4f496b9cf1d83f0bc8e495379496f65afa2f`.
- Engine and renderer: `getainode/jebadiah` at `678f6e2cc2bb1b51f34407d00255b5e13109ce3f`.
- Options: `repeat_payload=true`, `reverse_avg_below=0.7`, `prefix_cache=true`, `max_batch=64`, `batch_tokens=32768`, `cuda_alloc_conf=expandable_segments:True`.
- Published model scripts match the v2.1 full-run script hashes. Temperatures match: 9B choice/noul/score 1.0467/1.0284/0.6805; 27B 1.0608/1.0405/0.6524. Default BF16 CUDA SDPA loading and flash-linear-attention installation preserved.
- Runtime: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, `rtx-pro-6000`, USD 2.75/hour.
- Launcher: `scripts/track1_full_job.sh`, `MODEL_SIZE=9b` or `27b`. Full runs have no LIMIT or ROWS override. Smoke has `LIMIT=200`, `SMOKE_SUFFIX=-smoke`.
- Setup reads immutable kit code from the original dataset, writes results only to the new private dataset, and invokes `python -m decision_index pipeline` after explicitly downloading and verifying the pinned full suite.
- Results retain synchronized per-request `total_wall_ms`, excluding model load. Measure 27B mean, p50, p95, p99, maximum, and fraction at or above 1,000 ms; an early sample cannot establish the full-suite latency result.

| Job | Model revision | Run name | Job ID | Timeout | Maximum cost |
|---|---|---|---|---|---|
| Smoke 9B | de19e80e18aa23420be410d54fdee435587eaad3 | jebadiah-9b-v2-1-t1e-678f6e2-v03-smoke | 6acd3801de32f9b0f4321ead | 15 minutes | USD 0.6875 |
| Full 9B | de19e80e18aa23420be410d54fdee435587eaad3 | jebadiah-9b-v2-1-t1e-678f6e2-v03 | 6acd38c3dd8177968b42f711 | 13 hours | USD 35.75 |
| Full 27B | e38b48b82bcda8943d0958072c042c5867284838 | jebadiah-27b-v2-1-t1e-678f6e2-v03 | 6acd38c4dd8177968b42f713 | 30 hours | USD 82.50 |

Hard cap: USD 120 total. The three HF job timeouts sum to 43.25 hours, at most **USD 118.9375**, including the smoke. No automatic replacement jobs. Wrapper pipeline ATTEMPTS=1; periodic private uploads every 120 seconds and final sync retained.

Dry check: bash syntax passed; exact pinned wrapper adaptation assertions passed; proxy invocation removed; full pipeline and frozen suite pin verified. Smoke completed successfully: 200/200 requests OK, 29.0 seconds inference, HF running duration 155 seconds (estimated USD 0.1184). Both full jobs confirmed RUNNING and past model load at 2026-10-12T19:48:27.393462+00:00. Private environment records confirm both T1e options, the same frozen corpus hash as the v2.1 full runs, and all three edition 0.3 row files.

Initial 27B latency sample: 180/180 requests OK; mean 293.020 ms, p50 260.349 ms, p95 268.467 ms, p99 281.196 ms, maximum 7,972.277 ms. One of 180 requests is at or above 1,000 ms. The sample mean meets the 1,000 ms target; the per-request maximum does not. Full-suite latency compliance remains unconfirmed.

Estimated compute cost at this observation: USD 0.4173, including smoke. This is elapsed running time multiplied by USD 2.75/hour, not an invoice. The timeout maximum remains USD 118.9375.
