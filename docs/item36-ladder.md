# Item 36: N3 and R3 against A3

Both runs are private diagnostics, never submitted. The accepted comparator is
A3 at `frontier-infra/jebadiah-9b-v2-1-a3@9e69926a`, frozen proxy 46.97.
Use the same frozen 11,079 requests and paired complete-group bootstrap as
item32: 2,000 draws, seed 20261008, official metrics recomputed per draw.
Keep each rung only when its paired 95% difference interval versus A3 is above
zero. Compare both with v2 as well; do not infer full-suite scores.

N3 changes rank/alpha from 16/32 to 64/128, maintaining ratio 2. All other
recipe fields and the pinned A3 dataset are unchanged. R3 appends item33's
4,050 training questions to A3, without changing rank or any other recipe
field. Its 450 skill calibration questions remain a separate diagnostic
holdout; original A3 temperature-fitting calibration is byte-identical.

The builder is regenerated from the pinned original human source and procedural
code. The full-suite protected index is local/private and must never be uploaded.
`data/item36_skill_mix.py` verifies original and regenerated split hashes,
rejects cross-source ID/family/normalized-state collisions, and preserves A3
training bytes/order before appending the regenerated skill bytes.

Both use the item32 trainer and launcher pattern, seed 17, one full epoch,
2,048 tokens, batch 8, LR 1e-4, item27 speed flags, and backbone autocast off.
Run both training allocations concurrently on HF Jobs RTX PRO 6000, then run
both frozen proxies concurrently. Each training timeout is 160 minutes, each
proxy timeout is 35 minutes: 390 total maximum allocation minutes at $2.75/hour
is $17.875, leaving $2.125 within the shared $20 cap. Any failure/retry counts
against that cap; no automatic extra allocation is permitted without checking
remaining spend. Record reported running durations and estimated spend.

## Reproduction

Large local files use `~/bin/pro-g40-scratch item36`. Supply `HF_TOKEN` only
from the approved Bitwarden write field. Regenerate item33 with the documented
builder, then compose:

```sh
python data/item36_skill_mix.py --base "$TMPDIR/item36-a3-data/a3" \
  --skills "$TMPDIR/item36-skills" --output "$TMPDIR/item36-release/r3"
python scripts/item36_launch.py n3 bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed
python scripts/item36_launch.py r3 R3_DATASET_COMMIT
python scripts/item36_proxy.py n3 N3_MODEL_COMMIT
python scripts/item36_proxy.py r3 R3_MODEL_COMMIT
```

The run log in `docs/run-rules.md` was written before launch. Final results,
intervals versus A3 and v2, five area deltas, item31 worst five, immutable pins,
epoch/merge/coverage evidence and spend will be recorded after both jobs finish.
