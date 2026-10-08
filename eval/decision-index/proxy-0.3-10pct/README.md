# Frozen edition 0.3 ten-percent proxy

This is a proxy, not a leaderboard score. Use the same `run-ids.txt` for every model and size. No benchmark payloads or answers are included here.

`sample.py` reads the verified edition 0.3 suite after exclusions and retains the 37 benchmarks contributing to the official index. It groups by `(catalog_id, group_id)`, stratifies by published metadata domain and evaluation track, and selects `ceil(groups / 10)` in every stratum. Groups crossing tracks retain all tracks together in a combined stratum. Ordering is SHA256 of UTF-8 `20261008:catalog_id:group_id`, with a group-ID tie break. Every question, field, and candidate chunk in each selected group is retained. Unequal group sizes and upward rounding make this approximately ten percent of requests.

The manifest contains 11,079 run IDs across 10,380 whole groups. `run-ids.sha256` hashes the exact UTF-8 list, one ID per line including the final newline. `manifest.json` includes source revision and verification hashes, selected group IDs, and per-benchmark counts.

```sh
python sample.py --suite /private/suite-0.3 --out /tmp/item31-refreeze \
  --suite-revision e57106b5e0698e74bd1a88b3b4c19b94a0dc8328
cmp run-ids.txt /tmp/item31-refreeze/run-ids.txt
python -m unittest discover -s . -p 'test_*.py'
```

Install the supplied Decision Index 0.3 kit or expose it on `PYTHONPATH`. `score.py` calls its unchanged scorer through a suite view containing only manifest rows. It validates the list hash, missing IDs, and complete groups. A full existing result file is valid input: outside-manifest results contribute nothing to scores. Proxy coverage and failures replace full-suite completion counts. Edition 0.3 defines five scored areas: knowledge, language, retrieval, tools, and arts.

```sh
python score.py --suite /private/suite-0.3 \
  --results /private/baseline/results.jsonl.gz --engine baseline \
  --out /private/item31-baseline-score
python bootstrap.py --suite /private/suite-0.3 \
  --baseline /private/baseline/results.jsonl.gz \
  --candidate /private/candidate/results.jsonl.gz \
  --replicates 2000 --out /private/item31-bootstrap.json
```

The bootstrap draws the same complete groups for both models, with replacement within benchmark/domain/track strata, and recomputes the official benchmark metrics and area aggregation each time. Draws receive distinct group identities while retaining native song/user metadata clusters, so the official macro metrics keep their published definitions. Its percentile 95% interval is conditional on this proxy, not a measure of full-suite sampling uncertainty. Missing manifest rows prevent a paired interval; failed rows retain the official failure and coverage treatment.

The supplied wrapper's `ROWS_IN_SUITE` expects full benchmark-row JSONL, not an ID list. `run.py` privately materializes the validated manifest rows before calling the unchanged kit runner and scorer. It forces compact results, verifies the pinned suite revision, rejects limit/rows overrides and public uploads, and removes temporary row text afterward. Results go to a private dataset.

`prepare_runner.py` verifies the supplied wrapper and engine hashes before making narrow changes: invoke this proxy pipeline, enable `hf_xet`, remove process-pattern killing, and make engine provenance identify the actual model override. Engine inference and benchmark scoring remain unchanged. Stage the supplied kit source archive and `mps_delta.py`, the prepared wrapper and engine, and this directory's `sample.py`, `score.py`, `run.py`, manifest and ID files under `code/proxy/` in the private results dataset.

`verify_candidate.py` checks all indexed merged-weight shards, temperatures, prompt contract and the three inference scripts at one model revision. `submit.sh` accepts pinned results-code and candidate-model revisions and submits exactly one RTX PRO 6000 job with `-- bash -lc`, `hf_xet`, and a 100-minute timeout. At $2.75/hour this caps allocation at $4.59, below the $5 item cap. Supply Bitwarden-sourced `HF_TOKEN` through the environment and Jobs secrets; never put credentials in files or commands.
