#!/bin/bash
# usage: run_bench.sh LABEL PORT   -- the 231 public items through JevBench's unchanged typesafe adapter
set -e
L="$1"; PORT="$2"; D="$(cd "$(dirname "$0")" && pwd)"; R="$D/runs/$L"
cd "$D/repo"
t0=$(date +%s)
python3 -m jevbench.cli run --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:$PORT --model "$L" --key-env '' \
  --price-in-per-m 0 --price-out-per-m 0 --cost-basis self_hosted_mac_studio_m3_ultra_compute_excluded \
  --results "$R/results.jsonl" --raw-dir "$R/raw" --ledger "$R/ledger.jsonl" --manifest "$R/manifest.json" --run-label "$L"
echo "wall_s $(( $(date +%s) - t0 ))"
python3 -m jevbench.cli summarize --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --results "$R/results.jsonl" --public-export "$R/summary.json" > /dev/null
echo "summary written"
