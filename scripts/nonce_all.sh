#!/bin/bash
# Runs on the training box after the sweep: the nonce robustness test on every adapter and both bases,
# on the four head sets only (Jevals x3 + Nimble 324), K=3. Finished outputs are skipped on relaunch.
# JEB_NONCE_RUNS=9b-v1,4b-v1 restricts it to those runs and skips the two base models.
set -uo pipefail
ROOT=${JEB_ROOT:-/workspace/jeb}
source "$ROOT/env.sh" && source "$ROOT/venv/bin/activate" && cd "$ROOT/train"
export PYTHONUNBUFFERED=1
ATTN=$(cat $ROOT/ATTN 2>/dev/null || echo sdpa)
SETS="jevals-pubmedqa jevals-banking77 jevals-helpsteer2 nimble-eval"
ONLY=${JEB_NONCE_RUNS:-}
wanted() {  # run name -> 0 when it should run
  [[ -z $ONLY ]] && return 0
  local r; IFS=',' read -r -a r <<< "$ONLY"
  for w in "${r[@]}"; do [[ $1 == "$w" ]] && return 0; done
  return 1
}
run_one() {  # name base rev adapter
  local RUN=$1 BASE=$2 REV=$3 AD=$4
  mkdir -p $ROOT/runs/$RUN/nonce
  for SET in $SETS; do
    OUT=$ROOT/runs/$RUN/nonce/$SET.json
    [[ -f $OUT ]] && continue
    [[ -f $ROOT/data/test/$SET.jsonl ]] || { echo "no set $SET"; continue; }
    echo "== $RUN $SET $(date -u +%FT%TZ)"
    python nonce_eval.py --base "$BASE" --revision "$REV" --adapter "$AD" \
      --set $ROOT/data/test/$SET.jsonl --out "$OUT" --repeats 3 --batch 16 --attn "$ATTN" --seed 20260922 \
      2>&1 | grep --line-buffered -v -E "^(Loading weights|Fetching)|UserWarning|FutureWarning|warnings.warn"
  done
}
for RUN in $(ls $ROOT/runs); do
  [[ $RUN == smoke ]] && continue
  wanted "$RUN" || continue
  CFG=$ROOT/runs/$RUN/config.json
  [[ -f $CFG && -f $ROOT/runs/$RUN/adapter/adapter_config.json ]] || continue
  BASE=$(python -c "import json; print(json.load(open('$CFG'))['base_model'])")
  REV=$(python -c "import json; print(json.load(open('$CFG'))['base_revision'])")
  run_one "$RUN" "$BASE" "$REV" $ROOT/runs/$RUN/adapter
done
if [[ -z $ONLY ]]; then
  run_one base-4b Qwen/Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b none
  run_one base-9b Qwen/Qwen3.5-9B-Base 68c46c4b3498877f3ef123c856ecfde50c39f404 none
  echo NONCE_ALL_DONE; touch $ROOT/NONCE_ALL_DONE
else
  echo "NONCE_DONE for $ONLY"
fi
