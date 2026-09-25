#!/bin/bash
# One sweep run, start to finish: config -> train -> temperatures -> eval -> postprocess -> results.json.
# Every stage records its seconds and status in runs/<run>/stages.json and refreshes results.json
# and /workspace/jeb/sweep-status.json, so a box deleted mid-run still leaves what it had.
# Usage: run_one.sh <run-name>      (run inside the venv, from /workspace/jeb/train)
set -uo pipefail
RUN=${1:?run name}
ROOT=${JEB_ROOT:-/workspace/jeb}
export JEB_ROOT=$ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "$ROOT/train"
DIR=$ROOT/runs/$RUN
mkdir -p "$DIR"
LOG=$DIR/run.log
STAGES=$DIR/stages.json
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
filter() { grep --line-buffered -v -E "^(Loading weights|Fetching)|UserWarning|FutureWarning|warnings.warn" ; }

stage_set() {  # name status seconds [note]
  python - "$STAGES" "$1" "$2" "$3" "${4:-}" <<'EOF'
import json, os, sys
path, name, status, secs, note = sys.argv[1:6]
d = json.load(open(path)) if os.path.exists(path) else {}
d[name] = {"status": status, "seconds": float(secs), "note": note}
json.dump(d, open(path, "w"), indent=1)
EOF
}
stage_ok() { python -c "import json,sys; d=json.load(open('$STAGES')) if __import__('os').path.exists('$STAGES') else {}; sys.exit(0 if (d.get('$1') or {}).get('status')=='ok' else 1)"; }
collect() { python sweep_lib.py collect "$RUN" | tee -a "$LOG"; }

# ---- config (resolves best-of choices from finished runs)
out=$(python sweep_lib.py config "$RUN" 2>&1); rc=$?
echo "$out" | tee -a "$LOG"
if [[ $rc == 3 ]]; then python sweep_lib.py status "$RUN" skipped "$out"; exit 0; fi
if [[ $rc != 0 ]]; then python sweep_lib.py status "$RUN" failed "config: $out"; exit 1; fi
CFG=$DIR/config.json
BASE=$(python -c "import json; print(json.load(open('$CFG'))['base_model'])")
REV=$(python -c "import json; print(json.load(open('$CFG'))['base_revision'])")
ATTN=$(python -c "import json; print(json.load(open('$CFG'))['attn_implementation'])")
sw() { python -c "import json; v=json.load(open('$CFG'))['sweep'].get('$1'); print('' if v is None else v)"; }
dec() { python -c "import json; v=json.load(open('$CFG'))['decide'].get('$1'); print('' if v is None else v)"; }
DATA=$(python -c "import json; print(json.load(open('$CFG')).get('data_dir') or '')")
[[ -z "$DATA" ]] && DATA=${JEB_DATA_DIR:-$ROOT/data}
SETS_DIR=$DATA; [[ -f "$SETS_DIR/SETS" ]] || SETS_DIR=$ROOT/data   # the eval sets stay where they were built
SMOKE=$(sw smoke); EVAL_ONLY=$(sw eval_only); MAX_STEPS=$(sw max_steps); EVAL_LIMIT=$(sw eval_limit)
REPEATS=$(sw eval_repeats); REPEATS_FOR=$(sw eval_repeats_for); EBS=$(sw eval_batch_size); LAT=$(sw eval_latency_sample)
TTARGET=$(sw temperature_target); STARGETS=$(dec score_targets); SADJ=$(dec score_ordinal_adjacent)
python sweep_lib.py status "$RUN" running "attn=$ATTN base=$BASE"
log "== run $RUN base=$BASE attn=$ATTN smoke=$SMOKE eval_only=$EVAL_ONLY data=$DATA sets=$SETS_DIR"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

SETS=$(cat "$SETS_DIR/SETS" 2>/dev/null | sed "s#^#$SETS_DIR/test/#" | sed 's#$#.jsonl#' | tr '\n' ' ')
if [[ -z "$SETS" ]]; then log "no eval sets listed in $SETS_DIR/SETS"; python sweep_lib.py status "$RUN" failed "no eval sets"; exit 1; fi

# ---- train
if [[ "$EVAL_ONLY" != "True" ]]; then
  if ! stage_ok train; then
    t0=$(date +%s)
    args=(--config "$CFG"); [[ "$MAX_STEPS" != "" && "$MAX_STEPS" != "-1" ]] && args+=(--max-steps "$MAX_STEPS")
    python train_jebadiah.py "${args[@]}" 2>&1 | filter | tee -a "$DIR/train.log"; rc=${PIPESTATUS[0]}
    if [[ $rc != 0 && "$ATTN" == "flash_attention_2" ]]; then
      log "training failed with flash_attention_2 (rc=$rc); retrying once with sdpa"
      python - "$CFG" <<'EOF'
import json, sys; p = sys.argv[1]; c = json.load(open(p)); c["attn_implementation"] = "sdpa"; json.dump(c, open(p, "w"), indent=1)
EOF
      ATTN=sdpa
      python train_jebadiah.py "${args[@]}" 2>&1 | filter | tee -a "$DIR/train.log"; rc=${PIPESTATUS[0]}
    fi
    secs=$(( $(date +%s) - t0 ))
    if [[ $rc != 0 ]] || ! grep -q TRAINING_DONE "$DIR/train.log"; then
      stage_set train failed "$secs" "rc=$rc"; collect; python sweep_lib.py status "$RUN" failed "train rc=$rc"; exit 1
    fi
    stage_set train ok "$secs"; collect
  else
    log "train already done, skipping"
  fi

  # ---- temperatures (both fits recorded; the run's temperature_target is applied)
  if ! stage_ok temps; then
    t0=$(date +%s)
    python fit_temperature.py --base "$BASE" --base-revision "$REV" --adapter "$DIR/adapter" --calib "$DATA/calib.jsonl" \
      --target "$TTARGET" --score-targets "$STARGETS" --score-adjacent "$SADJ" --attn "$ATTN" --batch-size 16 2>&1 | filter | tee -a "$DIR/temps.log"; rc=${PIPESTATUS[0]}
    secs=$(( $(date +%s) - t0 ))
    if [[ $rc != 0 ]] || ! grep -q TEMPERATURES_DONE "$DIR/temps.log"; then
      stage_set temps failed "$secs" "rc=$rc"; collect; python sweep_lib.py status "$RUN" failed "temps rc=$rc"; exit 1
    fi
    stage_set temps ok "$secs"; collect
  fi
  ADAPTER_ARGS=(--adapter "$DIR/adapter" --temperatures adapter); EVAL_DIR=$DIR/eval-adapter; LABEL="$RUN, fitted temperatures"
else
  ADAPTER_ARGS=(); EVAL_DIR=$DIR/eval-base; LABEL="$BASE, no adapter"
fi

# ---- eval on every public set
if ! stage_ok eval; then
  t0=$(date +%s)
  args=(--base "$BASE" --base-revision "$REV" "${ADAPTER_ARGS[@]}" --sets $SETS --repeats "$REPEATS" --batch-size "$EBS" \
        --latency-sample "$LAT" --out "$EVAL_DIR" --label "$LABEL" --attn "$ATTN")
  [[ -n "$REPEATS_FOR" ]] && args+=(--repeats-for "$REPEATS_FOR")
  [[ "$EVAL_LIMIT" != "" && "$EVAL_LIMIT" != "0" ]] && args+=(--limit "$EVAL_LIMIT")
  python eval_jebadiah.py "${args[@]}" 2>&1 | filter | tee -a "$DIR/eval.log"; rc=${PIPESTATUS[0]}
  secs=$(( $(date +%s) - t0 ))
  if [[ $rc != 0 ]] || ! grep -q EVAL_DONE "$DIR/eval.log"; then
    stage_set eval failed "$secs" "rc=$rc"; collect; python sweep_lib.py status "$RUN" failed "eval rc=$rc"; exit 1
  fi
  stage_set eval ok "$secs"; collect
fi

# ---- postprocess (per-question floors, flips), no GPU
if ! stage_ok postprocess; then
  t0=$(date +%s)
  python postprocess.py "$EVAL_DIR" 2>&1 | tee -a "$DIR/postprocess.log"; rc=${PIPESTATUS[0]}
  secs=$(( $(date +%s) - t0 ))
  stage_set postprocess $([[ $rc == 0 ]] && echo ok || echo failed) "$secs" "rc=$rc"
fi
rm -rf "$DIR/checkpoints" 2>/dev/null || true
collect
python sweep_lib.py summary | tee -a "$LOG"
log "RUN_DONE $RUN"
