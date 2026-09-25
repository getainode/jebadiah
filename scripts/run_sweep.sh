#!/bin/bash
# Jebadiah sweep launcher for the training box (after scripts/setup.sh wrote READY). Runs the sweep.json list sequentially, one run at a
# time, detached (tmux when installed, else nohup), resumable: a run whose results.json says
# "complete" is skipped, so re-launching after a crash or a stop continues where it left off.
#
#   bash /workspace/jeb/run_sweep.sh                 launch detached (the normal way)
#   bash /workspace/jeb/run_sweep.sh --foreground    run in this shell (what the detached copy does)
#   bash /workspace/jeb/run_sweep.sh --only 9b-v0,4b-2ep   subset, in sweep order
#   bash /workspace/jeb/run_sweep.sh --skip-smoke    skip the smoke run (not recommended)
#   bash /workspace/jeb/run_sweep.sh --status        print sweep-status.json's table and exit
#   bash /workspace/jeb/run_sweep.sh --stop          stop the detached sweep after the current stage
#
# Outputs: /workspace/jeb/runs/<run>/results.json (rolling, per stage), /workspace/jeb/sweep-status.json
# (rolling), /workspace/jeb/sweep.log. The optional runs (sweep.json "optional": true) run last.
set -uo pipefail
ROOT=${JEB_ROOT:-/workspace/jeb}
export JEB_ROOT=$ROOT
# the environment setup.sh chose (HF_HOME on the big volume, allocator, triton cache)
# shellcheck disable=SC1091
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
export JEB_ROOT=$ROOT
ONLY=""; FOREGROUND=0; SKIP_SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1 ;;
    --only) ONLY=$2; shift ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --status) cd "$ROOT/train" && "$ROOT/venv/bin/python" sweep_lib.py summary; exit $? ;;
    --stop) touch "$ROOT/STOP"; echo "STOP requested: the sweep ends after the current stage"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
LOG=$ROOT/sweep.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

if [[ ! -f "$ROOT/READY" ]]; then
  echo "$ROOT/READY is missing: setup.sh has not finished (tail -f $ROOT/setup.log)"; exit 1
fi
if [[ ! -x "$ROOT/venv/bin/python" ]]; then echo "no venv at $ROOT/venv"; exit 1; fi

if [[ $FOREGROUND == 0 ]]; then
  if [[ -f "$ROOT/sweep.pid" ]] && kill -0 "$(cat "$ROOT/sweep.pid")" 2>/dev/null; then
    echo "sweep already running (pid $(cat "$ROOT/sweep.pid")); tail -f $LOG"; exit 0
  fi
  rm -f "$ROOT/STOP"
  args=(--foreground); [[ -n $ONLY ]] && args+=(--only "$ONLY"); [[ $SKIP_SMOKE == 1 ]] && args+=(--skip-smoke)
  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t jeb 2>/dev/null || true
    tmux new-session -d -s jeb "bash '$ROOT/run_sweep.sh' ${args[*]} >> '$LOG' 2>&1"
    echo "sweep launched in tmux session 'jeb' (tmux attach -t jeb); log: $LOG"
  else
    nohup bash "$ROOT/run_sweep.sh" "${args[@]}" >> "$LOG" 2>&1 &
    echo "sweep launched under nohup (pid $!); log: $LOG"
  fi
  exit 0
fi

# ---------------------------------------------------------------- foreground worker
echo $$ > "$ROOT/sweep.pid"
trap 'rm -f "$ROOT/sweep.pid"' EXIT
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
cd "$ROOT/train"
export PYTHONUNBUFFERED=1
log "== sweep start on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); attn=$(cat "$ROOT/ATTN" 2>/dev/null || echo sdpa)"
mapfile -t RUNS < <(python sweep_lib.py list --include-optional)
if [[ -n $ONLY ]]; then
  IFS=',' read -r -a want <<< "$ONLY"
  sel=(); for r in "${RUNS[@]}"; do for w in "${want[@]}"; do [[ $r == "$w" ]] && sel+=("$r"); done; done
  RUNS=("${sel[@]}")
fi
for run in "${RUNS[@]}"; do
  if [[ -f "$ROOT/STOP" ]]; then log "STOP file present; ending the sweep before $run"; break; fi
  if [[ $SKIP_SMOKE == 1 && $run == smoke ]]; then log "skipping smoke on request"; continue; fi
  if [[ -f "$ROOT/runs/$run/results.json" ]] && python -c "import json,sys; sys.exit(0 if json.load(open('$ROOT/runs/$run/results.json')).get('stage')=='complete' else 1)"; then
    log "$run already complete, skipping"; continue
  fi
  log "== $run"
  bash "$ROOT/train/run_one.sh" "$run"; rc=$?
  if [[ $rc != 0 ]]; then
    log "$run failed (rc=$rc)"
    if [[ $run == smoke ]]; then log "the smoke run failed: stopping the sweep before any paid run"; python sweep_lib.py status smoke failed "smoke failed, sweep stopped"; exit 1; fi
  fi
done
python sweep_lib.py summary | tee -a "$LOG"
log "SWEEP_DONE"
touch "$ROOT/SWEEP_DONE"
