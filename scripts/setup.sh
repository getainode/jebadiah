#!/bin/bash
# Jebadiah training box setup. Run it from a checkout of this repository on any Ubuntu machine with
# an NVIDIA GPU and the driver installed:
#
#   sudo JEB_ROOT=/workspace/jeb bash scripts/setup.sh
#
# (Generalised from the startup script we used on rented GPU boxes; nothing in it depends on a
# provider.) Idempotent: every phase leaves a marker under $JEB_ROOT/.setup and is skipped on a
# re-run; safe to run twice. Without root it skips the apt step and expects python3, git and jq.
#
# What it does, in order (all logged to /workspace/jeb/setup.log):
#   1. $JEB_ROOT (default /workspace/jeb; on the larger volume when /ephemeral exists), system packages
#   2. a Python 3.12 venv: torch (cu12x wheels, sm_80 + sm_90), transformers, peft, accelerate,
#      datasets, huggingface_hub, numpy, flash-linear-attention (Qwen3.5's DeltaNet kernels);
#      flash-attn 2 only if a prebuilt wheel installs and passes a smoke test, else SDPA.
#      Never FA3, never fp8, never transformer-engine; bf16 everywhere.
#   3. logs the GPU (no branching on it)
#   4. downloads every base model in configs/sweep.json "bases" (pinned revisions) in the background
#   5. stages this repository's code into $JEB_ROOT (train/, jevals/, sweep.json, run_sweep.sh,
#      nonce_all.sh) and refuses any path matching the forbidden pattern
#   6. builds the PUBLIC training pool and test sets with train/convert_data.py, lints, splits
#   7. writes $JEB_ROOT/READY
# Secrets: none needed. HF_TOKEN is honoured from the environment and never written anywhere.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
START=$(date +%s)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ---------------------------------------------------------------- 1. workspace
ROOT=${JEB_ROOT:-/workspace/jeb}
if [[ ! -e $ROOT ]]; then
  mkdir -p "$(dirname "$ROOT")"
  if [[ -d /ephemeral ]] && [[ $(df -Pk /ephemeral | awk 'NR==2{print $4}') -gt $(df -Pk "$(dirname "$ROOT")" | awk 'NR==2{print $4}') ]]; then
    mkdir -p /ephemeral/jeb && ln -s /ephemeral/jeb "$ROOT"
  else
    mkdir -p "$ROOT"
  fi
fi
mkdir -p "$ROOT/.setup" "$ROOT/runs" "$ROOT/raw" "$ROOT/hf"
exec > >(tee -a "$ROOT/setup.log") 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
done_marker() { [[ -f "$ROOT/.setup/$1.done" ]]; }
mark() { touch "$ROOT/.setup/$1.done"; }
fail() { log "SETUP_FAILED: $*"; echo "$*" > "$ROOT/SETUP_FAILED"; exit 1; }
rm -f "$ROOT/READY" "$ROOT/SETUP_FAILED"
log "== setup start (pid $$, $(id -un)); root=$ROOT -> $(readlink -f $ROOT)"
df -h / "$(dirname "$ROOT")" /ephemeral 2>/dev/null | sed 's/^/  /'
LOGIN_USER=${SUDO_USER:-$(id -un)}
log "files will be owned by: $LOGIN_USER"
export JEB_ROOT=$ROOT
export HF_HOME=$ROOT/hf
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=$ROOT/.triton
cat > "$ROOT/env.sh" <<EOF
export JEB_ROOT=$ROOT
export HF_HOME=$ROOT/hf
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/.triton
EOF
if [[ -n "${HF_TOKEN:-}" ]]; then log "HF_TOKEN present in the environment (not written to disk)"; else log "no HF_TOKEN (public repos, none needed)"; fi

if ! done_marker apt && [[ $(id -u) == 0 ]]; then
  log "== apt"
  apt-get update -qq || true
  apt-get install -y -qq python3 python3-venv python3-pip rsync tmux curl ca-certificates git jq >/dev/null 2>&1 || log "apt install had errors (continuing)"
  mark apt
fi

# ---------------------------------------------------------------- 3. GPU (log only)
log "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || log "nvidia-smi failed"
nvidia-smi -L || true

# ---------------------------------------------------------------- 2. venv + torch + deps
VENV=$ROOT/venv
PY=$VENV/bin/python
if ! done_marker venv; then
  log "== venv"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null 2>&1 || log "uv install failed; using python3 -m venv"
  fi
  rm -rf "$VENV"
  if command -v uv >/dev/null 2>&1 && uv venv --python 3.12 "$VENV" >/dev/null 2>&1; then
    log "venv: uv, $($PY --version)"
  else
    python3 -m venv "$VENV" || fail "could not create a venv"
    log "venv: python3 -m venv, $($PY --version)"
  fi
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PY" -m pip install -q --upgrade pip wheel setuptools >/dev/null 2>&1 || true
  mark venv
fi
PIP="$PY -m pip install -q --disable-pip-version-check"
PIPUN="$PY -m pip uninstall -y -q"
if command -v uv >/dev/null 2>&1; then PIP="uv pip install -q --python $PY"; PIPUN="uv pip uninstall -q --python $PY"; fi
cuda_ok() { "$PY" - <<'EOF'
import torch, sys
assert torch.cuda.is_available(), "cuda not available"
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
s = (a @ a).float().sum().item()
assert s == s, "nan from a bf16 matmul"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
EOF
}

if ! done_marker torch; then
  log "== torch (standard cu12x wheels)"
  ok=0
  for idx in cu128 cu126 cu129 cu124; do
    log "trying torch from https://download.pytorch.org/whl/$idx"
    if $PIP torch --index-url "https://download.pytorch.org/whl/$idx" 2>&1 | tail -3 && cuda_ok; then ok=1; log "torch from $idx works"; break; fi
    $PIPUN torch >/dev/null 2>&1 || true
  done
  if [[ $ok == 0 ]]; then
    log "cu12x indexes failed; trying PyPI's default torch"
    $PIP torch 2>&1 | tail -3 && cuda_ok && ok=1
  fi
  [[ $ok == 1 ]] || fail "no working torch build (see the CUDA test above)"
  mark torch
fi

if ! done_marker deps; then
  log "== core deps (v0's proven pins first)"
  if ! $PIP "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" "datasets>=3.0" safetensors huggingface_hub numpy jinja2 2>&1 | tail -3; then
    log "pinned set failed to resolve; installing latest"
    $PIP transformers peft accelerate "datasets>=3.0" safetensors huggingface_hub numpy jinja2 2>&1 | tail -3 || fail "core deps failed"
  fi
  "$PY" -c "import transformers, peft, accelerate, datasets, huggingface_hub, numpy; print('transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__, 'datasets', datasets.__version__)" || fail "core deps import failed"
  "$PY" -c "from transformers import Qwen3_5ForCausalLM" 2>/dev/null && log "Qwen3_5ForCausalLM available" || log "WARNING: transformers has no Qwen3_5ForCausalLM; jebadiah_model falls back to AutoModelForCausalLM"
  mark deps
fi

# ---------------------------------------------------------------- 4. models, in the background
MODELS_LOG=$ROOT/.setup/models.log
if ! done_marker models; then
  log "== base models: background download to $HF_HOME (log: $MODELS_LOG)"
  ( JEB_SWEEP_JSON="$REPO/configs/sweep.json" "$PY" - > "$MODELS_LOG" 2>&1 <<'EOF'
import json, os, time
from huggingface_hub import snapshot_download
root = os.environ["HF_HOME"]
bases = json.load(open(os.environ["JEB_SWEEP_JSON"]))["bases"]
models = sorted({(b["model"], b["revision"]) for b in bases.values()})
token = os.environ.get("HF_TOKEN") or None
out = {}
for repo, rev in models:
    for attempt in range(6):
        try:
            p = snapshot_download(repo, revision=rev, token=token, allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"])
            out[repo] = {"revision": rev, "path": p, "gb": round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) / 1e9, 2)}
            print("downloaded", repo, rev, out[repo]["gb"], "GB", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print("attempt", attempt, repo, "failed:", e, flush=True)
            time.sleep(20 * (attempt + 1))
    else:
        raise SystemExit(f"could not download {repo}")
json.dump(out, open(os.path.join(os.path.dirname(root), "models.json"), "w"), indent=1)
print("MODELS_DONE", flush=True)
EOF
  ) &
  MODELS_PID=$!
else
  MODELS_PID=""
fi

# ---------------------------------------------------------------- 2b. optional kernels
if ! done_marker kernels; then
  log "== flash-linear-attention (Qwen3.5 Gated DeltaNet kernels; triton ships with torch)"
  $PIP "flash-linear-attention==0.5.2" 2>&1 | tail -2 || $PIP flash-linear-attention 2>&1 | tail -2 || log "WARNING: fla install failed; the DeltaNet layers will use the slow reference path"
  "$PY" -c "import fla; print('fla', getattr(fla, '__version__', '?'))" || log "WARNING: fla import failed"
  log "== causal-conv1d (optional; prebuilt wheel or a bounded source build)"
  if ! "$PY" -c "import causal_conv1d" 2>/dev/null; then
    ( export MAX_JOBS=$(nproc); timeout 1500 $PIP causal-conv1d --no-build-isolation 2>&1 | tail -3 ) || log "causal-conv1d not installed (torch fallback conv, slower; fine)"
  fi
  "$PY" -c "import causal_conv1d; print('causal_conv1d ok')" 2>/dev/null || log "causal_conv1d absent: transformers uses its torch fallback"
  log "== flash-attn 2: prebuilt wheel only, never a source build"
  ATTN=sdpa
  WHL=$("$PY" - <<'EOF'
import json, sys, urllib.request
import torch
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
try:
    rels = json.load(urllib.request.urlopen("https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=15", timeout=30))
except Exception:
    sys.exit(1)
for r in rels:
    for a in r.get("assets", []):
        n = a["name"]
        if n.startswith("flash_attn-2") and "cu12" in n and f"torch{tv}" in n and f"cxx11abi{abi}" in n and n.endswith(f"{py}-{py}-linux_x86_64.whl"):
            print(a["browser_download_url"]); sys.exit(0)
sys.exit(1)
EOF
  ) || WHL=""
  if [[ -n "$WHL" ]]; then
    log "flash-attn wheel candidate: $WHL"
    if timeout 900 $PIP "$WHL" 2>&1 | tail -2 && "$PY" - <<'EOF'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q, causal=True)
assert o.shape == q.shape and torch.isfinite(o).all()
print("flash-attn", __import__("flash_attn").__version__, "smoke ok")
EOF
    then ATTN=flash_attention_2; else log "flash-attn wheel did not install or failed the smoke test; using sdpa"; $PIPUN flash-attn >/dev/null 2>&1 || true; fi
  else
    log "no matching flash-attn 2 wheel for this torch/python; using sdpa"
  fi
  echo "$ATTN" > "$ROOT/ATTN"
  log "attention implementation: $ATTN"
  mark kernels
fi
# Learned 2026-09-22: fla 0.5.2 refuses the DeltaNet backward on Hopper with triton 3.4 to 3.7.0, and the
# causal-conv1d source build needs nvcc (absent on rented boxes); a prebuilt wheel for torch 2.10 loads on 2.11.
if ! done_marker fixes; then
  log "== post-kernel fixes: triton >= 3.7.1, prebuilt causal-conv1d"
  $PIP "triton>=3.7.1" 2>&1 | tail -1 || log "triton upgrade failed (fine on Ampere)"
  if ! "$PY" -c "import causal_conv1d" 2>/dev/null; then
    $PIP https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1 || log "causal-conv1d wheel failed (torch fallback conv, slower)"
    $PIP "triton>=3.7.1" 2>&1 | tail -1 || true   # the wheel install re-pins torch's triton; put it back
  fi
  "$PY" -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)"; "$PY" -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__)" || true
  mark fixes
fi
log "ATTN=$(cat "$ROOT/ATTN")"
"$PY" -c "import torch, transformers, peft; print('env: torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)"

# ---------------------------------------------------------------- 5. stage the code from this checkout
FORBIDDEN='lab|helpdesk|help_desk|client-data|blind|cases-tuned|decide/data'
log "== staging code from $REPO"
mkdir -p "$ROOT/train" "$ROOT/jevals"
STAGE_LIST=$ROOT/.setup/stage.list
( cd "$REPO" && ls train/*.py train/*.sh data/convert_data.py data/split_pool.py data/lint_data.py eval/nonce_eval.py \
    data/jevals/* configs/sweep.json scripts/run_sweep.sh scripts/nonce_all.sh ) > "$STAGE_LIST" || fail "repository files missing"
if grep -E -i "$FORBIDDEN" "$STAGE_LIST"; then fail "a staged path matches the forbidden pattern ($FORBIDDEN)"; fi
cp "$REPO"/train/*.py "$REPO"/train/*.sh "$ROOT/train/"
cp "$REPO/data/convert_data.py" "$REPO/data/split_pool.py" "$REPO/data/lint_data.py" "$REPO/eval/nonce_eval.py" "$ROOT/train/"
cp "$REPO"/data/jevals/* "$ROOT/jevals/"
cp "$REPO/configs/sweep.json" "$ROOT/sweep.json"
cp "$REPO/scripts/run_sweep.sh" "$REPO/scripts/nonce_all.sh" "$ROOT/"
chmod +x "$ROOT/run_sweep.sh" "$ROOT/nonce_all.sh" "$ROOT/train/run_one.sh" 2>/dev/null || true
log "staged $(wc -l < "$STAGE_LIST") files"

# ---------------------------------------------------------------- 6. public data
if ! done_marker data; then
  log "== data: public sources only (typed-decisions, Kev kept sources; tests from HF, Kev, Nimble, Jevals)"
  cd "$ROOT/train" || fail "no train dir"
  "$PY" convert_data.py --out "$ROOT/data" --raw "$ROOT/raw" --jevals "$ROOT/jevals" 2>&1 | tee "$ROOT/.setup/convert.log"
  grep -q CONVERT_DONE "$ROOT/.setup/convert.log" || fail "convert_data.py failed (see .setup/convert.log)"
  "$PY" lint_data.py --train "$ROOT/data/pool.jsonl" || fail "the training pool failed the linter"
  "$PY" split_pool.py "$ROOT/data/pool.jsonl" "$ROOT/data/train.jsonl" "$ROOT/data/calib.jsonl" || fail "split failed"
  "$PY" lint_data.py --train "$ROOT/data/train.jsonl" "$ROOT/data/calib.jsonl" || fail "train/calib failed the linter"
  # eval sets: the head sets first, then the in-distribution ones, then Nimble's public subsets
  : > "$ROOT/data/SETS"
  for s in jevals-pubmedqa jevals-banking77 jevals-helpsteer2 nimble-eval kev-transfer-v4__test typed-decisions-test kev-decision-v7__test \
           $(ls "$ROOT/data/test" | grep '^nimble-public__' | sed 's/\.jsonl$//' | sort); do
    f=$ROOT/data/test/$s.jsonl
    if [[ -f "$f" ]] && "$PY" lint_data.py "$f" >/dev/null; then echo "$s" >> "$ROOT/data/SETS"; else log "eval set $s missing or failed the linter; excluded"; fi
  done
  log "eval sets: $(tr '\n' ' ' < "$ROOT/data/SETS")"
  log "data failures (manifest): $(jq -c .failures "$ROOT/data/manifest.json" 2>/dev/null || echo n/a)"
  # convert_data.py as committed builds the v1 pool (JEB_SCORE_CAP 0.45 default, HelpSteer2 and SummEval
  # train included), which the sweep calls data-v1. Its pool.jsonl sha256 is recorded in data/manifests/.
  ln -sfn data "$ROOT/data-v1"
  mark data
fi

# ---------------------------------------------------------------- wait for the models
if [[ -n "$MODELS_PID" ]]; then
  log "== waiting for the model downloads"
  wait "$MODELS_PID"; rc=$?
  tail -5 "$MODELS_LOG"
  if [[ $rc != 0 ]] || ! grep -q MODELS_DONE "$MODELS_LOG"; then fail "model download failed (see .setup/models.log)"; fi
  mark models
fi
"$PY" - <<'EOF' || fail "model load check failed"
import json, os, torch
from transformers import AutoConfig, AutoTokenizer
m = json.load(open(os.path.join(os.environ.get("JEB_ROOT", "/workspace/jeb"), "models.json")))
for repo, info in m.items():
    cfg = AutoConfig.from_pretrained(repo, revision=info["revision"])
    tok = AutoTokenizer.from_pretrained(repo, revision=info["revision"])
    print(repo, cfg.model_type, "layers", getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None), "vocab", len(tok))
print("model check ok")
EOF

# ---------------------------------------------------------------- 7. READY
chown -R "$LOGIN_USER":"$LOGIN_USER" "$ROOT/" 2>/dev/null || true
{
  echo "ready $(date -u +%FT%TZ) after $(( ($(date +%s) - START) / 60 )) min"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  echo "attn: $(cat "$ROOT/ATTN")"
  echo "torch: $("$PY" -c 'import torch; print(torch.__version__)')"
  echo "models: $(jq -c 'to_entries|map({(.key): .value.gb})|add' "$ROOT/models.json" 2>/dev/null)"
  echo "pool: $(jq -c '.files["pool.jsonl"] | {records, questions, types}' "$ROOT/data/manifest.json" 2>/dev/null)"
  echo "sets: $(tr '\n' ' ' < "$ROOT/data/SETS")"
  echo "data failures: $(jq -c .failures "$ROOT/data/manifest.json" 2>/dev/null)"
  echo "next: bash $ROOT/run_sweep.sh"
} > "$ROOT/READY"
cat "$ROOT/READY"
log "SETUP_DONE"
