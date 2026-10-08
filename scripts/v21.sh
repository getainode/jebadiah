#!/usr/bin/env bash
# Fresh Ubuntu CUDA box or existing CUDA container. HF_TOKEN is supplied by the operator.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JEB_ROOT=${JEB_ROOT:-/workspace/jeb-v21}
export JEB_ROOT HF_HOME="$JEB_ROOT/hf" TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$JEB_ROOT"
if [[ ${JEB_SKIP_INSTALL:-0} != 1 ]]; then
  if [[ ! -x "$JEB_ROOT/venv/bin/python" ]]; then
    if command -v python3.12 >/dev/null; then
      python3.12 -m venv "$JEB_ROOT/venv"
    else
      # Ubuntu 22.04 and CUDA containers may lack Python 3.12 in apt.
      if ! command -v curl >/dev/null; then
        [[ $(id -u) == 0 ]] || { echo 'Install curl first.' >&2; exit 1; }
        timeout 600 apt-get update -qq
        timeout 600 apt-get install -y curl ca-certificates
      fi
      curl --max-time 60 --retry 3 -LsSf https://astral.sh/uv/0.8.4/install.sh -o "$JEB_ROOT/item26-uv-install.sh"
      UV_INSTALL_DIR="$JEB_ROOT/bin" UV_NO_MODIFY_PATH=1 sh "$JEB_ROOT/item26-uv-install.sh"
      timeout 600 "$JEB_ROOT/bin/uv" venv --python 3.12 --seed "$JEB_ROOT/venv"
    fi
  fi
  PY="$JEB_ROOT/venv/bin/python"
  "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
  timeout 600 "$PY" -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
  timeout 600 "$PY" -m pip install -r "$REPO/configs/v21-cuda-requirements.txt"
  # Deliberate historical workaround: torch 2.11 metadata pins Triton 3.6,
  # but the published DeltaNet backward needed 3.7.1. Validate with cuda_preflight.
  timeout 600 "$PY" -m pip install --no-deps 'triton==3.7.1'
  "$PY" -m pip freeze > "$JEB_ROOT/environment.freeze.txt"
else
  PY=${JEB_PYTHON:-$JEB_ROOT/venv/bin/python}
fi
exec "$PY" "$REPO/train/v21_pipeline.py" --root "$JEB_ROOT" "$@"
