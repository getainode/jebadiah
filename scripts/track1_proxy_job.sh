#!/usr/bin/env bash
# Remote job payload only. The lead submits it after Desk decision #833.
set -euo pipefail
: "${TRACK1_REVISION:?immutable getainode/jebadiah source SHA required}"
: "${TRACK1_POLICY:?baseline, reverse_avg or repeat_payload required}"
[[ "$TRACK1_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
case "$TRACK1_POLICY" in baseline|reverse_avg|repeat_payload) ;; *) exit 2 ;; esac
export HF_HUB_DISABLE_XET=0
pip install -q 'huggingface_hub==1.33.0' hf_xet
hf download jbrashear/jebadiah-9b-v2-1-index-results \
  --revision 587f4f496b9cf1d83f0bc8e495379496f65afa2f \
  --include 'code/*' --repo-type dataset --local-dir /tmp/src
python - <<'PY'
import os
from pathlib import Path
from urllib.request import urlopen
root = Path('/tmp/src/code')
revision = os.environ['TRACK1_REVISION']
for name in ('jebadiah_engine.py', 'inference_renderer.py'):
    url = f'https://raw.githubusercontent.com/getainode/jebadiah/{revision}/eval/decision-index/{name}'
    (root / 'engine' / name).write_bytes(urlopen(url).read())
# Keep the pinned resumable wrapper, but prevent its setup from overwriting the variant
# engine. Pin its code refreshes too. No final prompt text is modified.
p = root / 'indexrun-job.sh'
s = p.read_text()
old = 'code/decision-index-src.tar.gz code/engine/jebadiah_engine.py code/engine/mps_delta.py'
assert s.count(old) == 1
s = s.replace(old, 'code/decision-index-src.tar.gz code/engine/mps_delta.py')
s = s.replace('retry hf download "$RESULTS_REPO" code/',
              'retry hf download "$RESULTS_REPO" --revision 587f4f496b9cf1d83f0bc8e495379496f65afa2f code/')
s = s.replace('retry hf download "$RESULTS_REPO" --include "code/proxy/*"',
              'retry hf download "$RESULTS_REPO" --revision 587f4f496b9cf1d83f0bc8e495379496f65afa2f --include "code/proxy/*"')
p.write_text(s)
PY
export ENGINE=jebadiah_engine:JebadiahEngine
export RUN_NAME="jebadiah-9b-v2-1-de19e80e-track1-${TRACK1_POLICY}-${TRACK1_REVISION:0:8}-proxy-0.3-10pct"
export ENGINE_OPTS='--option model=frontier-infra/jebadiah-9b-v2-1 --option revision=de19e80e18aa23420be410d54fdee435587eaad3 --option prefix_cache=true --option max_batch=64 --option cuda_alloc_conf=expandable_segments:True --option batch_tokens=32768'
if [[ "$TRACK1_POLICY" != baseline ]]; then
  ENGINE_OPTS+=" --option ${TRACK1_POLICY}=true"
fi
export RESULTS_REPO=jbrashear/jebadiah-9b-v2-1-index-results
export SUITE_DATASET=jbrashear/decision-index-suite-0.3
export SYNC_SEC=120 ROWS_IN_SUITE='' LIMIT='' ATTEMPTS=1
bash /tmp/src/code/indexrun-job.sh
