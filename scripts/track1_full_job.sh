#!/usr/bin/env bash
# Full-suite T1e job payload. Approved Desk #855. No automatic retries or relaunches.
set -euo pipefail
export TRACK1_REVISION=678f6e2cc2bb1b51f34407d00255b5e13109ce3f
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
# Keep setup and resumability, separate immutable code reads from private result writes.
p = root / 'indexrun-job.sh'
s = p.read_text()
old = 'code/decision-index-src.tar.gz code/engine/jebadiah_engine.py code/engine/mps_delta.py'
assert s.count(old) == 1
s = s.replace(old, 'code/decision-index-src.tar.gz code/engine/mps_delta.py')
s = s.replace('retry hf download "$RESULTS_REPO" code/',
              'retry hf download "jbrashear/jebadiah-9b-v2-1-index-results" --revision 587f4f496b9cf1d83f0bc8e495379496f65afa2f code/')
proxy = 'retry hf download "$RESULTS_REPO" --include "code/proxy/*" --repo-type dataset --local-dir /tmp/src || exit 2'
assert proxy in s
s = s.replace(proxy, '# Full suite: no proxy code or subset download.')
s = s.replace('python /tmp/src/code/proxy/run.py pipeline', 'python -m decision_index pipeline')
needle = '# resume from what an earlier job uploaded'
assert needle in s
s = s.replace(needle, 'python -m decision_index suite download --dataset "$SUITE_DATASET" --revision e57106b5e0698e74bd1a88b3b4c19b94a0dc8328 --edition 0.3 --dir "$WORKROOT/suite-0.3" || exit 2\n\n' + needle)
p.write_text(s)
PY
export ENGINE=jebadiah_engine:JebadiahEngine
: "${MODEL_SIZE:?9b or 27b required}"
case "$MODEL_SIZE" in
  9b) MODEL_REVISION=de19e80e18aa23420be410d54fdee435587eaad3 ;;
  27b) MODEL_REVISION=e38b48b82bcda8943d0958072c042c5867284838 ;;
  *) exit 2 ;;
esac
export RUN_NAME="jebadiah-${MODEL_SIZE}-v2-1-t1e-678f6e2-v03${SMOKE_SUFFIX:-}"
export ENGINE_OPTS="--option model=frontier-infra/jebadiah-${MODEL_SIZE}-v2-1 --option revision=${MODEL_REVISION} --option prefix_cache=true --option max_batch=64 --option cuda_alloc_conf=expandable_segments:True --option batch_tokens=32768 --option repeat_payload=true --option reverse_avg_below=0.7"
export RESULTS_REPO=frontier-infra/jebadiah-t1e-decision-index-results
export SUITE_DATASET=jbrashear/decision-index-suite-0.3
export SYNC_SEC=120 ROWS_IN_SUITE='' LIMIT="${LIMIT:-}" ATTEMPTS=1
bash /tmp/src/code/indexrun-job.sh 2>&1 | tee /tmp/indexrun-job.log
exit "${PIPESTATUS[0]}"
