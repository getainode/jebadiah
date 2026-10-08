#!/usr/bin/env bash
# Stage code in a PRIVATE results dataset first. Credentials must already be in HF_TOKEN.
# usage: submit.sh RESULTS_DATASET_REVISION MODEL_REVISION
set -euo pipefail
: "${HF_TOKEN:?Bitwarden-sourced HF_TOKEN required}"
[[ $# == 2 ]] || { echo 'usage: submit.sh RESULTS_DATASET_REVISION MODEL_REVISION' >&2; exit 2; }
[[ $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[0-9a-f]{40}$ ]] || { echo 'Both revisions must be pinned SHA1s' >&2; exit 2; }
REPO=jbrashear/jebadiah-9b-v2-1-index-results
RUN=jebadiah-9b-v2-1-r1-${2:0:8}-proxy-0.3-10pct
# 100m * $2.75/h = $4.5834 maximum allocation, under the item $5 cap.
CMD="set -o pipefail; export HF_HUB_DISABLE_XET=0; pip install -q 'huggingface_hub==1.33.0' hf_xet && hf download $REPO --revision $1 --include 'code/*' --repo-type dataset --local-dir /tmp/src && ENGINE=jebadiah_engine:JebadiahEngine RUN_NAME=$RUN ENGINE_OPTS='--option model=frontier-infra/jebadiah-9b-v2-1-r1 --option revision=$2 --option prefix_cache=true --option max_batch=64 --option cuda_alloc_conf=expandable_segments:True --option batch_tokens=32768' RESULTS_REPO=$REPO SUITE_DATASET=jbrashear/decision-index-suite-0.3 SYNC_SEC=120 ROWS_IN_SUITE='' LIMIT='' ATTEMPTS=1 bash /tmp/src/code/indexrun-job.sh 2>&1 | tee -a /tmp/indexrun-job.log"
hf jobs run --detach --flavor rtx-pro-6000 --timeout 100m --secrets HF_TOKEN pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -- bash -lc "$CMD"
