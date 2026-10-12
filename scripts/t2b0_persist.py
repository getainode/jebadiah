# SPDX-License-Identifier: Apache-2.0
"""Persist A3 training evidence and complete heldout answers to private checkpoints."""
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
repo = os.environ['ITEM52_MODEL_REPO'] + '-checkpoints'
assert repo in [f'frontier-infra/jebadiah-9b-t2b0-s{s}-checkpoints' for s in (17, 18)]
api.create_repo(repo, private=True, exist_ok=True)
assert api.model_info(repo).private
root = Path(os.environ['JEB_ROOT'])
names = ['t2b0-data-proof.json', 't2b0-heldout.json', 'item52-environment.json',
         'item52-final-environment.freeze.txt', 'environment.freeze.txt',
         'item52-seed18-runtime-proof.json', 'data_provenance.json', 'models.json',
         'runs/9b/config.json', 'runs/9b/trainer_state.json', 'runs/9b/log_history.json',
         'runs/9b/train_summary.json', 'runs/9b/merged/merge_verification.json',
         'runs/9b/merged/temperatures.json', 'runs/9b/merged/training_provenance.json']
for name in names:
    path = root / name
    if path.exists():
        api.upload_file(repo_id=repo, path_or_fileobj=path, path_in_repo='t2b0-initial/' + name,
                        commit_message='Preserve clean baseline training and heldout evidence')
log = Path('/workspace/item52-train.log')
if log.exists():
    api.upload_file(repo_id=repo, path_or_fileobj=log, path_in_repo='t2b0-initial/training.log',
                    commit_message='Preserve clean baseline training log')
