"""Persist lightweight control evidence, including a failed unchanged merge gate."""
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
repo = os.environ.get('ITEM52_MODEL_REPO', 'frontier-infra/jebadiah-9b-v2-1-a3-rep1') + '-checkpoints'
assert repo in ('frontier-infra/jebadiah-9b-v2-1-a3-rep1-checkpoints',
                'frontier-infra/jebadiah-9b-v2-1-a3-seed18-checkpoints')
api.create_repo(repo, private=True, exist_ok=True)
assert api.model_info(repo).private
root = Path(os.environ['JEB_ROOT'])
for name in ('item52-environment.json', 'item52-final-environment.freeze.txt', 'environment.freeze.txt',
             'item52-seed18-runtime-proof.json',
             'runs/9b/config.json', 'runs/9b/trainer_state.json', 'runs/9b/log_history.json',
             'runs/9b/train_summary.json', 'runs/9b/merged/merge_verification.json',
             'runs/9b/merged/temperatures.json', 'runs/9b/merged/training_provenance.json'):
    path = root / name
    if path.exists():
        api.upload_file(repo_id=repo, path_or_fileobj=path, path_in_repo='item52-initial/' + name,
                        commit_message='Preserve A3 replicate control evidence')
log = Path('/workspace/item52-train.log')
if log.exists():
    api.upload_file(repo_id=repo, path_or_fileobj=log, path_in_repo='item52-initial/training.log',
                    commit_message='Preserve A3 replicate training log')
