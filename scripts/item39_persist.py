"""Persist lightweight training and initial gate evidence, including failed gates."""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    args = p.parse_args()
    api = HfApi()
    assert api.model_info(args.repo).private
    root = Path(os.environ['JEB_ROOT']) / 'runs' / '27b'
    for relative in ('config.json', 'train_summary.json', 'log_history.json', 'merged/merge_verification.json'):
        path = root / relative
        if path.exists():
            api.upload_file(repo_id=args.repo, path_or_fileobj=path,
                            path_in_repo='item39-initial/' + relative,
                            commit_message='Preserve initial training and merge gate evidence')
    log = Path('/workspace/item39-train.log')
    if log.exists():
        api.upload_file(repo_id=args.repo, path_or_fileobj=log,
                        path_in_repo='item39-initial/training.log',
                        commit_message='Preserve initial training log')

    memory = Path('/workspace/item39-memory.csv')
    if memory.exists():
        api.upload_file(repo_id=args.repo, path_or_fileobj=memory,
                        path_in_repo='item39-initial/memory.csv',
                        commit_message='Preserve GPU memory measurements')
