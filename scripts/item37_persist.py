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
    root = Path(os.environ['JEB_ROOT']) / 'runs' / '9b'
    for relative in ('config.json', 'trainer_state.json', 'merged/merge_verification.json'):
        path = root / relative
        if path.exists():
            api.upload_file(repo_id=args.repo, path_or_fileobj=path,
                            path_in_repo='item37-initial/' + relative,
                            commit_message='Preserve initial training and merge gate evidence')
    log = Path('/workspace/item37-train.log')
    if log.exists():
        api.upload_file(repo_id=args.repo, path_or_fileobj=log,
                        path_in_repo='item37-initial/training.log',
                        commit_message='Preserve initial training log')
