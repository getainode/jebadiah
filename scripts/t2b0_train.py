# SPDX-License-Identifier: Apache-2.0
"""Route scrubbed data through the untouched A3 pipeline, then score the family holdout."""
import hashlib
import json
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path.cwd() / 'train'))
import v21_pipeline as pipeline
from t2b0_heldout import score


def main():
    api = HfApi()
    dataset = 'frontier-infra/jebadiah-data-t2b'
    revision = os.environ['T2B0_DATA_REVISION']
    assert api.dataset_info(dataset, revision=revision).private
    data = Path(snapshot_download(dataset, repo_type='dataset', revision=revision,
                                 allow_patterns=['t2b0/*', 'heldout.jsonl']))
    manifest = json.loads((data / 't2b0/manifest.json').read_text())
    assert manifest['removed_questions'] == 3553 and len(manifest['approved_families']) == 11
    assert manifest['added_questions'] == 0 and manifest['scan']['direct_hits'] == 0
    assert manifest['retained_rows_byte_identical_in_original_order']
    for name, entry in manifest['files'].items():
        assert pipeline.sha256(data / 't2b0' / name) == entry['sha256']
    report = pipeline.validate_data(data / 't2b0/train.jsonl', data / 't2b0/calib.jsonl')
    assert report['train']['questions'] == 17793 and report['calib']['questions'] == 356
    heldout = data / 'heldout.jsonl'
    assert hashlib.sha256(heldout.read_bytes()).hexdigest() == os.environ['T2B0_HELDOUT_SHA256']
    root = Path(os.environ['JEB_ROOT'])
    proof = {'dataset': dataset, 'dataset_revision': revision, 'data_validation': report,
             'seed': int(os.environ['ITEM52_CONTROL_SEED']),
             'original_runtime': '86a8203d792463536eb009a4d8da6c8681ffdd99',
             'calibration_policy': manifest['calibration_policy'],
             'manifest_sha256': pipeline.sha256(data / 't2b0/manifest.json')}
    (root / 't2b0-data-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    pipeline.main()
    base = json.loads((root / 'models.json').read_text())['9b']
    assert base['revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    result = score(Path(base['path']), root / 'runs/9b/adapter', heldout, root / 't2b0-heldout.json')
    result['dataset_revision'] = revision
    result['seed'] = proof['seed']
    (root / 't2b0-heldout.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
