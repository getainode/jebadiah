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
from t2b_heldout import score


def main():
    api = HfApi()
    dataset = 'frontier-infra/jebadiah-data-t2b'
    revision = os.environ['T2B_DATA_REVISION']
    assert api.dataset_info(dataset, revision=revision).private
    data = Path(snapshot_download(dataset, repo_type='dataset', revision=revision,
                                 allow_patterns=['t2b/*']))
    manifest = json.loads((data / 't2b/manifest.json').read_text())
    assert manifest['baseline_revision'] == '28f80a3c1430c27297e0944978bfda9310298623'
    assert manifest['tranche_revision'] == '092e5a61556ee473339d41090b66da3aa530d7a0'
    assert manifest['scan']['direct_hits'] == 0
    assert manifest['baseline_train_byte_prefix'] and manifest['calibration_byte_identical']
    for name, entry in manifest['files'].items():
        assert pipeline.sha256(data / 't2b' / name) == entry['sha256']
    report = pipeline.validate_data(data / 't2b/train.jsonl', data / 't2b/calib.jsonl')
    for split in ('train', 'calib'):
        for key in ('sha256', 'rows', 'questions'):
            assert manifest['files'][split + '.jsonl'][key] == report[split][key]
    assert report['train']['questions'] == 17793 + manifest['composition']['added_questions']
    assert report['calib']['questions'] == 356
    heldout = Path(snapshot_download(dataset, repo_type='dataset', revision=manifest['heldout_revision'],
                                     allow_patterns=['heldout.jsonl'])) / 'heldout.jsonl'
    assert hashlib.sha256(heldout.read_bytes()).hexdigest() == os.environ['T2B_HELDOUT_SHA256']
    root = Path(os.environ['JEB_ROOT'])
    proof = {'dataset': dataset, 'dataset_revision': revision, 'data_validation': report,
             'seed': int(os.environ['ITEM52_CONTROL_SEED']),
             'original_runtime': '86a8203d792463536eb009a4d8da6c8681ffdd99',
             'calibration_policy': manifest['calibration_policy'],
             'manifest_sha256': pipeline.sha256(data / 't2b/manifest.json')}
    (root / 't2b-data-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    pipeline.main()
    base = json.loads((root / 'models.json').read_text())['9b']
    assert base['revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    result = score(Path(base['path']), root / 'runs/9b/adapter', heldout, root / 't2b-heldout.json')
    result['dataset_revision'] = revision
    result['seed'] = proof['seed']
    (root / 't2b-heldout.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
