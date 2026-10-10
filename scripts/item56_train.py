# SPDX-License-Identifier: Apache-2.0
"""Data/notice routing wrapper around the original, otherwise unchanged A3 runtime."""
import json
import os
from pathlib import Path
import shutil
import sys
from huggingface_hub import HfApi, snapshot_download

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'train'))
import v21_pipeline as pipeline


def main():
    api=HfApi();dataset='frontier-infra/jebadiah-data-v2-1-item47'
    assert api.dataset_info(dataset).private
    data=Path(snapshot_download(dataset,repo_type='dataset',revision=os.environ['ITEM56_DATA_REVISION'],allow_patterns='probe-b/*'))/'probe-b'
    manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['probe'] and not manifest['preview_only']
    assert set(manifest['source_revisions'])=={'rung2b','rung3-add','rung4-add','rung5-add'}
    assert manifest['a3_train_prefix_byte_identical'] and manifest['calibration_bytes_unchanged']
    assert manifest['removed_questions']==0
    for name,entry in manifest['files'].items():
        assert pipeline.sha256(data/name)==entry['sha256'], name
    report=pipeline.validate_data(data/'train.jsonl',data/'calib.jsonl')
    assert report['train']['questions']==manifest['questions']
    assert report['calib']['questions']==512
    assert report['calib']['sha256']==manifest['a3_input_hashes']['calib.jsonl']
    root=Path(os.environ['JEB_ROOT'])
    proof={'dataset_revision':os.environ['ITEM56_DATA_REVISION'],'data_validation':report,
           'manifest_sha256':pipeline.sha256(data/'manifest.json'),'original_runtime': '86a8203d792463536eb009a4d8da6c8681ffdd99',
           'seed':int(os.environ['ITEM52_CONTROL_SEED']), 'temperature_policy':'Original A3 calibration command, bytes unchanged',
           'retained_notices':[p.name for p in sorted(data.glob('NOTICE*'))]}
    (root/'item56-data-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    original_run=pipeline.run
    def run(script,*args):
        original_run(script,*args)
        if script=='merge_export.py':
            merged=Path(args[args.index('--output')+1])
            for notice in data.glob('NOTICE*'):shutil.copyfile(notice,merged/notice.name)
            # Trainer checkpoints are private and retain the same attribution.
            checkpoint=os.environ['ITEM56_MODEL_REPO']+'-checkpoints'
            for notice in data.glob('NOTICE*'):
                api.upload_file(repo_id=checkpoint,path_or_fileobj=notice,path_in_repo=notice.name,
                                commit_message='Keep outside-source attribution with private checkpoints')
    pipeline.run=run
    pipeline.main()


if __name__=='__main__':main()
