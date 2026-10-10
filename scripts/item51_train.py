"""Rung 2 experiment wrapper: pinned hash-only manifest and unchanged A3 temperatures.

The shared trainer, renderer, merge verifier and optimizer remain unchanged.
The item49 manifest supplies hashes, so counts are independently validated here.
"""
import json
from pathlib import Path
import shutil
import sys
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import v21_pipeline as pipeline

DATA = 'frontier-infra/jebadiah-data-v2-1-item47'
REVISION = json.loads((ROOT / 'configs/item51-rung2b.json').read_text())['dataset_revision']
A3 = 'frontier-infra/jebadiah-9b-v2-1-a3'
A3_REVISION = '9e69926a007dd636e82d33485dcd48f9751c4248'


def main():
    data = Path(snapshot_download(DATA, repo_type='dataset', revision=REVISION, allow_patterns=['rung2b/*']))
    manifest = json.loads((data / 'rung2b/manifest.json').read_text())
    scan = json.loads((data / 'rung2b/overlap-scan-report.json').read_text())
    assert scan['status'] == 'passed' and scan['source_level']
    for phase, count in (('raw', 1035117), ('converted', 6300)):
        receipt = scan[phase]
        assert receipt['status'] == 'passed' and receipt['source_level']
        assert receipt['scanned_records'] == count and receipt['removed_records'] == 0
        assert receipt['direct_hits'] == 0 and receipt['remaining_hits_under_scanner'] == 0
        assert receipt['protected_index_sha256'] == 'cf54ade9013c05db74f4c70925287382708de965fbf3cba62145ee821adad0ff'
    assert pipeline.sha256(data / 'rung2b/overlap-scan-report.json') == manifest['source_scan_sha256']
    assert manifest['added_questions'] == 1000 and manifest['questions'] == 22190
    report = pipeline.validate_data(data / 'rung2b/train.jsonl', data / 'rung2b/calib.jsonl')
    assert report['train']['questions'] == 22190 and report['calib']['questions'] == 512
    for name, entry in manifest['files'].items():
        assert pipeline.sha256(data / 'rung2b' / name) == entry['sha256']
    assert report['calib']['sha256'] == manifest['a3_input_hashes']['calib.jsonl']
    assert manifest['upstream_questions'] == 1102 and manifest['aggregate_cap_questions'] == 2219
    proof = json.loads((data / 'rung2b/addition-proof.json').read_text())
    assert proof['a3_train_prefix_byte_identical'] and proof['added_rows'] == 1000
    diagnostic = pipeline.read_jsonl(data / 'rung2b/causal-diagnostic.jsonl')
    train = pipeline.read_jsonl(data / 'rung2b/train.jsonl')
    assert len(diagnostic) == 300
    assert not {r['id'] for r in diagnostic} & {r['id'] for r in train}
    assert not {r['family_id'] for r in diagnostic} & {r['family_id'] for r in train}
    original_entry = pipeline.split_manifest_entry
    def entry_with_counts(m, filename, manifest_file):
        entry = dict(original_entry(m, filename, manifest_file))
        split = 'train' if filename.endswith('/train.jsonl') else 'calib'
        entry.update({key: report[split][key] for key in ('rows', 'questions')})
        return entry
    pipeline.split_manifest_entry = entry_with_counts
    temperature = hf_hub_download(A3, 'temperatures.json', revision=A3_REVISION)
    original_run = pipeline.run
    def run(script, *args):
        if script == 'fit_temperature.py':
            output = Path(args[args.index('--output-dir') + 1])
            shutil.copyfile(temperature, output / 'temperatures.json')
            print('ITEM51_UNCHANGED_A3_TEMPERATURE', pipeline.sha256(output / 'temperatures.json'), flush=True)
        else:
            original_run(script, *args)
            if script == 'merge_export.py':
                output = Path(args[args.index('--output') + 1])
                shutil.copyfile(data / 'rung2b/NOTICE-Corr2Cause.txt', output / 'NOTICE-Corr2Cause.txt')
    pipeline.run = run
    pipeline.main()


if __name__ == '__main__':
    main()
