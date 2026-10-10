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
REVISION = '2242e85fae6f8aff3fc3cea3fa95869d99132522'
A3 = 'frontier-infra/jebadiah-9b-v2-1-a3'
A3_REVISION = '9e69926a007dd636e82d33485dcd48f9751c4248'


def main():
    data = Path(snapshot_download(DATA, repo_type='dataset', revision=REVISION, allow_patterns=['rung2/*']))
    manifest = json.loads((data / 'rung2/manifest.json').read_text())
    scan = json.loads((data / 'rung2/overlap-scan-report.json').read_text())
    assert scan['status'] == 'passed' and scan['source_level']
    for phase, count in (('raw', 1035117), ('converted', 6300)):
        receipt = scan[phase]
        assert receipt['status'] == 'passed' and receipt['source_level']
        assert receipt['scanned_records'] == count and receipt['removed_records'] == 0
        assert receipt['direct_hits'] == 0 and receipt['remaining_hits_under_scanner'] == 0
        assert receipt['protected_index_sha256'] == 'cf54ade9013c05db74f4c70925287382708de965fbf3cba62145ee821adad0ff'
    assert pipeline.sha256(data / 'rung2/overlap-scan-report.json') == manifest['source_scan_sha256']
    assert manifest['slots']['replaced_slots'] == 1000 and manifest['questions'] == 21190
    report = pipeline.validate_data(data / 'rung2/train.jsonl', data / 'rung2/calib.jsonl')
    assert report['train']['questions'] == 21190 and report['calib']['questions'] == 512
    for name, entry in manifest['files'].items():
        assert pipeline.sha256(data / 'rung2' / name) == entry['sha256']
    assert report['calib']['sha256'] == manifest['a3_input_hashes']['calib.jsonl']
    assert manifest['slots']['before']['area_type'] == manifest['slots']['after']['area_type']
    assert manifest['slots']['upstream_questions'] <= manifest['slots']['aggregate_cap_questions'] == 2119
    diagnostic = pipeline.read_jsonl(data / 'rung2/causal-diagnostic.jsonl')
    train = pipeline.read_jsonl(data / 'rung2/train.jsonl')
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
            print('ITEM50_UNCHANGED_A3_TEMPERATURE', pipeline.sha256(output / 'temperatures.json'), flush=True)
        else:
            original_run(script, *args)
            if script == 'merge_export.py':
                output = Path(args[args.index('--output') + 1])
                shutil.copyfile(data / 'rung2/NOTICE-Corr2Cause.txt', output / 'NOTICE-Corr2Cause.txt')
    pipeline.run = run
    pipeline.main()


if __name__ == '__main__':
    main()
