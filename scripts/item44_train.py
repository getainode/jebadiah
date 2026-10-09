"""N2 experiment wrapper: pinned hash-only manifest and unchanged A3 temperatures.

The shared trainer, renderer, merge verifier and optimizer remain unchanged.
The item43 manifest supplies hashes, so counts are independently validated here.
"""
import json
from pathlib import Path
import shutil
import sys
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import v21_pipeline as pipeline

DATA = 'frontier-infra/jebadiah-data-v2-1-item43'
REVISION = '0738c0ac008528511041e059b6c5453ab8198dbf'
A3 = 'frontier-infra/jebadiah-9b-v2-1-a3'
A3_REVISION = '9e69926a007dd636e82d33485dcd48f9751c4248'


def main():
    data = Path(snapshot_download(DATA, repo_type='dataset', revision=REVISION, allow_patterns=['n2/*']))
    manifest = json.loads((data / 'n2/manifest.json').read_text())
    scan = json.loads((data / 'n2/overlap-scan-report.json').read_text())
    assert scan['status'] == 'passed' and scan['scanned_records'] == 6144 and scan['removed_records'] == 0
    assert manifest['slots']['replaced_slots'] == 1170 and manifest['questions'] == 21190
    report = pipeline.validate_data(data / 'n2/train.jsonl', data / 'n2/calib.jsonl')
    assert report['train']['questions'] == 21190 and report['calib']['questions'] == 512
    for name, entry in manifest['files'].items():
        assert pipeline.sha256(data / 'n2' / name) == entry['sha256']
    assert report['calib']['sha256'] == manifest['a3_input_hashes']['calib.jsonl']
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
            print('ITEM44_UNCHANGED_A3_TEMPERATURE', pipeline.sha256(output / 'temperatures.json'), flush=True)
        else:
            original_run(script, *args)
    pipeline.run = run
    pipeline.main()


if __name__ == '__main__':
    main()
