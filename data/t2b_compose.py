# SPDX-License-Identifier: Apache-2.0
"""Compose pinned scrubbed A3 and T2b, preserving baseline bytes and calibration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from item32_ablation_mix import state_key
from item33_full_suite_scan import scan
from t2b0_scrub import sha, dump, training_manifest_entry, INDEX_SHA256, upstream

BASE_REVISION = '28f80a3c1430c27297e0944978bfda9310298623'
TRANCHE_REVISION = '092e5a61556ee473339d41090b66da3aa530d7a0'
HELDOUT_SHA256 = '000451d29bb99b67b9b3925fb1cef5859f723cc6f5fd7c01850ec15a59fc472b'


def records(path):
    with path.open('rb') as stream:
        for raw in stream:
            if raw.strip():
                yield raw, json.loads(raw)


def compose(base, tranche, index, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    manifest = json.loads((base / 'manifest.json').read_text())
    source = json.loads((tranche / 't2b-build-summary.json').read_text())
    for name in ('train.jsonl', 'calib.jsonl'):
        assert sha(base / name) == manifest['files'][name]['sha256']
    assert sha(tranche / 'tranche.jsonl') == source['files']['tranche.jsonl']['sha256']
    assert sha(tranche / 'heldout.jsonl') == HELDOUT_SHA256
    assert sha(index) == INDEX_SHA256
    output.mkdir(parents=True)
    verify = output / 'verification'; verify.mkdir()
    (verify / 'protected.pkl').symlink_to(index.resolve())
    protected_ids, protected_states, protected_families = set(), set(), set()
    for path in (base / 'calib.jsonl', tranche / 'heldout.jsonl'):
        for _, row in records(path):
            protected_ids.add(row['id']); protected_states.add(state_key(row)); protected_families.add(row.get('family_id', 't2b-state:' + state_key(row)))
    seen_ids, seen_states = set(), set()
    counts = Counter()
    with (output / 'train.jsonl').open('wb') as train, (verify / 'candidates.jsonl').open('w') as candidates:
        for path, baseline in ((base / 'train.jsonl', True), (tranche / 'tranche.jsonl', False)):
            for raw, row in records(path):
                key = state_key(row)
                if not baseline and not row.get('family_id'):
                    row['family_id'] = 't2b-state:' + key
                    raw = (json.dumps(row, ensure_ascii=False) + '\n').encode()
                if not baseline: counts['tranche_input_questions'] += len(row['questions'])
                if row['id'] in protected_ids or key in protected_states or row['family_id'] in protected_families:
                    raise ValueError('Training overlaps calibration or heldout')
                if row['id'] in seen_ids or key in seen_states:
                    if baseline:
                        # Baseline bytes and recipe are immutable; dedupe only incoming rows.
                        if row['id'] in seen_ids: raise ValueError('Duplicate baseline ID')
                    else:
                        counts['dedupe_removed_rows'] += 1; counts['dedupe_removed_questions'] += len(row['questions']); continue
                seen_ids.add(row['id']); seen_states.add(key)
                train.write(raw)
                if not baseline: counts['added_questions'] += len(row['questions'])
                row['family'] = upstream(row, manifest) if baseline else row['family']
                candidates.write(json.dumps(row, ensure_ascii=False) + '\n')
        for _, row in records(base / 'calib.jsonl'):
            row['family'] = upstream(row, manifest)
            candidates.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert counts['tranche_input_questions'] == 256965
    assert (output / 'train.jsonl').read_bytes().startswith((base / 'train.jsonl').read_bytes())
    shutil.copyfile(base / 'calib.jsonl', output / 'calib.jsonl')
    scan(verify)
    report = json.loads((verify / 'scan-report.json').read_text())
    assert report['direct_hits'] == report['removed_records'] == report['invalid_empty_state_records'] == 0
    report.update(status='passed', protected_index_sha256=INDEX_SHA256,
                  candidates_sha256=sha(verify / 'candidates.jsonl'), scanner_sha256=sha(Path(__file__).with_name('item33_full_suite_scan.py')))
    result = {'name': 't2b-scrubbed-A3-plus-tranche', 'private': True,
              'baseline_revision': BASE_REVISION, 'tranche_revision': TRANCHE_REVISION,
              'baseline_manifest_sha256': sha(base / 'manifest.json'), 'tranche_sha256': sha(tranche / 'tranche.jsonl'),
              'heldout_revision': TRANCHE_REVISION, 'heldout_sha256': HELDOUT_SHA256,
              'tranche_metadata': 'Add family_id=t2b-state:<house normalized state SHA256> for original pipeline schema; question, label, target and provenance unchanged', 'baseline_train_byte_prefix': True, 'calibration_byte_identical': True,
              'calibration_policy': manifest['calibration_policy'], 'dedupe_policy': 'Incoming row IDs and house normalized state SHA256; preserve baseline and first incoming occurrence; reject heldout/calibration ID, state or family_id overlap',
              'composition': dict(counts), 'files': {name: training_manifest_entry(output / name) for name in ('train.jsonl','calib.jsonl')},
              'source_manifests': manifest['source_manifests'], 'tranche_source_summary_sha256': sha(tranche / 't2b-build-summary.json'), 'scan': report}
    assert result['files']['calib.jsonl'] == manifest['files']['calib.jsonl']
    dump(output / 'manifest.json', result); dump(output / 'overlap-scan-report.json', report)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','tranche','index','output'): p.add_argument('--' + name, type=Path, required=True)
    a=p.parse_args(); r=compose(a.base,a.tranche,a.index,a.output)
    print(json.dumps({k:r[k] for k in ('composition','files','scan')},indent=2))
