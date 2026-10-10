# SPDX-License-Identifier: Apache-2.0
"""Append only the 1,000 already-scanned rung2 additions to byte-identical A3."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

A3_REVISION = 'bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed'
RUNG2_REVISION = '2242e85fae6f8aff3fc3cea3fa95869d99132522'
A3_HASHES = {'train.jsonl': '46fe2a743535a2e841d6b5aa5df479209c9e0c14f04ca1c987570acddd69e2ee',
             'calib.jsonl': '287dfda2c301bb836f1849c9c1a17738732576d7a3c1be2e4b9f7492f8febe88',
             'manifest.json': '2269f3721576869e7be1df1d945259d1bd6a644557fc6bb0eae2138dff0ae7a4'}
SCAN_SHA256 = 'b71e225d957ac4c4322d65e6c4430df82197ac025f34eb46f7db784107e0971c'
RUNG2_TRAIN_SHA256 = '796b89053f48f914341d50118a05bc54cbd01bffe91d61bf3f84d8525e0f0b0d'
SLOTS_SHA256 = '266058c4e9cf3607b360fbfbf4bc1d3329ea51328ebb1e3045b9f1337abd79ef'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [(json.loads(line), line) for line in path.read_bytes().splitlines(keepends=True) if line.strip()]


def additive_bytes(base_lines, rung_lines, mapping):
    ids = [r['replacement_id'] for r in mapping]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('Exactly 1,000 unique selected additions required')
    indexed = {r['id']: (r, raw) for r, raw in rung_lines}
    if len(indexed) != len(rung_lines):
        raise ValueError('Duplicate scanned row ID')
    base_ids = {r['id'] for r, raw in base_lines}
    selected = []
    hashes = {}
    for entry in mapping:
        rid = entry['replacement_id']
        if rid not in indexed or rid in base_ids:
            raise ValueError('Missing scanned row or A3 collision')
        row, raw = indexed[rid]
        if (len(row['questions']) != 1 or row['area'] != 'knowledge'
                or row['questions']['decision']['type'] != entry['type']
                or 'corr2cause' not in row['source'].lower()):
            raise ValueError('Unexpected selected addition')
        selected.append((row, raw))
        hashes[rid] = hashlib.sha256(raw).hexdigest()
    # Copy every original byte in original order, including mixed-question rows.
    original = b''.join(raw for row, raw in base_lines)
    if not original.endswith(b'\n'):
        raise ValueError('A3 train must end in newline')
    return original + b''.join(raw for row, raw in selected), selected, hashes


def compose(base, rung, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    for name, expected in A3_HASHES.items():
        if sha(base / name) != expected:
            raise ValueError('Wrong pinned A3 bytes: ' + name)
    for name, expected in {'train.jsonl': RUNG2_TRAIN_SHA256, 'replacement-slots.json': SLOTS_SHA256,
                           'overlap-scan-report.json': SCAN_SHA256}.items():
        if sha(rung / name) != expected:
            raise ValueError('Wrong pinned scanned rung2 bytes: ' + name)
    manifest = json.loads((rung / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if sha(rung / name) != entry['sha256']:
            raise ValueError('Rung2 manifest mismatch: ' + name)
    scan = json.loads((rung / 'overlap-scan-report.json').read_text())
    if manifest['source_scan_sha256'] != SCAN_SHA256 or scan['status'] != 'passed' or not scan['source_level']:
        raise ValueError('Scanned-set binding failed')
    for phase, count in [('raw', 1035117), ('converted', 6300)]:
        receipt = scan[phase]
        if (receipt['status'] != 'passed' or receipt['scanned_records'] != count
                or receipt['removed_records'] or receipt['direct_hits'] or receipt['remaining_hits_under_scanner']):
            raise ValueError('Source scan failed')
    base_lines = lines(base / 'train.jsonl')
    mapping = json.loads((rung / 'replacement-slots.json').read_text())['replacements']
    train_bytes, selected, hashes = additive_bytes(base_lines, lines(rung / 'train.jsonl'), mapping)
    if not train_bytes.startswith((base / 'train.jsonl').read_bytes()):
        raise ValueError('A3 bytes changed')
    questions = sum(len(r['questions']) for r, raw in base_lines) + len(selected)
    if questions != 22190:
        raise ValueError('Wrong question count')
    diag = [r for r, raw in lines(rung / 'causal-diagnostic.jsonl')]
    rows = [r for r, raw in base_lines + selected]
    if {r['id'] for r in rows} & {r['id'] for r in diag}:
        raise ValueError('Diagnostic ID leakage')
    if {r.get('family_id') for r, raw in selected} & {r['family_id'] for r in diag}:
        raise ValueError('Diagnostic graph leakage')
    exposure = sum(len(r['questions']) for r in rows if 'corr2cause' in json.dumps({k:r.get(k) for k in ('source','source_family','subset','provenance')}).lower())
    if exposure != 1102 or exposure > questions // 10:
        raise ValueError('Aggregate Corr2Cause source cap')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'train.jsonl').write_bytes(train_bytes)
    shutil.copyfile(base / 'calib.jsonl', output / 'calib.jsonl')
    for name in ('causal-diagnostic.jsonl', 'NOTICE-Corr2Cause.txt', 'source-licenses.json', 'overlap-scan-report.json'):
        shutil.copyfile(rung / name, output / name)
    receipt = {'a3_train_prefix_byte_identical': True, 'added_rows': len(selected), 'added_row_sha256': hashes,
               'selection': 'rung2/replacement-slots.json order; original scanned row bytes', 'rung2_train_sha256': RUNG2_TRAIN_SHA256,
               'replacement_slots_sha256': SLOTS_SHA256, 'source_scan_sha256': SCAN_SHA256}
    (output / 'addition-proof.json').write_text(json.dumps(receipt, indent=2) + '\n')
    report = {'name': 'item51-rung2b-additive-a3', 'private': True, 'a3_revision': A3_REVISION,
              'rung2_revision': RUNG2_REVISION, 'a3_input_hashes': A3_HASHES, 'source_scan_sha256': SCAN_SHA256,
              'questions': questions, 'calibration_questions': 512, 'added_questions': 1000,
              'upstream_questions': exposure, 'aggregate_cap_questions': questions // 10,
              'files': {p.name: {'sha256': sha(p)} for p in sorted(output.iterdir())},
              'base_policy': 'Existing A3 private exceptions unchanged; no shipping clearance'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('base', 'rung2', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compose(a.base, a.rung2, a.out), indent=2))
