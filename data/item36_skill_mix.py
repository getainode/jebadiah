"""Compose pinned A3 plus regenerated item33 skills without changing calibration."""
import argparse
import json
from pathlib import Path
import shutil

from item32_ablation_mix import sha, read, state_key

A3_REVISION = 'bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed'
A3_HASHES = {
    'train.jsonl': '46fe2a743535a2e841d6b5aa5df479209c9e0c14f04ca1c987570acddd69e2ee',
    'calib.jsonl': '287dfda2c301bb836f1849c9c1a17738732576d7a3c1be2e4b9f7492f8febe88',
}
SKILL_HASHES = {
    'train.jsonl': 'ca1e9a8c9f60033faf9ebedf96e1d32c0b9c6c7f7357cebf8dbf54d6c70ee430',
    'calib.jsonl': 'af4c205b37226916c133617b8607b67821a2b7f94311c875a53ac28448124196',
}


def compose(base, skills, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    for folder, expected in ((base, A3_HASHES), (skills, SKILL_HASHES)):
        for filename, digest in expected.items():
            if sha(folder / filename) != digest:
                raise ValueError(f'Wrong pinned input: {folder / filename}')
    a3 = {s: read(base / f'{s}.jsonl') for s in ('train', 'calib')}
    addition = {s: read(skills / f'{s}.jsonl') for s in ('train', 'calib')}
    # Reject cross-source collisions, including both held-out sets.
    for key in (lambda r: r['id'], lambda r: r['family_id'], state_key):
        old = {key(r) for rows in a3.values() for r in rows}
        new = {key(r) for rows in addition.values() for r in rows}
        if old & new:
            raise ValueError('A3/skill ID, family or normalized state collision')
        train = {key(r) for r in a3['train'] + addition['train']}
        heldout = {key(r) for r in a3['calib'] + addition['calib']}
        if train & heldout:
            raise ValueError('Composed train/heldout collision')
    all_ids = [r['id'] for r in a3['train'] + addition['train']]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Duplicate training IDs')
    output.mkdir(parents=True, exist_ok=True)
    # Preserve baseline bytes and order; append exactly the regenerated train bytes.
    with (output / 'train.jsonl').open('wb') as stream:
        stream.write((base / 'train.jsonl').read_bytes())
        stream.write((skills / 'train.jsonl').read_bytes())
    shutil.copyfile(base / 'calib.jsonl', output / 'calib.jsonl')
    shutil.copyfile(skills / 'calib.jsonl', output / 'skill-calib.jsonl')
    splits = {'train.jsonl': a3['train'] + addition['train'],
              'calib.jsonl': a3['calib'], 'skill-calib.jsonl': addition['calib']}
    counts = {name: {'sha256': sha(output / name), 'rows': len(rows),
                     'questions': sum(len(r['questions']) for r in rows)}
              for name, rows in splits.items()}
    if [counts[n]['questions'] for n in splits] != [25240, 512, 450]:
        raise ValueError('Unexpected composition question counts')
    manifest = {'name': 'item36-r3', 'private': True, 'files': counts,
                'a3_data_revision': A3_REVISION, 'a3_input_hashes': A3_HASHES,
                'skill_input_hashes': SKILL_HASHES,
                'a3_manifest_sha256': sha(base / 'manifest.json'),
                'skill_manifest_sha256': sha(skills / 'manifest.json'),
                'skill_source_scan_sha256': sha(skills / 'overlap-scan-report.json'),
                'calibration': 'A3 original unchanged; 450 skill questions diagnostic only',
                'base_policy': 'Private diagnostic A3 exceptions remain; not a shipping candidate',
                'cross_source_id_family_state_checks': 'passed'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'skills', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.base, args.skills, args.output), indent=2))
