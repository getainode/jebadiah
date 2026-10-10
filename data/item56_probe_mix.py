# SPDX-License-Identifier: Apache-2.0
"""Combine immutable scanned additive rungs, preserving every original A3 byte."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from item51_rung2b_mix import A3_HASHES, A3_REVISION

PINS = Path(__file__).with_name('manifests') / 'item56-input-pins.json'
CAP = 2119  # Frozen original A3 denominator, even when addition expands it.
INDEX = 'cf54ade9013c05db74f4c70925287382708de965fbf3cba62145ee821adad0ff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def lineage(row, base_manifest):
    """Count question exposure by upstream, combining HWU/SLURP/MASSIVE aliases."""
    meta = json.dumps({k:row.get(k) for k in ('source','source_family','subset','provenance')}).lower()
    for name, aliases in [('hwu-slurp-massive', ('hwu','slurp','massive')),
                          ('corr2cause', ('corr2cause',)), ('spacenli', ('spacenli',)),
                          ('deepmind-math', ('deepmind','mathematics_dataset','math_dataset'))]:
        if any(alias in meta for alias in aliases):
            return name
    info = base_manifest['source_manifests'].get(row.get('subset'))
    if not info:
        raise ValueError('Unresolved upstream source')
    match = re.search(r'(?:huggingface.co/datasets/)?([\w.-]+/[\w.-]+)', info['source_id'])
    return match.group(1).lower() if match else info['family'].lower()


def scan_gate(scan):
    stages = [scan] if 'raw' not in scan else [scan, scan['raw'], scan['converted']]
    for stage in stages:
        if stage.get('status') != 'passed' or stage.get('source_level') is not True:
            raise ValueError('Passed source-level Studio scan required')
        if stage.get('protected_index_sha256', INDEX) != INDEX:
            raise ValueError('Wrong protected index')
        for field in ('direct_hits','removed_records','removed_families','invalid_empty_state_records',
                      'remaining_hits_under_scanner'):
            if field in stage and stage[field] != 0:
                raise ValueError('Nonclean source scan')


def verify_source(folder, pin, original):
    if set(p.name for p in folder.iterdir() if p.is_file()) != set(pin['files']):
        raise ValueError('Pinned source file set changed')
    for name, expected in pin['files'].items():
        if sha(folder / name) != expected:
            raise ValueError('Pinned scanned source changed: ' + name)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if sha(folder / name) != entry['sha256']:
            raise ValueError('Source manifest hash mismatch')
    scan_gate(json.loads((folder / 'overlap-scan-report.json').read_text()))
    full = (folder / 'train.jsonl').read_bytes()
    if not original.endswith(b'\n') or not full.startswith(original):
        raise ValueError('Source did not preserve A3 train bytes')
    added = full[len(original):]
    parsed = rows(added)
    if len(parsed) != pin['added_questions'] or any(len(row['questions']) != 1 for row in parsed):
        raise ValueError('Unexpected added question count')
    return added, parsed


def compose(base, sources, output, pins=PINS, preview=False):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    spec = json.loads(pins.read_text())
    if preview:
        spec['sources'] = {k:v for k,v in spec['sources'].items() if k != 'rung5-add'}
    expected = {'rung2b','rung3-add','rung4-add'} | (set() if preview else {'rung5-add'})
    if set(spec['sources']) != expected:
        raise ValueError('All four scanned sources required for a launchable probe')
    for name, expected_hash in A3_HASHES.items():
        if sha(base / name) != expected_hash:
            raise ValueError('Wrong pinned A3 bytes: ' + name)
    original = (base / 'train.jsonl').read_bytes()
    calibration = (base / 'calib.jsonl').read_bytes()
    original_rows = rows(original)
    base_manifest = json.loads((base / 'manifest.json').read_text())
    parts = []; additions = []; diagnostics = []; proof = {}; copied = {}
    for name, pin in spec['sources'].items():
        folder = sources / name
        added, selected = verify_source(folder, pin, original)
        if (folder / 'calib.jsonl').read_bytes() != calibration:
            raise ValueError('Calibration changed')
        diagnostic = rows((folder / pin['diagnostic']).read_bytes())
        parts.append(added); additions.extend(selected); diagnostics.extend(diagnostic)
        proof[name] = {'revision':pin['revision'], 'added_questions':len(selected),
                       'append_sha256':hashlib.sha256(added).hexdigest(),
                       'row_sha256':{json.loads(line)['id']:hashlib.sha256(line).hexdigest()
                                     for line in added.splitlines(keepends=True)},
                       'manifest_sha256':sha(folder / 'manifest.json'),
                       'scan_sha256':sha(folder / 'overlap-scan-report.json')}
        # Keep every attribution notice and source receipt; preserve names for notices.
        for filename in pin['files']:
            if filename in ('train.jsonl','calib.jsonl','manifest.json'): continue
            target = filename if filename.startswith('NOTICE') or filename == pin['diagnostic'] else name + '-' + filename
            if target in copied and copied[target].read_bytes() != (folder / filename).read_bytes():
                raise ValueError('Conflicting retained notice or diagnostic')
            copied[target] = folder / filename
    result = original_rows + additions
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('Duplicate training IDs across sources')
    # Additional data must be disjoint from original calibration and every diagnostic.
    norm = lambda r: hashlib.sha256(' '.join(re.findall(r'\w+',json.dumps(r['state'],sort_keys=True,ensure_ascii=False).lower())).encode()).hexdigest()
    for key in (lambda r:r['id'], norm):
        if {key(r) for r in result} & {key(r) for r in diagnostics}:
            raise ValueError('Diagnostic leakage across compositions')
        if {key(r) for r in additions} & {key(r) for r in rows(calibration)}:
            raise ValueError('Calibration leakage')
    if {r.get('family_id') for r in additions if r.get('family_id')} & {r.get('family_id') for r in diagnostics if r.get('family_id')}:
        raise ValueError('Diagnostic family leakage')
    counts = Counter()
    for row in result:
        counts[lineage(row,base_manifest)] += len(row['questions'])
    # Legacy A1 upstreams predate the new-source cap. Every new source obeys it.
    capped = set(base_manifest['new_source_question_counts']) | {'corr2cause','spacenli','hwu-slurp-massive','deepmind-math'}
    if any(counts[source] > CAP for source in capped):
        raise ValueError('Aggregate upstream source cap exceeded')
    output.mkdir(parents=True,exist_ok=True)
    (output / 'train.jsonl').write_bytes(original + b''.join(parts))
    (output / 'calib.jsonl').write_bytes(calibration)
    for filename, source in copied.items(): shutil.copyfile(source, output / filename)
    # SpaceNLI's prior additive composition omitted a standalone notice; retain
    # the exact upstream MIT license recorded by its frozen source manifest.
    nli_license = Path(__file__).with_name('manifests') / 'item53-source-licenses.json'
    nli = json.loads(nli_license.read_text())['sources']['SpaceNLI']
    notice = nli['notice'].encode()
    if hashlib.sha256(notice).hexdigest() != nli['license_sha256']:
        raise ValueError('SpaceNLI license bytes changed')
    (output / 'NOTICE-SpaceNLI.txt').write_bytes(notice)
    shutil.copyfile(nli_license, output / 'rung3-add-source-licenses.json')
    write_json(output / 'addition-proof.json',proof)
    report = {'name':'item56-outside-data-probe-b', 'private':True, 'probe':True,
              'preview_only':preview,'training_launched':False,
              'a3_revision':A3_REVISION,'a3_input_hashes':A3_HASHES,
              'a3_train_prefix_byte_identical':True,'calibration_bytes_unchanged':True,
              'questions':sum(len(r['questions']) for r in result),'calibration_questions':512,
              'added_questions':len(additions),'removed_questions':0,
              'source_question_counts':dict(sorted(counts.items())), 'aggregate_cap_questions':CAP,
              'input_pins_sha256':sha(pins), 'source_revisions':{n:p['revision'] for n,p in spec['sources'].items()},
              'exposure_policy':'One epoch over untouched A3 plus all additions; original 50 percent mixture changes.',
              'base_policy':'Private A3 exceptions preserved; no shipping clearance.',
              'files':{p.name:{'sha256':sha(p),'rows':len(result) if p.name=='train.jsonl' else len(rows(calibration)) if p.name=='calib.jsonl' else None,
                                'questions':sum(len(r['questions']) for r in result) if p.name=='train.jsonl' else 512 if p.name=='calib.jsonl' else None}
                       for p in sorted(output.iterdir())}}
    write_json(output / 'manifest.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base','sources','out'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args()
    print(json.dumps(compose(args.base,args.sources,args.out,preview=args.preview),indent=2))
