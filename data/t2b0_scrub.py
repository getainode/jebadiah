# SPDX-License-Identifier: Apache-2.0
"""Remove the eleven approved whole source families from pinned A3, preserving retained bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from item51_rung2b_mix import A3_HASHES, A3_REVISION
from item33_full_suite_scan import leaves, scan
from pool_manifest import describe

FAMILIES = {
    'tasksource/math_qa': ('math_qa', 'math-QA', 'allenai/math_qa'),
    'nvidia/HelpSteer2': ('HelpSteer2',),
    'AndyChiang/dgen': ('dgen',),
    'GGLab/PARADISE': ('PARADISE',),
    'jaredfern/codah': ('codah',),
    'mteb/amazon_counterfactual': ('amazon_counterfactual',),
    'propsegment/nli': ('propsegment', 'sihaochen/propsegment'),
    'tasksource/arct': ('arct',),
    'tasksource/equate': ('equate',),
    'unigram/FOL-nli': ('FOL-nli',),
    'webis/args_me': ('args_me',),
}
INDEX_SHA256 = 'b92290742245132a59839fa565ecc70d691d26033533cb48eb3da702aebd9851'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def training_manifest_entry(path):
    entry = describe(str(path))
    # The pinned pipeline checks rows, while pool_manifest reports records.
    return {**entry, 'rows': entry['records']}


def ancestry(row, manifest):
    """Inspect identity and recorded parents, never question or state content."""
    meta = manifest['source_manifests'].get(row.get('subset'))
    if meta is None:
        raise ValueError('Unresolved A3 subset: ' + str(row.get('subset')))
    fields = ('source', 'source_id', 'source_family', 'subset', 'family',
              'upstream', 'upstream_originals', 'upstream_revisions', 'provenance')
    return [*leaves({k: row.get(k) for k in fields}),
            *leaves({k: meta.get(k) for k in fields}),
            *meta.get('upstream_revisions', {}),
            *meta.get('upstream', {}).get('revisions', {})]


def rejected_family(row, manifest):
    matches = set()
    for family, aliases in FAMILIES.items():
        for name in (family, *aliases):
            # Match identity tokens and child paths/rating views, not substrings of siblings.
            pattern = r'(?<![\w])' + re.escape(name).replace('_', '[-_]').replace('\\-', '[-_]') + r'(?![\w])'
            if any(re.search(pattern, value, re.I) for value in ancestry(row, manifest)):
                matches.add(family)
    if len(matches) > 1:
        raise ValueError('Ambiguous rejected ancestry: ' + str(sorted(matches)))
    return next(iter(matches), None)


def upstream(row, manifest):
    meta = manifest['source_manifests'][row['subset']]
    source = meta.get('source_id') or row['source']
    match = re.search(r'(?:huggingface.co/datasets/)?([\w.-]+/[\w.-]+)', source)
    return match.group(1) if match else source


def build(base, evidence, index, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    for name, expected in A3_HASHES.items():
        if sha(base / name) != expected:
            raise ValueError('Wrong pinned A3 bytes: ' + name)
    if sha(index) != INDEX_SHA256:
        raise ValueError('Wrong supplemented frozen-suite index')
    conflict = json.loads((evidence / 'a3-conflict.json').read_text())
    exposures = json.loads((evidence / 'legacy-raw-family-exposure.json').read_text())
    if set(conflict['whole_family_question_exposures']) | set(exposures) != set(FAMILIES):
        raise ValueError('Approved family set differs from evidence')
    manifest = json.loads((base / 'manifest.json').read_text())
    output.mkdir(parents=True)
    verification = output / 'verification'
    verification.mkdir()
    shutil.copyfile(index, verification / 'protected.pkl')
    removed = {f: {} for f in FAMILIES}
    retained_subsets = set()
    original_questions = {}
    with (verification / 'candidates.jsonl').open('w') as candidates:
        for name in ('train.jsonl', 'calib.jsonl'):
            original_questions[name] = 0
            with (output / name).open('wb') as target:
                for raw in (base / name).read_bytes().splitlines(keepends=True):
                    if not raw.strip():
                        target.write(raw)
                        continue
                    row = json.loads(raw)
                    original_questions[name] += len(row['questions'])
                    family = rejected_family(row, manifest)
                    if family:
                        count = removed[family].setdefault(name, {'records': 0, 'questions': 0, 'subsets': {}})
                        count['records'] += 1
                        count['questions'] += len(row['questions'])
                        count['subsets'][row['subset']] = count['subsets'].get(row['subset'], 0) + len(row['questions'])
                    else:
                        target.write(raw)
                        retained_subsets.add(row['subset'])
                        row['family'] = upstream(row, manifest)
                        candidates.write(json.dumps(row, ensure_ascii=False) + '\n')
    for family in FAMILIES:
        total = sum(v['questions'] for v in removed[family].values())
        lower_bound = conflict['whole_family_question_exposures'].get(family, exposures.get(family, 0))
        if total < lower_bound:
            raise ValueError('Incomplete family removal: ' + family)
    scan(verification)
    report = json.loads((verification / 'scan-report.json').read_text())
    if report['direct_hits'] or report['removed_records'] or report['invalid_empty_state_records']:
        raise ValueError('Scrubbed A3 failed supplemented full-suite scan')
    report.update(protected_index_sha256=INDEX_SHA256,
                  scanner_sha256=sha(Path(__file__).with_name('item33_full_suite_scan.py')),
                  candidates_sha256=sha(verification / 'candidates.jsonl'), status='passed')
    dump(output / 'overlap-scan-report.json', report)
    receipt = {'name': 't2b0-clean-A3', 'private': True, 'a3_revision': A3_REVISION,
               'a3_input_hashes': A3_HASHES, 'approved_families': FAMILIES,
               'removed_by_family': removed, 'original_questions': original_questions,
               'removed_questions': sum(v['questions'] for splits in removed.values() for v in splits.values()),
               'retained_rows_byte_identical_in_original_order': True,
               'added_questions': 0, 'protected_index_sha256': INDEX_SHA256,
               'source_policy': manifest['source_policy'],
               'source_manifests': {k: v for k, v in manifest['source_manifests'].items() if k in retained_subsets},
               'evidence_hashes': {name: sha(evidence / name) for name in ('a3-conflict.json', 'legacy-raw-family-exposure.json')},
               'files': {name: training_manifest_entry(output / name) for name in ('train.jsonl', 'calib.jsonl')},
               'calibration_policy': 'Original A3 procedure on scrubbed calibration; no heldout fitting',
               'scan': report}
    dump(output / 'manifest.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'evidence', 'index', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.base, args.evidence, args.index, args.output)
    print(json.dumps({k: result[k] for k in ('removed_by_family', 'removed_questions', 'files', 'scan')}, indent=2))
