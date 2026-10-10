"""Apache-2.0 deterministic SpaceNLI converter. CPU only, no model calls."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from item33_skill_data import digest, sha, write_json, write_rows, norm, canonical, validate_source
from item43_wide_choice import render_check
from item33_full_suite_scan import scan, ITEM25_INDEX_SHA256
from item36_skill_mix import A3_HASHES, A3_REVISION
from item32_ablation_mix import read

SOURCE = 'SpaceNLI'
SEED = 'item53-rung3-20261009'
REVISION = '8c10e94d238737be97142f5a7ffdffa49a6a6ab9'
LABELS = ('entailment', 'contradiction', 'neutral')
PROTECTED_PATH = '/Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl'


def state_norm(value):
    return norm(value if isinstance(value, str) else canonical(value))


def families(patterns):
    # Group related variants by numeric seed, including multi-seed XML groups.
    root = ET.parse(patterns).getroot()
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def seed(x): return re.match(r'\d+', x).group()
    for p in root.iter('problem'): find(seed(p.attrib['id']))
    for g in root.iter('group'):
        ids = [seed(p.attrib['id']) for p in g.iter('problem')]
        for x in ids[1:]: parent[find(x)] = find(ids[0])
    return {p.attrib['id']: 'item53:pattern:' + find(seed(p.attrib['id'])) for p in root.iter('problem')}


def convert(raw, family):
    if raw['label'] not in LABELS: raise ValueError('Unknown upstream label')
    return {'id': 'item53:SpaceNLI:' + raw['id'], 'set': 'item53-rung3', 'subset': SOURCE,
            'source': SOURCE, 'source_family': SOURCE, 'license': 'MIT', 'area': 'language',
            'family': family, 'family_id': family,
            'state': 'text_A: ' + raw['premises'] + '\ntext_B: ' + raw['hypothesis'],
            'questions': {'decision': {'type': 'choice', 'instructions': 'Does text_A entail text_B, contradict it, or neither?',
                                      'criteria': {x: None for x in LABELS}}},
            'label': {'decision': raw['label']},
            'provenance': {'repo': 'kovvalsky/SpaceNLI', 'revision': REVISION, 'original_id': raw['id'],
                           'original_split': None, 'pattern_id': raw['id'].rsplit('-', 1)[0]}}


def check_base(base):
    for name, expected in A3_HASHES.items():
        if sha(base / name) != expected: raise ValueError('Wrong pinned A3: ' + name)


def audit(splits):
    from lint_data import lint_record
    seen = set(); family_split = {}; states = set()
    for split, rows in splits.items():
        for r in rows:
            errors = []; lint_record(r, r['id'], errors)
            if errors: raise ValueError('; '.join(errors))
            if r['id'] in seen or state_norm(r['state']) in states: raise ValueError('Duplicate candidate')
            seen.add(r['id']); states.add(state_norm(r['state']))
            if family_split.setdefault(r['family_id'], split) != split: raise ValueError('Pattern leakage')
            if r['label']['decision'] not in LABELS or tuple(r['questions']['decision']['criteria']) != LABELS:
                raise ValueError('Invalid NLI mapping')


def select(raw, mapping, base_rows, train_limit=6000, diagnostic_count=300):
    if not 0 < train_limit <= 6000 or diagnostic_count != 300: raise ValueError('Invalid selection budget')
    pairs = [(norm(r['premises']), norm(r['hypothesis']), mapping[r['id'].rsplit('-', 1)[0]]) for r in raw]
    used = set()
    for old in base_rows:
        if old.get('subset', '').casefold() == 'spacenli':
            state = norm(old['state']); matches = {f for a,b,f in pairs if a in state and b in state}
            if not matches: raise ValueError('Cannot map existing A3 SpaceNLI ancestry')
            used.update(matches)
    old_states = {state_norm(r['state']) for r in base_rows}
    converted = [convert(r, mapping[r['id'].rsplit('-', 1)[0]]) for r in raw]
    eligible = []; seen = set()
    for r in sorted(converted, key=lambda r: digest(SEED + r['id'])):
        k = state_norm(r['state'])
        if r['family_id'] in used or k in old_states or k in seen: continue
        seen.add(k); eligible.append(r)
    held = set(); n = 0
    eligible_labels = {r['label']['decision'] for r in eligible}
    for f in sorted({r['family_id'] for r in eligible}, key=lambda f: digest(SEED + ':hold:' + f)):
        proposed = held | {f}
        remaining_labels = {r['label']['decision'] for r in eligible if r['family_id'] not in proposed}
        if remaining_labels != eligible_labels: continue
        held.add(f); n += sum(r['family_id'] == f for r in eligible)
        if n >= diagnostic_count: break
    pools = {label: [r for r in eligible if r['family_id'] in held and r['label']['decision'] == label] for label in LABELS}
    diagnostic = []
    while len(diagnostic) < diagnostic_count and any(pools.values()):
        for label in LABELS:
            if pools[label] and len(diagnostic) < diagnostic_count: diagnostic.append(pools[label].pop(0))
    train = [r for r in eligible if r['family_id'] not in held][:train_limit]
    if len(diagnostic) != diagnostic_count or not train: raise ValueError('Insufficient unseen patterns')
    splits = {'train': train, 'diagnostic': diagnostic}; audit(splits)
    return splits, {'a3_exposed_pattern_families': sorted(used), 'heldout_pattern_families': sorted(held),
                    'eligible_unique_rows': len(eligible), 'upstream_rows': len(raw),
                    'train_limit': train_limit, 'diagnostic_count': diagnostic_count,
                    'unused_holdout_siblings_excluded': n - diagnostic_count,
                    'diagnostic_missing_labels': sorted(set(LABELS) - {r['label']['decision'] for r in diagnostic}),
                    'split_policy': 'Preserve all eligible training labels; reserve whole unseen pattern families; round-robin heldout labels.'}


def build(upstream, base, tokenizer, output, licenses, index=None):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    check_base(base)
    policy = json.loads(licenses.read_text())
    validate_source(policy['sources'][SOURCE], policy['source_exclusions'])
    if policy['sources'][SOURCE]['revision'] != REVISION or policy['sources'][SOURCE]['license'] != 'MIT':
        raise ValueError('Wrong source revision or license')
    if policy['admission']['status'] != 'provenance_screen_passed': raise ValueError('Source not admitted')
    if policy['converter_sha256'] != sha(Path(__file__)): raise ValueError('Converter hash mismatch')
    for name, expected in policy['upstream_files'].items():
        if sha(upstream / name) != expected: raise ValueError('Upstream hash mismatch: ' + name)
    raw = json.loads((upstream / 'dataset/160x200.json').read_text())['data']
    if len(raw) != 32000 or len({r['id'] for r in raw}) != 32000: raise ValueError('Incomplete source')
    mapping = families(upstream / 'dataset/problem_patterns.xml')
    base_rows = read(base / 'train.jsonl') + read(base / 'calib.jsonl')
    # Scan all original rows and all mapped states/options/instructions before selection.
    scans = []
    for r in raw:
        scans.append({'id': 'raw:' + r['id'], 'family': SOURCE,
                      'state': {'premises': r['premises'], 'hypothesis': r['hypothesis']}, 'questions': {}})
        scans.append({**convert(r, mapping[r['id'].rsplit('-', 1)[0]]), 'family': SOURCE})
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'scan-candidates.jsonl', scans)
    scan_hash = sha(output / 'scan-candidates.jsonl')
    if scan_hash != policy['full_source_scan_candidates_sha256']: raise ValueError('Full-source scope hash mismatch')
    report = {'status': 'pending_studio_scan', 'source_level': True, 'candidates_sha256': scan_hash,
              'expected_records': len(scans), 'protected_index_sha256': ITEM25_INDEX_SHA256}
    if index is not None:
        if str(index.resolve()) != PROTECTED_PATH: raise ValueError('Studio-only protected index path required')
        if sha(index) != ITEM25_INDEX_SHA256: raise ValueError('Wrong protected index')
        work = output / 'source-scan'; work.mkdir()
        (work / 'candidates.jsonl').symlink_to((output / 'scan-candidates.jsonl').resolve())
        (work / 'protected.pkl').symlink_to(index.resolve())
        try: scan(work)
        finally: (work / 'protected.pkl').unlink()
        report = json.loads((work / 'scan-report.json').read_text())
        report.update(status='passed' if report['removed_records'] == 0 else 'rejected', source_level=True,
                      candidates_sha256=scan_hash, protected_index_sha256=ITEM25_INDEX_SHA256,
                      scanner_sha256=sha(Path(__file__).with_name('item33_full_suite_scan.py')))
        write_json(output / 'overlap-scan-report.json', report)
        if report['status'] != 'passed': raise ValueError('Entire SpaceNLI source rejected')
    write_json(output / 'overlap-scan-report.json', report)
    splits, selection = select(raw, mapping, base_rows)
    render = render_check(splits, tokenizer)
    for s, rs in splits.items(): write_rows(output / (s + '.jsonl'), rs)
    manifest = {'name': 'item53-spacenli', 'private': True, 'source': SOURCE, 'seed': SEED,
                'upstream_revision': REVISION, 'upstream_files': policy['upstream_files'],
                'converter_sha256': sha(Path(__file__)), 'license_manifest_sha256': sha(licenses),
                'a3_revision': A3_REVISION, 'a3_input_hashes': A3_HASHES, 'selection': selection,
                'splits': {s: {'questions': len(rs), 'labels': dict(Counter(r['label']['decision'] for r in rs)),
                               'pattern_families': len({r['family_id'] for r in rs})} for s,rs in splits.items()},
                'render_validation': render, 'overlap_scan': report,
                'files': {n: {'sha256': sha(output / n)} for n in ('train.jsonl','diagnostic.jsonl','scan-candidates.jsonl')},
                'training_launched': False}
    write_json(output / 'manifest.json', manifest); return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','base','tokenizer','out'): p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--licenses', type=Path, default=Path(__file__).with_name('manifests') / 'item53-source-licenses.json')
    p.add_argument('--index', type=Path)
    a = p.parse_args(); print(json.dumps(build(a.upstream,a.base,a.tokenizer,a.out,a.licenses,a.index), indent=2))
