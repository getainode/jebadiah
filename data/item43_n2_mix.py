"""Replace floor(10% of pinned A3 train choice questions), preserve other slots."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
from item33_skill_data import digest, sha, write_json, write_rows, canonical
from item36_skill_mix import A3_HASHES, A3_REVISION
from item32_ablation_mix import read, state_key
from item43_wide_choice import SOURCE, audit, ITEM25_INDEX_SHA256


def replace_slots(rows, candidates):
    # A row state applies to all its questions. Split only replaced questions out
    # into independent owned-state rows, retaining every other question unchanged.
    slots = [(i, qid) for i, r in enumerate(rows) for qid, q in r['questions'].items() if q['type'] == 'choice']
    slots.sort(key=lambda slot: digest('item43:replace:' + rows[slot[0]]['id'] + ':' + slot[1]))
    n = len(slots) // 10
    chosen = set(slots[:n])
    pools = {size: sorted([r for r in candidates if len(r['questions']['decision']['criteria']) == size], key=lambda r: digest('item43:select:' + r['id'])) for size in (32,64,128,255)}
    picked = []; offsets = Counter()
    for j in range(n):
        size = (32,64,128,255)[j % 4]; offset = offsets[size]
        if offset >= len(pools[size]): raise ValueError('Insufficient unique owned questions')
        picked.append(copy.deepcopy(pools[size][offset])); offsets[size] += 1
    result = []; replacements = []; position = 0
    for i, row in enumerate(rows):
        kept = copy.deepcopy(row)
        for qid in row['questions']:
            if (i, qid) in chosen:
                replacement = picked[position]; position += 1
                replacements.append({'original_id': row['id'], 'question_id': qid, 'replacement_id': replacement['id']})
                result.append(replacement)
                del kept['questions'][qid]; del kept['label'][qid]
                if qid in kept.get('target', {}): del kept['target'][qid]
        if kept['questions']: result.append(kept)
    if sum(len(r['questions']) for r in result) != sum(len(r['questions']) for r in rows): raise ValueError('Question presentations changed')
    return result, {'choice_slots': len(slots), 'replaced_slots': n, 'fraction': n / len(slots) if slots else 0,
                    'rounding': 'floor(choice_slots / 10)', 'per_size': dict(offsets), 'selected_domains': dict(Counter(r['skill'] for r in picked)),
                    'selected_labels': dict(Counter('none' if r['label']['decision'] == 'none' else 'match' for r in picked)),
                    'replacements': replacements}


def compose(base, wide, output):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    for filename, expected in A3_HASHES.items():
        if sha(base / filename) != expected: raise ValueError('Wrong pinned A3 input: ' + filename)
    manifest = json.loads((wide / 'manifest.json').read_text())
    if manifest['overlap_scan']['status'] != 'passed': raise ValueError('Full-suite Studio scan required before composition')
    if manifest['render_validation']['truncated'] or manifest['render_validation']['max_seq_length'] != 2048: raise ValueError('Render checks required')
    for filename, info in manifest['files'].items():
        if sha(wide / filename) != info['sha256']: raise ValueError('Owned data hash mismatch')
    if manifest['generator_sha256'] != sha(Path(__file__).with_name('item43_wide_choice.py')): raise ValueError('Wrong generator version')
    original = read(base / 'train.jsonl'); calib = read(base / 'calib.jsonl')
    owned = read(wide / 'train.jsonl'); diagnostic = read(wide / 'diagnostic.jsonl')
    audit({'train': owned, 'diagnostic': diagnostic})
    scan_report = manifest['overlap_scan']
    scanned = [{**r, 'family': SOURCE} for r in owned + diagnostic]
    candidate_hash = digest(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in scanned))
    if (scan_report.get('candidates_sha256') != candidate_hash or scan_report.get('scanned_records') != len(scanned)
            or scan_report.get('protected_index_sha256') != ITEM25_INDEX_SHA256 or scan_report.get('removed_records') != 0
            or scan_report.get('source_level') is not True): raise ValueError('Incomplete or unbound source scan')
    for key in (lambda r: r['id'], lambda r: r['family_id'], state_key):
        old = {key(r) for r in original + calib}; new = {key(r) for r in owned + diagnostic}
        if old & new: raise ValueError('A3/owned collision')
        if {key(r) for r in owned} & {key(r) for r in diagnostic}: raise ValueError('Owned diagnostic leakage')
    result, receipt = replace_slots(original, owned)
    for key in (lambda r: r['id'], lambda r: r['family_id'], state_key):
        if {key(r) for r in result} & {key(r) for r in calib + diagnostic}: raise ValueError('Train/heldout collision')
    output.mkdir(parents=True, exist_ok=True); write_rows(output / 'train.jsonl', result)
    shutil.copyfile(base / 'calib.jsonl', output / 'calib.jsonl')
    shutil.copyfile(wide / 'diagnostic.jsonl', output / 'wide-diagnostic.jsonl')
    write_json(output / 'replacement-slots.json', receipt)
    report = {'name': 'item43-n2-a3', 'private': True, 'a3_revision': A3_REVISION,
              'a3_input_hashes': A3_HASHES, 'a3_manifest_sha256': sha(base / 'manifest.json'),
              'wide_manifest_sha256': sha(wide / 'manifest.json'),
              'slots': {k:v for k,v in receipt.items() if k != 'replacements'},
              'questions': sum(len(r['questions']) for r in result),
              'calibration': 'A3 original bytes unchanged; wide diagnostic separate, no temperature fitting',
              'files': {n: {'sha256': sha(output / n)} for n in ('train.jsonl','calib.jsonl','wide-diagnostic.jsonl','replacement-slots.json')},
              'training_launched': False}
    write_json(output / 'manifest.json', report); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('base','wide','out'): p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(compose(a.base,a.wide,a.out), indent=2))
