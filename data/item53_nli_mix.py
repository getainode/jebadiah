"""Apache-2.0 scan-gated replacement and additive SpaceNLI compositions of A3."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
from item33_skill_data import digest, sha, write_json, write_rows, canonical, norm
from item32_ablation_mix import read
from item36_skill_mix import A3_HASHES, A3_REVISION
import item53_spacenli as S


def source_questions(rows):
    return sum(len(r['questions']) for r in rows if any('spacenli' in str(r.get(k, '')).casefold()
               for k in ('source','source_family','subset','family','family_id')))


def plan(rows, candidates, mode, requested=1000):
    if mode not in ('replace','add') or not 0 < requested <= 2017: raise ValueError('Invalid mode/budget')
    total = sum(len(r['questions']) for r in rows)
    exposure = source_questions(rows)
    slots = [(i,qid) for i,r in enumerate(rows) if r.get('area') == 'language' and not source_questions([r])
             for qid,q in r['questions'].items() if q['type'] == 'choice']
    slots.sort(key=lambda s: digest(S.SEED + ':donor:' + rows[s[0]]['id'] + ':' + s[1]))
    n = min(requested, len(candidates), max(0, total // 10 - exposure))
    if mode == 'replace': n = min(n, len(slots))
    if n == 0: raise ValueError('No admissible incoming slots')
    incoming = sorted(candidates, key=lambda r: digest(S.SEED + ':incoming:' + r['id']))[:n]
    selected = set(slots[:n]) if mode == 'replace' else set()
    result = []; replacements = []; position = 0
    for i,r in enumerate(rows):
        kept = copy.deepcopy(r)
        for qid in r['questions']:
            if (i,qid) in selected:
                new = copy.deepcopy(incoming[position]); position += 1; result.append(new)
                replacements.append({'original_id': r['id'], 'question_id': qid, 'replacement_id': new['id']})
                del kept['questions'][qid]; del kept['label'][qid]
                if qid in kept.get('target', {}): del kept['target'][qid]
        if kept['questions']: result.append(kept)
    if mode == 'add': result += copy.deepcopy(incoming)
    expected = total if mode == 'replace' else total + n
    if sum(len(r['questions']) for r in result) != expected: raise ValueError('Changed presentation counts')
    if source_questions(result) > total // 10: raise ValueError('Aggregate source cap exceeded')
    return result, {'mode': mode, 'requested': requested, 'incoming_questions': n, 'matched_donor_slots': len(slots),
                    'questions_before': total, 'questions_after': expected, 'source_questions_before': exposure,
                    'source_questions_after': exposure + n, 'cap_denominator': total, 'source_cap': total // 10,
                    'source_fraction_of_original_a3': (exposure + n) / total,
                    'source_fraction_of_composition': (exposure + n) / expected,
                    'incoming_labels': dict(Counter(r['label']['decision'] for r in incoming)),
                    'incoming_pattern_families': len({r['family_id'] for r in incoming}),
                    'selected_ids': [r['id'] for r in incoming], 'replacements': replacements,
                    'type_histogram': dict(Counter(q['type'] for r in result for q in r['questions'].values())),
                    'area_histogram': dict(Counter(r.get('area','unspecified') for r in result for q in r['questions']))}


def gate(folder, base_rows=None):
    m = json.loads((folder / 'manifest.json').read_text()); report = m['overlap_scan']
    if report.get('status') != 'passed': raise ValueError('Full-suite Studio scan required before composition')
    if m['converter_sha256'] != sha(Path(S.__file__)): raise ValueError('Converter version mismatch')
    licenses = Path(S.__file__).with_name('manifests') / 'item53-source-licenses.json'
    if m['license_manifest_sha256'] != sha(licenses): raise ValueError('License manifest mismatch')
    if m['a3_input_hashes'] != A3_HASHES or m['a3_revision'] != A3_REVISION: raise ValueError('Candidate A3 mismatch')
    if m['upstream_revision'] != S.REVISION: raise ValueError('Wrong upstream revision')
    for n, info in m['files'].items():
        if sha(folder / n) != info['sha256']: raise ValueError('Candidate hash mismatch')
    if json.loads((folder / 'overlap-scan-report.json').read_text()) != report: raise ValueError('Scan receipt mismatch')
    if (report.get('candidates_sha256') != sha(folder / 'scan-candidates.jsonl')
            or report.get('scanned_records') != 64000 or report.get('removed_records') != 0
            or report.get('direct_hits') != 0 or report.get('invalid_empty_state_records') != 0
            or report.get('source_level') is not True or report.get('protected_index_sha256') != S.ITEM25_INDEX_SHA256
            or report.get('scanner_sha256') != sha(Path(S.__file__).with_name('item33_full_suite_scan.py'))):
        raise ValueError('Incomplete or unbound full-source scan')
    policy = json.loads(licenses.read_text())
    if m['upstream_files'] != policy['upstream_files'] or report['candidates_sha256'] != policy['full_source_scan_candidates_sha256']:
        raise ValueError('Frozen upstream/scan mismatch')
    render = m['render_validation']
    if render['truncated'] != 0 or render['max_seq_length'] != 2048: raise ValueError('Render checks required')
    rs = {s: read(folder / (s + '.jsonl')) for s in ('train','diagnostic')}; S.audit(rs)
    if len(rs['diagnostic']) != 300 or not 0 < len(rs['train']) <= 6000: raise ValueError('Candidate counts')
    if render['checked_renders'] != 3 * sum(len(v) for v in rs.values()): raise ValueError('Incomplete render checks')
    scans = read(folder / 'scan-candidates.jsonl')
    if len(scans) != 64000 or len({r['id'] for r in scans}) != 64000: raise ValueError('Incomplete scan scope')
    scanned = {canonical(r) for r in scans}
    for r in rs['train'] + rs['diagnostic']:
        if canonical({**r, 'family': S.SOURCE}) not in scanned: raise ValueError('Unscanned candidate')
    if base_rows is not None:
        mapped = {r['provenance']['original_id']: r for r in scans if r['id'].startswith('item53:SpaceNLI:')}
        upstream = [{'id': r['id'][4:], **r['state'], 'label': mapped[r['id'][4:]]['label']['decision']}
                    for r in scans if r['id'].startswith('raw:')]
        mapping = {r['provenance']['pattern_id']:r['family_id'] for r in mapped.values()}
        expected, selection = S.select(upstream, mapping, base_rows)
        if expected != rs or selection != m['selection']: raise ValueError('Unbound A3/pattern selection')
    return m, rs


def compose(base, candidates, output, mode, requested=1000):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    S.check_base(base)
    original = read(base / 'train.jsonl'); calib = read(base / 'calib.jsonl')
    m, rs = gate(candidates, original + calib)
    if sum(len(r['questions']) for r in original) != 21190 or sum(len(r['questions']) for r in calib) != 512:
        raise ValueError('Unexpected pinned A3 counts')
    for key in (lambda r:r['id'], lambda r:r['family_id'], lambda r:S.state_norm(r['state'])):
        if {key(r) for r in original + calib} & {key(r) for r in rs['train'] + rs['diagnostic']}:
            raise ValueError('A3/candidate collision')
    result, receipt = plan(original, rs['train'], mode, requested)
    if {S.state_norm(r['state']) for r in result} & {S.state_norm(r['state']) for r in rs['diagnostic']}:
        raise ValueError('Diagnostic collision')
    output.mkdir(parents=True, exist_ok=True)
    if mode == 'add':
        selected = set(receipt['selected_ids'])
        # Preserve every original byte, then append exactly the selected questions.
        with (output / 'train.jsonl').open('wb') as f:
            original_bytes = (base / 'train.jsonl').read_bytes()
            if not original_bytes.endswith(b'\n'): raise ValueError('Missing A3 newline')
            f.write(original_bytes)
        with (output / 'train.jsonl').open('a') as f:
            for r in result[len(original):]:
                if r['id'] not in selected: raise ValueError('Unexpected addition')
                f.write(json.dumps(r,ensure_ascii=False,separators=(',',':')) + '\n')
    else: write_rows(output / 'train.jsonl', result)
    shutil.copyfile(base / 'calib.jsonl', output / 'calib.jsonl')
    shutil.copyfile(candidates / 'diagnostic.jsonl', output / 'nli-diagnostic.jsonl')
    write_json(output / 'composition-slots.json', receipt)
    report = {'name': 'item53-rung3-' + mode, 'private': True, 'a3_revision': A3_REVISION,
              'a3_input_hashes': A3_HASHES, 'candidate_manifest_sha256': sha(candidates / 'manifest.json'),
              'overlap_scan_report_sha256': sha(candidates / 'overlap-scan-report.json'),
              'slots': {k:v for k,v in receipt.items() if k not in ('replacements','selected_ids')},
              'calibration': 'Original A3 512 questions, bytes unchanged; separate 300-question diagnostic.',
              'files': {n:{'sha256':sha(output / n)} for n in ('train.jsonl','calib.jsonl','nli-diagnostic.jsonl','composition-slots.json')},
              'base_policy': 'Private A3 exceptions preserved; no public shipment authorized.', 'training_launched': False}
    write_json(output / 'manifest.json', report); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','candidates','out'): p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--mode', choices=('replace','add'), required=True)
    p.add_argument('--requested', type=int, default=1000)
    a = p.parse_args(); print(json.dumps(compose(a.base,a.candidates,a.out,a.mode,a.requested),indent=2))
