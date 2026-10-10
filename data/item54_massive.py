# SPDX-License-Identifier: Apache-2.0
"""Deterministic MASSIVE en-US converter, CPU only and no model calls.

The lead alone invokes --index. The complete frozen release across all locales
and splits is scanned before selecting en-US train rows in the admission path.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile

from item33_skill_data import digest, sha, norm, write_json, write_rows, validate_source
from item32_ablation_mix import read, state_key
from item36_skill_mix import A3_HASHES
from item47_rung1_mix import A3_MANIFEST_SHA256
from item49_corr2cause import renderer, check_index, ITEM25_INDEX_SHA256, TOKENIZER_SHA256
from item33_full_suite_scan import scan
from lint_data import lint_record

SOURCE = 'massive-en-US'
LINEAGE = 'hwu-slurp-massive'
REVISION = 'ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740'
SEED = 'item54-rung4-20261009'
POLICY = Path(__file__).with_name('manifests') / 'item54-source-licenses.json'
ARCHIVE = 'item54-massive-1.1.tar.gz'
CHOICE = 'Route this user request to its annotated intent in the supplied menu.'
VERIFY = 'Is the proposed intent the annotated intent for this user request?'
LINEAGE_NAMES = ('hwu', 'hwu64', 'slurp', 'massive', 'nlu-evaluation-data')


def is_lineage(row):
    metadata = json.dumps({k: row.get(k) for k in
                           ('source', 'source_family', 'subset', 'provenance')}).lower()
    return any(re.search(r'(?<![a-z0-9])' + re.escape(t) + r'(?![a-z0-9])', metadata)
               for t in LINEAGE_NAMES) or LINEAGE in metadata


def archive_rows(raw):
    with tarfile.open(raw / ARCHIVE) as archive:
        members = sorted((m for m in archive.getmembers()
                          if m.isfile() and m.name.startswith('1.1/data/') and m.name.endswith('.jsonl')),
                         key=lambda m: m.name)
        for member in members:
            for line in archive.extractfile(member):
                yield json.loads(line)


def upstream_rows(raw):
    with (raw / 'en-US.jsonl').open() as stream:
        for line in stream:
            yield json.loads(line)


def raw_scan_rows(raw):
    for row in archive_rows(raw):
        yield {'id': 'item54:raw:' + row['locale'] + ':' + row['id'], 'family': SOURCE,
               'state': {k: row[k] for k in ('utt', 'annot_utt', 'scenario', 'intent')},
               'questions': {}}


def validate_policy(raw, licenses=POLICY):
    policy = json.loads(licenses.read_text())
    source = policy['sources'][SOURCE]
    validate_source(source, policy['source_exclusions'])
    if (source['revision'] != REVISION or source['ancestry_verified'] is not True
            or source['generator_license'] != 'Apache-2.0'
            or source['generator_sha256'] != sha(Path(__file__))
            or source['model_written_training_text'] is not False
            or source['model_redistribution_ok'] is not True
            or source['aggregate_source_family'] != LINEAGE
            or source['ancestry_gate']['status'] != 'passed'):
        raise ValueError('Unverified source/converter ancestry or rights')
    for name, info in source['input_files'].items():
        if sha(raw / name) != info['sha256']:
            raise ValueError('Wrong frozen upstream bytes: ' + name)
    if digest(source['notice_text']) != source['notice_sha256']:
        raise ValueError('Notice hash mismatch')
    evidence_path = Path(__file__).resolve().parents[1] / 'results/research/item54-source-evidence.json'
    evidence = json.loads(evidence_path.read_text())
    if (source['source_evidence_sha256'] != sha(evidence_path)
            or source['ancestry_gate'] != evidence['ancestry_gate']):
        raise ValueError('Ancestry evidence binding failed')
    return policy


def existing_exposure(base, raw):
    for filename, expected in {**A3_HASHES, 'manifest.json': A3_MANIFEST_SHA256}.items():
        if sha(base / filename) != expected:
            raise ValueError('Wrong pinned A3: ' + filename)
    english = list(upstream_rows(raw))
    by_text = defaultdict(set)
    for r in english:
        by_text[norm(r['utt'])].add(r['id'])
    train = [r for r in english if r['partition'] == 'train']
    ids = set(); states = set(); counts = Counter()
    for split in ('train', 'calib'):
        for row in read(base / f'{split}.jsonl'):
            states.add(state_key(row))
            if not is_lineage(row):
                continue
            counts[split] += len(row['questions'])
            text = row['state'] if isinstance(row['state'], str) else row['state'].get('user_request', '')
            matches = by_text.get(norm(text), set())
            ids.update(matches)
            # The original Tasksource en-US train row locator is checked against
            # source text, never assumed to be a SLURP ID or another locale's ID.
            group = row.get('provenance', {}).get('group_id', '')
            match = re.search(r'multilingual-massive-[^:]+:train:(\d+)$', group)
            if match:
                i = int(match[1])
                if i >= len(train) or norm(train[i]['utt']) != norm(text):
                    raise ValueError('Unresolved A3 MASSIVE original-ID ancestry')
                ids.add(train[i]['id'])
            if not matches:
                raise ValueError('Unresolved inherited HWU/SLURP/MASSIVE original ID')
    return ids, states, dict(counts)


def convert(row, kind, intents):
    if row['locale'] != 'en-US' or row['partition'] != 'train' or kind not in ('choice', 'noul'):
        raise ValueError('Only original en-US train rows may become candidates')
    gold = row['intent']
    if gold not in intents or len(intents) < 16:
        raise ValueError('Unknown intent taxonomy')
    rng = random.Random(digest(SEED + ':menu:' + row['id']))
    # Same-scenario neighbors first, then unrelated intent distractors. All
    # labels are upstream annotations; no invented unknown/OOS label.
    other = [x for x in sorted(intents) if x != gold]
    rng.shuffle(other)
    neighbors = [x for x in other if intents[x] == row['scenario']]
    rest = [x for x in other if intents[x] != row['scenario']]
    distractors = (neighbors + rest)[:15]
    state = {'user_request': row['utt']}
    q = {'type': kind, 'instructions': CHOICE if kind == 'choice' else VERIFY}
    if kind == 'choice':
        menu = [gold] + distractors; rng.shuffle(menu)
        q['criteria'] = {x: x.replace('_', ' ') for x in menu}
        label = gold
    else:
        label = int(digest(SEED + ':verify:' + row['id'])[:8], 16) % 2 == 0
        state['proposed_intent'] = gold if label else distractors[0]
    family = 'item54:slurp-id:' + row['id']
    return {'id': 'item54:massive:en-US:train:' + row['id'], 'set': 'item54-rung4',
            'subset': SOURCE, 'source': SOURCE, 'source_family': LINEAGE,
            'license': 'CC-BY-4.0', 'area': 'retrieval', 'skill': 'intent-routing',
            'family': family, 'family_id': family, 'state': state,
            'questions': {'decision': q}, 'label': {'decision': label},
            'provenance': {'repo': 'AmazonScience/massive', 'revision': REVISION,
                           'release': '1.1', 'locale': 'en-US', 'original_split': 'train',
                           'slurp_id': row['id'], 'intent': gold, 'scenario': row['scenario']}}


def taxonomy(raw):
    intents = {}
    for row in upstream_rows(raw):
        if row['intent'] in intents and intents[row['intent']] != row['scenario']:
            raise ValueError('Inconsistent upstream intent/scenario')
        intents[row['intent']] = row['scenario']
    if len(intents) != 60:
        raise ValueError('Expected original 60-intent taxonomy')
    return intents


def select(raw, base, train_count=6000, diagnostic_count=300):
    if not 0 < train_count <= 6000 or not 0 < diagnostic_count <= 300:
        raise ValueError('Invalid question budget')
    excluded, states, exposure = existing_exposure(base, raw)
    rows = list(upstream_rows(raw)); intents = taxonomy(raw)
    # Entire duplicate-utterance groups across all en-US splits are excluded.
    # This avoids ambiguous annotations, cross-split repeats and alternate IDs.
    occurrences = Counter(norm(r['utt']) for r in rows)
    pools = defaultdict(list); skipped = Counter()
    for row in rows:
        if row['partition'] != 'train':
            continue
        if row['id'] in excluded or state_key({'state': row['utt']}) in states:
            skipped['a3_exposed'] += 1; continue
        if not norm(row['utt']) or occurrences[norm(row['utt'])] != 1:
            skipped['duplicate_or_empty_utterance'] += 1; continue
        kind = ('choice', 'noul')[int(digest(SEED + ':type:' + row['id'])[:8], 16) % 2]
        pools[(kind, row['intent'])].append(convert(row, kind, intents))
    splits = {}; selected_ids = set()
    for split, budget in (('diagnostic', diagnostic_count), ('train', train_count)):
        result = []
        for kind in ('choice', 'noul'):
            queues = [sorted(pools[(kind, intent)], key=lambda r: digest(SEED + ':row:' + r['id']))
                      for intent in sorted(intents)]
            wanted = budget // 2 + (budget % 2 if kind == 'choice' else 0)
            cursors = [0] * len(queues); added = 0
            while added < wanted:
                advanced = False
                for i, queue in enumerate(queues):
                    while cursors[i] < len(queue) and queue[cursors[i]]['id'] in selected_ids:
                        cursors[i] += 1
                    if cursors[i] == len(queue):
                        continue
                    advanced = True
                    r = queue[cursors[i]]; cursors[i] += 1
                    result.append(r); selected_ids.add(r['id']); added += 1
                    if added == wanted: break
                if not advanced:
                    raise ValueError('Insufficient unused original train candidates')
        splits[split] = result
    return splits, {'excluded_a3_slurp_ids': sorted(excluded, key=int), 'existing_lineage_questions': exposure,
                    'skipped_train_rows': dict(skipped),
                    'partition': 'Original SLURP-ID and normalized-utterance disjoint; diagnostic selected first by intent round robin from original train only'}


def counts(rows):
    return {'questions': len(rows), 'original_slurp_ids': len({r['provenance']['slurp_id'] for r in rows}),
            'types': dict(Counter(r['questions']['decision']['type'] for r in rows)),
            'intents': dict(sorted(Counter(r['provenance']['intent'] for r in rows).items())),
            'scenarios': dict(sorted(Counter(r['provenance']['scenario'] for r in rows).items())),
            'labels': dict(Counter(str(r['label']['decision']) for r in rows))}


def audit(splits):
    ids = set(); families = set(); texts = set()
    for rows in splits.values():
        for r in rows:
            errors = []; lint_record(r, r['id'], errors)
            if errors: raise ValueError('; '.join(errors))
            text = norm(r['state']['user_request']); family = r['family_id']
            if r['id'] in ids or family in families or text in texts:
                raise ValueError('Original ID or normalized-utterance leakage')
            ids.add(r['id']); families.add(family); texts.add(text)
            p = r['provenance']; q = r['questions']['decision']; label = r['label']['decision']
            if (r['source'] != SOURCE or r['source_family'] != LINEAGE or r['area'] != 'retrieval'
                    or r['license'] != 'CC-BY-4.0' or p['revision'] != REVISION
                    or p['original_split'] != 'train' or p['locale'] != 'en-US'
                    or p['repo'] != 'AmazonScience/massive' or p['release'] != '1.1'
                    or r['family'] != family or family != 'item54:slurp-id:' + p['slurp_id']
                    or r['id'] != 'item54:massive:en-US:train:' + p['slurp_id']
                    or q['type'] not in ('choice', 'noul')):
                raise ValueError('Invalid converted source provenance')
            if q['type'] == 'choice':
                if (q['instructions'] != CHOICE or len(q['criteria']) != 16
                        or label != p['intent'] or label not in q['criteria']
                        or any(v != k.replace('_', ' ') for k,v in q['criteria'].items())):
                    raise ValueError('Invalid annotated intent menu')
            elif (q['instructions'] != VERIFY or type(label) is not bool
                  or label != (r['state']['proposed_intent'] == p['intent'])):
                raise ValueError('Invalid intent verification label')


def render_check(splits, tok, rend, contract):
    maximum = Counter(); checked = 0
    for rows in splits.values():
        for row in rows:
            q = row['questions']['decision']; kind = q['type']
            if kind == 'choice':
                canonical = list(q['criteria']); shuffled = list(canonical)
                random.Random(digest(SEED + ':render:' + row['id'])).shuffle(shuffled)
                orders = (canonical, list(reversed(canonical)), shuffled)
            else: orders = (None,)
            for order in orders:
                out = rend.render(row['state'], q, order)
                n = len(tok.encode(out.prompt, add_special_tokens=False))
                expected = 16 if kind == 'choice' else 2
                if out.truncated or n > 1984 or len(out.cand_ids) != expected or len(set(out.cand_ids)) != expected:
                    raise ValueError('Truncation or incomplete answer menu')
                maximum[kind] = max(maximum[kind], n); checked += 1
    return {'checked_renders': checked, 'max_prompt_tokens_by_type': dict(maximum),
            'max_seq_length': 2048, 'prompt_budget': 1984, 'padding_reserve': 64, 'truncated': 0,
            'choice_orders': ['canonical', 'reversed', 'seeded-shuffled'], 'noul_order': 'true,false',
            'tokenizer_sha256': TOKENIZER_SHA256, 'prompt_source_sha256': contract['prompt_source_sha256'],
            'chat_template_sha256': contract['chat_template_sha256']}


def converted_scan_rows(splits):
    for rows in splits.values():
        for r in rows: yield {**r, 'family': SOURCE}


def stream_hash(rows):
    h = hashlib.sha256()
    for row in rows: h.update((json.dumps(row,ensure_ascii=False,separators=(',',':')) + '\n').encode())
    return h.hexdigest()


def scan_phase(rows, output, index, phase):
    """Lead-only source rejection, reusing the unchanged item25 scanner."""
    check_index(index)
    work = output / phase; work.mkdir(parents=True)
    candidate = work / 'candidates.jsonl'
    write_rows(candidate, rows)
    (work / 'protected.pkl').symlink_to(index.resolve())
    try:
        scan(work)
    finally:
        (work / 'protected.pkl').unlink()
    report = json.loads((work / 'scan-report.json').read_text())
    report.update(status='passed' if report['removed_records'] == 0 else 'rejected',
                  source_level=True, protected_index_sha256=ITEM25_INDEX_SHA256,
                  scanner_sha256=sha(Path(__file__).with_name('item33_full_suite_scan.py')),
                  candidates_sha256=sha(candidate))
    write_json(work / 'scan-report.json', report)
    if report['status'] != 'passed':
        write_json(output / 'overlap-scan-report.json',
                   {'status':'rejected','source_level':True,'failed_phase':phase,'report':report})
        raise ValueError('Entire upstream MASSIVE source rejected, no row salvage')
    return report


def build(raw, base, output, tokenizer, licenses=POLICY, index=None):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    policy = validate_policy(raw, licenses); source = policy['sources'][SOURCE]
    raw_report = None
    if index is not None:
        raw_report = scan_phase(raw_scan_rows(raw), output, index, 'raw-source-scan')
    splits, selection = select(raw, base); audit(splits)
    tok, rend, contract = renderer(tokenizer)
    render = render_check(splits, tok, rend, contract)
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'pending_studio_scan', 'source_level': True,
              'protected_index_sha256': ITEM25_INDEX_SHA256}
    if index is not None:
        converted_report = scan_phase(converted_scan_rows(splits), output, index, 'converted-source-scan')
    for split, rows in splits.items(): write_rows(output / f'{split}.jsonl', rows)
    (output / 'NOTICE-MASSIVE.txt').write_text(source['notice_text'])
    if index is not None:
        report = {'status': 'passed', 'source_level': True, 'raw': raw_report, 'converted': converted_report,
                  'protected_index_sha256': ITEM25_INDEX_SHA256,
                  'source_archive_sha256': source['input_files'][ARCHIVE]['sha256'],
                  'converted_sha256': {s: sha(output / f'{s}.jsonl') for s in splits},
                  'scanned_records': raw_report['scanned_records'] + converted_report['scanned_records']}
    write_json(output / 'overlap-scan-report.json', report)
    manifest = {'name': 'item54-rung4-massive-en-US', 'seed': SEED,
                'generator_sha256': sha(Path(__file__)), 'license_manifest_sha256': sha(licenses),
                'source_revision': REVISION, 'source_files': source['input_files'],
                'selection': selection, 'splits': {s: counts(rs) for s,rs in splits.items()},
                'render_validation': render, 'overlap_scan': report,
                'files': {f'{s}.jsonl': {'sha256': sha(output / f'{s}.jsonl')} for s in splits},
                'training_launched': False, 'model_calls': 0}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('raw', 'base', 'out', 'tokenizer'): p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--licenses', type=Path, default=POLICY); p.add_argument('--index', type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.raw,a.base,a.out,a.tokenizer,a.licenses,a.index),indent=2))
