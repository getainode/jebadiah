# SPDX-License-Identifier: Apache-2.0
"""Local Tasksource tranche adapter using the existing converter and pool tools.

Stages are resumable: prepare, scan, finish. Network fetches and publication are
separate from this builder. Inputs, index, tokenizer and output must be on PRO-G40.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import gzip
import json
import pickle
from pathlib import Path
import random
import re
import shutil
import sys

import pyarrow.parquet as pq
import yaml

from compose_pools import cat
from convert_data import tasksource_jev_row, write_jsonl
from item33_full_suite_scan import scan, norm, content, digest, shingle, ITEM25_INDEX_SHA256
from item51_rung2b_mix import A3_HASHES, A3_REVISION
from lint_data import lint_record, lint_file
from pool_manifest import describe

TYPED_REV = 'f8ff7e81ed820eeac1ad7afd63449fe99c8a76c3'
INSTRUCT_REV = '4cdba593d93bf1e32c0fd1731d84c6780136fe51'
TASK_CAP = 5000
TRAIN_TARGET = 300000
HOLDOUT_PER_FAMILY = 600
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
SUITE_REVISION = 'e57106b5e0698e74bd1a88b3b4c19b94a0dc8328'
SUITE_SHA256 = '1f92d9723b24ca5483474bf324d0ed8e5d542770f89694a579244c45a59cb7fe'
HOLDOUT = ('SDOH-NLI', 'circa', 'multilingual/x-stance',
           'poem_sentiment', 'twitter-financial-news-sentiment')
# Exclude generated responses, generated distractors, machine translations and
# sources whose synthetic augmentation cannot be separated in the recast.
MODEL_TEXT = ('helpsteer', 'ultrafeedback', 'wildguard', 'beavertail', 'saferlhf',
              'i2d2', 'synthetic-instruct', 'chatgpt', 'gpt', 'agent_action_safety',
              'dnd_style', 'it-support', 'procedural-typed-decisions', 'cicero',
              'cosmos_qa', 'swag', 'xcsr', 'turingbench', 'promptshield',
              'neuralchemy', 'dynasent', 'americas_nli', 'xnli',
              'sts-companion', 'translation-', 'flute')
ALLOWED = {'apache-2.0', 'mit', 'bsd', 'bsd-2-clause', 'bsd-3-clause',
           'cc-by-2.0', 'cc-by-3.0', 'cc-by-4.0', 'cc-by-sa-3.0',
           'cc-by-sa-4.0', 'cc0-1.0', 'odc-by'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2).replace('\u2014', '-') + '\n')


def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def parquet_rows(folder):
    for path in sorted(folder.glob('train-*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()


def licences(text):
    """All components must resolve to explicit permitted terms."""
    text = re.sub(r'\s*\((?:DPI|original source)\)', '', text, flags=re.I)
    text = text.lower().replace('apache license 2.0', 'apache-2.0')
    text = text.replace('mit license', 'mit').replace('bsd 2-clause license', 'bsd-2-clause')
    text = text.replace('bsd 3-clause license', 'bsd-3-clause')
    text = re.sub(r'cc by-sa (\d\.\d)', r'cc-by-sa-\1', text)
    text = re.sub(r'cc by (\d\.\d)', r'cc-by-\1', text).replace('cc0 1.0', 'cc0-1.0')
    parts = {p.strip() for p in text.split(',')}
    return sorted(parts) if parts and parts <= ALLOWED else []


def family(meta, task):
    return meta.get('dataset') or next(iter(meta.get('revisions', {})), task)


def prepare(research, output):
    output.mkdir(parents=True, exist_ok=True)
    registry = yaml.safe_load((research / 'tasksource-jev-typed-decisions/sources.yaml').read_text())['sources']
    base = json.loads((research / 'a3/a3/manifest.json').read_text())
    policy = base['source_policy']
    receipts = {r['repo']: r for r in json.loads((research / 'license-fetch-receipt.json').read_text())}
    eligible = {}; rejected = {}
    for task, meta in registry.items():
        ancestry = ' '.join([task, json.dumps(meta.get('originals', [])),
                             json.dumps(meta.get('revisions', {}))]).lower()
        if any(s in ancestry for s in policy['source_substrings']): reason = 'A3 whole-family panel policy'
        elif any(s in ancestry for s in MODEL_TEXT): reason = 'Model outputs or unresolved synthetic ancestry'
        elif meta.get('license_use') != 'commercial' or not licences(meta.get('license', '')): reason = 'Noncommercial, unknown or mixed unconfirmed terms'
        else:
            card_repos = meta.get('card_licenses') or meta.get('revisions', {})
            cards = [receipts.get(repo) for repo in card_repos]
            if not cards or any(not c or c['status'] != 'ok' for c in cards): reason = 'No independently retrievable licence card'
            else:
                evidence = []
                for c in cards:
                    path = Path(c['path'])
                    body = path.read_text()
                    front = yaml.safe_load(body.split('---', 2)[1]) if body.startswith('---') else {}
                    terms = (front or {}).get('license', [])
                    terms = [terms] if isinstance(terms, str) else terms
                    if not terms or any(not licences(t) for t in terms): break
                    evidence.append({'dataset': c['repo'], 'data_revision': meta.get('revisions', {}).get(c['repo']),
                                     'card_revision': c.get('evidence_revision', c['revision']),
                                     'license': terms, 'README_sha256': sha(path),
                                     'url': 'https://huggingface.co/datasets/' + c['repo'] + '/blob/' + c.get('evidence_revision', c['revision']) + '/README.md'})
                if len(evidence) != len(cards): reason = 'Licence card lacks confirmed commercial terms'
                else:
                    eligible[task] = {'source_id': task, 'family': family(meta, task),
                        'license': meta['license'], 'license_use': meta['license_use'],
                        'licenses': licences(meta['license']), 'commercial_use_ok': True,
                        'upstream': meta, 'evidence': evidence,
                        'dataset_revision': TYPED_REV, 'not_in_A3_as_collection': True}
                    continue
        rejected[task] = reason
    for task in HOLDOUT:
        if task not in eligible: raise ValueError('Required holdout lacks clearance: ' + task)
    counters = Counter(); task_counts = Counter(); hashes = {}
    raw_folder = output / 'raw-source-scan'; raw_folder.mkdir(exist_ok=True)
    with (output / 'candidates.jsonl').open('w') as out, (raw_folder / 'candidates.jsonl').open('w') as raw_out:
        for path in sorted((research / 'tasksource-jev-typed-decisions/full').glob('train-*.parquet')):
            hashes[str(path.relative_to(research))] = sha(path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    counters['typed_train_seen'] += 1
                    task = row['source']
                    if task not in eligible: counters['source_policy_removed'] += 1; continue
                    raw_out.write(json.dumps({'id': row['id'], 'family': eligible[task]['family'],
                        'state': row['state'], 'questions': {'source_text': {
                        'instructions': row['question'], 'criteria': row['options']}}}, ensure_ascii=False) + '\n')
                    counters['raw_source_scan_candidates'] += 1
                    if row['license_use'] != 'commercial' or not licences(row['license']): counters['row_license_removed'] += 1; continue
                    if row['variant'] != 'direct': counters['non_direct_variant_removed'] += 1; continue
                    try:
                        r = tasksource_jev_row(row, TYPED_REV)
                        problems = []; lint_record(r, 'candidate', problems, 20)
                        if problems: raise ValueError('Malformed')
                    except (ValueError, KeyError, TypeError): counters['malformed_removed'] += 1; continue
                    r['family'] = eligible[task]['family']
                    out.write(json.dumps(r, ensure_ascii=False) + '\n')
                    task_counts[task] += 1; counters['scan_candidates'] += 1
            print('prepared', path.name, counters['scan_candidates'], flush=True)
    # Audit the complete required instruct train release. Covered tasks and
    # renamed projections never contribute twice. Uncovered classification
    # tasks need independently confirmed licensing and an unambiguous menu.
    instruct_tasks = Counter(); instruct_commercial = Counter()
    for path in sorted((research / 'tasksource-instruct/data').glob('train-*.parquet')):
        hashes[str(path.relative_to(research))] = sha(path)
        for batch in pq.ParquetFile(path).iter_batches(columns=['task', 'license_use']):
            for row in batch.to_pylist():
                instruct_tasks[row['task']] += 1
                if row['license_use'] == 'commercial': instruct_commercial[row['task']] += 1
    receipt = {'status': 'prepared', 'typed_revision': TYPED_REV, 'instruct_revision': INSTRUCT_REV,
               'counters': dict(counters), 'eligible_tasks': eligible, 'excluded_tasks': rejected,
               'candidate_task_counts': dict(task_counts), 'input_hashes': hashes,
               'A3_revision': A3_REVISION, 'A3_hashes': A3_HASHES,
               'A3_source_policy': policy, 'holdout_tasks': HOLDOUT,
               'instruct_train_rows': sum(instruct_tasks.values()),
               'instruct_commercial_rows': sum(instruct_commercial.values()),
               'instruct_task_counts': dict(instruct_tasks),
               'instruct_uncovered_tasks': sorted(set(instruct_tasks) - set(registry)),
               'instruct_retained': 0,
               'instruct_policy': 'Prefer typed decisions for covered tasks and renamed source projections; unconfirmed or ambiguous uncovered recasts contribute zero.'}
    dump(output / 'prepare-receipt.json', receipt)
    return receipt


def supplement_index(research, output, index):
    """Cover every frozen selected row using item33's existing match functions.

    Preserve the pinned item25 index, including added public and GSM8K rows.
    Its public index omits the private ForecastBench binary payloads.
    """
    if sha(index) != ITEM25_INDEX_SHA256: raise ValueError('Wrong protected index')
    suite = research / 'suite/selected-rows.jsonl.gz'
    if sha(suite) != SUITE_SHA256: raise ValueError('Wrong frozen selected rows')
    with index.open('rb') as f: exact, grams, near, inv = pickle.load(f)
    before = len(exact); records = 0; missing = Counter(); added = Counter()
    with gzip.open(suite, 'rt') as f:
        for line in f:
            row = json.loads(line); records += 1
            run = 'selected:' + row['family'] + ':' + row['id']
            canonical = digest(json.dumps(row['state'], sort_keys=True, ensure_ascii=False))
            if canonical not in exact: missing[row['family']] += 1
            exact.setdefault(canonical, run)
            for text in content(row):
                t = norm(text); w = t.split()
                if len(t) >= 80: exact.setdefault(digest(t), run)
                new_span = False
                for h in shingle(w, 13):
                    if h not in grams: grams[h] = run; new_span = True
                if new_span and len(w) >= 20:
                    hs = shingle(w, 5); i = len(near); near.append((hs, run))
                    for h in sorted(hs)[:16]:
                        if h not in inv: inv[h] = []
                        inv[h].append(i)
                    added[row['family']] += 1
    destination = output / 'selected-plus-item25.pkl'
    with destination.open('wb') as f: pickle.dump((exact, grams, near, inv), f)
    receipt = {'base_index_sha256': ITEM25_INDEX_SHA256, 'index_sha256': sha(destination),
               'suite_revision': SUITE_REVISION, 'selected_rows_gzip_sha256': SUITE_SHA256,
               'selected_rows': records, 'canonical_states_added': sum(missing.values()),
               'exact_keys_added': len(exact) - before,
               'previously_missing_canonical_states_by_family': dict(missing),
               'new_span_leaves_by_family': dict(added),
               'all_selected_states_indexed': True, 'private_payloads_in_git': False}
    dump(output / 'index-receipt.json', receipt)
    return receipt


def scan_stage(output, index, supplement_receipt=None):
    expected = supplement_receipt['index_sha256'] if supplement_receipt else ITEM25_INDEX_SHA256
    if sha(index) != expected: raise ValueError('Wrong protected index')
    link = output / 'protected.pkl'
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(index.resolve())
    try: scan(output)
    finally: link.unlink()
    report = json.loads((output / 'scan-report.json').read_text())
    report.update(status='passed', source_level=True, protected_index_sha256=sha(index),
                  scanner_sha256=sha(Path(__file__).with_name('item33_full_suite_scan.py')),
                  candidates_sha256=sha(output / 'candidates.jsonl'),
                  clean_sha256=sha(output / 'clean.jsonl'))
    if supplement_receipt: report['index_coverage'] = supplement_receipt
    dump(output / 'scan-report.json', report)
    return report


def key(row):
    return hashlib.sha256(norm(json.dumps(row['state'], sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def finish(research, output):
    from transformers import AutoTokenizer
    from jebadiah_prompt import Renderer, wire_keys
    prep = json.loads((output / 'prepare-receipt.json').read_text())
    report = json.loads((output / 'scan-report.json').read_text())
    if report['status'] != 'passed' or report['remaining_hits_under_scanner']:
        raise ValueError('Source scan not passed')
    if report['clean_sha256'] != sha(output / 'clean.jsonl') or report['candidates_sha256'] != sha(output / 'candidates.jsonl'):
        raise ValueError('Scan bytes changed')
    base = research / 'a3/a3'
    for name, expected in A3_HASHES.items():
        if sha(base / name) != expected: raise ValueError('Wrong A3 bytes: ' + name)
    tokpath = research / 'tokenizer'
    if sha(tokpath / 'tokenizer.json') != TOKENIZER_SHA256: raise ValueError('Wrong tokenizer')
    tok = AutoTokenizer.from_pretrained(tokpath, local_files_only=True)
    renderer = Renderer(tok, max_tokens=1984)
    render_cache = set(); previous_max = {}
    previous_path = output / 't2b-build-summary.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous['render_validation']['tokenizer_sha256'] == TOKENIZER_SHA256:
            for name in ['tranche.jsonl', 'heldout.jsonl']:
                if sha(output / name) != previous['files'][name]['sha256']:
                    raise ValueError('Render cache bytes changed')
                for row in read_rows(output / name):
                    render_cache.add(digest(json.dumps(row, sort_keys=True, ensure_ascii=False)))
            previous_max = previous['render_validation']['max_prompt_tokens_by_type']
    seen = {key(r) for name in ['train.jsonl', 'calib.jsonl'] for r in read_rows(base / name)}
    pools = defaultdict(list); counters = Counter(); max_tokens = Counter(previous_max)
    holdout_families = {prep['eligible_tasks'][task]['family'] for task in HOLDOUT}
    # Hash-ranked selection is deterministic and independent of shard order.
    raw_bad = set()
    if (output / 'raw-source-scan/contamination-hits.json').exists():
        raw_bad = {row['family'] for row in json.loads((output / 'raw-source-scan/contamination-hits.json').read_text())}
    for r in read_rows(output / 'clean.jsonl'):
        if r['family'] in raw_bad: counters['raw_source_family_removed'] += 1; continue
        pools[r['subset']].append(r)
    for task in pools: pools[task].sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    selected = defaultdict(list); families = Counter(); checked = 0
    for task in sorted(pools):
        for row in pools[task]:
            held = row['family'] in holdout_families
            limit = HOLDOUT_PER_FAMILY if held else TASK_CAP
            if len(selected[task]) >= limit:
                counters['cap_removed'] += 1; continue
            identity = key(row)
            if identity in seen: counters['dedupe_removed'] += 1; continue
            q = row['questions']['decision']; keys = wire_keys(q)
            cached = digest(json.dumps(row, sort_keys=True, ensure_ascii=False)) in render_cache
            orders = [keys]
            if q['type'] == 'choice':
                shuffled = keys.copy(); random.Random(row['id']).shuffle(shuffled)
                orders += [list(reversed(keys)), shuffled]
            try:
                sizes = []
                for order in ([] if cached else orders):
                    rendered = renderer.render(row['state'], q, order if q['type'] == 'choice' else None)
                    n = len(tok.encode(rendered.prompt, add_special_tokens=False)); checked += 1
                    if rendered.truncated or n > 1984: raise ValueError('Over budget')
                    sizes.append(n)
            except (ValueError, TypeError): counters['prompt_budget_removed'] += 1; continue
            if cached: counters['reused_verified_render_records'] += 1
            else: max_tokens[q['type']] = max(max_tokens[q['type']], max(sizes))
            seen.add(identity); selected[task].append(row); families[row['family']] += 1
        print('selected', task, len(selected[task]), flush=True)
    heldout = [row for task in sorted(selected) for row in selected[task] if row['family'] in holdout_families]
    if set(row['family'] for row in heldout) != holdout_families or len(heldout) != 3000:
        raise ValueError('Holdout families need 600 clean questions each')
    # Round-robin truncation keeps breadth if more than the target survives.
    train_tasks = {task: rows for task, rows in selected.items() if rows and rows[0]['family'] not in holdout_families}
    additions = []
    for i in range(TASK_CAP):
        for task in sorted(train_tasks):
            if i < len(train_tasks[task]): additions.append(train_tasks[task][i])
            if len(additions) >= TRAIN_TARGET: break
        if len(additions) >= TRAIN_TARGET: break
    write_jsonl(str(output / 'tranche.jsonl'), additions)
    write_jsonl(str(output / 'heldout.jsonl'), heldout)
    composed = output / 'composition'
    cat(str(composed), [str(base / 'train.jsonl'), str(output / 'tranche.jsonl')])
    shutil.copyfile(composed / 'pool.jsonl', output / 'train.jsonl')
    shutil.copyfile(base / 'calib.jsonl', output / 'calib.jsonl')
    lint = {}
    for name in ['tranche.jsonl', 'heldout.jsonl', 'train.jsonl', 'calib.jsonl']:
        n, problems = lint_file(str(output / name), 20)
        if problems: raise ValueError(f'Lint failed: {name}: {problems[:3]}')
        lint[name] = {'records': n, 'problems': 0, 'status': 'passed'}
    tasks = Counter(r['subset'] for r in additions); family_counts = Counter(r['family'] for r in additions)
    license_manifest = {'sources': {t: {**prep['eligible_tasks'][t], 'train_questions': tasks[t],
            'heldout_questions': sum(r['subset'] == t for r in heldout)} for t in sorted(set(tasks) | set(HOLDOUT))},
        'source_exclusions': prep['A3_source_policy'], 'excluded_tasks': prep['excluded_tasks'],
        'required_datasets_not_in_A3': ['tasksource/tasksource-instruct', 'tasksource/tasksource-jev-typed-decisions'],
        'revisions': {'typed': TYPED_REV, 'instruct': INSTRUCT_REV},
        'instruct_retained': 0, 'A3_preserved_under_existing_manifest': A3_HASHES['manifest.json']}
    summary = {'status': 'locally_built', 'training_launched': False, 'task_cap': TASK_CAP,
        'upstream_family_cap': None, 'tranche_questions': len(additions),
        'tranche_tasks': len(tasks), 'tranche_upstream_families': len(family_counts),
        'source_question_counts': dict(sorted(tasks.items())), 'family_question_counts': dict(sorted(family_counts.items())),
        'heldout_questions': len(heldout), 'heldout_tasks': HOLDOUT,
        'heldout_families': sorted(holdout_families), 'scan': report,
        'quality_drops': {k: v for k, v in counters.items() if k != 'reused_verified_render_records'},
        'preparation_counts': prep['counters'], 'instruct_train_rows': prep['instruct_train_rows'],
        'instruct_commercial_rows': prep['instruct_commercial_rows'], 'instruct_retained': 0,
        'instruct_uncovered_tasks': prep['instruct_uncovered_tasks'], 'instruct_policy': prep['instruct_policy'],
        'A3_revision': A3_REVISION, 'A3_input_hashes': A3_HASHES,
        'A3_train_prefix_byte_identical': (output / 'train.jsonl').read_bytes().startswith((base / 'train.jsonl').read_bytes()),
        'A3_calibration_byte_identical': sha(output / 'calib.jsonl') == A3_HASHES['calib.jsonl'],
        'render_validation': {'prompt_budget': 1984, 'max_seq_length': 2048, 'padding_reserve': 64,
            'checked_renders': checked, 'max_prompt_tokens_by_type': dict(max_tokens),
            'reused_verified_records': counters.get('reused_verified_render_records', 0),
            'original_cache_checked_renders': previous['render_validation']['checked_renders'] if render_cache else 0,
            'tokenizer_sha256': TOKENIZER_SHA256, 'retained_truncated': 0},
        'files': {name: describe(str(output / name)) for name in lint}, 'lint': lint,
        'input_hashes': prep['input_hashes'], 'dataset': {'id': 'frontier-infra/jebadiah-data-t2b', 'revision': None, 'private': True}}
    dump(output / 't2b-build-summary.json', summary)
    dump(output / 't2b-source-licenses.json', license_manifest)
    return summary


def verify(research, output):
    """Verify selected new data and untouched A3 separately, then fail closed."""
    coverage = json.loads((output / 'index-receipt.json').read_text())
    index = output / 'selected-plus-item25.pkl'
    base = research / 'a3/a3'
    base_manifest = json.loads((base / 'manifest.json').read_text())
    summary = json.loads((output / 't2b-build-summary.json').read_text())
    raw_report = json.loads((output / 'raw-source-scan/scan-report.json').read_text())
    if (raw_report['candidates_sha256'] != sha(output / 'raw-source-scan/candidates.jsonl')
            or raw_report['protected_index_sha256'] != coverage['index_sha256']):
        raise ValueError('Raw-source scan binding failed')
    raw_bad = {row['family'] for row in json.loads((output / 'raw-source-scan/contamination-hits.json').read_text())}
    for name in ['tranche.jsonl', 'heldout.jsonl']:
        if any(row['family'] in raw_bad for row in read_rows(output / name)):
            raise ValueError('Raw-source gate excludes selected family; rerun finish')
    selected = output / 'selected-verification'; selected.mkdir(exist_ok=True)
    expected = hashlib.sha256()
    for name in ['tranche.jsonl', 'heldout.jsonl']:
        with (output / name).open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): expected.update(chunk)
    cached = json.loads((selected / 'scan-report.json').read_text()) if (selected / 'scan-report.json').exists() else {}
    if cached.get('candidates_sha256') != expected.hexdigest() or cached.get('protected_index_sha256') != coverage['index_sha256']:
        with (selected / 'candidates.jsonl').open('wb') as f:
            for name in ['tranche.jsonl', 'heldout.jsonl']:
                with (output / name).open('rb') as src: shutil.copyfileobj(src, f)
        cached = scan_stage(selected, index, coverage)
    if cached['direct_hits'] or cached['removed_records']: raise ValueError('Selected data failed full-suite scan')
    legacy = output / 'a3-verification'; legacy.mkdir(exist_ok=True)
    exposure = Counter(); legacy_raw_exposure = Counter(); split_ids = {}; held_families = set(summary['heldout_families'])
    with (legacy / 'candidates.jsonl').open('w') as f:
        for name in ['train.jsonl', 'calib.jsonl']:
            if sha(base / name) != A3_HASHES[name]: raise ValueError('A3 changed')
            for row in read_rows(base / name):
                meta = base_manifest['source_manifests'].get(row.get('subset'), {})
                source = meta.get('source_id') or row['source']
                match = re.search(r'(?:huggingface.co/datasets/)?([\w.-]+/[\w.-]+)', source)
                upstream = match.group(1) if match else source
                row['family'] = upstream
                if upstream in held_families: raise ValueError('Holdout family appeared in A3')
                exposure[upstream] += len(row['questions']); split_ids[row['id']] = name
                if upstream in raw_bad: legacy_raw_exposure[upstream] += len(row['questions'])
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    a3_scan = scan_stage(legacy, index, coverage)
    hits = json.loads((legacy / 'contamination-hits.json').read_text())
    bad = {row['family'] for row in hits}
    conflict = {'direct_hit_records': len(hits), 'direct_hits_by_split': dict(Counter(split_ids[r['id']] for r in hits)),
                'whole_family_question_exposures': {k: exposure[k] for k in sorted(bad)},
                'whole_family_records_removed_if_scrubbed': a3_scan['removed_records'],
                'A3_question_exposure_to_raw_rejected_families': dict(sorted(legacy_raw_exposure.items())),
                'unchanged_A3_zero_hit_gate': 'failed' if hits else 'passed', 'A3_modified': False}
    dump(output / 'a3-conflict.json', conflict)
    summary.update(raw_source_scan=raw_report, selected_scan=cached, A3_scan=a3_scan, A3_conflict=conflict,
                   heldout_families_absent_from_A3=True,
                   training_composition_zero_hit_gate='failed' if hits else 'passed',
                   status='blocked_A3_composition' if hits or legacy_raw_exposure else 'locally_verified',
                   target_tranche_questions=TRAIN_TARGET,
                   tranche_shortfall=TRAIN_TARGET - summary['tranche_questions'],
                   script_sha256=sha(Path(__file__)),
                   converter_sha256=sha(Path(__file__).with_name('convert_data.py')))
    summary['scan']['index_coverage'] = coverage
    summary['preparation_counts']['raw_source_scan_candidates'] = raw_report['scanned_records']
    summary['dataset'].update(published=False, revision=None)
    dump(output / 't2b-build-summary.json', summary)
    licenses = json.loads((output / 't2b-source-licenses.json').read_text())
    licenses['unchanged_A3_sources'] = base_manifest['source_manifests']
    licenses['required_collections'] = {
        'tasksource/tasksource-instruct': {'revision': INSTRUCT_REV, 'license': 'mixed source terms',
            'license_use_column': 'license_use', 'required_license_use': 'commercial',
            'commercial_train_rows': summary['instruct_commercial_rows'], 'retained_questions': 0,
            'policy': summary['instruct_policy']},
        'tasksource/tasksource-jev-typed-decisions': {'revision': TYPED_REV, 'license': 'mixed source terms',
            'license_use_column': 'license_use', 'required_license_use': 'commercial',
            'config': 'full', 'variants': ['direct'], 'retained_train_questions': summary['tranche_questions'],
            'retained_heldout_questions': summary['heldout_questions']}}
    dump(output / 't2b-source-licenses.json', licenses)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'index', 'raw-scan', 'scan', 'finish', 'verify'])
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=Path('/Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl'))
    args = parser.parse_args()
    for path in (args.research, args.output, args.index):
        if not str(path.resolve()).startswith('/Volumes/PRO-G40/'):
            parser.error('All build paths must stay on PRO-G40')
    if args.stage == 'prepare': result = prepare(args.research, args.output)
    elif args.stage == 'index': result = supplement_index(args.research, args.output, args.index)
    elif args.stage == 'raw-scan':
        receipt = json.loads((args.output / 'index-receipt.json').read_text())
        result = scan_stage(args.output / 'raw-source-scan', args.output / 'selected-plus-item25.pkl', receipt)
    elif args.stage == 'scan':
        receipt = json.loads((args.output / 'index-receipt.json').read_text()) if (args.output / 'index-receipt.json').exists() else None
        result = scan_stage(args.output, args.output / 'selected-plus-item25.pkl' if receipt else args.index, receipt)
    elif args.stage == 'finish': result = finish(args.research, args.output)
    else: result = verify(args.research, args.output)
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'tranche_questions', 'tranche_tasks', 'tranche_upstream_families', 'heldout_questions')}, indent=2))
    if args.stage == 'verify' and result['status'] == 'blocked_A3_composition': sys.exit(1)
