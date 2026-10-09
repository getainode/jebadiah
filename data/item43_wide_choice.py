"""Apache-2.0 owned finite taxonomies. CPU only; no model or external text inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from item33_skill_data import canonical, digest, sha, write_json, write_rows, validate_source
from lint_data import lint_record
from item33_full_suite_scan import scan, ITEM25_INDEX_SHA256

SEED = 'item43-n2-20261009'
SIZES = (32, 64, 128, 255)
SOURCE = 'item43-owned-taxonomies'
# Original Cartesian taxonomies. Each category requires both attributes.
TAXONOMIES = {
    'product': ('wood steel glass paper clay cork brass stone wool linen rubber silk foam wax tin felt',
                'tray cup box mat bowl plate jar rack bag lid bin pad tube case frame stand'),
    'routing': ('amber blue coral green ivory jade lilac navy olive peach pink red silver teal violet white',
                'desk gate dock hall yard bay room shop shed loft wing floor vault booth porch tower'),
    'intent': ('add cancel check copy count create delete edit find list move open print renew save send',
               'alert batch card chart file form group job label link note page plan report rule task'),
}


def oracle(request, criteria):
    matches = [key for key in criteria if key != 'none' and key == request]
    if len(matches) > 1:
        raise ValueError('Ambiguous taxonomy')
    return matches[0] if matches else 'none'


def generate():
    splits = {'train': [], 'diagnostic': []}
    for domain, (left, right) in TAXONOMIES.items():
        left, right = left.split(), right.split()
        universe = [a + ' ' + b for a in left for b in right]
        held = min(left, key=lambda a: digest(SEED + domain + a))
        for a in left:
            for b in right:
                target = a + ' ' + b
                family = f'item43:{domain}:{a}'
                for size in SIZES:
                    for mode in ('match', 'missing'):
                        key = f'{family}:{b}:{size}:{mode}'
                        rng = random.Random(digest(SEED + key))
                        # All one-attribute neighbors precede distant distractors.
                        hard = [x for x in universe if x != target and (x.split()[0] == a or x.split()[1] == b)]
                        far = [x for x in universe if x != target and x not in hard]
                        rng.shuffle(hard); rng.shuffle(far)
                        selected = ([target] if mode == 'match' else []) + (hard + far)[:size - 1 - (mode == 'match')]
                        selected.append('none'); rng.shuffle(selected)
                        criteria = {x: ('None of these' if x == 'none' else None) for x in selected}
                        q = {'type': 'choice', 'instructions': 'Select the exact two-attribute category. Both attributes must match. If absent, select none.', 'criteria': criteria}
                        row = {'id': 'item43:' + digest(key)[:24], 'set': 'item43-n2', 'subset': SOURCE,
                               'source': SOURCE, 'license': 'Apache-2.0', 'family': family, 'family_id': family,
                               'skill': domain, 'state': f'Request: {target}. Menu size: {size}.',
                               'questions': {'decision': q}, 'label': {'decision': oracle(target, criteria)}}
                        splits['diagnostic' if a == held else 'train'].append(row)
    return splits


def audit(splits):
    ids, families = set(), {}
    for split, rows in splits.items():
        for r in rows:
            errors = []; lint_record(r, r['id'], errors, max_options=255)
            if errors: raise ValueError('; '.join(errors))
            if r['id'] in ids: raise ValueError('Duplicate ID')
            ids.add(r['id'])
            if families.setdefault(r['family_id'], split) != split: raise ValueError('Family leakage')
            q = r['questions']['decision']; request = r['state'].split('Request: ', 1)[1].split('. Menu size:')[0]
            if oracle(request, q['criteria']) != r['label']['decision']: raise ValueError('Wrong executable label')


def render_check(splits, tokenizer_path):
    from transformers import AutoTokenizer
    from jebadiah_prompt import Renderer, PROMPT_SOURCE_SHA256
    # Importing model helpers would require torch; template hash is pure text.
    import hashlib
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / 'results/runs/9b-chat-v1/adapter/prompt_contract.json').read_text())
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    template_hash = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    if sha(tokenizer_path / 'tokenizer.json') != '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42': raise ValueError('Wrong tokenizer')
    if PROMPT_SOURCE_SHA256 != contract['prompt_source_sha256'] or template_hash != contract['chat_template_sha256']: raise ValueError('Wrong prompt contract')
    renderer = Renderer(tok, max_tokens=1984)
    if renderer.max_options_extended < 255: raise ValueError('Fewer than 255 single-token labels')
    maximum = {}; schemes = Counter(); checked = 0
    for rows in splits.values():
        for r in rows:
            q = r['questions']['decision']; size = len(q['criteria'])
            # Labels and content are order-independent in token length, but verify
            # canonical, reverse and seeded shuffled orders explicitly for every row.
            shuffled = list(q['criteria']); random.Random(digest(r['id'])).shuffle(shuffled)
            for order in (list(q['criteria']), list(reversed(q['criteria'])), shuffled):
                rendered = renderer.render(r['state'], q, order)
                n = len(tok.encode(rendered.prompt, add_special_tokens=False))
                if rendered.truncated or n > 1984: raise ValueError(f'Truncation/overflow at {r["id"]}: {n}')
                if len(set(rendered.cand_ids)) != size or len(rendered.keys) != size: raise ValueError('Incomplete candidate competition')
                maximum[str(size)] = max(maximum.get(str(size), 0), n); schemes[rendered.label_scheme] += 1; checked += 1
    return {'checked_renders': checked, 'max_prompt_tokens_by_size': maximum, 'max_seq_length': 2048,
            'prompt_budget': 1984, 'padding_reserve': 64, 'truncated': 0, 'order_checks': ['canonical', 'reversed', 'seeded_shuffle'],
            'label_schemes': dict(schemes), 'available_single_token_labels': renderer.max_options_extended,
            'prompt_source_sha256': PROMPT_SOURCE_SHA256, 'chat_template_sha256': template_hash,
            'tokenizer_sha256': sha(tokenizer_path / 'tokenizer.json')}


def counts(rows):
    return {'questions': len(rows), 'per_size': dict(Counter(str(len(r['questions']['decision']['criteria'])) for r in rows)),
            'domains': dict(Counter(r['skill'] for r in rows)), 'labels': dict(Counter('none' if r['label']['decision'] == 'none' else 'match' for r in rows))}


def build(output, tokenizer, licenses, index=None):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    policy = json.loads(licenses.read_text())
    for source in policy['sources'].values():
        validate_source(source, policy['source_exclusions'])
        if source['generator_sha256'] != sha(Path(__file__)): raise ValueError('Generator hash mismatch')
    splits = generate(); audit(splits)
    render = render_check(splits, tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'pending_studio_scan', 'protected_index_sha256': ITEM25_INDEX_SHA256, 'source_level': True}
    if index is not None:
        # Never copy the private index; Studio-only path and hash gate precede load.
        if str(index.resolve()) != '/Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl': raise ValueError('Scan must run on Studio at protected index location')
        if sha(index) != ITEM25_INDEX_SHA256: raise ValueError('Wrong protected index')
        work = output / 'source-scan'; work.mkdir()
        write_rows(work / 'candidates.jsonl', [{**r, 'family': SOURCE} for rows in splits.values() for r in rows])
        (work / 'protected.pkl').symlink_to(index.resolve())
        try: scan(work)
        finally: (work / 'protected.pkl').unlink()
        report = json.loads((work / 'scan-report.json').read_text())
        report.update(status='passed' if report['removed_records'] == 0 else 'rejected', source_level=True,
                      protected_index_sha256=ITEM25_INDEX_SHA256, scanner_sha256=sha(Path(__file__).with_name('item33_full_suite_scan.py')),
                      candidates_sha256=sha(work / 'candidates.jsonl'))
        write_json(output / 'overlap-scan-report.json', report)
        if report['status'] != 'passed': raise ValueError('Entire owned source rejected')
    write_json(output / 'overlap-scan-report.json', report)
    for split, rows in splits.items(): write_rows(output / f'{split}.jsonl', rows)
    manifest = {'name': 'item43-n2', 'seed': SEED, 'generator_sha256': sha(Path(__file__)),
                'license_manifest_sha256': sha(licenses), 'render_validation': render, 'overlap_scan': report,
                'splits': {s: counts(rs) for s, rs in splits.items()},
                'files': {f'{s}.jsonl': {'sha256': sha(output / f'{s}.jsonl')} for s in splits}}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path); p.add_argument('--tokenizer', required=True, type=Path)
    p.add_argument('--index', type=Path)
    p.add_argument('--licenses', type=Path, default=Path(__file__).with_name('manifests') / 'item43-source-licenses.json')
    args = p.parse_args(); print(json.dumps(build(args.out, args.tokenizer, args.licenses, args.index), indent=2))
