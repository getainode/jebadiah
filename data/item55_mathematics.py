# SPDX-License-Identifier: Apache-2.0
"""Fresh pinned DeepMind arithmetic, exact CPU conversion; lead-only scan gate."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch
from item33_skill_data import canonical, digest, sha, norm, write_json, write_rows, validate_source
from item32_ablation_mix import read
from item36_skill_mix import A3_HASHES, A3_REVISION
from item49_corr2cause import renderer, ITEM25_INDEX_SHA256, TOKENIZER_SHA256
from item54_massive import scan_phase, stream_hash
from lint_data import lint_record

SOURCE = 'deepmind-mathematics-fresh'
REVISION = '427f45075f84b8b9774950196ad63867ca20ffb3'
SEED = 55020261010
MODULES = ('add_sub_multiple', 'mul_div_multiple', 'mixed')
PER_MODULE = 4000
POLICY = Path(__file__).with_name('manifests') / 'item55-source-licenses.json'
DEPENDENCIES = {'numpy': '1.26.4', 'sympy': '1.12', 'absl-py': '2.1.0', 'six': '1.17.0'}
CHOICE = 'Select the exact answer to the supplied mathematics question.'
VERIFY = 'Is the proposed answer the exact answer to the supplied mathematics question?'


def exact(expression):
    """Independent whitelist AST interpreter, no eval or floating point."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError('Non-rational or unsupported arithmetic syntax')
    return visit(ast.parse(expression, mode='eval').body)


def tree(op):
    name = type(op).__name__
    if name == 'Constant': return ['Constant', str(op.value)]
    if name not in ('Add', 'Sub', 'Mul', 'Div', 'Neg'):
        raise ValueError('Unsupported upstream operation: ' + name)
    children = op.children
    if isinstance(children, dict):
        keys = {'Sub': ('left', 'right'), 'Div': ('numer', 'denom'), 'Neg': ('input',)}[name]
        children = [children[k] for k in keys]
    return [name, *[tree(x) for x in children]]


def structure(program):
    if program[0] == 'Constant': return 'C'
    # Sign of a numeric leaf is a number change, not a new program family.
    if program[0] == 'Neg' and program[1][0] == 'Constant': return 'C'
    children = [structure(x) for x in program[1:]]
    # Upstream prints nested Add/Mul without associative parentheses. Treat
    # those hidden grouping variants as the same expression family.
    if program[0] in ('Add', 'Mul'):
        children = [z for child in children for z in
                    (child[1:] if isinstance(child, list) and child[0] == program[0] else [child])]
    return [program[0], *children]


def tree_value(program):
    if program[0] == 'Constant': return Fraction(program[1])
    args = [tree_value(x) for x in program[1:]]
    if program[0] == 'Neg' and len(args) == 1: return -args[0]
    if len(args) < 2: raise ValueError('Invalid operation arity')
    result = args[0]
    for x in args[1:]:
        if program[0] == 'Add': result += x
        elif program[0] == 'Sub': result -= x
        elif program[0] == 'Mul': result *= x
        elif program[0] == 'Div': result /= x
        else: raise ValueError('Unknown operation')
    return result


def family(raw):
    return 'item55:program:' + digest(canonical(structure(raw['program'])))


def check_upstream(upstream, policy):
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION: raise ValueError('Wrong pinned upstream commit')
    # Verify the entire imported Python tree, not only the module entry point.
    actual = {str(p.relative_to(upstream)): sha(p) for p in sorted(upstream.rglob('*.py')) if '.git' not in p.parts}
    if actual != policy['upstream_python_files']: raise ValueError('Upstream Python tree modified')
    for name, expected in policy['upstream_files'].items():
        if sha(upstream / name) != expected: raise ValueError('Upstream rights/file hash mismatch')
    for name, version in DEPENDENCIES.items():
        if importlib.metadata.version(name) != version: raise ValueError('Wrong generator dependency: ' + name)


def is_lineage(row):
    import re
    text = canonical({k:row.get(k) for k in ('source','source_family','subset','family','provenance')}).lower()
    return SOURCE in text or 'mathematics_dataset' in text or bool(re.search(r'deepmind[-_/ ]?(?:mathematics|math)',text))


def audit_raw(rows):
    if len(rows) != PER_MODULE * len(MODULES): raise ValueError('Incomplete finite source')
    for position, r in enumerate(rows):
        module = MODULES[position // PER_MODULE]; i = position % PER_MODULE
        seed = int(digest(f'{SEED}:{module}:{i}')[:8],16)
        expected_question = {r['expression'], 'What is '+r['expression']+'?', 'Evaluate '+r['expression']+'.',
                             'Calculate '+r['expression']+'.', 'What is the value of '+r['expression']+'?'}
        if (r['id'] != f'item55:raw:{module}:{i}' or r['module'] != module or r['index'] != i
                or r['sample_seed'] != seed or r['question'] not in expected_question
                or exact(r['expression']) != Fraction(r['answer'])
                or tree_value(r['program']) != Fraction(r['answer'])):
            raise ValueError('Invalid finite-source provenance/answer')
        def steps(program):
            return 0 if program[0] == 'Constant' else 1+sum(steps(x) for x in program[1:])
        if steps(r['program']) < 3: raise ValueError('Single-step source is excluded')


def generate(upstream):
    sys.path.insert(0, str(upstream.resolve()))
    import numpy as np
    from mathematics_dataset.modules import arithmetic as modules
    from mathematics_dataset.sample import arithmetic as sampler
    from mathematics_dataset.util import composition
    original = sampler.arithmetic
    rows = []
    for module in MODULES:
        for i in range(PER_MODULE):
            seed = int(digest(f'{SEED}:{module}:{i}')[:8], 16)
            random.seed(seed); np.random.seed(seed)
            captured = []
            def capture(*args, **kwargs):
                op = original(*args, **kwargs); captured.append(op); return op
            # One pure module; 3 to 7 generated binary steps. No external
            # entities, data files, benchmark templates or pre-generated release.
            with patch.object(sampler, 'arithmetic', capture):
                problem = getattr(modules, module)(
                    'int' if module == 'add_sub_multiple' else 'rational',
                    composition.PreSampleArgs(1, 1, 6, 10), length=3 + i % 5)
            if len(captured) != 1: raise ValueError('Unexpected composed source call')
            op = captured[0]
            raw = {'id': f'item55:raw:{module}:{i}', 'module': module, 'index': i,
                   'sample_seed': seed, 'question': str(problem.question),
                   'answer': str(problem.answer), 'expression': str(op), 'program': tree(op)}
            if exact(raw['expression']) != Fraction(raw['answer']) or tree_value(raw['program']) != Fraction(raw['answer']):
                raise ValueError('Independent exact oracle disagrees with upstream')
            rows.append(raw)
    return rows


def convert(raw):
    gold = exact(raw['expression'])
    if tree_value(raw['program']) != gold or Fraction(raw['answer']) != gold:
        raise ValueError('Wrong upstream gold/program')
    key = raw['id']; h = int(digest(str(SEED) + key)[:8], 16)
    kind = ('choice', 'noul')[h % 2]
    state = {'problem': raw['question']}
    q = {'type': kind, 'instructions': CHOICE if kind == 'choice' else VERIFY}
    if kind == 'choice':
        answers = [str(gold + d) for d in (0, 1, -1, 2)]
        random.Random(h).shuffle(answers)
        q['criteria'] = {x: None for x in answers}; label = str(gold)
    else:
        label = (h // 2) % 2 == 0
        state['proposed_answer'] = str(gold if label else gold + (1 if h % 3 else -1))
    return {'id': key.replace(':raw:', ':question:'), 'set': 'item55-rung5',
            'source': SOURCE, 'source_family': SOURCE, 'subset': SOURCE,
            'license': 'Apache-2.0', 'area': 'knowledge', 'skill': 'multi-step-arithmetic',
            'family': family(raw), 'family_id': family(raw), 'state': state,
            'questions': {'decision': q}, 'label': {'decision': label},
            'provenance': {'repo': 'google-deepmind/mathematics_dataset', 'revision': REVISION,
                           'module': raw['module'], 'sample_seed': raw['sample_seed'],
                           'original_id': raw['id'], 'expression': raw['expression'],
                           'program': raw['program'], 'upstream_answer': raw['answer']}}


def audit(splits):
    ids = set(); texts = set(); families = {}
    for split, rows in splits.items():
        for r in rows:
            errors = []; lint_record(r, r['id'], errors)
            if errors: raise ValueError('; '.join(errors))
            p = r['provenance']
            raw = {'id': p['original_id'], 'module': p['module'], 'sample_seed': p['sample_seed'],
                   'question': r['state']['problem'], 'answer': p['upstream_answer'],
                   'expression': p['expression'], 'program': p['program']}
            if r != convert(raw): raise ValueError('Noncanonical or corrupted conversion')
            text = norm(raw['question'])
            if r['id'] in ids or text in texts: raise ValueError('Duplicate original question')
            ids.add(r['id']); texts.add(text)
            if families.setdefault(r['family_id'], split) != split: raise ValueError('Expression-structure leakage')


def select(raw, base_rows):
    pools = defaultdict(list); seen = set(); old = {norm(canonical(r['state'])) for r in base_rows}
    skipped = Counter()
    for upstream in raw:
        r = convert(upstream); text = norm(upstream['question'])
        if text in seen or norm(canonical(r['state'])) in old:
            skipped['duplicate_or_a3_state'] += 1; continue
        seen.add(text)
        held = int(r['family_id'].rsplit(':', 1)[1][:8], 16) % 5 == 0
        pools[('diagnostic' if held else 'train', upstream['module'], r['questions']['decision']['type'])].append(r)
    splits = {'train': [], 'diagnostic': []}
    for split, per_cell in (('train', 1000), ('diagnostic', 50)):
        for module in MODULES:
            for kind in ('choice', 'noul'):
                pool = sorted(pools[(split, module, kind)], key=lambda r: digest(str(SEED) + r['id']))
                if len(pool) < per_cell: raise ValueError(f'Insufficient program-disjoint candidates: {split}/{module}/{kind}: {len(pool)}')
                splits[split].extend(pool[:per_cell])
    audit(splits)
    return splits, {'policy': 'Global ordered operation-tree family, constants and numeric signs erased, associative Add/Mul flattened; hash mod 5 reserves diagnostic structures across all modules.',
                    'skipped': dict(skipped), 'available_by_split_module_type': {':'.join(k):len(v) for k,v in pools.items()}}


def counts(rows):
    return {'questions': len(rows), 'types': dict(Counter(r['questions']['decision']['type'] for r in rows)),
            'modules': dict(Counter(r['provenance']['module'] for r in rows)),
            'structure_families': len({r['family_id'] for r in rows})}


def render_check(splits, tok, rend, contract):
    maximum = Counter(); checked = 0
    for rows in splits.values():
        for row in rows:
            q = row['questions']['decision']; kind = q['type']
            if kind == 'choice':
                order = list(q['criteria']); shuffled = list(order)
                random.Random(digest(str(SEED) + row['id'])).shuffle(shuffled)
                orders = (order, list(reversed(order)), shuffled)
            else: orders = (None,)
            for order in orders:
                out = rend.render(row['state'], q, order)
                n = len(tok.encode(out.prompt, add_special_tokens=False))
                expected = 4 if kind == 'choice' else 2
                if out.truncated or n > 1984 or len(set(out.cand_ids)) != expected or len(out.cand_ids) != expected:
                    raise ValueError('Prompt truncation or incomplete single-token answer competition')
                maximum[kind] = max(maximum[kind], n); checked += 1
    return {'checked_renders':checked, 'max_prompt_tokens_by_type':dict(maximum),
            'max_seq_length':2048, 'prompt_budget':1984, 'padding_reserve':64, 'truncated':0,
            'choice_orders':['canonical','reversed','seeded-shuffled'], 'noul_order':'true,false',
            'tokenizer_sha256':TOKENIZER_SHA256, 'prompt_source_sha256':contract['prompt_source_sha256'],
            'chat_template_sha256':contract['chat_template_sha256']}


def check_base(base):
    for name, expected in A3_HASHES.items():
        if sha(base / name) != expected: raise ValueError('Wrong pinned A3: ' + name)


def scan_rows(raw):
    # Entire finite generated universe in raw and converted form, before sampling.
    for r in raw:
        yield {'id': r['id'], 'family': SOURCE, 'state': {'question': r['question'], 'expression': r['expression'], 'answer': r['answer']}, 'questions': {}}
        yield {**convert(r), 'family': SOURCE}


def build(upstream, base, tokenizer, output, licenses=POLICY, index=None):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    policy = json.loads(licenses.read_text()); source = policy['sources'][SOURCE]
    validate_source(source, policy['source_exclusions'])
    if (source['revision'] != REVISION or source['license'] != 'Apache-2.0'
            or source['generator_sha256'] != sha(Path(__file__))
            or source['ancestry_verified'] is not True or source['model_redistribution_ok'] is not True
            or source['model_written_training_text'] is not False): raise ValueError('Unverified converter/provenance')
    evidence = Path(__file__).resolve().parents[1] / 'results/research/item55-source-evidence.json'
    if sha(evidence) != source['source_evidence_sha256']: raise ValueError('Live evidence hash mismatch')
    check_upstream(upstream, policy); check_base(base)
    raw = generate(upstream)
    audit_raw(raw)
    if len(raw) != len(MODULES) * PER_MODULE: raise ValueError('Incomplete generated universe')
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'raw-source.jsonl', raw)
    write_rows(output / 'scan-candidates.jsonl', scan_rows(raw))
    for name, key in (('raw-source.jsonl','finite_source_raw_sha256'),('scan-candidates.jsonl','finite_source_scan_sha256')):
        if key in source and sha(output/name) != source[key]: raise ValueError('Frozen finite universe changed')
    report = {'status': 'pending_studio_scan', 'source_level': True, 'protected_index_sha256': ITEM25_INDEX_SHA256,
              'candidates_sha256': sha(output / 'scan-candidates.jsonl'), 'expected_records': 2 * len(raw)}
    if index is not None:
        report = scan_phase(scan_rows(raw), output, index, 'full-source-scan')
    write_json(output / 'overlap-scan-report.json', report)
    splits, selection = select(raw, read(base / 'train.jsonl') + read(base / 'calib.jsonl'))
    tok, rend, contract = renderer(tokenizer)
    render = render_check(splits, tok, rend, contract)
    for s, rs in splits.items(): write_rows(output / (s + '.jsonl'), rs)
    (output / 'NOTICE-DeepMind.txt').write_text(source['notice_text'])
    (output / 'LICENSE-DeepMind.txt').write_bytes((upstream / 'LICENSE').read_bytes())
    manifest = {'name':'item55-rung5-mathematics','seed':SEED,'source_revision':REVISION,
                'generator_sha256':sha(Path(__file__)),'license_manifest_sha256':sha(licenses),
                'a3_revision':A3_REVISION,'a3_input_hashes':A3_HASHES,
                'upstream_python_files':policy['upstream_python_files'],'upstream_files':policy['upstream_files'],
                'dependencies':DEPENDENCIES,'raw_questions':len(raw),'selection':selection,
                'splits':{s:counts(rs) for s,rs in splits.items()},'render_validation':render,'overlap_scan':report,
                'files':{n:{'sha256':sha(output/n)} for n in ('train.jsonl','diagnostic.jsonl','raw-source.jsonl','scan-candidates.jsonl','NOTICE-DeepMind.txt','LICENSE-DeepMind.txt')},
                'model_calls':0,'training_launched':False}
    write_json(output/'manifest.json',manifest); return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','base','tokenizer','out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--licenses',type=Path,default=POLICY); p.add_argument('--index',type=Path)
    a=p.parse_args(); print(json.dumps(build(a.upstream,a.base,a.tokenizer,a.out,a.licenses,a.index),indent=2))
