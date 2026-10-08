"""Local-only rung 3 data builder: human intent routing and executable scenarios.

This program never calls a model, launches training, downloads data or reads
benchmark text outside the contamination scanner. Outputs stay private/local.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
from lint_data import lint_record
from item33_full_suite_scan import ITEM25_INDEX_SHA256, ITEM25_SCANNER_SHA256, scan

SEED = 'item33-rung3-20261008'
SKILLS = ('intent_routing', 'tool_timing', 'grounding')
HUMAN = 'hwu-original'
TOOLS = 'item33-tool-worlds'
GROUND = 'item33-evidence-worlds'
HWU_REVISION = 'f6071b496b17d71e6eb43f543af0707f4ff30557'
HWU_FILES = ('AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv',
             'Collected-Original-Data/paraphrases_and_intents_26k_normalised_all.csv')
DOMAINS = ('parcel', 'reservation', 'invoice', 'device', 'inventory',
           'appointment', 'subscription', 'shipment', 'access_pass', 'expense')
ACTIONS = {
    'call_tool': 'Invoke the available tool now.',
    'ask_user': 'Ask the user for the missing required argument.',
    'answer_context': 'Answer using the fresh, complete context already available.',
    'decline': 'Decline the action forbidden by the explicit policy.',
    'explain_unavailable': 'Explain that the required capability is unavailable.',
}
SUPPORT = ['No response claims are supported by the supplied evidence.',
           'One of the two response claims is supported by the supplied evidence.',
           'Both response claims are supported by the supplied evidence.']


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def norm(value):
    return ' '.join(re.findall(r'\w+', value.casefold()))


def ordered(values, salt):
    return sorted(values, key=lambda x: digest(SEED + ':' + salt + ':' + str(x)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_rows(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def record(source, skill, family, key, state, kind, instruction, criteria, label):
    q = {'type': kind, 'instructions': instruction}
    if criteria is not None:
        q['criteria'] = criteria
    return {'id': 'item33:' + source + ':' + digest(key)[:24],
            'set': 'item33-rung3', 'subset': source, 'source': source,
            'license': 'CC-BY-4.0' if source == HUMAN else 'Apache-2.0',
            'skill': skill, 'family': family, 'family_id': family,
            'state': state, 'questions': {'decision': q}, 'label': {'decision': label}}


def validate_source(info, exclusions):
    for field in ('source_id', 'url', 'revision', 'license', 'ancestry', 'origin'):
        if not info.get(field):
            raise ValueError('Missing source provenance: ' + field)
    if info.get('commercial_use_ok') is not True:
        raise ValueError('Commercial permission required')
    if info['license'].lower() not in exclusions['license_allowlist']:
        raise ValueError('License not allowlisted')
    ancestry = canonical([info['source_id'], info['url'], info['ancestry']]).lower()
    for term in exclusions['source_substrings']:
        if re.search(r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])', ancestry):
            raise ValueError('Protected source ancestry: ' + term)
    if info['origin'] == 'procedural':
        if info.get('generator_license') != 'Apache-2.0' or not info.get('generator_sha256'):
            raise ValueError('Procedural generator license/hash required')


def raw_human(path):
    """Both complete master tables, including rows the annotators rejected.

    CrossValidation is a shuffled conversion of these masters, not another
    collection. Do not sample, filter statuses or discard overlaps here.
    """
    result = []
    for relative in HWU_FILES:
        with (path / relative).open() as f:
            for i, row in enumerate(csv.DictReader(f, delimiter=';')):
                result.append({'id': 'hwu-raw:' + relative + ':' + str(i),
                               'subset': HUMAN, 'family': HUMAN,
                               'state': {k: v for k, v in row.items()
                                         if k not in ('userid', 'answerid')},
                               'questions': {'check': {'type': 'noul',
                                              'instructions': 'Assess this source record.'}},
                               'label': {'check': True}})
    return result


def human_questions(path):
    with (path / HWU_FILES[0]).open() as f:
        masters = list(csv.DictReader(f, delimiter=';'))
    # Ambiguous normalized utterances are dropped before any split is assigned.
    utterances = defaultdict(set)
    for r in masters:
        if not r['status'].startswith('IRR_') and r['answer_normalised'].strip():
            utterances[norm(r['answer_normalised'])].add(r['scenario'] + '.' + r['intent'])
    intents = sorted({next(iter(v)) for v in utterances.values() if len(v) == 1})
    rows, seen = [], set()
    for r in masters:
        text = r['answer_normalised'].strip()
        if r['status'].startswith('IRR_') or not text:
            continue
        key = norm(text)
        if len(utterances[key]) != 1 or key in seen:
            continue
        seen.add(key)
        gold = r['scenario'] + '.' + r['intent']
        # The elicitation intent is the family: every human paraphrase and both
        # question variants stay together, stricter than utterance-only splits.
        family = 'item33:hwu-intent:' + gold
        rng = random.Random(digest(SEED + ':menu:' + key))
        distractors = rng.sample([x for x in intents if x != gold], 15)
        menu = distractors + [gold]
        rng.shuffle(menu)
        descriptions = {x: x.replace('.', ': ').replace('_', ' ') for x in menu}
        state = {'user_request': text}
        rows.append(record(HUMAN, 'intent_routing', family, key + ':choice', state,
                           'choice', 'Route this user request to the most specific intent in the supplied menu.',
                           descriptions, gold))
        proposed = gold if int(digest(key)[:8], 16) % 2 == 0 else rng.choice(distractors)
        rows.append(record(HUMAN, 'intent_routing', family, key + ':noul',
                           {'user_request': text, 'proposed_intent': proposed}, 'noul',
                           'Does the proposed intent describe the action requested by the user?',
                           None, proposed == gold))
    return rows


def tool_oracle(world):
    """Precedence is policy, complete fresh context, capability, arguments, call.

    It deliberately does not use the generation mode or a gold hint.
    """
    request = world['request']
    if request['operation'] not in world['policy']['allowed_operations']:
        return 'decline'
    if (request['operation'] == 'read' and world['context']['complete']
            and world['context']['record_id'] == request['arguments'].get('record_id')
            and world['context']['age_minutes'] <= world['policy']['max_age_minutes']):
        return 'answer_context'
    compatible = [t for t in world['tools'] if t['operation'] == request['operation']]
    if not compatible:
        return 'explain_unavailable'
    if any(any(not request['arguments'].get(k) for k in t['required_arguments']) for t in compatible):
        return 'ask_user'
    return 'call_tool'


def tool_questions():
    rows = []
    for domain in DOMAINS:
        for mode in range(5):
            family = f'item33:tool:{domain}:{mode}'
            for i in range(10):
                rng = random.Random(digest(SEED + ':' + family + ':' + str(i)))
                operation = 'read' if mode in (0, 2, 4) else 'update'
                identifier = domain[:3].upper() + '-' + str(rng.randrange(10000, 99999))
                world = {
                    'request': {'message': f'{"Check the current state of" if operation == "read" else "Set the notification method to email for"} {domain.replace("_", " ")} {identifier}.',
                                'operation': operation, 'arguments': {'record_id': identifier}},
                    'tools': [{'name': domain + '_' + operation, 'operation': operation,
                               'description': f'{operation.capitalize()} an external {domain.replace("_", " ")} record.',
                               'required_arguments': ['record_id']}],
                    'context': {'complete': False, 'age_minutes': rng.randrange(20, 60),
                                'record_id': identifier, 'record_state': 'queued'},
                    'policy': {'allowed_operations': ['read', 'update'], 'max_age_minutes': 5,
                               'decision_precedence': 'Refuse forbidden operations first. Use complete fresh context for reads of the same record. Otherwise check capability, then arguments, before calling.'},
                }
                if operation == 'update':
                    world['request']['arguments']['channel'] = 'email'
                    world['tools'][0]['required_arguments'].append('channel')
                if mode == 1:
                    world['request']['arguments'] = {'channel': 'email'}
                    world['context'].update(record_id=None, record_state=None)
                    world['request']['message'] = f'Set the notification method to email for my {domain.replace("_", " ")}.'
                elif mode == 2:
                    world['context'].update(complete=True, age_minutes=rng.randrange(0, 6))
                elif mode == 3:
                    world['policy']['allowed_operations'] = ['read']
                elif mode == 4:
                    world['tools'][0].update(operation='update', name=domain + '_update',
                                             description='Update the notification method of an external record.',
                                             required_arguments=['record_id', 'channel'])
                # Vary independent competing conditions to exercise precedence.
                if mode == 3 and i % 2 == 0:
                    world['tools'] = []
                    world['request']['arguments'] = {}
                if mode == 2 and i % 2 == 0:
                    world['tools'] = []
                if mode == 0 and i % 2 == 0:
                    world['context'].update(complete=True, age_minutes=6 + i)
                if mode == 0 and i % 3 == 0:
                    world['context'].update(complete=True, age_minutes=0, record_id='unrelated-record')
                gold = tool_oracle(world)
                menu = list(ACTIONS)
                rng.shuffle(menu)
                rows.append(record(TOOLS, 'tool_timing', family, family + f':{i}:action', world,
                                   'choice', 'What should the assistant do next under the supplied policy?',
                                   {x: ACTIONS[x] for x in menu}, gold))
                # A second choice avoids making this a binary call/no-call toy.
                candidate = gold if i % 2 == 0 else rng.choice([x for x in ACTIONS if x != gold])
                rows.append(record(TOOLS, 'tool_timing', family, family + f':{i}:proposal',
                                   {**world, 'proposed_next_step': candidate}, 'choice',
                                   'Assess the proposed next step under the supplied policy.',
                                   {'appropriate': 'The proposed action is appropriate.',
                                    'inappropriate': 'A different action is required.'},
                                   'appropriate' if candidate == gold else 'inappropriate'))
                proposal = gold if i % 2 == 0 else rng.choice([x for x in ACTIONS if x != gold])
                rows.append(record(TOOLS, 'tool_timing', family, family + f':{i}:noul',
                                   {**world, 'proposed_next_step': proposal}, 'noul',
                                   'Is the proposed next step appropriate under the supplied policy?', None, proposal == gold))
    return rows


def evidence_oracle(world):
    """Evaluate atomic claims from source evidence with a closed evidence scope.

    Claims after 'not' mean explicit inequality with a present fact; absence
    never supplies support for either a positive or a negative assertion.
    """
    facts = {(f['entity'], f['attribute']): f['value'] for f in world['evidence']}
    supported = []
    for claim in world['response_claims']:
        key = (claim['entity'], claim['attribute'])
        known = key in facts
        matches = known and facts[key] == claim['value']
        supported.append((known and not matches) if claim['negated'] else matches)
    return sum(supported)


def evidence_text(world, domain):
    attributes = {'status': 'processing status', 'amount': 'billed amount',
                  'arrival_date': 'arrival date', 'approval': 'approval status'}
    def sentence(fact, negated=False):
        verb = 'is not' if negated else 'is'
        return f"The {attributes[fact['attribute']]} of {domain.replace('_', ' ')} {fact['entity']} {verb} {fact['value']}."
    return {'source_evidence': [sentence(f) for f in world['evidence']],
            'assistant_response': ' '.join(sentence(c, c['negated']) for c in world['response_claims']),
            'evidence_scope': 'The supplied records are the only evidence. An absent fact supports neither an assertion nor its negation.'}


def evidence_questions():
    rows = []
    for domain in DOMAINS:
        for mode in range(5):
            family = f'item33:evidence:{domain}:{mode}'
            for i in range(10):
                rng = random.Random(digest(SEED + ':' + family + ':' + str(i)))
                entity = domain[:3].upper() + '-' + str(rng.randrange(10000, 99999))
                other = domain[:3].upper() + '-' + str(rng.randrange(10000, 99999))
                amount = str(rng.randrange(10, 900)) + ' credits'
                facts = [dict(entity=entity, attribute='status', value='queued'),
                         dict(entity=entity, attribute='amount', value=amount),
                         dict(entity=other, attribute='status', value='complete')]
                claims = [dict(entity=entity, attribute='status', value='queued', negated=False),
                          dict(entity=entity, attribute='amount', value=amount, negated=False)]
                # Rotate 0/1/2 supported claims within every template family.
                desired = (i + mode) % 3
                for j, claim in enumerate(claims):
                    supported = j < desired
                    if mode == 0:
                        if not supported:
                            claim['value'] = 'complete' if j == 0 else str(int(amount.split()[0]) + 3) + ' credits'
                    elif mode == 1:
                        if not supported:
                            claim.update(attribute='arrival_date', value='next Tuesday')
                    elif mode == 2:
                        if not supported:
                            claim['entity'] = other  # different entity, different status; no amount
                    elif mode == 3:
                        claim['negated'] = True
                        if supported:
                            claim['value'] = 'complete' if j == 0 else str(int(amount.split()[0]) + 3) + ' credits'
                    else:
                        claim['negated'] = True
                        if supported:
                            claim['value'] = 'complete' if j == 0 else str(int(amount.split()[0]) + 3) + ' credits'
                        else:
                            claim.update(attribute='approval', value='approved')
                rng.shuffle(facts)
                rng.shuffle(claims)
                world = {'evidence': facts, 'response_claims': claims,
                         'scope': 'Judge only the supplied evidence. Do not infer absent facts. Claims use entity, attribute, value and explicit negation.'}
                gold = evidence_oracle(world)
                world = evidence_text(world, domain)
                rows.append(record(GROUND, 'grounding', family, family + f':{i}:choice', world,
                                   'choice', 'How well is the response grounded in the supplied evidence?',
                                   {'none': SUPPORT[0], 'partial': SUPPORT[1], 'full': SUPPORT[2]},
                                   ('none', 'partial', 'full')[gold]))
                rows.append(record(GROUND, 'grounding', family, family + f':{i}:noul', world,
                                   'noul', 'Is every response claim supported by the supplied evidence?', None, gold == 2))
                rows.append(record(GROUND, 'grounding', family, family + f':{i}:score', world,
                                   'score', 'Rate the response by the number of evidence-supported claims.', SUPPORT, gold))
    return rows


def split_select(rows, skill):
    families = ordered({r['family_id'] for r in rows}, 'calibration:' + skill)
    # Ten percent of whole elicitation/template families, rather than IDs.
    if skill in ('tool_timing', 'grounding'):
        # One complete domain/template family per scenario mode keeps all five
        # boundary conditions represented in calibration.
        calibration = {next(f for f in families if f.endswith(':' + str(mode))) for mode in range(5)}
    else:
        calibration = set(families[:max(1, len(families) // 10)])
    splits = {'train': [], 'calib': []}
    quotas = {'intent_routing': {'choice': 1000, 'noul': 500},
              'tool_timing': {'choice': 1000, 'noul': 500},
              'grounding': {'choice': 500, 'noul': 500, 'score': 500}}[skill]
    for kind, total in quotas.items():
        for split in splits:
            quota = total // 10 if split == 'calib' else total - total // 10
            groups = defaultdict(list)
            for r in rows:
                actual = 'calib' if r['family_id'] in calibration else 'train'
                if actual == split and r['questions']['decision']['type'] == kind:
                    groups[r['family_id']].append(r)
            # Round-robin across source families, not a random prefix.
            groups = {f: sorted(rs, key=lambda r: digest(SEED + ':select:' + r['id']))
                      for f, rs in groups.items()}
            balanced = []
            for i in range(max((len(v) for v in groups.values()), default=0)):
                for f in ordered(groups, 'balance:' + kind):
                    if i < len(groups[f]):
                        balanced.append(groups[f][i])
            if len(balanced) < quota:
                raise ValueError(f'Insufficient whole-family {skill}/{split}/{kind} data')
            splits[split].extend(balanced[:quota])
    return splits


def audit_splits(splits):
    seen_ids, families, states = {}, {}, {}
    for split, rows in splits.items():
        for r in rows:
            errors = []
            lint_record(r, r['id'], errors, max_options=20)
            if errors:
                raise ValueError('; '.join(errors))
            if r['id'] in seen_ids:
                raise ValueError('Duplicate record ID')
            seen_ids[r['id']] = split
            for mapping, key in ((families, r['family_id']), (states, canonical(r['state']))):
                if key in mapping and mapping[key] != split:
                    raise ValueError('Train/calibration family or state leakage')
                mapping[key] = split


def render_check(splits, tokenizer_path):
    from transformers import AutoTokenizer
    from jebadiah_prompt import Renderer, reorder_choice, PROMPT_SOURCE_SHA256
    from jebadiah_model import template_sha256
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / 'results/runs/9b-chat-v1/adapter/prompt_contract.json').read_text())
    if PROMPT_SOURCE_SHA256 != contract['prompt_source_sha256']:
        raise ValueError('Wrong baseline prompt source')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if sha(tokenizer_path / 'tokenizer.json') != '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42':
        raise ValueError('Wrong frozen 9B tokenizer')
    if template_sha256(tokenizer) != contract['chat_template_sha256']:
        raise ValueError('Wrong baseline chat template')
    renderer = Renderer(tokenizer, max_tokens=1984)
    maximum = 0
    for rows in splits.values():
        for r in rows:
            for q in r['questions'].values():
                variants = [q]
                if q['type'] == 'choice':
                    variants.append(reorder_choice(q, list(reversed(q['criteria']))))
                for variant in variants:
                    rendered = renderer.render(r['state'], variant)
                    if rendered.truncated:
                        raise ValueError('Rung 3 state truncation is forbidden')
                    maximum = max(maximum, len(tokenizer.encode(rendered.prompt)))
    return {'checked_questions': sum(len(r['questions']) for rs in splits.values() for r in rs),
            'max_prompt_tokens': maximum, 'budget_with_64_token_reserve': 1984,
            'prompt_source_sha256': contract['prompt_source_sha256'],
            'chat_template_sha256': contract['chat_template_sha256'],
            'tokenizer_sha256': sha(tokenizer_path / 'tokenizer.json'),
            'canonical_and_reversed_choice_order': True, 'truncated': 0}


def summary(rows):
    return {'questions': sum(len(r['questions']) for r in rows),
            'skills': dict(Counter(r['skill'] for r in rows)),
            'types': dict(Counter(q['type'] for r in rows for q in r['questions'].values())),
            'families': len({r['family_id'] for r in rows}),
            'labels': {skill: dict(Counter(str(r['label']['decision']) for r in rows if r['skill'] == skill))
                       for skill in SKILLS}}


def build(hwu, output, index_path, license_path, tokenizer_path):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory to avoid stale data')
    manifest = json.loads(license_path.read_text())
    for info in manifest['sources'].values():
        validate_source(info, manifest['source_exclusions'])
    human = manifest['sources'][HUMAN]
    for name, expected in human['input_sha256'].items():
        if sha(hwu / name) != expected:
            raise ValueError('Human source revision/hash mismatch: ' + name)
    expected_generator = sha(Path(__file__))
    for source in (TOOLS, GROUND):
        if manifest['sources'][source]['generator_sha256'] != expected_generator:
            raise ValueError('Generator revision/hash mismatch')
    if sha(index_path) != ITEM25_INDEX_SHA256:
        raise ValueError('Wrong full-suite item 25 index')
    output.mkdir(parents=True, exist_ok=True)
    candidates = human_questions(hwu) + tool_questions() + evidence_questions()
    # Scan the complete raw source AND every converted/generated question,
    # before selection. Set family to source ID for item 25's removal filter.
    scan_rows = raw_human(hwu) + [{**r, 'family': r['subset']} for r in candidates]
    scan_dir = output / 'source-scan'
    scan_dir.mkdir()
    write_rows(scan_dir / 'candidates.jsonl', scan_rows)
    (scan_dir / 'protected.pkl').symlink_to(index_path.resolve())
    scan(scan_dir)
    hits = json.loads((scan_dir / 'contamination-hits.json').read_text())
    blocked = {h['family'] for h in hits}
    report = {
        'edition': '0.3', 'scanner_origin_sha256': ITEM25_SCANNER_SHA256,
        'protected_index_sha256': ITEM25_INDEX_SHA256,
        'scanner_sha256': sha(Path(__file__).with_name('item33_full_suite_scan.py')),
        'suite_revision': 'e57106b5e0698e74bd1a88b3b4c19b94a0dc8328',
        'source_level': True, 'word_span': 13,
        'sources': {s: {'scanned_records': sum(r['subset'] == s for r in scan_rows),
                         'input_sha256': digest(''.join(canonical(r) + '\n' for r in scan_rows if r['subset'] == s)),
                         'rejected': s in blocked,
                         'overlap_ids': [h['id'] for h in hits if h['family'] == s]}
                    for s in manifest['sources']},
        'hits': [{**h, 'record_sha256': digest(canonical(next(r for r in scan_rows if r['id'] == h['id'])))} for h in hits],
        'benchmark_payloads_in_report': False,
        'semantic_independence_proven': False,
    }
    write_json(output / 'overlap-scan-report.json', report)
    if blocked:
        # No partial salvage: report persists, training output is not created.
        raise ValueError('Whole sources rejected: ' + ', '.join(sorted(blocked)))
    splits = {'train': [], 'calib': []}
    for skill in SKILLS:
        chosen = split_select([r for r in candidates if r['skill'] == skill], skill)
        for name in splits:
            splits[name].extend(chosen[name])
    audit_splits(splits)
    render = render_check(splits, tokenizer_path)
    built = {'name': 'item33-rung3-skills', 'seed': SEED,
             'base_mix': 'pending measured A1/A3 winner; no training launched',
             'source_level_contamination_rule': True,
             'license_manifest_sha256': sha(license_path), 'render_validation': render,
             'splits': {}, 'files': {}, 'source_manifests': manifest['sources']}
    for split, rows in splits.items():
        rows.sort(key=lambda r: digest(SEED + ':output:' + r['id']))
        path = output / (split + '.jsonl')
        write_rows(path, rows)
        built['splits'][split] = summary(rows)
        built['files'][path.name] = {'sha256': sha(path), 'questions': len(rows)}
    built['files']['overlap-scan-report.json'] = {'sha256': sha(output / 'overlap-scan-report.json')}
    write_json(output / 'manifest.json', built)
    print(json.dumps(built['splits'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hwu', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--licenses', type=Path, default=Path(__file__).with_name('manifests') / 'item33-source-licenses.json')
    args = parser.parse_args()
    build(args.hwu, args.out, args.index, args.licenses, args.tokenizer)
