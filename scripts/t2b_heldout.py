# SPDX-License-Identifier: Apache-2.0
"""Score all heldout questions with the training evaluator's untempered candidate argmax."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


def score(model_path, adapter_path, heldout, output):
    from jebadiah_model import Scorer, load_adapter, load_base, load_tokenizer
    from train_jebadiah import label_index
    records = [json.loads(line) for line in heldout.read_text().splitlines() if line.strip()]
    if len(records) != 3000 or sum(len(r['questions']) for r in records) != 3000:
        raise ValueError('Expected exactly 3000 heldout questions')
    scorer = Scorer(load_adapter(load_base(str(model_path)), str(adapter_path)),
                    load_tokenizer(str(model_path)), 2048)
    groups = defaultdict(lambda: {'n': 0, 'correct': 0})
    items = [(r, qid, q) for r in records for qid, q in r['questions'].items()]
    answers = []
    started = time.monotonic()
    truncated = 0
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        rendered = [(scorer.render(r['state'], q), q['type']) for r, qid, q in chunk]
        probs = scorer.score_rendered(rendered)
        for (r, qid, q), (render, _), p in zip(chunk, rendered, probs):
            gold = label_index(q, r['label'][qid], render.keys)
            pick = max(range(len(p)), key=lambda j: p[j])
            family = r['family']
            truncated += int(render.truncated)
            for key in ('overall', family):
                groups[key]['n'] += 1
                groups[key]['correct'] += int(pick == gold)
            answers.append({'id': r['id'], 'qid': qid, 'family': family,
                            'gold_index': gold, 'pick_index': pick, 'correct': pick == gold})
    families = {k: v for k, v in groups.items() if k != 'overall'}
    if len(families) != 5 or any(v['n'] != 600 for v in families.values()) or truncated:
        raise ValueError('Incomplete family coverage or truncated heldout prompt')
    for value in groups.values():
        value['accuracy'] = value['correct'] / value['n']
    report = {'scoring': 'Training CalibEval Scorer, batch 8, canonical menus, no temperatures, argmax candidate logits',
              'model': str(model_path), 'adapter': str(adapter_path),
              'heldout_sha256': hashlib.sha256(heldout.read_bytes()).hexdigest(),
              'metrics': dict(groups), 'family_macro_accuracy': sum(v['accuracy'] for v in families.values()) / 5,
              'questions': len(answers), 'truncated_prompts': truncated,
              'wall_seconds': time.monotonic() - started, 'answers': answers}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('T2B_HELDOUT', json.dumps({k: v for k, v in report.items() if k != 'answers'}), flush=True)
    return report
