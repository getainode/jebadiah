"""Paired graph-disjoint causal diagnostic with unchanged A3 temperatures.

Primary diagnostic improvement is balanced accuracy over necessarily-valid and
not-necessarily-valid labels. Report ordinary accuracy and all requested strata.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = 'frontier-infra/jebadiah-data-v2-1-item47'
DATA_REV = '2242e85fae6f8aff3fc3cea3fa95869d99132522'
A3 = 'frontier-infra/jebadiah-9b-v2-1-a3'
A3_REV = '9e69926a007dd636e82d33485dcd48f9751c4248'
REPO = 'frontier-infra/jebadiah-9b-v2-1-rung2'


def summarize(rows, records):
    lookup = {r['id']: r for r in records}
    groups = defaultdict(list)
    confusion = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    seen = set()
    for row in rows:
        r = lookup[row['id']]
        q = r['questions'][row['qid']]
        key = (row['id'], row['qid'])
        assert key not in seen and row['repeat'] == 0
        seen.add(key)
        positive = 'valid' if q['type'] == 'choice' else 'true'
        negative = 'not_valid' if q['type'] == 'choice' else 'false'
        assert row['label'] in (positive, negative) and row['pick'] in (positive, negative)
        expected = str(r['label'][row['qid']]).lower() if q['type'] == 'noul' else r['label'][row['qid']]
        assert row['label'] == expected
        gold, predicted = row['label'] == positive, row['pick'] == positive
        cell = 'tp' if gold and predicted else 'fn' if gold else 'fp' if predicted else 'tn'
        for name in ('overall', 'type:' + q['type'], 'template:' + r['provenance']['template'],
                     'variables:' + str(r['provenance']['num_variables']), 'family:' + r['family_id']):
            groups[name].append(row['pick'] == row['label'])
            confusion[name][cell] += 1
    assert seen == {(r['id'], qid) for r in records for qid in r['questions']}
    metrics = {}
    for name, values in groups.items():
        c = confusion[name]
        tp, fp, fn, tn = (c[k] for k in ('tp', 'fp', 'fn', 'tn'))
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        metrics[name] = {'n': len(values), 'correct': sum(values), 'accuracy': sum(values) / len(values),
                         'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
                         'valid_precision': tp / (tp + fp) if tp + fp else None, 'valid_recall': recall,
                         'valid_count': tp + fn, 'not_valid_count': tn + fp, **c}
    return {'metrics': metrics, 'rows': rows}


def main():
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    sys.path.insert(0, str(ROOT / 'train'))
    import eval_jebadiah as evaluate
    from jebadiah_model import Scorer, load_base, load_tokenizer, read_temperatures
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate_revision')
    a = p.parse_args()
    api = HfApi()
    assert api.model_info(REPO).private and api.model_info(REPO + '-checkpoints').private
    source = hf_hub_download(DATA, 'rung2/causal-diagnostic.jsonl', repo_type='dataset', revision=DATA_REV)
    with open(source) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    assert len(records) == 300 and len({r['family_id'] for r in records}) == 12
    temperatures = read_temperatures(Path(hf_hub_download(A3, 'temperatures.json', revision=A3_REV)).parent)
    result = {'data_revision': DATA_REV, 'candidate_revision': a.candidate_revision, 'a3_revision': A3_REV,
              'temperatures': temperatures, 'primary_metric': 'overall balanced accuracy',
              'scoring': 'eval_jebadiah.run_set; full menus; one repeat; no temperature fitting'}
    for name, repo, revision in [('a3', A3, A3_REV), ('rung2', REPO, a.candidate_revision)]:
        path = snapshot_download(repo, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
        assert read_temperatures(path) == temperatures
        scorer = Scorer(load_base(path), load_tokenizer(path), 2048, temperatures=temperatures)
        rows, extra = evaluate.run_set(scorer, records, 1, False, 8, 0)
        result[name] = summarize(rows, records)
        result[name]['run_metrics'] = extra['timing']
        assert extra['timing']['questions_scored'] == 300 and extra['timing']['truncated_prompts'] == 0
        partial = Path('/workspace/item50-causal-diagnostic-partial-' + name + '.json')
        partial.write_text(json.dumps(result, indent=2) + '\n')
        api.upload_file(repo_id=REPO + '-checkpoints', path_or_fileobj=partial, path_in_repo=partial.name)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
        print('ITEM50_DIAGNOSTIC_MODEL', name, json.dumps(result[name]['metrics']['overall']), flush=True)
    for metric in ('accuracy', 'balanced_accuracy'):
        result[metric + '_difference'] = result['rung2']['metrics']['overall'][metric] - result['a3']['metrics']['overall'][metric]
    result['diagnostic_improves'] = result['balanced_accuracy_difference'] > 0
    output = Path('/workspace/item50-causal-diagnostic.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    api.upload_file(repo_id=REPO + '-checkpoints', path_or_fileobj=output, path_in_repo=output.name)
    print('ITEM50_DIAGNOSTIC', json.dumps({k: v for k, v in result.items() if k not in ('a3', 'rung2')}), flush=True)


if __name__ == '__main__':
    main()
