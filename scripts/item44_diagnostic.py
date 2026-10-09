"""Score the owned holdout through eval_jebadiah's full-menu local Scorer."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, hf_hub_download
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import eval_jebadiah as evaluate
from jebadiah_model import Scorer, load_base, load_tokenizer, read_temperatures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--repo', default='frontier-infra/jebadiah-9b-v2-1-n2-checkpoints')
    a = p.parse_args()
    source = hf_hub_download('frontier-infra/jebadiah-data-v2-1-item43', 'n2/wide-diagnostic.jsonl', repo_type='dataset', revision='0738c0ac008528511041e059b6c5453ab8198dbf')
    records = [json.loads(line) for line in Path(source).read_text().split('\n') if line.strip()]
    assert len(records) == 384
    scorer = Scorer(load_base(a.model), load_tokenizer(a.model), 2048, temperatures=read_temperatures(a.model))
    assert scorer.temperatures
    rows, extra = evaluate.run_set(scorer, records, 1, False, 8, 0)
    lookup = {r['id']: r for r in records}
    groups = defaultdict(list)
    tp = fp = fn = 0
    for r in rows:
        original = lookup[r['id']]
        correct = r['pick'] == r['label']
        mode = 'no-match' if r['label'] == 'none' else 'match'
        size = str(len(original['questions']['decision']['criteria']))
        for name in ('overall', 'domain:' + original['skill'], 'options:' + size, 'mode:' + mode):
            groups[name].append(correct)
        tp += r['pick'] == 'none' and r['label'] == 'none'
        fp += r['pick'] == 'none' and r['label'] != 'none'
        fn += r['pick'] != 'none' and r['label'] == 'none'
    result = {'accuracy': {k: {'n': len(v), 'correct': sum(v), 'accuracy': sum(v)/len(v)} for k,v in groups.items()},
              'no_match': {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn)},
              'temperatures': scorer.temperatures, 'rows': rows, 'scoring': 'eval_jebadiah.run_set; all candidates jointly; one repeat'}
    output = Path(a.model).parent / 'item44-wide-diagnostic.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    api = HfApi(); assert api.model_info(a.repo).private
    api.upload_file(repo_id=a.repo, path_or_fileobj=output, path_in_repo='item44-wide-diagnostic.json')
    print('ITEM44_WIDE_DIAGNOSTIC', json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
