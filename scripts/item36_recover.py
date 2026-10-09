"""Export the saved final R3 checkpoint under the lead's private-only exception.

Never trains, changes weights, or relaxes the shared merge gate. Records the
unchanged gate result and per-question diagnostics before private publication.
"""
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))


def main():
    from merge_export import merge
    import verify_merge
    import subprocess

    work = Path(os.environ.get('JEB_ROOT', '/workspace/item36-recovery'))
    work.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model = 'frontier-infra/jebadiah-9b-v2-1-r3'
    assert api.model_info(model).private
    checkpoint_repo = model + '-checkpoints'
    checkpoint_revision = '8578df33929a7a50d3111df0d47afec1e8803c9d'
    cp_root = Path(snapshot_download(checkpoint_repo, revision=checkpoint_revision,
                                    allow_patterns=['checkpoints/checkpoint-3155/adapter_model.safetensors',
                                                    'checkpoints/checkpoint-3155/adapter_config.json',
                                                    'checkpoints/checkpoint-3155/run_config.json',
                                                    'checkpoints/checkpoint-3155/trainer_state.json']))
    cp = cp_root / 'checkpoints/checkpoint-3155'
    cfg = json.loads((cp / 'run_config.json').read_text())
    state = json.loads((cp / 'trainer_state.json').read_text())
    assert state['global_step'] == 3155 and state['epoch'] == 1
    assert cfg['dataset_sha256'] == '86ddd8bd210ae7cefd795ee5093a2f4e98b0d9c5cdf0296d448b335ca3d97a9d'
    base = snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE*'])
    data_revision = '7430dae8b94337f3f9561ca98ecf7c6fc63dbecf'
    calib = hf_hub_download('frontier-infra/jebadiah-data-v2-1-item36', 'r3/calib.jsonl',
                            repo_type='dataset', revision=data_revision)
    contract = hf_hub_download('frontier-infra/jebadiah-9b-v2-1-a3', 'prompt_contract.json',
                               revision='9e69926a007dd636e82d33485dcd48f9751c4248')
    adapter = work / 'adapter'
    adapter.mkdir(exist_ok=True)
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        shutil.copyfile(cp / name, adapter / name)
    shutil.copyfile(contract, adapter / 'prompt_contract.json')
    merged = work / 'merged'
    merge(base, adapter, merged)
    captured = []
    original = verify_merge.Scorer.score_rendered

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.extend(result)
        return result

    verify_merge.Scorer.score_rendered = capture
    try:
        verify_merge.verify(base, adapter, merged, calib, device='cuda', max_tokens=2048)
    except ValueError:
        report = json.loads((merged / 'merge_verification.json').read_text())
        # The explicit exception is private scoring with zero confident flips.
        # A new failure outside the observed numerical range is not authorized.
        if report['clear_margin_flips'] or report['max_probability_delta'] > 0.1:
            raise
    finally:
        verify_merge.Scorer.score_rendered = original
    report = json.loads((merged / 'merge_verification.json').read_text())
    gate_passed = report['clear_margin_flips'] == 0 and report['max_probability_delta'] <= 0.05
    rows = [json.loads(line) for line in Path(calib).read_text().split('\n') if line.strip()]
    items = [(r, qid, q) for r in rows for qid, q in r['questions'].items()][:260]
    assert len(captured) == 2 * len(items)
    diagnostics = []
    for i, (row, qid, q) in enumerate(items):
        before, after = captured[i], captured[i + len(items)]
        delta = max(abs(a-b) for a, b in zip(before, after))
        top = sorted(before, reverse=True)
        diagnostics.append({'record_id': row['id'], 'question_id': qid, 'type': q['type'],
                            'max_probability_delta': delta, 'margin_before': top[0]-top[1],
                            'pick_before': max(range(len(before)), key=before.__getitem__),
                            'pick_after': max(range(len(after)), key=after.__getitem__),
                            'probabilities_before': before, 'probabilities_after': after})
    diagnostics.sort(key=lambda x: x['max_probability_delta'], reverse=True)
    (merged / 'merge_question_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print('MERGE_DIAGNOSTICS', json.dumps(diagnostics[:3]), flush=True)
    exception = {'private_only': True, 'shipping_allowed': False,
                 'unchanged_merge_gate_passed': gate_passed, 'gate': report,
                 'approval': 'Lead Orca message msg_c4892b870ed6, 2026-10-09T00:42:42Z',
                 'checkpoint_revision': checkpoint_revision, 'optimizer_steps_added': 0}
    (merged / 'private_merge_exception.json').write_text(json.dumps(exception, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'train/fit_temperature.py'), '--base', str(merged),
                    '--output-dir', str(merged), '--calib', calib, '--max-tokens', '2048',
                    '--batch-size', '1', '--target', 'train', '--score-fit', 'hard',
                    '--score-targets', 'ordinal', '--device', 'cuda', '--dtype', 'bfloat16'], check=True)
    clean_cfg = {k: v for k, v in cfg.items() if k not in
                 ('base_model', 'dataset_path', 'eval_dataset_path', 'output_dir', 'checkpoint_repo', 'resume', 'max_steps')}
    provenance = {'base_model': 'Qwen/Qwen3.5-9B',
                  'base_revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                  'dataset': 'frontier-infra/jebadiah-data-v2-1-item36', 'dataset_revision': data_revision,
                  'effective_batch_size': 8, 'config': clean_cfg}
    (merged/'training_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (merged/'README.md').write_text('---\nbase_model: Qwen/Qwen3.5-9B\nlibrary_name: transformers\n---\n\n'
        '# Private R3 diagnostic\n\nMerge-gate failed: max shift 0.0836, 259/260 picks, 0 confident flips.\n'
        f'Recovered unchanged gate: max shift {report["max_probability_delta"]:.6f}, passed={gate_passed}.\n'
        'Private diagnostic only; inherited A3 source-policy exceptions also block shipping.\n'
        'See private_merge_exception.json and merge_question_diagnostics.json.\n')
    assert api.model_info(model).private
    result = api.upload_folder(repo_id=model, folder_path=merged,
                               commit_message='Private R3 diagnostic with explicit failed-merge-gate evidence')
    print('ITEM36_RECOVERY_DONE', result.oid, flush=True)


if __name__ == '__main__':
    main()
