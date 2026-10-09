"""Reuse item36 export recovery with a strict unchanged 0.05 gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))


def main():
    p = argparse.ArgumentParser(description="Strict export-only recovery of an item37 final checkpoint")
    p.add_argument("run", choices=["g1", "d0", "l5"])
    p.add_argument("checkpoint_revision")
    p.add_argument("data_revision")
    args = p.parse_args()
    experiment = json.loads((ROOT / f"configs/item37-{args.run}.json").read_text())
    step = 2818 if args.run == "g1" else 2649
    from merge_export import merge
    import verify_merge
    import subprocess

    work = Path(os.environ.get('JEB_ROOT', '/workspace/item37-recovery'))
    work.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model = experiment['model_repo']
    assert api.model_info(model).private
    checkpoint_repo = model + '-checkpoints'
    checkpoint_revision = args.checkpoint_revision
    cp_root = Path(snapshot_download(checkpoint_repo, revision=checkpoint_revision,
                                    allow_patterns=[f'checkpoints/checkpoint-{step}/adapter_model.safetensors',
                                                    f'checkpoints/checkpoint-{step}/adapter_config.json',
                                                    f'checkpoints/checkpoint-{step}/run_config.json',
                                                    f'checkpoints/checkpoint-{step}/trainer_state.json']))
    cp = cp_root / f'checkpoints/checkpoint-{step}'
    cfg = json.loads((cp / 'run_config.json').read_text())
    state = json.loads((cp / 'trainer_state.json').read_text())
    assert state['global_step'] == step and state['epoch'] == 1
    assert cfg['decide']['lora_dropout'] == experiment['decide']['lora_dropout']
    assert cfg['learning_rate'] == experiment['learning_rate']
    base = snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE*'])
    data_revision = args.data_revision
    calib = hf_hub_download(experiment['dataset_repo'], experiment['calib_file'],
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
    finally:
        verify_merge.Scorer.score_rendered = original
        gate_path = merged / 'merge_verification.json'
        if gate_path.exists():
            api.upload_file(repo_id=checkpoint_repo, path_or_fileobj=gate_path,
                            path_in_repo='item37-recovery/merge_verification.json',
                            commit_message='Preserve strict recovery merge gate evidence')
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
    assert gate_passed
    (merged / 'export_recovery.json').write_text(json.dumps({
        'checkpoint_revision': checkpoint_revision, 'optimizer_steps_added': 0,
        'unchanged_merge_gate_passed': gate_passed, 'gate': report}, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'train/fit_temperature.py'), '--base', str(merged),
                    '--output-dir', str(merged), '--calib', calib, '--max-tokens', '2048',
                    '--batch-size', '1', '--target', 'train', '--score-fit', 'hard',
                    '--score-targets', 'ordinal', '--device', 'cuda', '--dtype', 'bfloat16'], check=True)
    clean_cfg = {k: v for k, v in cfg.items() if k not in
                 ('base_model', 'dataset_path', 'eval_dataset_path', 'output_dir', 'checkpoint_repo', 'resume', 'max_steps')}
    provenance = {'base_model': 'Qwen/Qwen3.5-9B',
                  'base_revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                  'dataset': experiment['dataset_repo'], 'dataset_revision': data_revision,
                  'effective_batch_size': 8, 'config': clean_cfg}
    (merged/'training_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (merged/'README.md').write_text('---\nbase_model: Qwen/Qwen3.5-9B\nlibrary_name: transformers\n---\n\n'
        f'# Private {args.run.upper()} diagnostic\n\n'
        'Strict export-only recovery; zero additional optimizer steps.\n'
        'Inherited A3 source-policy exceptions block shipping.\n'
        'See export_recovery.json and merge_question_diagnostics.json.\n')
    assert api.model_info(model).private
    result = api.upload_folder(repo_id=model, folder_path=merged,
                               commit_message='Strict item37 final-checkpoint export with unchanged merge gate')
    print('ITEM37_RECOVERY_DONE', result.oid, flush=True)


if __name__ == '__main__':
    main()
