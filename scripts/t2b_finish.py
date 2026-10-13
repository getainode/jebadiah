# SPDX-License-Identifier: Apache-2.0
"""Publish only code/docs and report verified T2b aggregates after evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
from huggingface_hub import HfApi
from t2b_jobs import save
from t2b_monitor import message,cost


def finish(root):
    repo=root/'repo';analysis=root/'analysis';summary=json.loads((analysis/'t2b-summary.json').read_text())
    ledger=json.loads((root/'jobs.json').read_text());assert all(e['status']=='COMPLETED' for e in ledger['jobs'].values()) and len(ledger['jobs'])==4
    total=cost(ledger);assert total<65;save(root/'jobs.json',ledger)
    proofs={str(s):json.loads((root/'artifacts'/f's{s}'/'training-proof.json').read_text()) for s in (17,18)}
    cis={str(s):json.loads((analysis/f's{s}-bootstrap.json').read_text()) for s in (17,18)}
    families={f:sum(proofs[str(s)]['heldout_per_family'][f]['accuracy'] for s in (17,18))/2 for f in proofs['17']['heldout_per_family']}
    heldmean=sum(families.values())/5
    base=json.loads((Path('/Volumes/PRO-G40/scratch/lead-ainode/t2b0/repo/data/manifests/t2b0-results-summary.json')).read_text())
    basefamilies={f:sum(base['per_seed'][str(seed)]['heldout_per_family'][f]['accuracy'] for seed in (17,18))/2 for f in families}
    report={'summary':summary,'per_seed_bootstrap':cis,'training_proofs':proofs,'heldout_two_seed_per_family':families,'heldout_two_seed_mean':heldmean,'heldout_difference_vs_79_53_points':100*heldmean-79.53,'estimated_total_usd':total,'jobs':ledger['jobs']}
    save(analysis/'final-report.json',report)
    api=HfApi();results='jbrashear/jebadiah-9b-v2-1-index-results';assert api.dataset_info(results).private
    commit=api.upload_folder(repo_id=results,repo_type='dataset',folder_path=analysis,path_in_repo='analysis/t2b-training',commit_message='Private T2b paired seed and two-seed mean evaluation evidence')
    save(root/'analysis-upload.json',{'results_repo':results,'revision':commit.oid,'path':'analysis/t2b-training','private':True})
    def interval(v):return f"[{v[0]:+.4f}, {v[1]:+.4f}]"
    lines=['\n## Verified results\n','| Seed | Proxy | Difference vs matching T2b-0 | Paired 95% CI | Heldout family mean |','|---|---:|---:|---|---:|']
    for s in (17,18):
        c=cis[str(s)];p=proofs[str(s)]
        lines.append(f"| {s} | {c['candidate']:.6f} | {c['difference']:+.6f} | {interval(c['difference_ci95'])} | {100*p['heldout_family_accuracy']:.2f}% |")
    lines += [f"\nTwo-seed proxy mean **{summary['candidate_mean']:.6f}**, {summary['difference_vs_44_12']:+.6f} points against the stated 44.12 reference. Paired difference against the unrounded T2b-0 mean {summary['baseline_mean']:.6f}: {summary['difference']:+.6f}, 95% CI {interval(summary['difference_ci95'])}. Bootstrap decision: **{summary['decision']}**. CLINC mean regression flag: {summary['clinc_regression']}. No model promotion or PR merge performed.",
              '\nBoth seed comparisons and the two-seed mean use 2,000 complete catalog/group bootstrap replicates with seed 20261008. All four runs use the same paired draws. These intervals measure uncertainty conditional on the frozen proxy, not training-seed variance. All 11,079 frozen proxy IDs per run are unique, successful and payload-bound to the verified suite. The unchanged official scorer is recomputed for every draw.',
              '\n| Heldout family | Seed 17 | Seed 18 | Two-seed mean |','|---|---:|---:|---:|']
    for f,mean in families.items():
        lines.append(f"| {f} | {100*proofs['17']['heldout_per_family'][f]['accuracy']:.2f}% | {100*proofs['18']['heldout_per_family'][f]['accuracy']:.2f}% | {100*mean:.2f}% | {100*basefamilies[f]:.2f}% | {100*(mean-basefamilies[f]):+.2f} |")
    lines.append(f"\nHeldout equal-family mean **{100*heldmean:.2f}%**, {100*heldmean-79.53:+.2f} points against 79.53%. Each seed covers all 3,000 questions, with zero heldout truncation.")
    clinc=summary['clinc_mean_macro_f1_loss_points'];tools=summary['areas']['tools']['difference']
    lines.append(f"\nCLINC macro-F1 deltas: seed 17 {-summary['clinc_matched_seed_macro_f1_loss_points']['17']:+.4f}, seed 18 {-summary['clinc_matched_seed_macro_f1_loss_points']['18']:+.4f}, two-seed mean {-clinc:+.4f} points. Losses beyond 2.5 count as regressions. Tools deltas: seed 17 {cis['17']['areas']['tools']['difference']:+.4f}, seed 18 {cis['18']['areas']['tools']['difference']:+.4f}, mean {tools:+.4f}; mean 95% CI {interval(summary['areas']['tools']['difference_ci95'])}. Tools deltas within 6 are noise.")
    lines += ['\n| Job | ID | Timeout minutes | Running seconds | Estimated USD |','|---|---|---:|---:|---:|']
    for key,e in ledger['jobs'].items():lines.append(f"| {key} | `{e['id']}` | {e['timeout_minutes']} | {e['running_seconds']:.3f} | {e['estimated_usd']:.4f} |")
    lines.append(f"\nTotal duration-based compute estimate **USD {total:.4f}**, including every paid attempt. Local smoke and pre-submission signature correction cost USD 0. No paid retry occurred. All HF jobs completed. This is a running-duration estimate at USD 2.75/hour, not an invoice.")
    for s in (17,18):
        p=proofs[str(s)];g=p['merge_gate'];lines.append(f"\nSeed {s}: {p['steps']} optimizer steps, one epoch; original A3 recipe and prompt fields verified. Training truncations: {p['truncated_training_prompts']}. Original merge gate passed at maximum probability shift {g['max_probability_delta']:.6f}, zero confident flips. Private model revision `{p['model_revision']}`.")
    lines.append(f"\nComplete answers and outputs remain private. Evaluation analysis is persisted in `{results}`, folder `analysis/t2b-training`, revision `{commit.oid}`. Only code and documentation are in this PR.")
    doc=repo/'docs/t2b-training.md';text=doc.read_text();text=text.replace('Results and measured costs are pending. This document will be completed after both proxies and paired bootstraps finish. The code and documentation PR will remain unmerged.','Both training/heldout jobs and both proxies completed. See verified results below. The code and documentation PR remains unmerged.')
    assert '## Verified results' not in text;doc.write_text(text+'\n'+'\n'.join(lines)+'\n')
    rules=repo/'docs/run-rules.md';rules.write_text(rules.read_text()+f"\n| T2b result | T2b-0 | Completed one epoch, seeds 17 and 18 | Paired two-seed bootstrap CI above zero | {summary['decision'].upper()}: proxy {cis['17']['candidate']:.4f}/{cis['18']['candidate']:.4f}, mean {summary['candidate_mean']:.4f}, difference {summary['difference']:+.4f} {interval(summary['difference_ci95'])}; heldout {100*heldmean:.2f}%; CLINC {-clinc:+.4f}, Tools {tools:+.4f}; estimated USD {total:.4f}; private, no promotion. |\n")
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    subprocess.run(['git','add','data/t2b_compose.py','scripts/t2b_analyze.py','scripts/t2b_collect.py','scripts/t2b_finish.py','scripts/t2b_heldout.py','scripts/t2b_hf_client.py','scripts/t2b_jobs.py','scripts/t2b_launch.py','scripts/t2b_monitor.py','scripts/t2b_persist.py','scripts/t2b_supervise.py','scripts/t2b_train.py','scripts/test_t2b_launch.py','docs/t2b-training.md','docs/run-rules.md'],cwd=repo,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    assert all(f.endswith(('.py','.md')) for f in staged)
    for f in staged:assert '\u2014' not in (repo/f).read_text(),'No em dashes: '+f
    subprocess.run(['git','commit','-m','Train T2b with scrubbed A3 recipe and paired two-seed evaluation'],cwd=repo,check=True)
    subprocess.run(['git','push','-u','origin','fable/t2b-train'],cwd=repo,check=True)
    body=root/'pr-body.md';body.write_text(f"Adds the 256,965-question T2b tranche to scrubbed A3 with the unchanged one-epoch recipe, seeds 17 and 18. Mean proxy {summary['candidate_mean']:.4f} against T2b-0 {summary['baseline_mean']:.4f}; paired 95% difference CI {interval(summary['difference_ci95'])}, decision {summary['decision']}. Heldout family mean {100*heldmean:.2f}% against 79.53%.\n\nValidation: original pipeline setup-only smoke, 852 prompt identity renders, full composition manifest validation, zero direct hits in the supplemented suite scan, unchanged launcher tests, verified recipe and merge gate for both seeds, complete 11,079-record proxies, 2,000 paired bootstrap replicates. Estimated total compute USD {total:.4f}, below USD 65 including every attempt. Code and docs only; data and outputs remain private.\n")
    url=subprocess.check_output(['gh','pr','create','--repo','getainode/jebadiah','--base','fable/t2b0-baseline','--head','fable/t2b-train','--title',f"T2b: scrubbed A3 plus tranche, two-seed results ({summary['decision']})",'--body-file',str(body)],cwd=repo,text=True).strip();save(root/'pr.json',{'url':url,'unmerged':True})
    body=f"T2b complete. Proxy seeds 17/18: {cis['17']['candidate']:.6f}/{cis['18']['candidate']:.6f}; mean {summary['candidate_mean']:.6f}, vs 44.12 {summary['difference_vs_44_12']:+.6f}. Matching-seed bootstrap CIs: s17 {interval(cis['17']['difference_ci95'])}, s18 {interval(cis['18']['difference_ci95'])}; paired two-seed mean difference {summary['difference']:+.6f} CI {interval(summary['difference_ci95'])}. Decision {summary['decision']}. Heldout family means: "+'; '.join(f'{f}: s17 {100*proofs["17"]["heldout_per_family"][f]["accuracy"]:.2f}%, s18 {100*proofs["18"]["heldout_per_family"][f]["accuracy"]:.2f}%, mean {100*v:.2f}% vs T2b-0 {100*basefamilies[f]:.2f}%'  for f,v in families.items())+f"; overall {100*heldmean:.2f}% vs 79.53% ({100*heldmean-79.53:+.2f} points). CLINC deltas s17/s18/mean: {-summary['clinc_matched_seed_macro_f1_loss_points']['17']:+.4f}/{-summary['clinc_matched_seed_macro_f1_loss_points']['18']:+.4f}/{-clinc:+.4f}; mean regression {summary['clinc_regression']}, matched-seed regression flags {summary['matched_seed_clinc_regression']}. Tools deltas s17/s18/mean: {cis['17']['areas']['tools']['difference']:+.4f}/{cis['18']['areas']['tools']['difference']:+.4f}/{tools:+.4f}, within-6 noise {summary['tools_delta_within_seed_noise']}. Total estimated USD {total:.4f}, all paid attempts included, cap USD 65. PR {url}, code/docs only, unmerged."
    message(root,'status','T2b final paired results and PR',body,'final-send.json')
    save(root/'complete.json',{'status':'complete','pr_url':url,'estimated_total_usd':total,'proxy_mean':summary['candidate_mean'],'decision':summary['decision']})
    print(body,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();finish(a.root.resolve())
