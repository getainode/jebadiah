"""Render the final ITEM 52 report from verified result and provenance artifacts."""
from pathlib import Path
import json
repo=Path(__file__).resolve().parents[1]
r=repo/'results/runs/item52'
s=json.loads((r/'analysis/item52-summary.json').read_text())
j=json.loads((r/'job-record.json').read_text())
m=json.loads((r/'model-proof.json').read_text())
m18=json.loads((r/'seed18/model-proof.json').read_text())
e=json.loads((r/'item52-environment.json').read_text())
e18=json.loads((r/'seed18/item52-environment.json').read_text())
labels={'a3':'Original A3, seed 17','replicate':'A3 repeat, seed 17','seed18':'A3, seed 18','rung1':'Rung 1','rung2':'Rung 2','rung2b':'Rung 2b'}
area_labels={'knowledge':'Knowledge','language':'Language','retrieval':'Retrieval','tools':'Tools','arts':'Arts'}
def ci(pair):
 lo,hi=pair['difference_ci95'];return f"{pair['difference']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
p=s['comparisons']['replicate-vs-a3'];q=s['comparisons']['seed18-vs-a3']
lines=['# ITEM 52: A3 repeatability and seed-18 controls','',
 f"The exact seed-17 repeat scores {p['candidate']:.2f}, with paired difference {ci(p)} against original A3. All four served BF16 weight shards match original A3 SHA256s, refitted temperature values match, and all 11,079 scored answer maps match exactly. It loses neither Tools nor CLINC. The adapter checkpoint file SHA256 differs, while the merged served weights are identical.",'',
 f"The approved seed-18 control scores {q['candidate']:.2f}, with paired difference {ci(q)} against A3 seed 17. Seed 17 to 18 is the only intended recipe change. The original seed-17 result is reproducible under the same seed; the different-seed control measures the observed spread under the recorded HF host assignments.",'',
 '| Model | Proxy | Knowledge | Language | Retrieval | Tools | Arts |','|---|---:|---:|---:|---:|---:|---:|']
headlines={'a3':p['baseline'],'replicate':p['candidate'],'seed18':q['candidate']}
for name in ['rung1','rung2','rung2b']:headlines[name]=s['comparisons'][name+'-vs-replicate']['candidate']
for name in labels:
 vals=[headlines[name]]+[s['areas'][name][a] for a in area_labels]
 lines.append('| '+labels[name]+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines+=['','All area values are official chance-corrected skill points. Deltas and intervals use unrounded scores.','',
 '| Rung | Delta vs seed-17 repeat, paired 95% interval | Delta vs seed-18 A3, paired 95% interval | In observed two-seed headline range? |',
 '|---|---:|---:|---|']
spread=s['a3_two_seed_spread'];low,high=spread['headline_range']
for name in ['rung1','rung2','rung2b']:
 value=headlines[name];position='within' if low<=value<=high else 'below' if value<low else 'above'
 lines.append(f"| {labels[name]} | {ci(s['comparisons'][name+'-vs-replicate'])} | {ci(s['comparisons'][name+'-vs-seed18'])} | {position} |")
lines+=['',f"The observed A3 two-seed range is [{low:.2f}, {high:.2f}], width {spread['headline_width']:.2f} points. It is a descriptive range from two seeds, not a training-variance confidence interval. The seed-17 repeat is not a third independent seed. No rung beats the accepted seed-17 A3 baseline beyond paired proxy noise.",'',
 '| Area | Seed 18 minus seed 17 | Paired 95% interval |','|---|---:|---|']
for a,label in area_labels.items():
 v=q['areas'][a];lo,hi=v['difference_ci95'];lines.append(f"| {label} | {v['difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
lines+=['','| Model | CLINC macro-F1 / skill | CLadder accuracy / skill | POP909 cluster macro accuracy / skill |','|---|---:|---:|---:|']
for name in labels:
 cells=[]
 for benchmark in ['CLINC150','CLadder','POP909']:
  v=s['benchmarks'][benchmark][name];cells.append(f"{100*v['raw']:.2f} / {100*v['skill']:.2f}")
 lines.append('| '+labels[name]+' | '+' | '.join(cells)+' |')
clinc_delta=100*(s['benchmarks']['CLINC150']['seed18']['raw']-s['benchmarks']['CLINC150']['a3']['raw'])
lines += ['',
 'Tools illustrates why the overall score hides seed sensitivity: seed 18 reproduces a -5.44-point Tools loss without any data change. Rung 1 and rung 2b Tools scores are inside the observed two-seed Tools range [62.29, 67.73]; rung 2 is slightly below it at 61.71. All three Tools paired intervals against seed 18 cross zero. This is consistent with a material seed/order component to the original Tools losses, but two seeds and different assigned drivers/kernels do not establish its cause or variance.', '',
 '| Rung | Tools delta vs seed 18, paired 95% interval | CLINC macro-F1 delta vs seed 17 | CLINC macro-F1 delta vs seed 18 |',
 '|---|---:|---:|---:|']
for name in ['rung1', 'rung2', 'rung2b']:
 v=s['comparisons'][name+'-vs-seed18']['areas']['tools'];lo,hi=v['difference_ci95']
 clinc=s['benchmarks']['CLINC150']
 d17=100*(clinc[name]['raw']-clinc['a3']['raw'])
 d18=100*(clinc[name]['raw']-clinc['seed18']['raw'])
 lines.append(f"| {labels[name]} | {v['difference']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {d17:+.2f} | {d18:+.2f} |")
lines += ['',
 'All rungs remain below both observed A3 seeds on CLINC, whose control range is [74.35, 76.82] macro-F1 percent. Their CLINC losses exceed this observed spread: -12.82, -7.54 and -15.80 percentage points against seed 18. The two-seed result therefore weakens attribution of Tools loss to the added data, while it does not explain away the much larger CLINC losses. No rung satisfies the unchanged keep rule against accepted A3.']
lines+=['',f"Seed 18's Tools delta is {q['areas']['tools']['difference']:+.2f} skill points and its CLINC delta is {clinc_delta:+.2f} macro-F1 percentage points. CLINC150+OOS has 550 requests, CLadder 500, and POP909 200. POP909 uses the official cluster macro accuracy for its index raw value. Both controls have all 11,079 proxy requests successful and zero whole-case failures.",'',
 'All intervals use the unchanged official scorer and 2,000 paired complete catalog/group draws within benchmark/domain/track strata, seed 20261008. They are conditional on the frozen proxy and do not include training-seed uncertainty. The manifest ID-list SHA256 is 74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405; suite revision is e57106b5e0698e74bd1a88b3b4c19b94a0dc8328. Existing rung results were reused with no new GPU inference.','',
 'Both controls use byte-identical a3/ data from frontier-infra/jebadiah-data-v2-1-item32 at bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed, Qwen3.5-9B parent c202236235762e1c871ad0ccb60c8ee5ba337b9a, and original runtime commit 86a8203d792463536eb009a4d8da6c8681ffdd99. Rank 16 / alpha 32, dropout 0.05, max length 2048, LR 1e-4, one epoch, ordinal targets, item27 speed flags and autocast off remain fixed. The seed-18 runtime guard changes exactly one byte in the seed literal; checkpoints prove the sole recipe difference is seed 18. Both completed exactly 2,649 optimizer steps and one epoch. Each control was logged and committed before launch; the lead approved the additional $8 in msg_76e46223b4db. No causal diagnostic was run.','',
 '| Environment | Seed-17 repeat | Seed 18 |','|---|---|---|']
for label,key in [('Python','python'),('torch','torch'),('CUDA runtime','cuda_runtime'),('cuDNN numeric version','cudnn'),('NVIDIA driver','driver'),('Host platform','platform')]:
 lines.append(f'| {label} | {e[key]} | {e18[key]} |')
for package in ['transformers','peft','accelerate','triton','flash-linear-attention','fla-core','causal-conv1d','tokenizers','safetensors']:
 def version(facts):
  return next(v for k,v in facts['packages'].items() if k.lower().replace('_','-')==package)
 lines.append(f'| {package} | {version(e)} | {version(e18)} |')
lines+=['',
 'Both training images use pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel, with the original dependency setup and nvcc 12.8.93. Final Triton is 3.6.0 after the setup initially requests 3.7.1 and then installs causal-conv1d; it matches the original requirements lock. Full installed inventories and freezes are retained. The original job retained Python 3.12.11 and torch 2.11.0+cu128, matching the repeat; its final installed freeze, driver and immutable image digest were not retained, so exact historical host/add-on versions cannot be asserted. HF assigned different drivers and host kernels to the two controls, so the observed seed spread includes those recorded host assignments. Proxy image/runner and inference code hashes match the original run, with torch 2.8.0+cu128, transformers 5.17.0 and Python 3.11.13; the repeat proxy kernel is 6.12.100 versus original 6.12.103.','',
 f"Private models are frontier-infra/jebadiah-9b-v2-1-a3-rep1 at {m['model_revision']} and frontier-infra/jebadiah-9b-v2-1-a3-seed18 at {m18['model_revision']}, with private -checkpoints repositories. Both unchanged merge gates require maximum probability shift <= 0.05 and zero confident flips. Seed-17 gate shift is {m['gate']['max_probability_delta']:.8f}; seed-18 is {m18['gate']['max_probability_delta']:.8f}. Temperatures were refitted with the original A3 command in both controls.",'',
 '| Job | ID | Running seconds | Estimated USD |','|---|---|---:|---:|']
for label,record,key in [('Seed-17 training/export',j,'training'),('Seed-17 proxy',j,'proxy'),('Seed-18 training/export',j['seed18'],'training'),('Seed-18 proxy',j['seed18'],'proxy')]:
 seconds=record[key+'_running_seconds'];lines.append(f"| {label} | {record[key+'_job']} | {seconds} | {seconds/3600*2.75:.4f} |")
total17=j['estimated_seed17_total_usd'];total18=j['seed18']['estimated_total_usd']
lines+=['',f"At $2.75/hour, seed-17 estimated compute is ${total17:.4f} under its $8 cap, seed-18 is ${total18:.4f} under the additional $8 cap, total ${total17+total18:.4f}. Each control's 135-minute training and 35-minute proxy timeout allocation ceiling was $7.7917. These are duration-based compute estimates, not billing invoices. Results and provenance are retained in PR 29 and private results dataset jbrashear/jebadiah-9b-v2-1-index-results under analysis/item52-controls. All local scratch downloads, suite/data copies and analysis working files in /Volumes/PRO-G40/scratch/item52 are deleted after persistence. No merged model was downloaded to the Mac. Lead review/merge remain pending; no GitHub-hosted CI was run.",'']
(repo/'docs/item52-a3-replicate.md').write_text('\n'.join(lines))
