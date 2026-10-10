# SPDX-License-Identifier: Apache-2.0
"""Studio-scan-gated matched replacement or byte-preserving addition to A3."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

from item32_ablation_mix import read, state_key
from item33_skill_data import digest, sha, write_json, write_rows
from item36_skill_mix import A3_HASHES, A3_REVISION
from item47_rung1_mix import A3_MANIFEST_SHA256, histograms
import item55_mathematics as M

PRESENTATIONS = 21190
CALIBRATION = 512
CAP = 2119
REQUESTED = 1000


def plan_slots(original, candidates):
    """One deterministic matched selection feeds both experimental forms."""
    total = sum(len(r['questions']) for r in original)
    existing = sum(len(r['questions']) for r in original if M.is_lineage(r))
    budget = min(REQUESTED, CAP - existing, total // 10 - existing)
    if budget <= 0: raise ValueError('No aggregate DeepMind mathematics cap headroom')
    pools = {k: sorted(((i,qid) for i,r in enumerate(original) for qid,q in r['questions'].items()
                        if r.get('area') == 'knowledge' and q['type'] == k),
                       key=lambda slot: digest(str(M.SEED) + ':donor:' + original[slot[0]]['id'] + ':' + slot[1]))
             for k in ('choice','noul')}
    n = min(budget, sum(map(len,pools.values())))
    if not n: raise ValueError('No matched knowledge donor slots')
    allocation = {k:min(n//2,len(v)) for k,v in pools.items()}
    for k in pools: allocation[k] += min(n-sum(allocation.values()),len(pools[k])-allocation[k])
    selected = {}; mapping = []
    for kind,number in allocation.items():
        incoming = sorted((r for r in candidates if r['questions']['decision']['type'] == kind),
                          key=lambda r:digest(str(M.SEED) + ':replace:' + r['id']))
        if len(incoming) < number: raise ValueError('Insufficient unique matching candidates')
        for slot,row in zip(pools[kind][:number],incoming[:number]):
            if row['area'] != 'knowledge' or row['source'] != M.SOURCE or len(row['questions']) != 1:
                raise ValueError('Unmatched incoming row')
            selected[slot] = copy.deepcopy(row)
            donor = original[slot[0]]
            mapping.append({'original_id':donor['id'],'question_id':slot[1],
                            'replacement_id':row['id'],'area':'knowledge','type':kind,
                            'donor_source':donor['source'],'donor_lineage':M.is_lineage(donor)})
    if len({r['id'] for r in selected.values()}) != n: raise ValueError('Duplicate selected IDs')
    return selected, {'requested_questions':REQUESTED,'selected_questions':n,
                      'matched_donor_pool':{k:len(v) for k,v in pools.items()},
                      'selected_types':allocation,'existing_lineage_questions':existing,
                      'matched_donor_lineage_questions':sum(x['donor_lineage'] for x in mapping),
                      'aggregate_cap_questions':min(CAP,total//10),
                      'selection_counts':M.counts(list(selected.values())),'slots':mapping}


def compose_rows(original, candidates, mode):
    if mode not in ('replace','add'): raise ValueError('Unknown composition mode')
    selected,receipt = plan_slots(original,candidates)
    result = []
    if mode == 'add': result = copy.deepcopy(original) + list(selected.values())
    else:
        for i,row in enumerate(original):
            kept = copy.deepcopy(row)
            for qid in row['questions']:
                if (i,qid) not in selected: continue
                result.append(selected[(i,qid)])
                for field in ('questions','label','target'): kept.get(field,{}).pop(qid,None)
            if kept['questions']: result.append(kept)
    total = sum(len(r['questions']) for r in original)
    expected = total + (len(selected) if mode == 'add' else 0)
    exposure = sum(len(r['questions']) for r in result if M.is_lineage(r))
    if exposure > min(CAP,total//10): raise ValueError('Aggregate lineage cap exceeded')
    if sum(len(r['questions']) for r in result) != expected: raise ValueError('Presentation count changed')
    if mode == 'replace' and histograms(result)['area_type'] != histograms(original)['area_type']:
        raise ValueError('Replacement area/type counts changed')
    if len({r['id'] for r in result}) != len(result): raise ValueError('Composition ID collision')
    receipt.update(mode=mode,presentations=expected,lineage_questions=exposure,
                   removed_questions=len(selected) if mode == 'replace' else 0,
                   removed_lineage_questions=receipt['matched_donor_lineage_questions'] if mode == 'replace' else 0,
                   incoming_fraction_of_a3=len(selected)/total,
                   before=histograms(original),after=histograms(result))
    return result,receipt


def additive_bytes(original_bytes, additions):
    if not original_bytes.endswith(b'\n'): raise ValueError('A3 lacks final newline')
    return original_bytes + b''.join((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                                     for row in additions)

def validate_scan(manifest, incoming, licenses=M.POLICY):
    report = manifest.get('overlap_scan', {})
    if report.get('status') != 'passed': raise ValueError('Full-source Studio scan required before final composition')
    if json.loads((incoming/'overlap-scan-report.json').read_text()) != report:
        raise ValueError('Scan report mismatch')
    policy = json.loads(licenses.read_text()); source = policy['sources'][M.SOURCE]
    M.validate_source(source, policy['source_exclusions'])
    evidence = Path(__file__).resolve().parents[1]/'results/research/item55-source-evidence.json'
    if (source['source_evidence_sha256'] != sha(evidence) or source['ancestry_verified'] is not True
            or source['model_redistribution_ok'] is not True or source['model_written_training_text'] is not False
            or source['license'] != 'Apache-2.0' or source['revision'] != M.REVISION):
        raise ValueError('Invalid rights/ancestry evidence')
    names = {'train.jsonl','diagnostic.jsonl','raw-source.jsonl','scan-candidates.jsonl','NOTICE-DeepMind.txt','LICENSE-DeepMind.txt'}
    if (manifest['generator_sha256'] != sha(Path(M.__file__)) or source['generator_sha256'] != sha(Path(M.__file__))
            or manifest['license_manifest_sha256'] != sha(licenses)
            or manifest['source_revision'] != M.REVISION or manifest['seed'] != M.SEED
            or manifest['a3_revision'] != A3_REVISION or manifest['a3_input_hashes'] != A3_HASHES
            or manifest['dependencies'] != M.DEPENDENCIES or manifest['raw_questions'] != 12000
            or manifest['upstream_files'] != policy['upstream_files']
            or manifest['upstream_python_files'] != policy['upstream_python_files']
            or set(manifest['files']) != names): raise ValueError('Frozen converter/source binding failed')
    for name in names:
        if sha(incoming/name) != manifest['files'][name]['sha256']: raise ValueError('Candidate/source hash mismatch')
    if (sha(incoming/'raw-source.jsonl') != source['finite_source_raw_sha256']
            or sha(incoming/'scan-candidates.jsonl') != source['finite_source_scan_sha256']
            or sha(incoming/'LICENSE-DeepMind.txt') != source['license_sha256']
            or (incoming/'NOTICE-DeepMind.txt').read_text() != source['notice_text']
            or M.digest(source['notice_text']) != source['notice_sha256']):
        raise ValueError('Frozen finite source or attribution changed')
    if (report.get('source_level') is not True or report.get('protected_index_sha256') != M.ITEM25_INDEX_SHA256
            or report.get('scanner_sha256') != sha(Path(M.__file__).with_name('item33_full_suite_scan.py'))
            or report.get('candidates_sha256') != source['finite_source_scan_sha256']
            or report.get('scanned_records') != 24000 or report.get('retained_records') != 24000
            or any(report.get(k) != 0 for k in ('direct_hits','removed_records','removed_families',
                                               'invalid_empty_state_records','remaining_hits_under_scanner'))):
        raise ValueError('Incomplete/unbound full-source scan')
    raw = read(incoming/'raw-source.jsonl')
    M.audit_raw(raw)
    if M.stream_hash(M.scan_rows(raw)) != source['finite_source_scan_sha256']:
        raise ValueError('Unscanned or changed raw/converted universe')
    splits = {s:read(incoming/(s+'.jsonl')) for s in ('train','diagnostic')}; M.audit(splits)
    if ({s:len(rs) for s,rs in splits.items()} != {'train':6000,'diagnostic':300}
            or manifest['splits'] != {s:M.counts(rs) for s,rs in splits.items()}):
        raise ValueError('Candidate counts changed')
    from jebadiah_prompt import PROMPT_SOURCE_SHA256
    contract = json.loads((Path(__file__).resolve().parents[1]/'results/runs/9b-chat-v1/adapter/prompt_contract.json').read_text())
    render = manifest['render_validation']
    if (render.get('checked_renders') != 12600 or render.get('truncated') != 0
            or render.get('max_seq_length') != 2048 or render.get('prompt_budget') != 1984
            or render.get('padding_reserve') != 64 or render.get('tokenizer_sha256') != M.TOKENIZER_SHA256
            or render.get('prompt_source_sha256') != PROMPT_SOURCE_SHA256
            or render.get('chat_template_sha256') != contract['chat_template_sha256']
            or render.get('choice_orders') != ['canonical','reversed','seeded-shuffled']
            or render.get('noul_order') != 'true,false'
            or set(render.get('max_prompt_tokens_by_type',{})) != {'choice','noul'}
            or any(not 0 < n <= 1984 for n in render['max_prompt_tokens_by_type'].values())):
        raise ValueError('Invalid render/prompt-contract receipt')
    return splits, raw


def compose(base, incoming, output, mode, licenses=M.POLICY):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    manifest = json.loads((incoming/'manifest.json').read_text())
    # Fail closed before writing files or interpreting source payloads.
    if manifest.get('overlap_scan',{}).get('status') != 'passed':
        raise ValueError('Full-source Studio scan required before final composition')
    for name,expected in {**A3_HASHES,'manifest.json':A3_MANIFEST_SHA256}.items():
        if sha(base/name) != expected: raise ValueError('Wrong pinned A3: '+name)
    original = read(base/'train.jsonl'); calib = read(base/'calib.jsonl')
    if (sum(len(r['questions']) for r in original) != PRESENTATIONS
            or sum(len(r['questions']) for r in calib) != CALIBRATION): raise ValueError('Wrong pinned A3 counts')
    splits,raw = validate_scan(manifest,incoming,licenses)
    expected,selection = M.select(raw,original+calib)
    if splits != expected or manifest['selection'] != selection:
        raise ValueError('Candidates disagree with finite source/deterministic partition')
    for key in (lambda r:r['id'],lambda r:r['family_id'],state_key):
        if {key(r) for r in original+calib} & {key(r) for r in splits['train']+splits['diagnostic']}:
            raise ValueError('A3/candidate collision')
    result,receipt = compose_rows(original,splits['train'],mode)
    for key in (lambda r:r['id'],lambda r:r['family_id'],state_key):
        if {key(r) for r in result} & {key(r) for r in splits['diagnostic']}:
            raise ValueError('Diagnostic leakage')
    output.mkdir(parents=True,exist_ok=True)
    if mode == 'add':
        additions = result[len(original):]
        (output/'train.jsonl').write_bytes(additive_bytes((base/'train.jsonl').read_bytes(),additions))
    else: write_rows(output/'train.jsonl',result)
    shutil.copyfile(base/'calib.jsonl',output/'calib.jsonl')
    shutil.copyfile(incoming/'diagnostic.jsonl',output/'math-diagnostic.jsonl')
    for name in ('NOTICE-DeepMind.txt','LICENSE-DeepMind.txt'):
        shutil.copyfile(incoming/name,output/name)
    write_json(output/'composition-slots.json',receipt)
    report = {'name':'item55-rung5-'+mode,'private':True,'a3_revision':A3_REVISION,
              'a3_input_hashes':A3_HASHES,'candidate_manifest_sha256':sha(incoming/'manifest.json'),
              'overlap_scan_report_sha256':sha(incoming/'overlap-scan-report.json'),
              'slots':{k:v for k,v in receipt.items() if k != 'slots'},
              'calibration':'Original A3 512 questions and bytes unchanged; diagnostic separate; temperatures unchanged.',
              'base_policy':'Existing private A3 exceptions unchanged; no public shipping clearance.',
              'files':{p.name:{'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file()},
              'training_launched':False}
    write_json(output/'manifest.json',report); return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','incoming','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('replace','add'),required=True)
    p.add_argument('--licenses',type=Path,default=M.POLICY)
    a=p.parse_args();print(json.dumps(compose(a.base,a.incoming,a.out,a.mode,a.licenses),indent=2))
