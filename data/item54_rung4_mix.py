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
import item54_massive as M

PRESENTATIONS = 21190
CALIBRATION = 512
CAP = 2119
REQUESTED = 1000


def plan_slots(original, candidates):
    """One deterministic matched selection feeds both experimental forms."""
    total = sum(len(r['questions']) for r in original)
    existing = sum(len(r['questions']) for r in original if M.is_lineage(r))
    budget = min(REQUESTED, CAP - existing, total // 10 - existing)
    if budget <= 0: raise ValueError('No aggregate HWU/SLURP/MASSIVE cap headroom')
    pools = {k: sorted(((i,qid) for i,r in enumerate(original) for qid,q in r['questions'].items()
                        if r.get('area') == 'retrieval' and q['type'] == k),
                       key=lambda slot: digest(M.SEED + ':donor:' + original[slot[0]]['id'] + ':' + slot[1]))
             for k in ('choice','noul')}
    n = min(budget, sum(map(len,pools.values())))
    if not n: raise ValueError('No matched retrieval donor slots')
    allocation = {k:min(n//2,len(v)) for k,v in pools.items()}
    for k in pools: allocation[k] += min(n-sum(allocation.values()),len(pools[k])-allocation[k])
    selected = {}; mapping = []
    for kind,number in allocation.items():
        incoming = sorted((r for r in candidates if r['questions']['decision']['type'] == kind),
                          key=lambda r:digest(M.SEED + ':replace:' + r['id']))
        if len(incoming) < number: raise ValueError('Insufficient unique matching candidates')
        for slot,row in zip(pools[kind][:number],incoming[:number]):
            if row['area'] != 'retrieval' or row['source'] != M.SOURCE or len(row['questions']) != 1:
                raise ValueError('Unmatched incoming row')
            selected[slot] = copy.deepcopy(row)
            donor = original[slot[0]]
            mapping.append({'original_id':donor['id'],'question_id':slot[1],
                            'replacement_id':row['id'],'area':'retrieval','type':kind,
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


def validate_scan(manifest, raw, incoming, licenses):
    report = manifest.get('overlap_scan',{})
    if report.get('status') != 'passed':
        raise ValueError('Full-source and converted Studio scan required before final composition')
    if json.loads((incoming/'overlap-scan-report.json').read_text()) != report:
        raise ValueError('Scan report mismatch')
    source = M.validate_policy(raw,licenses)['sources'][M.SOURCE]
    if (incoming/'NOTICE-MASSIVE.txt').read_text() != source['notice_text']:
        raise ValueError('Attribution notice changed')
    if (manifest['generator_sha256'] != sha(Path(M.__file__))
            or manifest['license_manifest_sha256'] != sha(licenses)
            or manifest['source_revision'] != M.REVISION
            or manifest['source_files'] != source['input_files']
            or set(manifest['files']) != {'train.jsonl','diagnostic.jsonl'}):
        raise ValueError('Frozen source/converter binding failed')
    splits = {}
    for split in ('train','diagnostic'):
        p = incoming/f'{split}.jsonl'
        if sha(p) != manifest['files'][p.name]['sha256']: raise ValueError('Candidate hash mismatch')
        splits[split] = read(p)
    M.audit(splits)
    if ({s:len(rs) for s,rs in splits.items()} != {'train':6000,'diagnostic':300}
            or manifest['splits'] != {s:M.counts(rs) for s,rs in splits.items()}
            or M.counts(splits['train'])['types'] != {'choice':3000,'noul':3000}
            or M.counts(splits['diagnostic'])['types'] != {'choice':150,'noul':150}):
        raise ValueError('Candidate counts changed')
    raw_count = source['complete_records']
    if (report.get('source_level') is not True or report.get('protected_index_sha256') != M.ITEM25_INDEX_SHA256
            or report.get('source_archive_sha256') != source['input_files'][M.ARCHIVE]['sha256']
            or report.get('converted_sha256') != {s:sha(incoming/f'{s}.jsonl') for s in splits}
            or report.get('scanned_records') != raw_count+6300):
        raise ValueError('Unbound full-source scan')
    for phase,number,rows in (('raw',raw_count,M.raw_scan_rows(raw)),
                              ('converted',6300,M.converted_scan_rows(splits))):
        stage = report.get(phase,{})
        if (stage.get('status') != 'passed' or stage.get('source_level') is not True
                or stage.get('protected_index_sha256') != M.ITEM25_INDEX_SHA256
                or stage.get('scanner_sha256') != sha(Path(M.__file__).with_name('item33_full_suite_scan.py'))
                or stage.get('scanned_records') != number or stage.get('retained_records') != number
                or any(stage.get(k) != 0 for k in ('direct_hits','removed_records','removed_families',
                                                   'invalid_empty_state_records','remaining_hits_under_scanner'))
                or stage.get('candidates_sha256') != M.stream_hash(rows)):
            raise ValueError('Incomplete/unbound ' + phase + ' scan')
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
        raise ValueError('Invalid prompt-contract render receipt')
    return splits


def compose(raw,base,incoming,output,mode,licenses=M.POLICY):
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    manifest = json.loads((incoming/'manifest.json').read_text())
    # Fail closed before loading source text or creating any output.
    if manifest.get('overlap_scan',{}).get('status') != 'passed':
        raise ValueError('Full-source and converted Studio scan required before final composition')
    for name,expected in {**A3_HASHES,'manifest.json':A3_MANIFEST_SHA256}.items():
        if sha(base/name) != expected: raise ValueError('Wrong pinned A3: '+name)
    splits = validate_scan(manifest,raw,incoming,licenses)
    expected,selection = M.select(raw,base)
    if splits != expected or manifest['selection'] != selection:
        raise ValueError('Candidates disagree with original train rows or deterministic selection')
    original = read(base/'train.jsonl');calib = read(base/'calib.jsonl')
    if (sum(len(r['questions']) for r in original) != PRESENTATIONS
            or sum(len(r['questions']) for r in calib) != CALIBRATION):
        raise ValueError('Wrong A3 presentation counts')
    result,receipt = compose_rows(original,splits['train'],mode)
    for key in (lambda r:r['id'],state_key):
        if {key(r) for r in result} & {key(r) for r in splits['diagnostic']}:
            raise ValueError('Held-out diagnostic leakage')
        new = [r for r in result if r['source'] == M.SOURCE]
        if {key(r) for r in new} & {key(r) for r in original+calib}:
            raise ValueError('A3/calibration candidate collision')
    output.mkdir(parents=True,exist_ok=True)
    if mode == 'replace': write_rows(output/'train.jsonl',result)
    else:
        original_bytes = (base/'train.jsonl').read_bytes()
        by_id = {r['id']:r for r in splits['train']}
        addition = [by_id[s['replacement_id']] for s in receipt['slots']]
        # Selection insertion order matches plan_slots and compose_rows.
        write_rows(output/'item54-additions.jsonl',addition)
        (output/'train.jsonl').write_bytes(additive_bytes(original_bytes,addition))
        if not (output/'train.jsonl').read_bytes().startswith(original_bytes):
            raise ValueError('A3 original prefix changed')
        receipt['a3_train_prefix_byte_identical'] = True
    shutil.copyfile(base/'calib.jsonl',output/'calib.jsonl')
    shutil.copyfile(incoming/'diagnostic.jsonl',output/'intent-diagnostic.jsonl')
    shutil.copyfile(licenses,output/'source-licenses.json')
    shutil.copyfile(incoming/'NOTICE-MASSIVE.txt',output/'NOTICE-MASSIVE.txt')
    shutil.copyfile(incoming/'overlap-scan-report.json',output/'overlap-scan-report.json')
    write_json(output/'composition-proof.json',receipt)
    report = {'name':'item54-rung4-'+mode+'-a3','private':True,'a3_revision':A3_REVISION,
              'a3_input_hashes':{**A3_HASHES,'manifest.json':A3_MANIFEST_SHA256},
              'candidate_manifest_sha256':sha(incoming/'manifest.json'),
              'source_scan_sha256':sha(incoming/'overlap-scan-report.json'),
              'mode':mode,'questions':receipt['presentations'],'calibration_questions':CALIBRATION,
              'incoming_questions':receipt['selected_questions'],'lineage_questions':receipt['lineage_questions'],
              'aggregate_cap_questions':CAP,
              'calibration':'Original A3 bytes and temperatures unchanged; intent diagnostic separate',
              'base_policy':'Existing A3 private exceptions unchanged; no shipping clearance',
              'files':{p.name:{'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file()},
              'training_launched':False}
    write_json(output/'manifest.json',report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('raw','base','incoming','out'): p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--mode',choices=('replace','add'),required=True)
    p.add_argument('--licenses',type=Path,default=M.POLICY)
    a=p.parse_args()
    print(json.dumps(compose(a.raw,a.base,a.incoming,a.out,a.mode,a.licenses),indent=2))
