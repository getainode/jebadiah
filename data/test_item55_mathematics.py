# SPDX-License-Identifier: Apache-2.0
"""Exact oracles, program isolation, area/type composition and fail-closed gates."""
import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import item55_mathematics as M
import item55_rung5_mix as C
from item32_ablation_mix import read
from item33_skill_data import sha, write_json


def raw(expression='(20 - 4)/2 + 3', program=None, key='mixed:0'):
    program = program or ['Add',['Div',['Sub',['Constant','20'],['Constant','4']],['Constant','2']],['Constant','3']]
    return {'id':'item55:raw:'+key,'question':'Calculate '+expression+'.',
            'module':key.split(':')[0], 'sample_seed':55, 'answer':str(M.exact(expression)),
            'expression':expression,'program':program}


def candidate(kind, index):
    for i in range(index*20,index*20+100):
        r = M.convert(raw(key=f'mixed:{i}'))
        if r['questions']['decision']['type'] == kind: return r
    raise AssertionError('No candidate')


def base_row(index, kind='choice', area='knowledge'):
    q={'type':kind,'instructions':'Select the supplied value.'}
    if kind=='choice':q['criteria']={'a':None,'b':None}
    return {'id':f'base:{index}','source':'original-test','area':area,'family_id':f'base-family:{index}',
            'state':f'base state {index}','questions':{'q':q},'label':{'q':'a' if kind=='choice' else True},
            'target':{'q':{'a':1,'b':0}}}


class ExactOracleTests(unittest.TestCase):
    def test_nested_exact_fraction_and_precedence(self):
        self.assertEqual(M.exact('-(9 - 5)/3*(7/2) + 1'),Fraction(-11,3))
        self.assertEqual(M.tree_value(raw()['program']),Fraction(11))

    def test_no_eval_and_no_float_or_power(self):
        for text in ("__import__('os').system('echo unsafe')",'1.25 + 2','2**3','x+1','True+1'):
            with self.subTest(text=text),self.assertRaises(ValueError):M.exact(text)
        with self.assertRaises(ZeroDivisionError):M.exact('1/(3-3)')

    def test_mismatched_upstream_gold_and_tree_rejected(self):
        r=raw();r['answer']='10'
        with self.assertRaisesRegex(ValueError,'Wrong upstream'):M.convert(r)
        r=raw();r['program'][1][2][1]='3'
        with self.assertRaisesRegex(ValueError,'Wrong upstream'):M.convert(r)

    def test_number_changes_share_structure_across_modules(self):
        a=raw();b=raw('(50 - 10)/4 + 8',key='add_sub_multiple:0')
        b['program']=['Add',['Div',['Sub',['Constant','50'],['Constant','10']],['Constant','4']],['Constant','8']]
        self.assertEqual(M.family(a),M.family(b))
        a['program']=['Neg',['Constant','20']];b['program']=['Constant','-7']
        self.assertEqual(M.family(a),M.family(b))

    def test_printed_associative_variants_share_family(self):
        a=['Add',['Add',['Constant','1'],['Constant','2']],['Constant','3']]
        b=['Add',['Constant','9'],['Add',['Constant','8'],['Constant','7']]]
        self.assertEqual(M.structure(a),M.structure(b))
        a=['Mul',['Mul',['Constant','1'],['Constant','2']],['Constant','3']]
        b=['Mul',['Constant','9'],['Mul',['Constant','8'],['Constant','7']]]
        self.assertEqual(M.structure(a),M.structure(b))


    def test_independent_choice_and_boolean_gold(self):
        for kind in ('choice','noul'):
            r=candidate(kind,1);q=r['questions']['decision'];gold=M.exact(r['provenance']['expression'])
            if kind=='choice':
                self.assertEqual(len(q['criteria']),4)
                self.assertEqual(Fraction(r['label']['decision']),gold)
                self.assertEqual(sum(Fraction(k)==gold for k in q['criteria']),1)
            else:self.assertEqual(r['label']['decision'],Fraction(r['state']['proposed_answer'])==gold)

    def test_audit_gold_mutation_and_cross_split_family_leakage(self):
        a=candidate('choice',1);b=candidate('choice',5)
        # Make changed numeric values but same expression structure and distinct text.
        p=b['provenance'];p['expression']='(50 - 10)/4 + 8';p['program']=['Add',['Div',['Sub',['Constant','50'],['Constant','10']],['Constant','4']],['Constant','8']];p['upstream_answer']='18'
        b=M.convert({'id':p['original_id'],'module':p['module'],'sample_seed':p['sample_seed'],
                     'question':'Calculate '+p['expression']+'.','answer':'18','expression':p['expression'],'program':p['program']})
        with self.assertRaisesRegex(ValueError,'structure leakage'):M.audit({'train':[a],'diagnostic':[b]})
        a['label']['decision']='999'
        with self.assertRaises(ValueError):M.audit({'train':[a]})

    def test_lineage_variants_aggregate(self):
        for s in ('deepmind_math','deepmind-mathematics','google-deepmind/mathematics_dataset',M.SOURCE):
            self.assertTrue(M.is_lineage({'source':s}))
        self.assertFalse(M.is_lineage({'source':'unrelated-arithmetic'}))


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.base=[base_row(i,'choice' if i%2 else 'noul') for i in range(20)]
        self.candidates=[candidate('choice',1),candidate('noul',5)]

    def test_matched_replace_add_share_ids_and_preserve_presentations(self):
        before=copy.deepcopy(self.base)
        replace,rr=C.compose_rows(self.base,self.candidates,'replace')
        add,ar=C.compose_rows(self.base,self.candidates,'add')
        self.assertEqual(self.base,before)
        self.assertEqual(rr['presentations'],20);self.assertEqual(ar['presentations'],22)
        self.assertEqual(rr['slots'],ar['slots'])
        self.assertEqual(rr['before']['area_type'],rr['after']['area_type'])
        self.assertEqual(add[:len(before)],before)
        self.assertEqual(rr['lineage_questions'],2)

    def test_mixed_row_keeps_state_labels_targets_and_other_questions(self):
        self.base=[base_row(0)]+[base_row(i,area='arts') for i in range(1,20)]
        self.base[0]['questions']['kept']={'type':'score','instructions':'Score the value.','criteria':['low','high']}
        self.base[0]['label']['kept']=0;self.base[0]['target']['kept']=[1,0]
        result,receipt=C.compose_rows(self.base,[candidate('choice',1)],'replace')
        kept=next(r for r in result if r['id']=='base:0')
        self.assertEqual(kept['state'],self.base[0]['state'])
        self.assertEqual(kept['questions'],{'kept':self.base[0]['questions']['kept']})
        self.assertEqual(kept['label'],{'kept':0});self.assertEqual(kept['target'],{'kept':[1,0]})
        self.assertEqual(receipt['selected_questions'],1)

    def test_additive_original_bytes_exact(self):
        original=b'{"original": "spacing retained"}\n'
        out=C.additive_bytes(original,[self.candidates[0]])
        self.assertTrue(out.startswith(original))
        with self.assertRaises(ValueError):C.additive_bytes(b'no-newline',[])

    def test_cap_and_insufficient_candidates_fail_closed(self):
        self.base[0]['source']=M.SOURCE;self.base[1]['source']='deepmind_math'
        with self.assertRaisesRegex(ValueError,'cap headroom'):C.compose_rows(self.base,self.candidates,'add')
        with self.assertRaisesRegex(ValueError,'Insufficient'):C.compose_rows([base_row(i) for i in range(20)],[],'replace')
        with self.assertRaisesRegex(ValueError,'matched'):C.compose_rows([base_row(i,area='arts') for i in range(20)],self.candidates,'replace')

    def test_pending_scan_blocks_both_forms_without_creating_output(self):
        with tempfile.TemporaryDirectory(prefix='item55-test-') as td:
            folder=Path(td);incoming=folder/'incoming';incoming.mkdir()
            write_json(incoming/'manifest.json',{'overlap_scan':{'status':'pending_studio_scan'}})
            for mode in ('replace','add'):
                out=folder/mode
                with self.assertRaisesRegex(ValueError,'Studio scan'):C.compose(folder/'missing-base',incoming,out,mode)
                self.assertFalse(out.exists())


@unittest.skipUnless(os.environ.get('ITEM55_CANDIDATES'),'Set ITEM55_CANDIDATES for full live finite-source checks')
class LiveFiniteSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=Path(os.environ['ITEM55_CANDIDATES'])
        cls.raw=read(cls.folder/'raw-source.jsonl')
        cls.splits={s:read(cls.folder/(s+'.jsonl')) for s in ('train','diagnostic')}

    def test_complete_source_exact_oracles_and_partition(self):
        M.audit_raw(self.raw);M.audit(self.splits)
        self.assertEqual(len(self.raw),12000)
        self.assertEqual({s:len(rs) for s,rs in self.splits.items()},{'train':6000,'diagnostic':300})
        for split,rows in self.splits.items():
            for r in rows:
                held=int(r['family_id'].rsplit(':',1)[1][:8],16)%5==0
                self.assertEqual(held,split=='diagnostic')

    def test_bound_scan_stream_and_render_receipt(self):
        manifest=json.loads((self.folder/'manifest.json').read_text())
        policy=json.loads(M.POLICY.read_text());src=policy['sources'][M.SOURCE]
        self.assertEqual(M.stream_hash(M.scan_rows(self.raw)),sha(self.folder/'scan-candidates.jsonl'))
        self.assertEqual(sha(self.folder/'raw-source.jsonl'),src['finite_source_raw_sha256'])
        self.assertEqual(sha(self.folder/'scan-candidates.jsonl'),src['finite_source_scan_sha256'])
        self.assertEqual(manifest['render_validation']['checked_renders'],12600)
        self.assertEqual(manifest['render_validation']['truncated'],0)

    def test_unmodified_upstream_seed_replay(self):
        if not os.environ.get('ITEM55_UPSTREAM'):self.skipTest('Set ITEM55_UPSTREAM')
        with patch.object(M,'PER_MODULE',2):
            replay=M.generate(Path(os.environ['ITEM55_UPSTREAM']))
        by_id={r['id']:r for r in self.raw}
        self.assertTrue(all(r==by_id[r['id']] for r in replay))

    def test_scan_gate_rejects_incomplete_or_unbound_reports(self):
        # Test-only passed report fixture. Never touches a protected index and
        # never writes a passed production receipt or a final composition.
        manifest=json.loads((self.folder/'manifest.json').read_text())
        pending=manifest['overlap_scan']
        with self.assertRaisesRegex(ValueError,'Studio scan'):C.validate_scan(manifest,self.folder)
        with tempfile.TemporaryDirectory(prefix='item55-scan-test-') as td:
            folder=Path(td)
            for p in self.folder.iterdir():
                if p.is_file():(folder/p.name).symlink_to(p.resolve())
            (folder/'overlap-scan-report.json').unlink()
            report={'status':'passed','source_level':True,'protected_index_sha256':M.ITEM25_INDEX_SHA256,
                    'scanner_sha256':sha(Path(M.__file__).with_name('item33_full_suite_scan.py')),
                    'candidates_sha256':pending['candidates_sha256'],'scanned_records':24000,'retained_records':24000,
                    **{k:0 for k in ('direct_hits','removed_records','removed_families','invalid_empty_state_records','remaining_hits_under_scanner')}}
            for key,value in (('scanned_records',23999),('candidates_sha256','wrong'),('direct_hits',1),
                              ('scanner_sha256','wrong'),('protected_index_sha256','wrong')):
                changed={**report,key:value};m={**manifest,'overlap_scan':changed}
                write_json(folder/'overlap-scan-report.json',changed)
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Incomplete/unbound'):C.validate_scan(m,folder)
            # A complete fixture can validate payload and contract, but cannot
            # establish actual clearance. Only the lead's real scanner can.
            write_json(folder/'overlap-scan-report.json',report)
            rs,_=C.validate_scan({**manifest,'overlap_scan':report},folder)
            self.assertEqual(rs,self.splits)


if __name__=='__main__':unittest.main()
