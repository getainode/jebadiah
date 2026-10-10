# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks. Synthetic scanner fixtures never read the index."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import item54_massive as M
import item54_rung4_mix as C


def upstream(i=1, **kw):
    row = {'id':str(i),'locale':'en-US','partition':'train','scenario':'alarm',
           'intent':'alarm_set','utt':f'test request {i}'}
    return {**row,**kw}


def intents():
    return {'alarm_set':'alarm','alarm_remove':'alarm',
            **{f'category_{i}':'other' for i in range(18)}}


def candidate(i,kind='choice'):
    return M.convert(upstream(i),kind,intents())


def base_row(i,kind='choice',area='retrieval',source='original',mixed=False):
    q={'type':kind,'instructions':'Original instruction'}
    if kind == 'choice':q['criteria']={'one':'First','two':'Second'}
    row={'id':f'base:{i}','state':{'original':i},'source':source,'area':area,
         'questions':{'main':q},'label':{'main':'one' if kind=='choice' else True},
         'target':{'main':{'one':1.0} if kind=='choice' else {'true':1.0}}}
    if mixed:
        row['questions']['untouched']={'type':'score','instructions':'Preserved','criteria':['low','high']}
        row['label']['untouched']=1;row['target']['untouched']={'1':1.0}
    return row


class ConverterTests(unittest.TestCase):
    def test_only_original_en_US_train_and_known_labels(self):
        for changes in ({'partition':'dev'},{'partition':'test'},{'locale':'de-DE'}, {'intent':'unlisted'}):
            with self.assertRaises(ValueError):M.convert(upstream(**changes),'choice',intents())
        with self.assertRaises(ValueError):M.convert(upstream(),'score',intents())

    def test_source_intent_menu_and_hard_neighbors(self):
        row=candidate(1)
        q=row['questions']['decision']
        self.assertEqual(len(q['criteria']),16)
        self.assertIn('alarm_remove',q['criteria'])
        self.assertEqual(row['label']['decision'],'alarm_set')
        self.assertNotIn('none',q['criteria'])
        self.assertEqual(row,candidate(1))
        self.assertEqual(row['family_id'],'item54:slurp-id:1')
        self.assertNotIn('intent',row['state'])

    def test_verification_uses_upstream_label_without_OOS(self):
        labels=set()
        for i in range(40):
            row=candidate(i,'noul');labels.add(row['label']['decision'])
            self.assertIs(row['label']['decision'],row['state']['proposed_intent']=='alarm_set')
            self.assertNotIn('criteria',row['questions']['decision'])
        self.assertEqual(labels,{True,False})

    def test_protected_names_and_ancestors_are_rejected(self):
        policy=json.loads(M.POLICY.read_text());src=policy['sources'][M.SOURCE]
        for ancestor in ('CLINC150','BANKING77','banking-77','clinc-150','ToolRet','schema-guided'):
            bad=copy.deepcopy(src);bad['ancestry']=[ancestor]
            with self.assertRaisesRegex(ValueError,'Protected source ancestry'):
                M.validate_source(bad,policy['source_exclusions'])
        M.validate_source(src,policy['source_exclusions'])
        self.assertEqual(src['aggregate_source_family'],M.LINEAGE)

    def test_lineage_aliases_and_no_substring_false_positive(self):
        for name in ('multilingual/massive','hwu-original','HWU64','pswietojanski/slurp',
                     'xliuhw/NLU-Evaluation-Data',M.LINEAGE):
            self.assertTrue(M.is_lineage({'source':name}))
        self.assertTrue(M.is_lineage({'provenance':{'parent':'SLURP'}}))
        self.assertFalse(M.is_lineage({'source':'massively-unrelated'}))

    def test_diagnostic_id_and_utterance_leakage_fails(self):
        row=candidate(2)
        for changed in (copy.deepcopy(row),candidate(3)):
            changed['state']['user_request']=row['state']['user_request']
            with self.assertRaises(ValueError):M.audit({'train':[row],'diagnostic':[changed]})
        bad=copy.deepcopy(row);bad['label']['decision']='alarm_remove'
        with self.assertRaises(ValueError):M.audit({'train':[bad]})

    def test_pending_scan_cannot_create_either_composition(self):
        with tempfile.TemporaryDirectory(prefix='item54-test-') as folder:
            root=Path(folder);incoming=root/'incoming';incoming.mkdir()
            M.write_json(incoming/'manifest.json',{'overlap_scan':{'status':'pending_studio_scan'}})
            for mode in ('replace','add'):
                output=root/mode
                with self.assertRaisesRegex(ValueError,'Studio scan required'):
                    C.compose(root/'missing-raw',root/'missing-base',incoming,output,mode)
                self.assertFalse(output.exists())

    def test_source_hit_rejects_whole_source_and_removes_symlink(self):
        # check_index and scanner are replaced only in this temporary fixture.
        # This is not an overlap clearance and does not open any protected file.
        with tempfile.TemporaryDirectory(prefix='item54-test-') as folder:
            root=Path(folder);output=root/'out';fake=root/'item54-fake-index'
            fake.write_bytes(b'fixture')
            def reject(work):
                M.write_json(work/'scan-report.json',{'removed_records':2,'direct_hits':1})
            with patch.object(M,'check_index'),patch.object(M,'scan',reject):
                with self.assertRaisesRegex(ValueError,'Entire upstream MASSIVE source rejected'):
                    M.scan_phase([{'id':'fixture','state':'fixture','family':M.SOURCE}],output,fake,'raw')
            self.assertFalse((output/'raw/protected.pkl').exists())
            self.assertFalse((output/'train.jsonl').exists())
            self.assertEqual(json.loads((output/'overlap-scan-report.json').read_text())['status'],'rejected')

    def test_scanner_exception_removes_symlink(self):
        with tempfile.TemporaryDirectory(prefix='item54-test-') as folder:
            root=Path(folder);fake=root/'item54-fake-index';fake.write_bytes(b'fixture')
            with patch.object(M,'check_index'),patch.object(M,'scan',side_effect=RuntimeError('fixture failure')):
                with self.assertRaises(RuntimeError):M.scan_phase([],root/'out',fake,'raw')
            self.assertFalse((root/'out/raw/protected.pkl').exists())


class CompositionTests(unittest.TestCase):
    def test_replacement_preserves_mixed_siblings_and_type_counts(self):
        original=[base_row(0,mixed=True),base_row(1,'noul')]+[base_row(i,area='language') for i in range(2,31)]
        before=copy.deepcopy(original)
        rows,rc=C.compose_rows(original,[candidate(100),candidate(101,'noul')],'replace')
        self.assertEqual(original,before)
        self.assertEqual(rc['presentations'],32)
        self.assertEqual(rc['selected_questions'],2)
        kept=next(r for r in rows if r['id']=='base:0')
        self.assertEqual(kept['questions'],{'untouched':original[0]['questions']['untouched']})
        self.assertEqual(kept['label'],{'untouched':1})
        self.assertEqual(kept['target'],{'untouched':{'1':1.0}})
        self.assertEqual(C.histograms(rows)['area_type'],C.histograms(original)['area_type'])

    def test_same_additions_and_no_removals_in_add_mode(self):
        original=[base_row(0,source='hwu-original'),base_row(1,'noul')]+[base_row(i,area='language') for i in range(2,40)]
        candidates=[candidate(100),candidate(101,'noul')]
        replacement,r=C.compose_rows(original,candidates,'replace')
        additive,a=C.compose_rows(original,candidates,'add')
        self.assertEqual(r['slots'],a['slots'])
        self.assertEqual(additive[:len(original)],original)
        self.assertEqual(a['removed_questions'],0)
        self.assertEqual(a['removed_lineage_questions'],0)
        self.assertEqual(r['removed_lineage_questions'],1)
        self.assertEqual(a['lineage_questions'],3)
        self.assertEqual(a['presentations'],42)
        self.assertEqual({x['id'] for x in replacement if x['source']==M.SOURCE},
                         {x['id'] for x in additive if x['source']==M.SOURCE})

    def test_lineage_cap_counts_inherited_exposure(self):
        rows=[base_row(i,source='SLURP') for i in range(10)] + [base_row(i,area='language') for i in range(10,100)]
        with self.assertRaisesRegex(ValueError,'cap headroom'):C.compose_rows(rows,[candidate(101)],'add')
        # The fixed 2,119 cap remains even if additive total would permit more.
        rows=[base_row(i,source='HWU64') for i in range(2119)]
        rows += [base_row(i,area='language') for i in range(2119,22000)]
        with self.assertRaisesRegex(ValueError,'cap headroom'):C.compose_rows(rows,[candidate(30000)],'replace')

    def test_additive_keeps_exact_original_bytes_and_order(self):
        original=b'{ "id": "first", "state": "noncanonical spacing" }\n\n'
        addition=[candidate(100),candidate(101,'noul')]
        result=C.additive_bytes(original,addition)
        self.assertTrue(result.startswith(original))
        suffix=result[len(original):].splitlines()
        self.assertEqual([json.loads(x) for x in suffix],addition)
        with self.assertRaises(ValueError):C.additive_bytes(original.rstrip(),addition)

    def test_wrong_area_missing_type_or_id_collision_fails(self):
        original=[base_row(i) for i in range(20)]
        wrong=candidate(100);wrong['area']='language'
        with self.assertRaises(ValueError):C.compose_rows(original,[wrong],'replace')
        with self.assertRaises(ValueError):C.compose_rows(original,[candidate(100,'noul')],'replace')
        bad=candidate(100);bad['id']='base:0'
        with self.assertRaises(ValueError):C.compose_rows(original,[bad,candidate(101)],'add')


@unittest.skipUnless(os.environ.get('ITEM54_SCRATCH'),'Set ITEM54_SCRATCH for frozen-source CPU checks')
class FrozenSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['ITEM54_SCRATCH']);cls.raw=cls.root/'item54-raw';cls.base=cls.root/'item54-a3-download/a3'
        cls.splits,cls.selection=M.select(cls.raw,cls.base)

    def test_original_id_mapping_and_candidate_labels(self):
        M.validate_policy(self.raw);M.audit(self.splits)
        originals={r['id']:r for r in M.upstream_rows(self.raw)};taxonomy=M.taxonomy(self.raw)
        exposed,_,count=M.existing_exposure(self.base,self.raw)
        self.assertEqual(count,{'train':102})
        for split,rows in self.splits.items():
            for r in rows:
                p=r['provenance'];self.assertNotIn(p['slurp_id'],exposed)
                self.assertEqual(r,M.convert(originals[p['slurp_id']],r['questions']['decision']['type'],taxonomy))
        self.assertEqual([len(self.splits[x]) for x in ('train','diagnostic')],[6000,300])

    def test_live_preview_counts_both_forms(self):
        original=C.read(self.base/'train.jsonl')
        r,rc=C.compose_rows(original,self.splits['train'],'replace')
        a,ac=C.compose_rows(original,self.splits['train'],'add')
        self.assertEqual(rc['selected_types'],{'choice':583,'noul':166})
        self.assertEqual((rc['presentations'],rc['lineage_questions']),(21190,749))
        self.assertEqual((ac['presentations'],ac['lineage_questions']),(21939,851))
        self.assertEqual(rc['slots'],ac['slots'])
        self.assertEqual(C.histograms(r)['area_type'],C.histograms(original)['area_type'])
        self.assertEqual(a[:len(original)],original)
        self.assertEqual(C.sha(self.base/'calib.jsonl'),C.A3_HASHES['calib.jsonl'])

    def test_synthetic_pass_receipts_exercise_both_final_writers(self):
        # Synthetic clearance exists only inside this temporary test. It proves
        # the final writers and bindings, never real source overlap clearance.
        import shutil
        incoming_source=self.root/'item54-preview-release'
        manifest=json.loads((incoming_source/'manifest.json').read_text())
        policy=json.loads(M.POLICY.read_text())
        policy['sources'][M.SOURCE]['complete_records']=1
        fixture=[{'id':'item54:scanner-fixture','family':M.SOURCE,'state':'fixture','questions':{}}]
        splits={s:C.read(incoming_source/f'{s}.jsonl') for s in ('train','diagnostic')}
        def phase(rows,number):
            return {'status':'passed','source_level':True,'protected_index_sha256':M.ITEM25_INDEX_SHA256,
                    'scanner_sha256':M.sha(Path(M.__file__).with_name('item33_full_suite_scan.py')),
                    'scanned_records':number,'retained_records':number,'direct_hits':0,'removed_records':0,
                    'removed_families':0,'invalid_empty_state_records':0,'remaining_hits_under_scanner':0,
                    'candidates_sha256':M.stream_hash(rows)}
        report={'status':'passed','source_level':True,'protected_index_sha256':M.ITEM25_INDEX_SHA256,
                'source_archive_sha256':policy['sources'][M.SOURCE]['input_files'][M.ARCHIVE]['sha256'],
                'converted_sha256':{s:M.sha(incoming_source/f'{s}.jsonl') for s in splits},
                'scanned_records':6301,'raw':phase(fixture,1),
                'converted':phase(M.converted_scan_rows(splits),6300)}
        with tempfile.TemporaryDirectory(prefix='item54-test-') as folder:
            root=Path(folder);incoming=root/'incoming';shutil.copytree(incoming_source,incoming)
            manifest['overlap_scan']=report
            M.write_json(incoming/'manifest.json',manifest);M.write_json(incoming/'overlap-scan-report.json',report)
            with patch.object(M,'validate_policy',return_value=policy),patch.object(M,'raw_scan_rows',side_effect=lambda raw:iter(fixture)):
                for mode,count,exposure in [('replace',21190,749),('add',21939,851)]:
                    out=root/mode
                    result=C.compose(self.raw,self.base,incoming,out,mode)
                    self.assertEqual((result['questions'],result['lineage_questions']),(count,exposure))
                    self.assertEqual((out/'calib.jsonl').read_bytes(),(self.base/'calib.jsonl').read_bytes())
                    if mode=='add':self.assertTrue((out/'train.jsonl').read_bytes().startswith((self.base/'train.jsonl').read_bytes()))
                    self.assertEqual((out/'intent-diagnostic.jsonl').read_bytes(),(incoming/'diagnostic.jsonl').read_bytes())
                # A reported pass with only partial upstream coverage must fail.
                report['raw']['scanned_records']=0
                manifest['overlap_scan']=report
                M.write_json(incoming/'manifest.json',manifest);M.write_json(incoming/'overlap-scan-report.json',report)
                with self.assertRaisesRegex(ValueError,'Incomplete/unbound raw scan'):
                    C.compose(self.raw,self.base,incoming,root/'partial','replace')
                self.assertFalse((root/'partial').exists())

    def test_whole_release_scan_input_covers_every_locale_and_split(self):
        policy=json.loads(M.POLICY.read_text())['sources'][M.SOURCE]
        counts={};number=0
        for r in M.archive_rows(self.raw):
            counts.setdefault(r['locale'],{}).setdefault(r['partition'],0)
            counts[r['locale']][r['partition']]+=1;number+=1
        self.assertEqual(number,859092)
        self.assertEqual(len(counts),52)
        self.assertEqual(counts,{k:v['partitions'] for k,v in policy['complete_release'].items()})
        self.assertEqual(counts['en-US'],{'train':11514,'dev':2033,'test':2974})


if __name__=='__main__':unittest.main()
