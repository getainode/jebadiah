"""CPU tests for upstream labels, family isolation and both composition modes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import item53_spacenli as S
import item53_nli_mix as M


def raw(i, family='1', label='entailment'):
    return {'id': f'{family}-{i}', 'premises': f'Object {i} is in container {family}.',
            'hypothesis': f'Container {family} contains object {i}.', 'label': label}


def base_rows(n=20):
    return [{'id': f'base:{i}', 'family_id': f'base:{i}', 'area': 'language', 'state': {'original': i},
             'questions': {'c': {'type':'choice','instructions':'Choose','criteria':{'x':None,'y':None}},
                           'b': {'type':'noul','instructions':'Is it true?'}},
             'label': {'c':'x','b':True}, 'target': {'c':{'x':1,'y':0}}} for i in range(n)]


class SpaceTests(unittest.TestCase):
    def test_converter_preserves_text_and_expert_labels(self):
        for label in S.LABELS:
            r = raw(0,label=label); q = S.convert(r,'item53:pattern:1')
            self.assertEqual(q['state'], 'text_A: '+r['premises']+'\ntext_B: '+r['hypothesis'])
            self.assertEqual(q['label']['decision'],label)
            self.assertEqual(tuple(q['questions']['decision']['criteria']),S.LABELS)
            self.assertIsNone(q['provenance']['original_split'])
            S.audit({'train':[q]})
        with self.assertRaises(ValueError): S.convert(raw(0,label='unknown'),'f')

    def test_related_seeds_and_suffixes_are_one_family(self):
        with tempfile.TemporaryDirectory(prefix='item53-test-') as d:
            p=Path(d)/'patterns.xml'
            p.write_text('<root><group><problem id="2"/><problem id="3a"/></group><problem id="2b"/><problem id="4"/></root>')
            fs=S.families(p)
            self.assertEqual(fs['2'],fs['3a']); self.assertEqual(fs['2'],fs['2b'])
            self.assertNotEqual(fs['2'],fs['4'])

    def test_selection_excludes_a3_families_and_diagnostic_siblings(self):
        rows=[raw(i,str(f),S.LABELS[i%3]) for f in range(1,7) for i in range(220)]
        mapping={str(f):f'item53:pattern:{f}' for f in range(1,7)}
        old=S.convert(rows[0],mapping['1']); old['state']='Passage A:\n'+rows[0]['premises']+'\n\nPassage B:\n'+rows[0]['hypothesis']
        splits,receipt=S.select(rows,mapping,[old])
        self.assertEqual(splits,S.select(rows,mapping,[old])[0]); S.audit(splits)
        self.assertEqual(len(splits['diagnostic']),300)
        self.assertEqual(len(splits['train']),660)
        self.assertNotIn(mapping['1'],{r['family_id'] for rs in splits.values() for r in rs})
        self.assertEqual(receipt['unused_holdout_siblings_excluded'],140)
        self.assertFalse({r['family_id'] for r in splits['train']} & {r['family_id'] for r in splits['diagnostic']})
        bad=copy.deepcopy(splits); bad['diagnostic'].append(bad['train'][0])
        with self.assertRaises(ValueError): S.audit(bad)
        old['state']='Unknown origin text'
        with self.assertRaisesRegex(ValueError,'ancestry'): S.select(rows,mapping,[old])

    def test_replacement_preserves_nonmatching_questions_and_state(self):
        rows=base_rows(); before=copy.deepcopy(rows)
        incoming=[S.convert(raw(i),'newfamily') for i in range(20)]
        result,receipt=M.plan(rows,incoming,'replace',requested=3)
        self.assertEqual(rows,before); self.assertEqual(receipt['incoming_questions'],3)
        self.assertEqual(sum(len(r['questions']) for r in result),40)
        for old in rows:
            kept=next(r for r in result if r['id']==old['id'])
            self.assertEqual(kept['state'],old['state']);self.assertEqual(kept['questions']['b'],old['questions']['b'])
            if 'c' not in kept['questions']: self.assertNotIn('c',kept['target'])
        self.assertEqual(receipt['questions_after'],40)

    def test_add_keeps_all_rows_and_aggregate_cap(self):
        rows=base_rows(30)
        rows[0]['subset']='SpaceNLI'  # Two pre-existing source presentations.
        incoming=[S.convert(raw(i),'newfamily') for i in range(30)]
        result,receipt=M.plan(rows,incoming,'add',requested=1000)
        self.assertEqual(result[:len(rows)],rows)
        self.assertEqual(receipt['incoming_questions'],4)
        self.assertEqual(receipt['source_questions_after'],6)
        self.assertEqual(receipt['questions_after'],64)
        self.assertEqual(receipt['source_fraction_of_original_a3'],.1)
        replaced,receipt=M.plan(rows,incoming,'replace',requested=1000)
        self.assertEqual(sum(len(r['questions']) for r in replaced),60)
        self.assertEqual(M.source_questions(replaced),6)
        self.assertNotIn(rows[0]['id'],{r['original_id'] for r in receipt['replacements']})

    def test_pending_source_scan_blocks_both_modes(self):
        with tempfile.TemporaryDirectory(prefix='item53-test-') as d:
            p=Path(d);candidate=p/'candidate';candidate.mkdir()
            S.write_rows(p/'train.jsonl',[]); S.write_rows(p/'calib.jsonl',[])
            S.write_json(candidate/'manifest.json',{'overlap_scan':{'status':'pending_studio_scan'}})
            for mode in ('replace','add'):
                with patch.object(S,'check_base'):
                    with self.assertRaisesRegex(ValueError,'Studio scan'): M.compose(p,candidate,p/mode,mode)
                self.assertFalse((p/mode).exists())

    def test_bound_scan_gate_rejects_tampered_file_and_wrong_index(self):
        with tempfile.TemporaryDirectory(prefix='item53-test-') as d:
            p=Path(d);(p/'train.jsonl').write_text('tampered\n')
            licenses=Path(S.__file__).with_name('manifests')/'item53-source-licenses.json'
            m={'overlap_scan':{'status':'passed'},'converter_sha256':S.sha(Path(S.__file__)),
               'license_manifest_sha256':S.sha(licenses),'a3_input_hashes':M.A3_HASHES,
               'a3_revision':M.A3_REVISION,'upstream_revision':S.REVISION,
               'files':{'train.jsonl':{'sha256':'not-the-real-hash'}}}
            S.write_json(p/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'hash mismatch'): M.gate(p)
            m['files']={};S.write_json(p/'overlap-scan-report.json',m['overlap_scan'])
            (p/'scan-candidates.jsonl').write_text('')
            S.write_json(p/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'full-source scan'): M.gate(p)

    def test_compositions_copy_calibration_and_additive_prefix_exactly(self):
        # Gate mocked here only to test file-writing behavior; never a scan claim.
        with tempfile.TemporaryDirectory(prefix='item53-test-') as d:
            p=Path(d);base=p/'base';candidate=p/'candidate';base.mkdir();candidate.mkdir()
            rows=base_rows(10595)
            calib=[{'id':'calib','family_id':'calib','state':'calibration state','questions':{str(i):{'type':'noul'} for i in range(512)}}]
            S.write_rows(base/'train.jsonl',rows);S.write_rows(base/'calib.jsonl',calib)
            incoming=[S.convert(raw(i),'newfamily') for i in range(1000)]
            held=[S.convert(raw(i,'3'),'heldfamily') for i in range(300)]
            S.write_rows(candidate/'diagnostic.jsonl',held);S.write_json(candidate/'manifest.json',{})
            S.write_json(candidate/'overlap-scan-report.json',{})
            for mode in ('replace','add'):
                with patch.object(S,'check_base'),patch.object(M,'gate',return_value=({}, {'train':incoming,'diagnostic':held})):
                    report=M.compose(base,candidate,p/mode,mode)
                self.assertEqual((p/mode/'calib.jsonl').read_bytes(),(base/'calib.jsonl').read_bytes())
                self.assertEqual(report['slots']['questions_after'],21190+(1000 if mode=='add' else 0))
            self.assertTrue((p/'add/train.jsonl').read_bytes().startswith((base/'train.jsonl').read_bytes()))

    def test_license_and_ancestry_rejection(self):
        from item33_skill_data import validate_source
        p=json.loads(Path(S.__file__).with_name('manifests').joinpath('item53-source-licenses.json').read_text())
        source=p['sources'][S.SOURCE];validate_source(source,p['source_exclusions'])
        self.assertEqual(p['converter_sha256'],S.sha(Path(S.__file__)))
        for ancestor in ('ANLI','SNLI','MNLI','CLadder','FEVER','BBH'):
            bad=copy.deepcopy(source);bad['ancestry']=[ancestor]
            with self.assertRaisesRegex(ValueError,'Protected'): validate_source(bad,p['source_exclusions'])


if __name__ == '__main__': unittest.main()
