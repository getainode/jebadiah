"""CPU tests for N2 oracle, competition, source isolation and question-slot replacement."""
import copy
import json
from pathlib import Path
import tempfile
import os
import random
import unittest
from unittest.mock import patch
import item43_wide_choice as W
import item43_n2_mix as M


class WideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.splits = W.generate()

    def test_determinism_counts_and_all_labels(self):
        self.assertEqual(self.splits, W.generate()); W.audit(self.splits)
        self.assertEqual(W.counts(self.splits['train'])['per_size'], {'32':1440,'64':1440,'128':1440,'255':1440})
        self.assertEqual(W.counts(self.splits['diagnostic'])['per_size'], {'32':96,'64':96,'128':96,'255':96})
        for rows in self.splits.values():
            for r in rows:
                q=r['questions']['decision']; c=q['criteria']; target=r['state'].split('Request: ')[1].split('. Menu')[0]
                self.assertIn('none',c); self.assertEqual(W.oracle(target,c),r['label']['decision'])
                self.assertEqual(len(c),int(r['state'].split('Menu size: ')[1].strip('.')))
                # Full set of hard one-attribute neighbors survives at all sizes.
                a,b=target.split(); hard=[k for k in c if k!='none' and k!=target and (k.split()[0]==a or k.split()[1]==b)]
                self.assertGreaterEqual(len(hard),29)
                self.assertEqual(W.oracle(target,dict(reversed(list(c.items())))),r['label']['decision'])

    def test_family_and_label_validation_fail_closed(self):
        bad=copy.deepcopy(self.splits)
        bad['diagnostic'].append(bad['train'][0])
        with self.assertRaises(ValueError): W.audit(bad)
        bad=copy.deepcopy(self.splits); bad['train'][0]['label']['decision']='invalid'
        with self.assertRaises(ValueError): W.audit(bad)

    def test_replacement_preserves_nonchoice_and_mixed_row_state(self):
        rows=[]
        for i in range(21):
            rows.append({'id':f'a3:{i}','family_id':f'a3family:{i}','state':f'original {i}',
                         'questions': {'c':{'type':'choice','instructions':'Choose','criteria':{'x':None,'y':None}},
                                       'b':{'type':'noul','instructions':'True?'}},
                         'label':{'c':'x','b':True}, 'target':{'c':{'x':1,'y':0}}})
        before=copy.deepcopy(rows); result, receipt=M.replace_slots(rows,self.splits['train'])
        self.assertEqual(rows,before); self.assertEqual(receipt['replaced_slots'],2)
        self.assertEqual(sum(len(r['questions']) for r in result),42)
        for original in rows:
            retained=next(r for r in result if r['id']==original['id'])
            self.assertEqual(retained['state'],original['state']); self.assertEqual(retained['questions']['b'],original['questions']['b'])
            self.assertEqual(retained['label']['b'],True)
            if 'c' not in retained['questions']: self.assertNotIn('c',retained['target'])

    def test_pending_scan_blocks_composition(self):
        with tempfile.TemporaryDirectory(prefix='item43-test-') as d:
            root=Path(d); base=root/'base'; wide=root/'wide'; base.mkdir();wide.mkdir()
            for name in M.A3_HASHES: (base/name).write_text('')
            (wide/'manifest.json').write_text(json.dumps({'overlap_scan':{'status':'pending_studio_scan'}}))
            with patch.object(M,'A3_HASHES',{n:M.sha(base/n) for n in M.A3_HASHES}):
                with self.assertRaisesRegex(ValueError,'Studio scan'): M.compose(base,wide,root/'out')
            self.assertFalse((root/'out').exists())

    def test_license_manifest_and_benchmark_aliases(self):
        from item33_skill_data import validate_source, sha
        policy=json.loads(Path(__file__).with_name('manifests').joinpath('item43-source-licenses.json').read_text())
        source=policy['sources'][W.SOURCE]
        validate_source(source,policy['source_exclusions'])
        self.assertEqual(source['generator_sha256'],sha(Path(W.__file__)))
        self.assertEqual(source['external_inputs'],[]);self.assertIsNone(source['generator_model'])
        for name in ('CLINC150','BANKING77','POP909','HWU64'):
            bad=copy.deepcopy(source);bad['ancestry']=['conversion of '+name]
            with self.assertRaisesRegex(ValueError,'Protected source'): validate_source(bad,policy['source_exclusions'])

    def test_composition_writes_original_calibration_and_separate_holdout(self):
        # Tiny synthetic fixtures stand in for pinned files; no real scan claim.
        with tempfile.TemporaryDirectory(prefix='item43-compose-test-') as d:
            root=Path(d);base=root/'base';wide=root/'wide';base.mkdir();wide.mkdir()
            rows=[{'id':f'base:{i}','family_id':f'base:{i}','state':f'state {i}',
                   'questions':{'c':{'type':'choice','instructions':'Choose','criteria':{'x':None,'y':None}}},'label':{'c':'x'}} for i in range(20)]
            calibration=[{'id':'calib','family_id':'calib','state':'calibration state','questions':{'b':{'type':'noul','instructions':'True?'}},'label':{'b':True}}]
            W.write_rows(base/'train.jsonl',rows);W.write_rows(base/'calib.jsonl',calibration);W.write_json(base/'manifest.json',{})
            for split in self.splits: W.write_rows(wide/f'{split}.jsonl',self.splits[split])
            candidate_hash=W.digest(''.join(json.dumps({**r,'family':W.SOURCE},ensure_ascii=False,separators=(',',':'))+'\n' for rs in self.splits.values() for r in rs))
            W.write_json(wide/'manifest.json',{'generator_sha256':W.sha(Path(W.__file__)),'overlap_scan':{'status':'passed','candidates_sha256':candidate_hash,'scanned_records':6144,'protected_index_sha256':W.ITEM25_INDEX_SHA256,'removed_records':0,'source_level':True},'render_validation':{'truncated':0,'max_seq_length':2048},
                         'files':{f'{split}.jsonl':{'sha256':W.sha(wide/f'{split}.jsonl')} for split in self.splits}})
            with patch.object(M,'A3_HASHES',{n:W.sha(base/n) for n in M.A3_HASHES}):
                report=M.compose(base,wide,root/'out')
            self.assertEqual(report['questions'],20);self.assertEqual(report['slots']['replaced_slots'],2)
            self.assertEqual((base/'calib.jsonl').read_bytes(),(root/'out/calib.jsonl').read_bytes())
            self.assertEqual((wide/'diagnostic.jsonl').read_bytes(),(root/'out/wide-diagnostic.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError,'empty'): M.compose(base,wide,root/'out')

    @unittest.skipUnless(os.environ.get('ITEM43_TOKENIZER'), 'Pinned tokenizer path required for live collator check')
    def test_live_training_collator_full_menus_and_gold_mapping(self):
        from transformers import AutoTokenizer
        from jebadiah_prompt import Renderer
        from train_jebadiah import DecideCollator, DecideDataset
        tok=AutoTokenizer.from_pretrained(os.environ['ITEM43_TOKENIZER'],local_files_only=True)
        rows=[next(r for r in self.splits['train'] if len(r['questions']['decision']['criteria'])==n and
                   (r['label']['decision']=='none')==none) for n,none in ((32,False),(255,True))]
        renderer=Renderer(tok,max_tokens=1984)
        collator=DecideCollator(tok,renderer,shuffle_choice=True,pad_to_multiple_of=64)
        saved=random.getstate()
        try:
            random.seed(17);expected_rng=random.Random();expected_rng.setstate(random.getstate())
            gold_positions=[]
            for row in rows:
                order=list(row['questions']['decision']['criteria']);expected_rng.shuffle(order)
                gold_positions.append(order.index(row['label']['decision']))
            batch=collator(list(DecideDataset(rows)))
        finally: random.setstate(saved)
        self.assertEqual(batch['label_idx'].tolist(),gold_positions)
        self.assertEqual(tuple(batch['cand_ids'].shape),(2,255))
        self.assertTrue((batch['cand_ids'][0,32:]==-1).all())
        self.assertEqual(batch['target'].argmax(dim=-1).tolist(),gold_positions)
        self.assertEqual(batch['target'].sum(dim=-1).tolist(),[1.0,1.0])
        self.assertLessEqual(batch['input_ids'].shape[1],2048);self.assertEqual(collator.truncated,0)

    def test_255_logits_and_padding_cpu(self):
        import torch
        from types import SimpleNamespace
        from jebadiah_model import option_logits
        class Core:
            def __init__(self):
                self.lm_head=torch.nn.Linear(2,300,bias=False)
                with torch.no_grad():
                    self.lm_head.weight[:,0]=torch.arange(300);self.lm_head.weight[:,1]=0
            def model(self,**kwargs): return SimpleNamespace(last_hidden_state=torch.ones(2,2,2))
        core=Core()
        model=SimpleNamespace(config=SimpleNamespace(model_type='mock'),get_base_model=lambda:core)
        # core_of understands get_base_model and unwraps the base model.
        cand=torch.full((2,255),-1,dtype=torch.long);cand[0]=torch.arange(255);cand[1,:32]=torch.arange(32)
        with patch('jebadiah_model.core_of',return_value=core):
            logits=option_logits(model,torch.ones(2,2,dtype=torch.long),torch.ones(2,2,dtype=torch.long),cand)
        self.assertEqual(tuple(logits.shape),(2,255));self.assertEqual(int(logits[0].argmax()),254)
        self.assertTrue(torch.isneginf(logits[1,32:]).all())
        probs=torch.softmax(logits,dim=-1)
        self.assertAlmostEqual(float(probs[0].sum().detach()),1.0,places=6);self.assertEqual(float(probs[1,32:].sum().detach()),0)
        from jebadiah_model import Scorer
        from jebadiah_prompt import Rendered, answer_from_probs
        class Encoding(SimpleNamespace):
            def to(self, device): return self
        scorer=Scorer.__new__(Scorer)
        scorer.tok=lambda *args,**kwargs: Encoding(input_ids=torch.ones(2,2,dtype=torch.long),attention_mask=torch.ones(2,2,dtype=torch.long))
        scorer.model=model;scorer.device='cpu';scorer.temperatures={'choice':2.179078721266035}
        keys=[f'k{i}' for i in range(255)]
        rendered=Rendered('prompt',keys,keys,list(range(255)),False,'extended')
        short=Rendered('prompt',keys[:32],keys[:32],list(range(32)),False)
        with patch('jebadiah_model.core_of',return_value=core):
            scored=scorer.score_rendered([(rendered,'choice'),(short,'choice')])
        self.assertEqual([len(p) for p in scored],[255,32])
        self.assertEqual(answer_from_probs({'type':'choice'},keys,scored[0])['choice'],'k254')
        self.assertAlmostEqual(sum(scored[0]),1.0,places=6)
        # A 255th-position gold must participate in the candidate CE, including its gradient.
        logits=logits.detach().requires_grad_();loss=torch.nn.functional.cross_entropy(logits,torch.tensor([254,31]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertNotEqual(float(logits.grad[0,254]),0)


if __name__=='__main__': unittest.main()
