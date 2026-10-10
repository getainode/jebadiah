# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import tempfile
import unittest
import item54_massive as M

spec=importlib.util.spec_from_file_location('item54_report',Path(__file__).resolve().parents[1]/'scripts/item54_diagnostic_report.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)

class ReportTests(unittest.TestCase):
    def records(self):
        intents={'alarm_set':'alarm',**{f'intent_{i}':'other' for i in range(16)}}
        return [M.convert({'id':str(i),'locale':'en-US','partition':'train','intent':'alarm_set',
                           'scenario':'alarm','utt':f'fixture {i}'},kind,intents)
                for i,kind in enumerate(('choice','noul','noul'))]

    def rows(self,records):
        return [{'id':r['id'],'qid':'decision','repeat':0,
                 'label':str(r['label']['decision']).lower() if r['questions']['decision']['type']=='noul' else r['label']['decision'],
                 'pick':str(r['label']['decision']).lower() if r['questions']['decision']['type']=='noul' else r['label']['decision']}
                for r in records]

    def test_timing_only_wrapper_persists_each_completed_pass(self):
        records=self.records();rows=self.rows(records);report={}
        extra={'timing':{'questions_scored':3,'truncated_prompts':0},'question_map':{('tuple','key'):object()}}
        with tempfile.TemporaryDirectory(prefix='item54-test-') as folder:
            path=Path(folder)/'item54-report.json'
            R.save_completed_pass(path,report,'a3',rows,extra,records)
            self.assertTrue(path.exists());self.assertEqual(report['a3']['choice_macro_intent_accuracy'],1.0)
            R.save_completed_pass(path,report,'rung4',rows,extra,records)
            self.assertEqual(set(report),{'a3','rung4'})
            self.assertNotIn('question_map',path.read_text())
            with self.assertRaises(ValueError):R.save_completed_pass(path,report,'rung4',rows,extra,records)

    def test_missing_duplicate_wrong_label_and_truncation_fail(self):
        records=self.records();rows=self.rows(records)
        for bad in (rows[:-1],rows+[rows[0]], [{**r,'label':'wrong'} for r in rows]):
            with self.assertRaises(ValueError):R.summarize(bad,records)
        with tempfile.TemporaryDirectory(prefix='item54-test-') as folder:
            path=Path(folder)/'item54-report.json'
            with self.assertRaises(ValueError):
                R.save_completed_pass(path,{},'rung4',rows,{'timing':{'questions_scored':3,'truncated_prompts':1}},records)
            self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
