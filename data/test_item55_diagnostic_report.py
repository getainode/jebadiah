# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from item55_diagnostic_report import summarize, save_completed_pass


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.records=[{'id':'math:1','family_id':'program:1','provenance':{'module':'mixed'},
                       'questions':{'decision':{'type':'choice','criteria':{'1/3':None,'2/3':None}}},'label':{'decision':'1/3'}},
                      {'id':'math:2','family_id':'program:2','provenance':{'module':'mixed'},
                       'questions':{'decision':{'type':'noul'}},'label':{'decision':False}}]
        self.rows=[{'id':'math:1','qid':'decision','repeat':0,'pick':'1/3','label':'1/3'},
                   {'id':'math:2','qid':'decision','repeat':0,'pick':'false','label':'false'}]

    def test_exact_choice_and_boolean_wire_report(self):
        out=summarize(self.rows,self.records)
        self.assertEqual(out['choice_macro_module_accuracy'],1)
        self.assertEqual(out['accuracy']['overall']['correct'],2)
        self.assertEqual(out['verification']['tn'],1)

    def test_incomplete_duplicate_wrong_gold_and_invalid_pick_rejected(self):
        variants=[self.rows[:1],self.rows+[self.rows[0]],
                  [{**self.rows[0],'label':'wrong'},self.rows[1]],
                  [{**self.rows[0],'pick':'wrong'},self.rows[1]]]
        for rows in variants:
            with self.assertRaises(ValueError):summarize(rows,self.records)

    def test_each_completed_pass_saves_only_timing(self):
        with tempfile.TemporaryDirectory(prefix='item55-report-test-') as td:
            path=Path(td)/'item55-report.json';report={}
            extra={'timing':{'questions_scored':2,'truncated_prompts':0},'questions':{('tuple','key'):object()}}
            save_completed_pass(path,report,'a3',self.rows,extra,self.records)
            self.assertEqual(json.loads(path.read_text()),report)
            self.assertNotIn('questions',report['a3'])
            save_completed_pass(path,report,'rung5',self.rows,extra,self.records)
            self.assertEqual(set(json.loads(path.read_text())),{'a3','rung5'})
            before=path.read_bytes()
            with self.assertRaises(ValueError):save_completed_pass(path,report,'a3',self.rows,extra,self.records)
            with self.assertRaises(ValueError):save_completed_pass(path,report,'bad',self.rows,{'timing':{'questions_scored':1,'truncated_prompts':0}},self.records)
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
