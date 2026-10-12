import unittest
import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from t2b0_scrub import rejected_family, FAMILIES, training_manifest_entry


class AncestryTest(unittest.TestCase):
    def check_family(self, identity, expected, **meta):
        manifest = {'source_manifests': {'child': {'source_id': identity, **meta}}}
        row = {'subset': 'child', 'state': 'HelpSteer2 math_qa', 'questions': {}}
        self.assertEqual(rejected_family(row, manifest), expected)

    def test_all_families_and_rating_views(self):
        for name in FAMILIES:
            self.check_family(name, name)
        self.check_family('helpsteer2-helpfulness', 'nvidia/HelpSteer2')
        self.check_family('codah/codah', 'jaredfern/codah')
        self.check_family('amazon_counterfactual/en', 'mteb/amazon_counterfactual')

    def test_recorded_parent_and_alias(self):
        self.check_family('other/derivative', 'tasksource/math_qa', upstream_originals=['allenai/math_qa'])
        self.check_family('other/derivative', 'propsegment/nli', upstream={'originals': ['sihaochen/propsegment']})
        self.check_family('math-QA', 'tasksource/math_qa')

    def test_siblings_and_content_do_not_trigger(self):
        for identity in ('nvidia/HelpSteer', 'nvidia/HelpSteer3', 'other/arctic', 'other/codah_extra'):
            self.check_family(identity, None)

    def test_ambiguous_parent_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_family('tasksource/arct', None, upstream_originals=['tasksource/equate'])

    def test_original_pipeline_manifest_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', 'show',
            '86a8203d792463536eb009a4d8da6c8681ffdd99:train/v21_pipeline.py'], cwd=root)
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                    and [v.id for v in n.target.elts] == ['split', 'filename'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text(json.dumps({'set': 'fixture', 'questions': {'one': {'type': 'choice'}, 'two': {'type': 'score'}}}) + '\n')
            entry = training_manifest_entry(path)
            # Execute the unchanged original pipeline's actual validation block.
            scope = {'Path': Path, 'args': SimpleNamespace(train_file='t2b0/train.jsonl',
                      calib_file='t2b0/calib.jsonl', manifest_file='t2b0/manifest.json'),
                     'manifest': {'files': {'train.jsonl': entry, 'calib.jsonl': entry}},
                     'data_report': {split: {'sha256': entry['sha256'], 'rows': 1, 'questions': 2}
                                     for split in ('train', 'calib')}}
            exec(compile(ast.Module(body=[loop], type_ignores=[]), '<original-manifest-contract>', 'exec'), scope)
            del entry['rows']
            with self.assertRaises(KeyError):
                exec(compile(ast.Module(body=[loop], type_ignores=[]), '<original-manifest-contract>', 'exec'), scope)


if __name__ == '__main__':
    unittest.main()
