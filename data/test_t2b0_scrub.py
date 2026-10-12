import unittest
from t2b0_scrub import rejected_family, FAMILIES


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


if __name__ == '__main__':
    unittest.main()
