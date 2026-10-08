"""Source-boundary checks for the item 32 ablation builder."""
import copy
import tempfile
import unittest
from pathlib import Path
from item32_ablation_mix import check_hash, root_source, source_check


class SourcePolicyTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            'source_exclusions': {'source_substrings': ['mnli', 'summeval', 'banking77']},
            'source_manifests': {
                'ratings/a': {'license': 'cc-by-4.0', 'allowed_reason': 'approved',
                              'source_id': 'owner/ratings', 'family': 'ratings'},
                'ratings/b': {'license': 'cc-by-4.0', 'allowed_reason': 'approved',
                              'source_id': 'owner/ratings', 'family': 'ratings'},
            },
        }

    def test_rubric_variants_share_upstream_cap(self):
        self.assertEqual(root_source('ratings/a', self.manifest),
                         root_source('ratings/b', self.manifest))

    def test_indirect_benchmark_parent_fails_closed(self):
        m = copy.deepcopy(self.manifest)
        m['source_manifests']['ratings/a']['upstream_originals'] = ['MultiNLI/mnli']
        with self.assertRaisesRegex(ValueError, 'Blocked source ancestry'):
            source_check('ratings/a', m)

    def test_missing_license_approval_fails_closed(self):
        for key in ('license', 'allowed_reason'):
            m = copy.deepcopy(self.manifest)
            del m['source_manifests']['ratings/a'][key]
            with self.assertRaises(ValueError):
                source_check('ratings/a', m)
        with self.assertRaises(ValueError):
            source_check('unknown', self.manifest)

    def test_changed_private_input_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix='item32-test-') as d:
            p = Path(d) / 'train.jsonl'
            p.write_text('changed input\n')
            with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
                check_hash(p, '0' * 64)


if __name__ == '__main__':
    unittest.main()
