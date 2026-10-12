"""CPU checks for source annotation preservation and fail-closed validation."""
import unittest
from convert_data import tasksource_jev_row
from lint_data import lint_record
from t2b_build import licences, key


def source(**changes):
    return {**{'state': 'A bird flies over a tree.', 'kind': 'choice', 'id': 'source:train:7',
               'options': ['entailment', 'contradiction', 'neutral'], 'target': [0.1, 0.2, 0.7],
               'question': 'Which relationship holds?', 'source': 'test-nli', 'variant': 'direct',
               'split': 'train', 'group_id': 'source:train:7', 'license': 'mit',
               'license_use': 'commercial'}, **changes}


class TasksourceConversion(unittest.TestCase):
    def test_choice_preserves_distribution_without_payload_labels(self):
        r = tasksource_jev_row(source(), 'pinned')
        self.assertEqual(r['label']['decision'], '2')
        self.assertEqual(r['questions']['decision']['criteria']['2'], 'neutral')
        self.assertEqual(r['target']['decision'], {'0': 0.1, '1': 0.2, '2': 0.7})
        self.assertEqual(r['license_use'], 'commercial')
        problems = []; lint_record(r, 'test', problems, 20)
        self.assertEqual(problems, [])

    def test_noul_probability_is_not_an_option_index(self):
        r = tasksource_jev_row(source(kind='noul', options=[], target=[0.2]), 'pinned')
        self.assertIs(r['label']['decision'], False)
        self.assertEqual(r['target']['decision'], {'true': 0.2, 'false': 0.8})

    def test_score_keeps_ordered_rubric(self):
        r = tasksource_jev_row(source(kind='score', options=['poor', 'adequate', 'good'],
                                      target=[0, 0.25, 0.75]), 'pinned')
        self.assertEqual(r['label']['decision'], 2)
        self.assertEqual(r['questions']['decision']['criteria'], ['poor', 'adequate', 'good'])

    def test_bad_annotations_and_noncommercial_rows_fail(self):
        for changes in [{'license_use': 'non-commercial'}, {'split': 'test'},
                        {'options': ['same', 'same'], 'target': [1, 0]},
                        {'target': [float('nan'), 0, 1]}, {'target': [0.5, 0.5, 0]},
                        {'target': [0.1, 0.1, 0.1]}, {'options': list(map(str, range(21)))},
                        {'state': ''}, {'kind': 'chat'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tasksource_jev_row(source(**changes), 'pinned')

    def test_mixed_licence_cannot_hide_restriction(self):
        self.assertEqual(licences('apache-2.0, CC BY-SA 4.0 (DPI)'), ['apache-2.0', 'cc-by-sa-4.0'])
        for text in ['mit, Custom (DPI)', 'cc-by-nc-4.0', 'research-only', 'other', 'unspecified']:
            self.assertEqual(licences(text), [])

    def test_dedup_identity_ignores_source_and_punctuation(self):
        self.assertEqual(key({'state': 'Hello, WORLD!'}), key({'state': 'hello world'}))


if __name__ == '__main__': unittest.main()
