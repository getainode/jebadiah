"""Local tests for rung 3 source policy, executable labels and family isolation."""
import copy
import json
from pathlib import Path
import pickle
import re
from collections import Counter
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import item33_skill_data as B
import item33_full_suite_scan as S


class ToolDecisionTests(unittest.TestCase):
    def setUp(self):
        self.world = {'request': {'operation': 'read', 'arguments': {'record_id': 'R1'}},
                      'tools': [{'operation': 'read', 'required_arguments': ['record_id']}],
                      'context': {'complete': False, 'age_minutes': 30, 'record_id': 'R1'},
                      'policy': {'allowed_operations': ['read', 'update'], 'max_age_minutes': 5}}

    def test_tool_required_for_stale_context(self):
        self.world['context']['complete'] = True
        self.assertEqual(B.tool_oracle(self.world), 'call_tool')

    def test_fresh_context_avoids_call_even_without_tool(self):
        self.world['context'].update(complete=True, age_minutes=5)
        self.world['tools'] = []
        self.assertEqual(B.tool_oracle(self.world), 'answer_context')

    def test_missing_argument_requires_clarification(self):
        self.world['request']['arguments'] = {}
        self.assertEqual(B.tool_oracle(self.world), 'ask_user')

    def test_policy_denial_precedes_capability_and_arguments(self):
        self.world['policy']['allowed_operations'] = []
        self.world['tools'] = []
        self.world['request']['arguments'] = {}
        self.assertEqual(B.tool_oracle(self.world), 'decline')

    def test_wrong_capability_is_unavailable(self):
        self.world['tools'][0]['operation'] = 'update'
        self.assertEqual(B.tool_oracle(self.world), 'explain_unavailable')

    def test_unrelated_fresh_context_cannot_answer(self):
        self.world['context'].update(complete=True, age_minutes=0, record_id='OTHER')
        self.assertEqual(B.tool_oracle(self.world), 'call_tool')

    def test_generated_action_and_proposal_labels_are_executable(self):
        rows = B.tool_questions()
        self.assertEqual(len(rows), 1500)
        self.assertEqual(Counter(r['label']['decision'] for r in rows
                                 if len(r['questions']['decision'].get('criteria', {})) == 5),
                         Counter({x: 100 for x in B.ACTIONS}))
        for row in rows:
            if len(row['questions']['decision'].get('criteria', {})) == 5:
                self.assertEqual(row['label']['decision'], B.tool_oracle(row['state']))


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.world = {'evidence': [{'entity': 'A', 'attribute': 'status', 'value': 'queued'}],
                      'response_claims': [{'entity': 'A', 'attribute': 'status', 'value': 'queued', 'negated': False}]}

    def test_entity_confusion_is_not_supported(self):
        self.world['response_claims'][0]['entity'] = 'B'
        self.assertEqual(B.evidence_oracle(self.world), 0)

    def test_absence_never_supports_negation(self):
        self.world['response_claims'][0].update(attribute='date', value='Tuesday', negated=True)
        self.assertEqual(B.evidence_oracle(self.world), 0)

    def test_explicit_negative_claim_uses_known_fact(self):
        self.world['response_claims'][0].update(value='complete', negated=True)
        self.assertEqual(B.evidence_oracle(self.world), 1)

    def test_changed_value_and_negated_true_fact_fail(self):
        self.world['response_claims'][0]['value'] = 'complete'
        self.assertEqual(B.evidence_oracle(self.world), 0)
        self.world['response_claims'][0].update(value='queued', negated=True)
        self.assertEqual(B.evidence_oracle(self.world), 0)

    def test_natural_response_and_type_labels_agree(self):
        rows = B.evidence_questions()
        self.assertEqual(len(rows), 1500)
        families = {}
        for r in rows:
            self.assertIsInstance(r['state']['assistant_response'], str)
            self.assertNotIn('response_claims', r['state'])
            key = B.canonical(r['state'])
            families.setdefault(key, {})[r['questions']['decision']['type']] = r['label']['decision']
        for labels in families.values():
            self.assertEqual(labels['noul'], labels['score'] == 2)
            self.assertEqual(labels['choice'], ('none', 'partial', 'full')[labels['score']])

    def test_labels_match_rendered_prose_not_only_structured_oracle(self):
        pattern = r'The (.+?) of (.+?) is (not )?(.+?)\.'
        for row in B.evidence_questions():
            if row['questions']['decision']['type'] != 'score':
                continue
            facts = {}
            for sentence in row['state']['source_evidence']:
                attribute, entity, negation, value = re.fullmatch(pattern, sentence).groups()
                self.assertIsNone(negation)
                facts[entity + ':' + attribute] = value
            supported = 0
            for match in re.finditer(pattern, row['state']['assistant_response']):
                attribute, entity, negation, value = match.groups()
                observed = facts.get(entity + ':' + attribute)
                supported += observed is not None and ((observed != value) if negation else (observed == value))
            self.assertEqual(row['label']['decision'], supported)


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((Path(__file__).parent / 'manifests/item33-source-licenses.json').read_text())
        self.info = copy.deepcopy(self.manifest['sources'][B.HUMAN])

    def test_approved_sources_pass(self):
        for info in self.manifest['sources'].values():
            B.validate_source(info, self.manifest['source_exclusions'])

    def test_every_protected_name_and_parent_is_rejected_in_ancestry(self):
        for term in self.manifest['source_exclusions']['source_substrings']:
            with self.subTest(term=term):
                self.info['ancestry'] = ['mirror/derived-from/' + term]
                with self.assertRaisesRegex(ValueError, 'Protected source ancestry'):
                    B.validate_source(self.info, self.manifest['source_exclusions'])

    def test_missing_license_or_generator_terms_rejected(self):
        self.info['commercial_use_ok'] = False
        with self.assertRaisesRegex(ValueError, 'Commercial'):
            B.validate_source(self.info, self.manifest['source_exclusions'])
        generator = copy.deepcopy(self.manifest['sources'][B.TOOLS])
        del generator['generator_license']
        with self.assertRaisesRegex(ValueError, 'generator license/hash'):
            B.validate_source(generator, self.manifest['source_exclusions'])


class FamilySplitTests(unittest.TestCase):
    def test_generated_counts_and_siblings_are_disjoint(self):
        splits = {'train': [], 'calib': []}
        for skill, rows in [('tool_timing', B.tool_questions()), ('grounding', B.evidence_questions())]:
            chosen = B.split_select(rows, skill)
            self.assertEqual(len(chosen['train']), 1350)
            self.assertEqual(len(chosen['calib']), 150)
            if skill == 'tool_timing':
                actions = {r['label']['decision'] for r in chosen['calib']
                           if len(r['questions']['decision'].get('criteria', {})) == 5}
                self.assertEqual(actions, set(B.ACTIONS))
            for name in splits:
                splits[name].extend(chosen[name])
        B.audit_splits(splits)
        self.assertEqual(len({r['family_id'] for r in splits['calib']}), 10)

    def test_duplicate_ids_and_cross_split_states_rejected(self):
        row = B.tool_questions()[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate record ID'):
            B.audit_splits({'train': [row, row], 'calib': []})
        sibling = {**row, 'id': 'sibling', 'family_id': 'different'}
        with self.assertRaisesRegex(ValueError, 'leakage'):
            B.audit_splits({'train': [row], 'calib': [sibling]})

    def test_deterministic_family_selection(self):
        rows = B.evidence_questions()
        self.assertEqual(B.split_select(rows, 'grounding'), B.split_select(list(reversed(rows)), 'grounding'))


class ScannerTests(unittest.TestCase):
    def test_one_unselected_record_drops_whole_source_and_siblings(self):
        # Original test phrases, not benchmark records or reconstructions.
        phrase = 'violet lantern quartz meadow copper orbit spruce lichen beacon velvet canyon pebble harvest'
        idx = ({}, {h: 'fixture:protected-id' for h in S.shingle(phrase.split(), 13)}, [], {})
        with tempfile.TemporaryDirectory(prefix='item33-test-') as name:
            root = Path(name)
            with (root / 'protected.pkl').open('wb') as f:
                pickle.dump(idx, f)
            rows = [
                {'id': 'selected-clean', 'family': 'upstream-A', 'state': 'Unrelated clean material', 'questions': {}},
                {'id': 'unselected-hit', 'family': 'upstream-A', 'state': 'Distinct material',
                 'questions': {'q': {'criteria': {'x': phrase}, 'instructions': 'Check fixture.'}}},
                {'id': 'other-source', 'family': 'upstream-B', 'state': 'Another independent fixture', 'questions': {}},
            ]
            B.write_rows(root / 'candidates.jsonl', rows)
            S.scan(root)
            clean = [json.loads(line) for line in (root / 'clean.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in clean], ['other-source'])
            hits = json.loads((root / 'contamination-hits.json').read_text())
            self.assertEqual(hits[0]['method'], '13_word_span')
            self.assertNotIn(phrase, (root / 'contamination-hits.json').read_text())


if __name__ == '__main__':
    unittest.main()
