# SPDX-License-Identifier: Apache-2.0
"""Verify additive composition preserves mixed-question bytes and rejects collisions."""
import hashlib
import json
import unittest
from item51_rung2b_mix import additive_bytes


class AdditiveMixTest(unittest.TestCase):
    def fixtures(self):
        base_raw = b'{ "id": "a3", "questions": {"first": {}, "second": {}} }\n'
        base = [(json.loads(base_raw), base_raw)]
        scanned = []
        mapping = []
        for i in range(1000):
            row = {'id': 'causal-' + str(i), 'source': 'corr2cause', 'area': 'knowledge',
                   'questions': {'decision': {'type': 'choice' if i % 2 else 'noul'}}}
            raw = (json.dumps(row) + '\n').encode()
            scanned.append((row, raw))
            mapping.append({'replacement_id': row['id'], 'type': row['questions']['decision']['type']})
        return base, scanned, mapping

    def test_preserves_bytes_and_selects_exact_scanned_rows(self):
        base, scanned, mapping = self.fixtures()
        # An unselected scanned row must never enter the output.
        extra = ({'id': 'unused'}, b'{"id":"unused"}\n')
        result, selected, hashes = additive_bytes(base, scanned + [extra], mapping)
        self.assertEqual(result, base[0][1] + b''.join(raw for row, raw in scanned))
        self.assertEqual(len(selected), 1000)
        self.assertEqual(hashes, {row['id']: hashlib.sha256(raw).hexdigest() for row, raw in scanned})

    def test_rejects_missing_duplicate_and_a3_collision(self):
        base, scanned, mapping = self.fixtures()
        with self.assertRaises(ValueError):
            additive_bytes(base, scanned[:-1], mapping)
        with self.assertRaises(ValueError):
            additive_bytes(base, scanned + [scanned[0]], mapping)
        with self.assertRaises(ValueError):
            additive_bytes([(scanned[0][0], base[0][1])], scanned, mapping)
        mapping[1] = mapping[0]
        with self.assertRaises(ValueError):
            additive_bytes(base, scanned, mapping)


if __name__ == '__main__':
    unittest.main()
