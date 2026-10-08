import hashlib
import importlib.util
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).parent))
from score import load_proxy
from bootstrap import group_strata, resample
import random
import unittest

spec = importlib.util.spec_from_file_location('proxy_sample', Path(__file__).with_name('sample.py'))
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)


def row(b, g, chunk=0, track='all', domain=''):
    return dict(_evaluation=dict(catalog_id=b, group_id=g, run_id=f'{b}:{g}:{chunk}', track=track), metadata=dict(domain=domain))


class SamplingTest(unittest.TestCase):
    def test_whole_groups_per_benchmark_stratum_and_order_invariance(self):
        rows = [row(b, str(g), c, track=t, domain=d) for b in (1, 2) for t, d in (('a', 'x'), ('b', 'y')) for g in range(11 if t == 'a' else 30, 22 if t == 'a' else 40) for c in (0, 1)]
        a = sample.select(rows, {1, 2})
        self.assertEqual(a, sample.select(reversed(rows), {1, 2}))
        self.assertEqual(len(a[0]), 12)  # ceil(11/10) + ceil(10/10), two benchmarks, two chunks
        self.assertEqual(len(a[1]), 6)
        for b, g in a[1]:
            self.assertTrue({f'{b}:{g}:0', f'{b}:{g}:1'} <= set(a[0]))
        rank = sorted([str(g) for g in range(11, 22)], key=lambda g: hashlib.sha256(f'20261008:1:{g}'.encode()).hexdigest())
        self.assertEqual({g for b, g in a[1] if b == 1 and 11 <= int(g) < 22}, set(rank[:2]))

    def test_excluded_benchmark_missing_duplicate_and_hash(self):
        self.assertEqual(sample.select([row(1, 'a'), row(9, 'b')], {1})[0], ['1:a:0'])
        with self.assertRaises(ValueError):
            sample.select([row(1, 'a')], {1, 2})
        with self.assertRaises(ValueError):
            sample.select([row(1, 'a'), row(1, 'a')], {1})
        with tempfile.TemporaryDirectory(prefix='item31-test-') as d:
            p = Path(d) / 'ids'
            p.write_bytes(sample.id_bytes(['a', 'b']))
            self.assertEqual(sample.read_ids(p, hashlib.sha256(p.read_bytes()).hexdigest()), {'a', 'b'})
            with self.assertRaises(ValueError):
                sample.read_ids(p, '0' * 64)

    def test_cross_track_group_is_never_split(self):
        rows = [row(1, 'a', 0, 'x'), row(1, 'a', 1, 'y')]
        self.assertEqual(sample.select(rows, {1})[0], ['1:a:0', '1:a:1'])

    def test_proxy_rejects_partial_groups_and_missing_ids(self):
        class Suite:
            edition = {'id': '0.3'}
            def rows(self, apply_exclusions=False):
                return iter([row(1, 'a', 0), row(1, 'a', 1), row(2, 'b')])
        with self.assertRaises(ValueError):
            load_proxy(Suite(), {'1:a:0'})
        with self.assertRaises(ValueError):
            load_proxy(Suite(), {'missing'})
        proxy = load_proxy(Suite(), {'1:a:0', '1:a:1'})
        self.assertEqual(len(proxy.selected), 2)

    def test_bootstrap_draw_preserves_case_chunks_and_native_clusters(self):
        rows = [row(22, 'song', c) for c in (0, 1)]
        for r in rows:
            r['metadata']['song_id'] = 'song'
        sampled = resample(group_strata(rows), random.Random(20261008))
        self.assertEqual(len(sampled), 2)
        self.assertEqual(len({r['_evaluation']['group_id'] for r in sampled}), 1)
        self.assertEqual({r['_evaluation']['run_id'] for r in sampled}, {'22:song:0', '22:song:1'})
        self.assertEqual({r['metadata']['song_id'] for r in sampled}, {'song'})


if __name__ == '__main__':
    unittest.main()
