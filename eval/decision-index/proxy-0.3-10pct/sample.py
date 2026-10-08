#!/usr/bin/env python3
"""Freeze IDs only from the verified, exclusion-applied Decision Index 0.3 suite."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SEED = 20261008


def select(rows, contributing, seed=SEED):
    groups = collections.defaultdict(list)
    seen = set()
    for row in rows:
        e = row['_evaluation']
        if e['catalog_id'] not in contributing:
            continue
        rid = e['run_id']
        if rid in seen:
            raise ValueError(f'duplicate run ID: {rid}')
        seen.add(rid)
        domain = row.get('metadata', {}).get('domain', e.get('domain', ''))
        groups[(e['catalog_id'], e['group_id'])].append((rid, str(domain), str(e.get('track', ''))))
    strata = collections.defaultdict(list)
    for key, members in groups.items():
        # A cross-track group stays intact in a combined stratum.
        signature = tuple(sorted({(d, t) for _, d, t in members}))
        strata[(key[0], signature)].append(key)
    chosen, counts = [], collections.defaultdict(lambda: collections.Counter())
    for (benchmark, _), keys in sorted(strata.items()):
        ordered = sorted(keys, key=lambda k: (hashlib.sha256(f'{seed}:{k[0]}:{k[1]}'.encode()).hexdigest(), str(k[1])))
        take = (len(keys) + 9) // 10
        chosen.extend(ordered[:take])
        counts[benchmark].update(groups=len(keys), selected_groups=take, strata=1)
    ids = sorted(rid for key in chosen for rid, _, _ in groups[key])
    for b in contributing:
        if b not in counts:
            raise ValueError(f'contributing benchmark {b} absent')
        counts[b]['rows'] = sum(len(v) for (n, _), v in groups.items() if n == b)
        counts[b]['selected_rows'] = sum(len(groups[k]) for k in chosen if k[0] == b)
    return ids, sorted(chosen), {str(b): dict(c) for b, c in sorted(counts.items())}


def id_bytes(ids):
    return (''.join(rid + '\n' for rid in ids)).encode('utf-8')


def read_ids(path, expected_hash=None):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_hash and digest != expected_hash:
        raise ValueError('ID list SHA256 mismatch')
    ids = data.decode('utf-8').splitlines()
    if len(ids) != len(set(ids)) or any(not rid for rid in ids):
        raise ValueError('empty or duplicate ID')
    return set(ids)


def materialize(suite, ids, output):
    found = set()
    with Path(output).open('w') as f:
        for row in suite.rows(apply_exclusions=True):
            rid = row['_evaluation']['run_id']
            if rid in ids:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                found.add(rid)
    if found != ids:
        Path(output).unlink()
        raise ValueError(f'{len(ids - found)} manifest IDs absent from suite')
    return len(found)


def main():
    from decision_index.suite.io import Suite
    from decision_index.scoring.index02 import spec
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--materialize', type=Path, help='private scratch rows, never commit')
    p.add_argument('--ids', type=Path)
    p.add_argument('--sha256')
    p.add_argument('--suite-revision')
    args = p.parse_args()
    suite = Suite(args.suite, '0.3')
    verification = suite.verify()
    if args.materialize:
        if not args.ids or not args.sha256:
            p.error('materialize requires --ids and --sha256')
        print(materialize(suite, read_ids(args.ids, args.sha256), args.materialize))
        return
    if not args.out:
        p.error('freeze requires --out')
    contributing = {n for area in spec('0.3')['areas'] for n in area['benchmarks']}
    ids, groups, counts = select(suite.rows(apply_exclusions=True), contributing)
    data = id_bytes(ids)
    digest = hashlib.sha256(data).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'run-ids.txt').write_bytes(data)
    (args.out / 'run-ids.sha256').write_text(digest + '  run-ids.txt\n')
    manifest = dict(edition='0.3', seed=SEED, fraction='1/10',
                    ordering='SHA256(UTF8(seed:catalog_id:group_id)); group_id tie break',
                    allocation='ceil(group_count/10) per benchmark/domain/track stratum',
                    suite_dataset='jbrashear/decision-index-suite-0.3', suite_revision=args.suite_revision,
                    suite_verification=verification, rows=len(ids), groups=len(groups),
                    run_ids_sha256=digest, selected_groups=groups, benchmarks=counts)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(rows=len(ids), groups=len(groups), sha256=digest, benchmarks=counts)))


if __name__ == '__main__':
    main()
