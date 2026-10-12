# SPDX-License-Identifier: Apache-2.0
"""Reuse exact A3 launch/install/proxy commands with a $12 aggregate timeout ceiling."""
import argparse
import re
import shlex
from item52_launch import command as control_command, DATA_REVISION

CAP_MINUTES = {'train': 90, 'proxy': 35}
TOTAL_MAX_USD = 2 * sum(CAP_MINUTES.values()) * 2.75 / 60
assert TOTAL_MAX_USD < 12


def command(stage, revision, code_revision, seed, data_revision, heldout_sha256):
    for value in (revision, code_revision, data_revision):
        if not re.fullmatch('[0-9a-f]{40}', value):
            raise ValueError('Immutable revisions required')
    if not re.fullmatch('[0-9a-f]{64}', heldout_sha256):
        raise ValueError('Pinned heldout hash required')
    if stage not in CAP_MINUTES or seed not in (17, 18):
        raise ValueError('Unapproved stage or seed')
    cmd = control_command(stage, code_revision if stage == 'train' else revision, seed)
    setup = cmd[-1]
    old_slug = 'a3-rep1' if seed == 17 else 'a3-seed18'
    old_model = 'frontier-infra/jebadiah-9b-v2-1-' + old_slug
    model = f'frontier-infra/jebadiah-9b-t2b0-s{seed}'
    setup = setup.replace(old_model, model).replace(old_slug, f't2b0-s{seed}')
    if stage == 'train':
        setup = setup.replace('frontier-infra/jebadiah-data-v2-1-item32', 'frontier-infra/jebadiah-data-t2b')
        setup = setup.replace(DATA_REVISION, data_revision)
        for name in ('train.jsonl', 'calib.jsonl', 'manifest.json'):
            setup = setup.replace('a3/' + name, 't2b0/' + name)
        marker = 'set +e\n'
        bootstrap = f'''"$JEB_ROOT/venv/bin/hf" download frontier-infra/jebadiah-data-t2b --repo-type dataset --revision {data_revision} --include 't2b0/code/*' --local-dir /workspace/t2b0-source
cp /workspace/t2b0-source/t2b0/code/t2b0_persist.py /workspace/item52-persist.py
cp /workspace/t2b0-source/t2b0/code/t2b0_train.py /workspace/t2b0_train.py
cp /workspace/t2b0-source/t2b0/code/t2b0_heldout.py /workspace/t2b0_heldout.py
export T2B0_DATA_REVISION={data_revision} T2B0_HELDOUT_SHA256={heldout_sha256}
'''
        assert setup.count(marker) == 1
        setup = setup.replace(marker, bootstrap + marker)
        setup = setup.replace('JEB_SKIP_INSTALL=1 bash scripts/v21.sh',
                              'JEB_SKIP_INSTALL=1 "$JEB_ROOT/venv/bin/python" /workspace/t2b0_train.py')
    cmd[-1] = setup
    cmd[cmd.index('--name') + 1] = f't2b0-{stage}-s{seed}'
    cmd[cmd.index('--timeout') + 1] = f'{CAP_MINUTES[stage]}m'
    return cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=CAP_MINUTES)
    parser.add_argument('revision')
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--data-revision', required=True)
    parser.add_argument('--heldout-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=[17, 18], required=True)
    args = parser.parse_args()
    print(shlex.join(command(args.stage, args.revision, args.code_revision, args.seed,
                            args.data_revision, args.heldout_sha256)))
