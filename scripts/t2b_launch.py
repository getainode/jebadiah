# SPDX-License-Identifier: Apache-2.0
"""Reuse T2b-0 launch path with T2b inputs and approved aggregate ceilings."""
from t2b0_launch import command as baseline_command

CAP_MINUTES = {'train': 630, 'proxy': 40}
SMOKE_MINUTES = 15
RATE = 2.75
TOTAL_MAX_USD = (2 * sum(CAP_MINUTES.values()) + SMOKE_MINUTES) * RATE / 60
assert TOTAL_MAX_USD < 65


def command(stage, revision, code_revision, seed, data_revision, heldout_sha256):
    cmd = baseline_command(stage, revision, code_revision, seed, data_revision, heldout_sha256)
    setup = cmd[-1].replace('t2b0-s', 't2b-s')
    if stage == 'train':
        setup = setup.replace('t2b0/', 't2b/').replace('t2b0-source', 't2b-source')
        for name in ('train', 'heldout', 'persist'):
            setup = setup.replace('t2b0_' + name + '.py', 't2b_' + name + '.py')
        setup = setup.replace('T2B0_', 'T2B_')
    cmd[-1] = setup
    cmd[cmd.index('--name') + 1] = f't2b-{stage}-s{seed}'
    cmd[cmd.index('--timeout') + 1] = f'{CAP_MINUTES[stage]}m'
    return cmd
