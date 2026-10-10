# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import item56_launch as L


class RecipeContract(unittest.TestCase):
    def test_original_runtime_seed_guard_and_recipe_flags(self):
        revision='a'*40
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            (root/'configs/item56-probe-b.json').write_text(json.dumps({'dataset_revision':revision}))
            with patch.object(L,'ROOT',root):
                for seed in (17,18):
                    command=L.command('train',revision,revision,seed,revision)
                    setup=command[-1]
                    for required in ('git checkout -q 86a8203d792463536eb009a4d8da6c8681ffdd99',
                                     '--rank 16 --alpha 32 --max-seq-length 2048 --epochs 1 --lr 0.0001',
                                     '--microbatch 8 --accumulation 1 --group-by-length --pad-to-multiple-of 64',
                                     '--checkpoint-min-tokens 512 --eval-steps 200 --calib-eval-limit 358',
                                     'probe-b/train.jsonl','probe-b/calib.jsonl','probe-b/manifest.json',
                                     f'frontier-infra/jebadiah-9b-v2-1-probe-b-s{seed}',
                                     'item56_train.py','item56_persist.py'):
                        # Shell calls use hyphen names for /workspace wrappers.
                        required=required.replace('item56_train.py','item56-train.py')
                        self.assertIn(required,setup)
                    self.assertNotIn('--backbone-autocast',setup)
                    self.assertEqual('item52_seed18_patch.py' in setup,seed==18)
                    self.assertEqual(command[command.index('--flavor')+1],'rtx-pro-6000')
                    self.assertEqual(command[command.index('--timeout')+1],'150m')
                command=L.command('proxy',revision,revision,17)
                self.assertIn('587f4f496b9cf1d83f0bc8e495379496f65afa2f',command[-1])
                self.assertIn('LIMIT=',command[-1])

    def test_reserved_jobs_cannot_exceed_twenty_dollars(self):
        self.assertLessEqual((2*L.CAP_MINUTES['train']+2*L.CAP_MINUTES['proxy']+L.CAP_MINUTES['diagnostic'])/60*2.75,20)


if __name__=='__main__':unittest.main()
