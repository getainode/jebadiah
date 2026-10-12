import unittest
from t2b0_launch import command, CAP_MINUTES, TOTAL_MAX_USD
from item52_launch import command as original

CODE = '8f5fd57' + '0' * 33
DATA = '1' * 40
MODEL = '2' * 40
HELDOUT = '3' * 64


class LaunchTest(unittest.TestCase):
    def test_budget_and_unchanged_recipe(self):
        self.assertLessEqual(TOTAL_MAX_USD, 12)
        for seed in (17, 18):
            cmd = command('train', CODE, CODE, seed, DATA, HELDOUT)
            self.assertEqual(cmd[cmd.index('--timeout') + 1], '90m')
            self.assertEqual(cmd[cmd.index('--flavor') + 1], 'rtx-pro-6000')
            setup = cmd[-1]
            for flags in ('--rank 16 --alpha 32', '--epochs 1 --lr 0.0001',
                          '--microbatch 8 --accumulation 1', '--calib-eval-limit 358',
                          '--save-steps 500 --resume none', '86a8203d792463536eb009a4d8da6c8681ffdd99'):
                self.assertIn(flags, setup)
            self.assertIn('--train-file t2b0/train.jsonl', setup)
            self.assertIn('--calib-file t2b0/calib.jsonl', setup)
            self.assertIn('/workspace/item52-persist.py', setup)
            self.assertEqual('item52-seed18-patch.py\npython3' in setup, seed == 18)

    def test_proxy_changes_only_model_identity(self):
        for seed in (17, 18):
            cmd = command('proxy', MODEL, CODE, seed, DATA, HELDOUT)
            prior = original('proxy', MODEL, seed)
            slug = 'a3-rep1' if seed == 17 else 'a3-seed18'
            expected = prior[-1].replace('frontier-infra/jebadiah-9b-v2-1-' + slug,
                                        f'frontier-infra/jebadiah-9b-t2b0-s{seed}').replace(slug, f't2b0-s{seed}')
            self.assertEqual(cmd[-1], expected)
            self.assertEqual(cmd[cmd.index('--timeout') + 1], '35m')


if __name__ == '__main__':
    unittest.main()
