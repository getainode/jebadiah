import unittest
from t2b_launch import command, TOTAL_MAX_USD
from t2b0_launch import command as baseline

class LaunchTest(unittest.TestCase):
    def test_unchanged_training_recipe_and_proxy(self):
        self.assertLess(TOTAL_MAX_USD,65)
        for seed in (17,18):
            for stage,minutes in [('train',630),('proxy',40)]:
                args=(stage,'1'*40,'2'*40,seed,'3'*40,'4'*64)
                got=command(*args); old=baseline(*args)
                self.assertEqual(got[got.index('--timeout')+1],f'{minutes}m')
                expected=old[-1].replace('t2b0-s','t2b-s')
                if stage=='train':
                    expected=expected.replace('t2b0/','t2b/').replace('t2b0-source','t2b-source')
                    for n in ('train','heldout','persist'): expected=expected.replace('t2b0_'+n+'.py','t2b_'+n+'.py')
                    expected=expected.replace('T2B0_','T2B_')
                self.assertEqual(got[-1],expected)
                self.assertIn('rtx-pro-6000',got)

if __name__=='__main__': unittest.main()
