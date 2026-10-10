# SPDX-License-Identifier: Apache-2.0
"""Contract and mutation tests for exact scanned additions and original A3 bytes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from item56_probe_mix import compose, scan_gate, verify_source, lineage


class ProbeContracts(unittest.TestCase):
    def test_source_level_scan_cannot_be_partial_or_salvaged(self):
        clean = {'status':'passed','source_level':True,'direct_hits':0}
        scan_gate(clean)
        for changed in ({'source_level':False},{'status':'pending'},{'direct_hits':1},
                        {'removed_records':1},{'protected_index_sha256':'wrong'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                scan_gate({**clean,**changed})
        with self.assertRaises(ValueError):
            scan_gate({**clean,'raw':clean,'converted':{**clean,'status':'rejected'}})

    def test_hwu_slurp_massive_aggregate(self):
        for source in ('HWU64','SLURP English','MASSIVE en-US','translated MASSIVE'):
            self.assertEqual(lineage({'source':source},{'source_manifests':{}}),'hwu-slurp-massive')

    def test_exact_append_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            original=b'{"id":"a3"}\n'
            addition=b'{"id":"added","questions":{"decision":{"type":"choice"}}}\n'
            (folder/'train.jsonl').write_bytes(original+addition)
            (folder/'overlap-scan-report.json').write_text(json.dumps({'status':'passed','source_level':True}))
            (folder/'manifest.json').write_text(json.dumps({'files':{}}))
            pin={'added_questions':1,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
            actual, selected=verify_source(folder,pin,original)
            self.assertEqual(actual,addition)
            self.assertEqual(selected[0]['id'],'added')
            with self.assertRaises(ValueError): verify_source(folder,pin,b'changed\n')
            (folder/'train.jsonl').write_bytes(original+addition.replace(b'added',b'other'))
            with self.assertRaises(ValueError): verify_source(folder,pin,original)

    def test_math_required_before_launchable_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pins=root/'pins.json'
            pins.write_text(json.dumps({'sources':{'rung2b':{},'rung3-add':{},'rung4-add':{}}}))
            with self.assertRaisesRegex(ValueError,'All four'):
                compose(root,root,root/'out',pins=pins)
            self.assertFalse((root/'out').exists())

    @unittest.skipUnless(os.environ.get('ITEM56_SCRATCH'), 'Requires downloaded private pinned data')
    def test_live_three_source_bytes_caps_and_diagnostics(self):
        root=Path(os.environ['ITEM56_SCRATCH']);base=root/'item56-a3/a3';sources=root/'item56-sources'
        with tempfile.TemporaryDirectory(prefix='item56-test-',dir=root) as tmp:
            out=Path(tmp)/'preview'
            report=compose(base,sources,out,preview=True)
            self.assertEqual(report['questions'],23939)
            self.assertEqual(report['added_questions'],2749)
            self.assertEqual(report['removed_questions'],0)
            original=(base/'train.jsonl').read_bytes()
            expected=original+b''.join((sources/n/'train.jsonl').read_bytes()[len(original):]
                                      for n in ('rung2b','rung3-add','rung4-add'))
            self.assertEqual((out/'train.jsonl').read_bytes(),expected)
            self.assertEqual((out/'calib.jsonl').read_bytes(),(base/'calib.jsonl').read_bytes())
            for source,count in [('corr2cause',1102),('spacenli',1102),('hwu-slurp-massive',851)]:
                self.assertEqual(report['source_question_counts'][source],count)
            with self.assertRaisesRegex(ValueError,'empty'): compose(base,sources,out,preview=True)


if __name__=='__main__': unittest.main()
