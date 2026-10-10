"""Change exactly the original pipeline's seed literal, preserving every other byte."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

original = subprocess.check_output(['git', 'show',
    '86a8203d792463536eb009a4d8da6c8681ffdd99:train/v21_pipeline.py'])
source = Path('train/v21_pipeline.py')
assert source.read_bytes() == original
assert original.count(b'"seed": 17') == 1
modified = original.replace(b'"seed": 17', b'"seed": 18', 1)
assert len(original) == len(modified)
assert sum(a != b for a, b in zip(original, modified)) == 1
source.write_bytes(modified)
root = Path(os.environ['JEB_ROOT'])
root.mkdir(parents=True, exist_ok=True)
proof = {'control': 'A3 seed 18', 'original_seed': 17, 'control_seed': 18,
         'runtime_changed_bytes': 1, 'original_runtime_commit': '86a8203d792463536eb009a4d8da6c8681ffdd99',
         'original_pipeline_sha256': hashlib.sha256(original).hexdigest(),
         'seed18_pipeline_sha256': hashlib.sha256(modified).hexdigest(),
         'only_change': 'v21_pipeline.py config seed literal 17 to 18'}
(root / 'item52-seed18-runtime-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print('ITEM52_SEED18_PATCH', json.dumps(proof), flush=True)
