"""Record installed versions and CUDA facts in a separate process before training."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import torch

root = Path(os.environ['JEB_ROOT'])
facts = {
    'python': platform.python_version(),
    'platform': platform.platform(),
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(),
    'gpu': torch.cuda.get_device_name(),
    'capability': list(torch.cuda.get_device_capability()),
    'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
    'nvcc': subprocess.check_output(['nvcc', '--version'], text=True).strip(),
    'packages': {d.metadata['Name']: d.version for d in metadata.distributions()},
    'original_runtime_commit': '86a8203d792463536eb009a4d8da6c8681ffdd99',
    'control_seed': int(os.environ.get('ITEM52_CONTROL_SEED', '17')),
}
(root / 'item52-environment.json').write_text(json.dumps(facts, indent=2) + '\n')
subprocess.run([str(root / 'venv/bin/python'), '-m', 'pip', 'freeze'],
               stdout=(root / 'item52-final-environment.freeze.txt').open('w'), check=True)
print('ITEM52_ENVIRONMENT', json.dumps({k: v for k, v in facts.items() if k != 'packages'}), flush=True)
