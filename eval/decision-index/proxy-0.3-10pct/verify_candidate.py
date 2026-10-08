#!/usr/bin/env python3
"""Pin and check merged candidate artifacts before allocating the single GPU job."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='frontier-infra/jebadiah-9b-v2-1-r1')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    info = HfApi().model_info(args.model, files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    required = ['config.json', 'tokenizer_config.json', 'prompt_contract.json', 'temperatures.json',
                'scripts/ainode_prompt_verbatim.py', 'scripts/jebadiah_prompt.py', 'scripts/jebadiah_model.py']
    if 'model.safetensors.index.json' in files:
        required.append('model.safetensors.index.json')
    elif 'model.safetensors' not in files:
        raise ValueError('merged model weights not uploaded yet')
    missing = set(required) - files.keys()
    if missing:
        raise ValueError(f'missing engine artifacts: {sorted(missing)}')
    hashes, downloaded = {}, {}
    for name in required:
        path = Path(hf_hub_download(args.model, name, revision=info.sha))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        downloaded[name] = path
    if 'model.safetensors.index.json' in downloaded:
        shards = set(json.loads(downloaded['model.safetensors.index.json'].read_text())['weight_map'].values())
    else:
        shards = {'model.safetensors'}
    if shards - files.keys() or any(not files[s].size for s in shards):
        raise ValueError('merged weight shards missing or empty')
    temperatures = json.loads(downloaded['temperatures.json'].read_text())['temperatures']
    if set(temperatures) != {'choice', 'noul', 'score'} or not all(isinstance(v, (int, float)) and v > 0 for v in temperatures.values()):
        raise ValueError('invalid temperatures')
    out = dict(model=args.model, revision=info.sha, merged_weight_files=sorted(shards),
               weight_bytes=sum(files[s].size for s in shards), temperatures=temperatures, artifact_sha256=hashes)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
