# SPDX-License-Identifier: Apache-2.0
"""Fetch only pinned source data, tokenizer files and private A3, never weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

POLICY=Path(__file__).with_name('manifests')/'item54-source-licenses.json'


def fetch(root):
    root=Path(root).resolve()
    if not root.is_relative_to(Path('/Volumes/PRO-G40/scratch/item54')):
        raise ValueError('Item54 downloads must use its PRO-G40 scratch directory')
    root.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(root)
    source=json.loads(POLICY.read_text())['sources']['massive-en-US']
    raw=root/'item54-raw';raw.mkdir(exist_ok=True)
    archive=raw/'item54-massive-1.1.tar.gz'
    if not archive.exists():urllib.request.urlretrieve(source['archive_url'],archive)
    with archive.open('rb') as stream:
        actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual != source['input_files'][archive.name]['sha256']:
        raise ValueError('Downloaded archive differs from frozen MASSIVE 1.1 bytes')
    with tarfile.open(archive) as data:
        for name in ('en-US.jsonl','LICENSE','NOTICE.md','CITATION.md'):
            member='1.1/data/'+name if name=='en-US.jsonl' else '1.1/'+name
            (raw/name).write_bytes(data.extractfile(member).read())
    subprocess.run(['hf','download','AmazonScience/massive','README.md','massive.py','CITATION.cff',
                    '--repo-type','dataset','--revision',source['revision'],'--local-dir',str(raw)],check=True)
    for name,info in source['input_files'].items():
        with (raw/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != info['sha256']:raise ValueError('Downloaded source metadata mismatch: '+name)
    subprocess.run(['hf','download','Qwen/Qwen3.5-9B','tokenizer.json','tokenizer_config.json',
                    'chat_template.jinja','merges.txt','vocab.json','--revision',
                    'c202236235762e1c871ad0ccb60c8ee5ba337b9a','--local-dir',str(root/'item54-tokenizer')],check=True)
    subprocess.run(['hf','download','frontier-infra/jebadiah-data-v2-1-item32','--repo-type','dataset',
                    '--revision','bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed','--include','a3/*',
                    '--local-dir',str(root/'item54-a3-download')],check=True)
    print('ITEM54 pinned source, tokenizer and A3 ready; no model weights downloaded')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True)
    fetch(p.parse_args().scratch)
