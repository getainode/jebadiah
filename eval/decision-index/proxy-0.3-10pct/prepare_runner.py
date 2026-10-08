#!/usr/bin/env python3
"""Small explicit changes to the supplied resumable wrapper and engine provenance."""
import argparse
import hashlib
from pathlib import Path

WRAPPER_SHA256 = 'c5e4e29c2ba2d2ffeb5c30367d6f4ae337f811b89fa0b2a2eeee9a470170aed6'
ENGINE_SHA256 = '52230541c49ca888801de378b415a4e8da32f488069894d9bc0da71b967be532'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected exactly one source match: {old!r}')
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='directory containing supplied code/')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    wrapper_path = args.source / 'indexrun-job.sh'
    engine_path = args.source / 'engine/jebadiah_engine.py'
    for path, expected in ((wrapper_path, WRAPPER_SHA256), (engine_path, ENGINE_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'supplied runner source changed: {path}')
    wrapper = wrapper_path.read_text()
    wrapper = replace_once(wrapper, 'export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false',
                           'export HF_HUB_DISABLE_XET=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false')
    wrapper = replace_once(wrapper, 'final() { log "exit trap: final sync"; [ -n "${LOOP_PID:-}" ] && { pkill -P "$LOOP_PID" 2>/dev/null; kill "$LOOP_PID" 2>/dev/null; }; sync_up final || log "WARN final sync failed"; }',
                           'final() { log "exit trap: final sync"; [ -n "${LOOP_PID:-}" ] && kill "$LOOP_PID" 2>/dev/null; sync_up final || log "WARN final sync failed"; }')
    wrapper = replace_once(wrapper, 'python -m decision_index pipeline --engine',
                           'python /tmp/src/code/proxy/run.py pipeline --engine')
    wrapper = replace_once(wrapper, 'setup_once || {',
                           'retry hf download "$RESULTS_REPO" --include "code/proxy/*" --repo-type dataset --local-dir /tmp/src || exit 2\nsetup_once || {')
    engine = engine_path.read_text()
    engine = replace_once(engine, 'self.model_id = f"{MODEL_REPO}@{revision[:8]}"', 'self.model_id = f"{model}@{revision[:8]}"')
    engine = replace_once(engine, '"model_repo": MODEL_REPO, "model_revision": revision,', '"model_repo": model, "model_revision": revision,')
    engine = replace_once(engine, 'f"https://huggingface.co/{MODEL_REPO}/tree/{revision}/scripts"', 'f"https://huggingface.co/{model}/tree/{revision}/scripts"')
    (args.out / 'engine').mkdir(parents=True, exist_ok=True)
    (args.out / 'indexrun-job.sh').write_text(wrapper)
    (args.out / 'engine/jebadiah_engine.py').write_text(engine)


if __name__ == '__main__':
    main()
