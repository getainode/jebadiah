#!/usr/bin/env python3
"""Run the kit's engine and scorer over the frozen proxy, storing no row text."""
import json
import os
import sys
from pathlib import Path
from sample import materialize, read_ids
from score import load_proxy, score


def main(argv=None):
    from decision_index.cli import build_parser, engine_options
    from decision_index.pipeline import upload_run
    from decision_index.runner import run
    from decision_index.suite.download import download
    from decision_index.suite.io import Suite
    args = build_parser().parse_args(argv)
    if args.command != 'pipeline' or args.edition != '0.3' or args.limit or args.rows or args.public:
        raise ValueError('proxy requires pipeline edition 0.3, private results, and no LIMIT/ROWS override')
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    directory = Path(args.suite_dir)
    download(directory, dataset=manifest['suite_dataset'], revision=manifest['suite_revision'], edition='0.3')
    suite = Suite(directory, '0.3')
    suite.verify()
    ids = read_ids(root / 'run-ids.txt', manifest['run_ids_sha256'])
    proxy = load_proxy(suite, ids)
    rows_path = directory.parent / 'item31-proxy-rows.jsonl'
    materialize(suite, ids, rows_path)
    out = Path(args.out)
    try:
        run(args.engine, engine_options(args), rows_path, out, compact=True, resume=not args.fresh,
            corpus_sha256=manifest['run_ids_sha256'])
        result = score(proxy, out / 'results.jsonl', out, args.engine)
        result['manifest_sha256'] = manifest['run_ids_sha256']
        (out / 'scores.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('decision_index', 'counts', 'completed', 'expected_rows')}), flush=True)
        if args.upload:
            print(upload_run(args.upload, out, path_in_repo=args.upload_path, private=True), flush=True)
    finally:
        rows_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main(sys.argv[1:])
