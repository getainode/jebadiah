# eval/

| Path | Purpose |
|---|---|
| `nonce_eval.py` | the nonce robustness test: a random id planted in the state or the instructions, K passes, compared with the clean read (run by `scripts/nonce_all.sh`) |
| `decision-index/jebadiah_engine.py` | a Decision Index `Engine` that loads a Jebadiah model at a pinned Hub revision and calls its published scripts unchanged; `mps_delta.py` is an Apple-silicon-only workaround for one DeltaNet op |
| `jevbench/` | the JevBench runner: `shim/systemone_shim.py` (a loopback `/v1/systemone` server over the Decision Index engine, with `score` enabled), `run_bench.sh` (the benchmark's own CLI, unchanged), `analyze.py` (public-item metrics with the benchmark's own functions), `overlap_check.py` (training pool against the public items) |

The main evaluator, `train/eval_jebadiah.py`, lives with the trainer because every sweep run calls it.

## JevBench

```bash
git clone https://github.com/fstandhartinger/jevbench eval/jevbench/repo && git -C eval/jevbench/repo checkout 1bcc55e
pip install -e ./path/to/decision-index      # apolinario/decision-index, for decision_index.engines
python eval/jevbench/shim/systemone_shim.py --adapter frontier-infra/jebadiah-9b-v1 --adapter-revision <sha> \
    --base Qwen/Qwen3.5-9B-Base --base-revision 68c46c4b3498877f3ef123c856ecfde50c39f404 --name jebadiah-9b-v1 --port 8790 &
bash eval/jevbench/run_bench.sh jebadiah-9b-v1 8790
python3 eval/jevbench/analyze.py eval/jevbench/runs/jebadiah-9b-v1 0.10
```

Our records are in `results/jevbench/`.
