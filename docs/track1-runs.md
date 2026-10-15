# Track 1 inference proxy runs

Prepared only. The lead launches after Jason's Desk decision #833. No training,
HF Job or Brev job was launched for this change. Do not combine the policies
until each passes rule 3 independently.

**The baseline already exists. Do not launch T1a.** The rule 6 run log's A3
row scores 46.97 on the frozen proxy. The published v2.1 model card identifies
its source as `frontier-infra/jebadiah-9b-v2-1-a3@9e69926a007dd636e82d33485dcd48f9751c4248`.
Read-only Hub checks on 2026-10-12 confirmed that all four safetensors LFS SHA256
values at the published revision `de19e80e18aa23420be410d54fdee435587eaad3`
match that A3 source, without downloading weights. The published
`eval/sha256-verification.json` also records matching local and Hub checks.
Temperatures and all three inference scripts match A3 byte for byte;
`temperatures.json` SHA256 is `48424adbce6d97faa73dc7a6201df939e4d0119d97b90c4ceadfd3e1578c741b`.
Reuse `runs/jebadiah-9b-v2-1-a3-9e69926a-proxy-0.3-10pct/results.jsonl.gz`
in the private results dataset at revision
`f92b3d7250a0f02f93b4cce1ce34967d0cf87296` as the paired baseline
(the file was verified present at that immutable revision). The separate 38.12 r1
run and the A3 two-seed mean are not this single-checkpoint baseline.

Both variant jobs use the ITEM 31 frozen runner/kit snapshot at results dataset
revision `587f4f496b9cf1d83f0bc8e495379496f65afa2f`, the 11,079-row proxy,
manifest/run-id SHA256 `74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`,
and suite revision `e57106b5e0698e74bd1a88b3b4c19b94a0dc8328`.
The new engine adds explicit merged-model loading so these commands load the
published weights, rather than the engine's historical default v1 adapter.
Prefix caching, batch size 64 and batch token budget 32,768 match earlier ITEM
proxy runs. The reused baseline has both new options off and each variant sets
only its own option. Temperatures remain the model's published temperatures.

Run these two exact HF Jobs commands from the checked-out PR commit after it is pushed.
`TRACK1_REVISION` captures that immutable commit. `TRACK1_JOB` is the committed
remote payload, which downloads the pinned wrapper and installs the new engine
and renderer before running it. Its explicit wrapper adjustments prevent setup
from replacing the variant engine with the old Hub engine and pin code refreshes.
Results remain private in the existing results dataset with distinct run names.
The HF secret is passed by name; no token is printed.

```bash
TRACK1_REVISION=$(git rev-parse HEAD)
TRACK1_JOB=$(cat scripts/track1_proxy_job.sh)
# Use the logged-in local HF token without displaying its value.
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

# T1a: baseline through the NEW engine, both options off. Run it so every variant differs from
# its baseline by its option alone (the new engine also adds explicit merged-model loading).
hf jobs run --detach --name track1-baseline-proxy --flavor rtx-pro-6000 --timeout 50m --secrets HF_TOKEN --env TRACK1_REVISION="$TRACK1_REVISION" --env TRACK1_POLICY=baseline pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -- bash -lc "$TRACK1_JOB"

# T1b: reverse_avg only.
hf jobs run --detach --name track1-reverse-avg-proxy --flavor rtx-pro-6000 --timeout 70m --secrets HF_TOKEN --env TRACK1_REVISION="$TRACK1_REVISION" --env TRACK1_POLICY=reverse_avg pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -- bash -lc "$TRACK1_JOB"

# T1c: repeat_payload only.
hf jobs run --detach --name track1-repeat-payload-proxy --flavor rtx-pro-6000 --timeout 70m --secrets HF_TOKEN --env TRACK1_REVISION="$TRACK1_REVISION" --env TRACK1_POLICY=repeat_payload pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -- bash -lc "$TRACK1_JOB"

# T1d: reversed read averaged only when the forward top probability is under 0.7 (Decider gate).
hf jobs run --detach --name track1-reverse-avg-below-proxy --flavor rtx-pro-6000 --timeout 70m --secrets HF_TOKEN --env TRACK1_REVISION="$TRACK1_REVISION" --env TRACK1_POLICY=reverse_avg_below pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -- bash -lc "$TRACK1_JOB"
```

The lead added T1a back (a fresh baseline through the new engine; the reused A3 result stays as a cross-check) and T1d (the gated variant).

At the [run rules](run-rules.md) rate of USD 2.75/h:

| Run | Expected runtime including setup | Expected cost | Timeout cost ceiling |
|---|---|---|---|
| T1a baseline | existing A3 result reused | $0 | no job |
| T1b reverse_avg | 40 to 60 minutes | $1.83 to $2.75 | 70 minutes, $3.21 |
| T1c repeat_payload | 40 to 60 minutes | $1.83 to $2.75 | 70 minutes, $3.21 |

These are estimates, not measurements or authorization. ITEM 31's evaluation
alone took 18.7 minutes; earlier ITEM 51/52 proxies used 35-minute allocations.
Reverse averaging adds one read per choice question; repetition nearly doubles
user payload tokens and may reduce the shared prefix benefit. Setup/model
loading and prompt length distribution can change runtime. Total expected cost
is $3.67 to $5.50; the two timeout ceilings total $6.42.
[ITEM 27](item27-training-speed.md) measures training throughput, so its
4.18 questions/s is not used as an inference throughput estimate.

For each variant, compare the same groups against T1a using the frozen proxy's
`bootstrap.py` paired whole-group bootstrap. Keep if the paired bootstrap 95%
interval of the difference is above zero; Tools deltas within 6 are noise.
Record job IDs, immutable result revisions, runtime and bootstrap outputs in
the rule 6 run log before any keep decision. Use the pinned reused A3 result
revision above for analysis; no new baseline job is needed.

`repeat_payload` duplicates the complete rendered user turn after extended-label
assignment, verbatim with one blank line between copies. The system turn, chat
template kwargs, assistant generation prefix and label read stay unchanged.
The CPU tests verify this construction. There is no known rendering ambiguity;
GPU runtime and actual proxy effects remain unmeasured. Over-context prompts
are refused, including either order's prompt in reverse averaging.

## Local validation

The nine new inference tests include the published 9B v2.1 tokenizer, a frozen
copy of the original engine call, positional bias with key mapping and log
averaging, normalization, noul single reads, both policies together, extended
labels, actual shared-prefix execution on a tiny CPU model, prefix fallback,
and OOM retry through both scoring paths. No weights were downloaded.

The repo uses separate client and server pytest configurations and imports
`conftest` directly in both; its train/server tests also share a module name.
Run suite partitions as separate processes rather than one root collection:
`train`, `data`, `scripts`, `eval/decision-index`, `clients/python`, `server`.
Set `PYTHONPATH` to the absolute train/data/scripts/eval/decision-index paths,
use Python 3.12 with the existing pytest/unittest dependencies and `ijson`,
and set `TMPDIR` to a resolved path without a macOS symlink. Set
`TRACK1_TOKENIZER` to the local tokenizer snapshot at the published revision
above to enable the real-tokenizer check. The data tests skip external-input
checks when those inputs are absent; live-model tests remain opt-in.

The standalone variant check is `python -m unittest discover -s eval/decision-index -p 'test_*.py' -v`.
The shell payload is syntax-checked with `bash -n scripts/track1_proxy_job.sh`.
Suite processes ran directly without pipes, and their real exit codes were
propagated by the parent subprocess runner.

## Launched

2026-10-12 13:06 CDT by the lead, after Desk #833 (default A). Commit c860af0, flavor rtx-pro-6000 at USD 2.75/h, spend ceiling USD 11.46 (timeouts 50 m and 3 x 70 m).

| Run | HF job id |
|---|---|
| T1a baseline | 6acd2198de32f9b0f431e85b |
| T1b reverse_avg | 6acd2198dd8177968b42bcda |
| T1c repeat_payload | 6acd2199de32f9b0f431e85d |
| T1d reverse_avg_below 0.7 | 6acd2199de32f9b0f431e85f |
