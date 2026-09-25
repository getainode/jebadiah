# data/

Builders for every training pool, and the manifests that pin them. The built pools are not committed; each one's
sha256, record and question counts, and counts by set and type are in `manifests/pools.json`, and
`pool_manifest.py check <root> [dir ...]` compares a rebuilt pool against them.

| File | Purpose |
|---|---|
| `convert_data.py` | downloads the public sources at pinned revisions and writes the v1 pool plus the 20 test sets (`--out`, `--raw`, `--jevals jevals/`) |
| `split_pool.py` | 95/5 train/calibration split by family hash |
| `lint_data.py` | rejects label-like keys under questions, labels that do not fit, malformed questions, empty states |
| `pool-v2/` | the synthetic pool: `families.py` (24 state families), `generate.py` (author and judge through an AINode endpoint; set `JEB_TEACHER_BASE_URL` and `JEB_TEACHER_KEY`), `finalize.py` (dedupe, overlap check, split, lint, manifest), `DATASET-CARD.md` |
| `dose/make_dose_pool.py` | the c20 and c33 synthetic-choice dose pools |
| `human-noul/` | `convert_human_noul.py` (CondaQA, ContractNLI, ShARC, UNFAIR-ToS to yes/no records, with overlap checks) and `make_c20h_pools.py`; the two manifests record every count and sha256 |
| `human/prolific-r1.jsonl` | 54 human-labelled yes/no questions from our round-1 reader study (aggregated targets only) |
| `compose_pools.py` | `data-v3-c20fh-alt`, `data-v3-full`, `data-v3-full-s` from their parts |
| `jevals/` | Jevals suite 0.1.0 task files (ids, row indices, hashes and labels; no item text), CC BY 4.0 |
| `manifests/` | `pools.json`, the converter manifests for the v0 and v1 pools, the synthetic generator's manifest |

Environment variables: `JEB_ROOT` (where `data-*` pool directories live, default `/workspace/jeb`) and
`JEB_POOL_V2` (the pool-v2 working directory, default `data/pool-v2`).
