# Run Jeb locally

This guide sets Jebadiah up on the runtime you already use: Ollama, LM Studio, llama.cpp's `llama-server`,
vLLM, MLX on a Mac, or AINode. Every section has the install, the load, the exact request, the output you
should see, and the things that trip people up. It ends with how to plug Jeb into
[JDE](https://github.com/Titanium-Devops/jde) as its judge.

## What Jeb needs from a runtime

Jeb does not write text. You give it a state and typed questions, it runs one forward pass per question,
and the answer is the probability of each option label ("A", "B", ...) at the answer position. So a
runtime has to do three things:

1. take the prompt exactly as AINode renders it, with thinking off;
2. return the log probabilities of the next token before sampling;
3. return enough of them to cover every option.

That is why the chat window doesn't work in any of these apps: chat runs with thinking on and shows text.
Everything below goes through the runtime's API instead.

## The request

Every section uses the same request: the Jev wire format, which is also what AINode's `/v1/systemone` takes.
It is `scripts/example-request.json` in every model repository.

```json
{"state": {"ticket": "Customer says the invoice total does not match the quote."},
 "questions": {
  "route": {"type": "choice", "instructions": "Which team should take this ticket?",
            "criteria": {"billing": "an invoice, a charge or a refund", "support": "a product question",
                         "sales": "a quote or a renewal"}},
  "urgent": {"type": "noul", "instructions": "The customer is blocked from working.",
             "criteria": {"true": "work has stopped", "false": "it can wait"}}}}
```

- `choice` picks one option (2 to 20 of them) and returns a probability for each.
- `noul` is a yes or no statement and returns the probability that it is true.
- `score` places the state on an ordered scale of 2 to 10 levels.

## Pick a model and a file

| Model | Start with | Size | Same pick as bf16 (260 questions) |
|---|---|---:|---:|
| [Jebadiah 27B](https://huggingface.co/frontier-infra/jebadiah-27b-GGUF) | `Q8_0` | 29 GB | 260 |
| [Jebadiah 9B v2](https://huggingface.co/frontier-infra/jebadiah-9b-v2-GGUF) | `Q8_0` | 9.8 GB | 257 |
| [Jebadiah 4B v2](https://huggingface.co/frontier-infra/jebadiah-4b-v2-GGUF) | `Q8_0` | 4.6 GB | 256 |

Use Q8_0 if it fits. The 9B and 4B lose more to 4-bit than the 27B does (each card says how many answers
change). A file needs about its own size in GPU or unified memory, plus about 1 GB for a 4k context. On a
Mac, the [MLX builds](https://huggingface.co/collections/frontier-infra/jebadiah-open-system-one-decision-models-6ab80765ddd3fa0b3eba5213)
are the other choice.

## Two ways to talk to it

**The client (recommended).** [`jebadiah-decide`](../clients/python) is one small Python package with a
backend for every runtime in this guide. It renders the prompt exactly as AINode does, applies the model's
temperatures, checks that the runtime counted the same prompt tokens, and returns the `/v1/systemone`
response shape. It also serves `/v1/systemone` in front of any backend, which is how JDE uses a local Jeb.

```bash
pip install "jebadiah-decide[render] @ git+https://github.com/getainode/jebadiah#subdirectory=clients/python"
```

**The scripts in each model repository.** `decide_gguf.py` (llama-server), `decide_ollama.py`,
`decide_lmstudio.py` and `decide_mlx.py` do the same thing for one runtime each, with no package to
install beyond `transformers` (MLX: `mlx-lm`).

```bash
hf download frontier-infra/jebadiah-9b-v2-GGUF --include "scripts/*" --include "*.json" --include "*.jinja" --include "*.txt" --local-dir jebadiah-9b-v2-GGUF
pip install transformers        # the tokenizer only, no torch
```

In both, `--repo` (or the script's folder) must be the repository of the size you loaded, because every
size has its own `temperatures.json`.

---

## Ollama

Works up to 20 options per question. Checked on Ollama 0.34.4.

**Load.** Ollama pulls the GGUF straight from Hugging Face; nothing else to set up.

```bash
ollama pull hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0
```

**Ask.**

```bash
jebadiah-decide ask --backend ollama --model hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0 \
  --repo frontier-infra/jebadiah-9b-v2-GGUF --request example-request.json
# or: python jebadiah-9b-v2-GGUF/scripts/decide_ollama.py --model hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0 \
#       --request jebadiah-9b-v2-GGUF/scripts/example-request.json
```

Under the hood that is `POST /api/generate` with `"raw": true` (so Ollama's template never touches the
prompt), `"think": false`, `"num_predict": 1`, `"logprobs": true` and `"top_logprobs": 20`.

**Expect** (9B v2 Q8_0):

```json
"route":  {"type": "choice", "choice": "billing", "confidence": 0.461713,
           "probabilities": {"billing": 0.641142, "support": 0.036095, "sales": 0.322763}},
"urgent": {"type": "noul", "noul": 0.167016}
```

**Checked.** On the 260 held-out questions, Ollama gave the same pick as llama-server on the same file every
time, for the 4B v2, the 9B v2 and the 27B. On questions of 20 options or fewer the probabilities match
llama-server to 0.0003.

**Gotchas.**
- `ollama run` is the chat window. It won't give you decisions.
- 20 options at most: Ollama caps `top_logprobs` at 20. On the 77-option Banking77 questions the pick was
  still right, but the probabilities moved by up to 0.16. The client warns when a label falls outside the
  top 20.
- Logprobs need a recent Ollama (they arrived in late 2025).

## LM Studio

Works up to 20 options per question, through the local server. Checked on LM Studio 0.4.21, 9B v2 only.

**Load.** Search for `jebadiah-9b-v2` in LM Studio and download the Q8_0 from
`frontier-infra/jebadiah-9b-v2-GGUF`. Open the **Developer** tab, start the server, load the model, and
note the identifier it shows (for example `jebadiah-9b-v2`). With the CLI:

```bash
lms server start
lms load jebadiah-9b-v2 --identifier jebadiah-9b-v2
```

**Ask.**

```bash
jebadiah-decide ask --backend lmstudio --model jebadiah-9b-v2 \
  --repo frontier-infra/jebadiah-9b-v2-GGUF --request example-request.json
# or: python jebadiah-9b-v2-GGUF/scripts/decide_lmstudio.py --model jebadiah-9b-v2 --request ...
```

That sends the chat messages to `/v1/chat/completions` with `"reasoning_effort": "none"`, one token and
the top 20 log probabilities. With reasoning off, LM Studio renders the same prompt text as the other runtimes.

**Expect** (9B v2 Q8_0): billing 0.64114, support 0.036056, sales 0.322804; urgent 0.16703.

**Gotchas.**
- `reasoning_effort: "none"` is the only switch that turns thinking off here. `chat_template_kwargs` is
  ignored, and LM Studio's raw `/v1/completions` returns no log probabilities.
- If **Require Authentication** is on in the server settings, create a token there and pass `--api-key`
  (or `export LM_API_TOKEN=...` for the script).
- 20 options at most, the same as Ollama.
- **Ignore a "Vision" tag.** Some third-party quants ship a vision file (mmproj) from the Qwen base, and LM
  Studio labels the model Vision because of it. Jeb was not trained on images. The mmproj isn't needed.

## llama.cpp (`llama-server`)

No option cap. This is the reference path every GGUF was checked on.

**Load.** You need llama.cpp v0.5.0 or later; older builds don't know the `qwen35` architecture.

```bash
hf download frontier-infra/jebadiah-9b-v2-GGUF jebadiah-9b-v2-Q8_0.gguf --local-dir .
llama-server -m jebadiah-9b-v2-Q8_0.gguf -c 4096 -np 1 --port 8080
```

**Ask.**

```bash
jebadiah-decide ask --backend llama-server --url http://127.0.0.1:8080 \
  --repo frontier-infra/jebadiah-9b-v2-GGUF --request example-request.json
# or: python jebadiah-9b-v2-GGUF/scripts/decide_gguf.py --server http://127.0.0.1:8080 --request ...
```

That sends the rendered text to `/completion` with `"n_predict": 1`, `"n_probs": 1000`,
`"post_sampling_probs": false` and `"cache_prompt": false`, and reads each label by token id.

**Expect** (9B v2 Q8_0): the same numbers as Ollama above. 4B v2 Q8_0: billing 0.598244, support 0.082792,
sales 0.318963; urgent 0.153519. 27B Q8_0: billing 0.764669, support 0.034001, sales 0.20133; urgent 0.209376.

**Gotchas.**
- Keep `post_sampling_probs` off. With it off, llama-server returns a plain softmax of the logits, which is
  what the model's temperatures were fitted on.
- Keep `cache_prompt` off when you compare numbers across runs; a reused cache can move the last digits.

## vLLM

This is the engine AINode serves Jeb on. Checked with vLLM 0.17 on a DGX Spark, through AINode.

**Load.** Serve the full-weight repository. `--language-model-only` is needed because the Qwen3.5 base
carries a vision tower that Jeb doesn't use.

```bash
vllm serve frontier-infra/jebadiah-9b-v2 --max-model-len 4096 --language-model-only \
  --served-model-name frontier-infra/jebadiah-9b-v2
```

**Ask.**

```bash
jebadiah-decide ask --backend vllm --url http://127.0.0.1:8000 --model frontier-infra/jebadiah-9b-v2 \
  --repo frontier-infra/jebadiah-9b-v2 --request example-request.json
```

That sends the rendered text to `/v1/completions` with `"max_tokens": 1` and `"logprobs": 20`.

**Expect** (9B v2, bf16 on a GB10): billing 0.627868, support 0.038476, sales 0.333656; urgent 0.167275.
The small differences from the GGUF numbers are the bf16 engine's own rounding.

**Gotchas.**
- vLLM caps logprobs at `--max-logprobs`, 20 by default. Raise it, and pass `--top-n` to the client, to go
  past 20 options. We haven't run Jeb that way yet.
- On a DGX Spark with the vLLM 0.17 image, AINode also passes `--enforce-eager`; that is about that image
  on GB10, not about Jeb.

## MLX on Apple silicon

No option cap. Runs in-process: one forward pass, then the output-head rows of the label tokens in fp32.

**Load and ask.** The client downloads the precision folder you ask for (8-bit by default).

```bash
pip install "jebadiah-decide[mlx] @ git+https://github.com/getainode/jebadiah#subdirectory=clients/python"
jebadiah-decide ask --backend mlx --model frontier-infra/jebadiah-9b-v2-MLX --request example-request.json
# or: python jebadiah-9b-v2-MLX/scripts/decide_mlx.py --model frontier-infra/jebadiah-9b-v2-MLX --precision 8bit --request ...
```

**Expect** (9B v2, 8-bit): billing 0.640554, support 0.037693, sales 0.321753; urgent 0.165308.
4B v2 8-bit: billing 0.596195, support 0.0839, sales 0.319905; urgent 0.147991.

**Gotchas.**
- Needs `mlx-lm` 0.31 or newer (Qwen3.5 support).
- `mlx_lm.server` is not a path: it returns at most 11 log probabilities.

## AINode

AINode serves Jeb on vLLM and answers `POST /v1/systemone` (the Jev wire) and `POST /v1/decide` itself,
with the model's temperatures applied. Up to 20 options per question.

**Install and load.** Install AINode on an NVIDIA box (`curl -fsSL https://ainode.dev/install | bash`), then
load the model with the flags from the vLLM section. `gpu_memory_utilization` is the share of GPU memory
the engine takes; 0.3 is what we run the 9B at on a 128 GB DGX Spark, so set it for your card.

```bash
curl -s http://<node>:3000/api/models/load -H "Authorization: Bearer $AINODE_API_KEY" \
  -H 'content-type: application/json' -d '{"model": "frontier-infra/jebadiah-9b-v2", "max_model_len": 4096,
  "gpu_memory_utilization": 0.3, "extra_vllm_args": ["--language-model-only"]}'
```

**Ask.**

```bash
jebadiah-decide ask --backend ainode --url http://<node>:3000 --model frontier-infra/jebadiah-9b-v2 \
  --api-key "$AINODE_API_KEY" --request example-request.json
# or plain HTTP: POST http://<node>:3000/v1/systemone with the request plus "model"
```

**Expect** (9B v2 on a DGX Spark): billing 0.62911, support 0.036574, sales 0.334317; urgent 0.167275, about
250 ms for the request.

**Gotchas.**
- A choice with more than 20 options is refused with a 422 that names the cap. It isn't truncated.
- The first request after a load is slow while the engine warms up.

## jebadiah-serve (the standalone server)

[`server/`](../server) in this repository loads the full weights with transformers on one CUDA GPU or a Mac,
and serves `/v1/systemone`, `/v1/decide` and a browser playground. It is the simplest way to get a Jev-wire
endpoint without another runtime. See [`server/README.md`](../server/README.md). The client talks to it with
`--backend jebadiah-serve --url http://127.0.0.1:8000`.

---

## Things that trip people up, in one place

- **No chat window.** Every runtime's chat UI runs with thinking on and shows text. Use the API.
- **Thinking off.** Ollama `"think": false` (with `"raw": true`), LM Studio `"reasoning_effort": "none"`; the
  raw-text paths (llama-server, vLLM) carry the empty think block in the rendered prompt.
- **The prompt must be exact.** Jeb was trained on one rendering. The client and the scripts check the
  runtime's prompt token count against the local template on every call and stop if they differ.
- **20 options** on Ollama, LM Studio, AINode and default vLLM. llama-server and MLX have no cap.
- **Temperatures.** Each size ships `temperatures.json` (choice, noul, score), fitted on held-out data. The
  client applies it by default; `--no-temperatures` gives the raw probabilities.
- **Quantization.** Stay on Q8_0 or 8-bit unless memory forces you lower.
- **The Vision tag** in LM Studio comes from a stray mmproj file. Ignore it.

## Use Jeb as JDE's judge

[JDE](https://github.com/Titanium-Devops/jde), the Jev Decision Engine, asks its questions through a
*judge* and turns the answers into actions with policy bands. Its `jevJudge` posts the Jev wire to one
endpoint, so any `/v1/systemone` server can be the judge: AINode, `jebadiah-serve`, or `jebadiah-decide
serve` in front of Ollama, LM Studio, llama-server, vLLM or MLX.

**1. A local endpoint.** For example, Jeb in Ollama behind the client's server:

```bash
jebadiah-decide serve --backend ollama --model hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0 \
  --repo frontier-infra/jebadiah-9b-v2-GGUF --port 8100
```

**2. Point JDE at it.**

```ts
import { ask, jevJudge } from "@titanium/jde";

const judge = jevJudge({ endpoint: "http://127.0.0.1:8100/v1/systemone", model: "jebadiah" });
const outcome = await ask({ decision, state, questions }, { judge });
```

`jevJudge` reads a key from `TYPESAFE_API_KEY` and refuses to run without one. Set it to anything for a
local server with no key, or to your server's key (`jebadiah-decide serve --key`, AINode's API key).

**3. Raise the deadline.** JDE's shipped policy gives a judgment 750 ms, which is a hosted budget. Set
`timeout_ms` to your own endpoint's p95 in `policy.json`, or pass a `policyEntry`.

[`examples/jde-local-jeb.mjs`](examples/jde-local-jeb.mjs) runs JDE's completion check over one of JDE's own
case files with a local Jeb as the judge:

```bash
git clone https://github.com/Titanium-Devops/jde && (cd jde && npm install && npx tsc -p tsconfig.json)
export TYPESAFE_API_KEY=local-jeb
JDE_DIR=./jde JEB_ENDPOINT=http://127.0.0.1:8100/v1/systemone JEB_MODEL=jebadiah \
  node docs/examples/jde-local-jeb.mjs jde/cases/completion-check-tuned.json
```

That was run end to end against Jeb 9B v2 on AINode and against Jeb 4B v2 in Ollama through `jebadiah-decide
serve`. It is a smoke test that the wiring works, not a measurement. The JDE blind-test result for the 9B v2
is on its [model card](https://huggingface.co/frontier-infra/jebadiah-9b-v2-GGUF).

**Judge Jeb.** An adapter specialised for JDE's judging questions, and a `jebJudge` provider so JDE can ask
Jeb directly, are planned but not published; they are listed under
[What's next](../README.md#whats-next). Until then, `jevJudge` with a local endpoint, as above, is the way in.

## Links

- Models: the [Hugging Face collection](https://huggingface.co/collections/frontier-infra/jebadiah-open-system-one-decision-models-6ab80765ddd3fa0b3eba5213)
- Client: [`clients/python`](../clients/python)
- Standalone server: [`server/`](../server)
- JDE: [github.com/Titanium-Devops/jde](https://github.com/Titanium-Devops/jde)
- AINode: [github.com/getainode/ainode](https://github.com/getainode/ainode)
- Project site: [jebadiah.ai](https://jebadiah.ai)
