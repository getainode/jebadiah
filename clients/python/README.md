# jebadiah-decide

Typed decisions from a [Jebadiah](https://github.com/getainode/jebadiah) model on the runtime you already
run. You send a state and typed questions (choice, noul, score) in the Jev wire format, and you get a
probability for every option. Nothing is generated.

One request format, six backends:

| Backend | Runtime | Options per question |
|---|---|---|
| `llama-server` | llama.cpp's `llama-server` (v0.5.0 or later) | no cap |
| `ollama` | Ollama, `/api/generate` in raw mode | 20 |
| `lmstudio` | LM Studio's local server | 20 |
| `vllm` | vLLM's OpenAI server, `/v1/completions` | 20, or its `--max-logprobs` |
| `mlx` | MLX in this process, Apple silicon | no cap |
| `systemone` (`ainode`, `jebadiah-serve`) | any `POST /v1/systemone` server | the server's own (20 on AINode) |

For the first four the prompt is rendered here, exactly as AINode renders it, with the tokenizer and chat
template from the model's repository, and the model's own `temperatures.json` is applied. The package
checks on every call that the runtime counted the same prompt tokens as the local template.

The setup for each runtime (install, load, what to expect, the gotchas) is in the
[Run Jeb locally](https://github.com/getainode/jebadiah/blob/main/docs/run-locally.md) guide.

## Install

```bash
pip install "jebadiah-decide[render]"     # llama-server, Ollama, LM Studio, vLLM: the tokenizer only, no torch
pip install "jebadiah-decide[mlx]"        # MLX on Apple silicon
pip install jebadiah-decide               # a /v1/systemone server only: no dependencies
```

Until it is on PyPI, install it from this repository:

```bash
pip install "jebadiah-decide[render] @ git+https://github.com/getainode/jebadiah#subdirectory=clients/python"
```

## Use it

```bash
jebadiah-decide ask --backend ollama --model hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0 \
  --repo frontier-infra/jebadiah-9b-v2-GGUF --request example-request.json
```

```python
from jebadiah_decide import Jeb

jeb = Jeb("ollama", model="hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0",
          repo="frontier-infra/jebadiah-9b-v2-GGUF")
out = jeb.decide(
    {"ticket": "Customer says the invoice total does not match the quote."},
    {"route": {"type": "choice", "instructions": "Which team should take this ticket?",
               "criteria": {"billing": "an invoice, a charge or a refund",
                            "support": "a product question", "sales": "a quote or a renewal"}}})
print(out["answers"]["route"])
```

`--model` is the name the runtime knows the model by (an Ollama tag, LM Studio's identifier, vLLM's served
name, an AINode model id, or an MLX repo or folder). `--repo` is the Jebadiah repository the loaded file came
from; only its tokenizer, chat template and `temperatures.json` are downloaded. Use the repo that matches the
size you loaded, because every size has its own temperatures.

The response has the `/v1/systemone` shape (`answers`, `usage`, `latency_ms`, `calibration`) plus `backend`
and `warnings`. A warning means a label fell outside the runtime's top 20, so that question's probabilities
are upper bounds: keep questions to 20 options on the capped backends.

## Serve the Jev wire in front of any backend

```bash
jebadiah-decide serve --backend ollama --model hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0 \
  --repo frontier-infra/jebadiah-9b-v2-GGUF --port 8100
```

That answers `POST http://127.0.0.1:8100/v1/systemone`, so a client written for the Jev wire, such as
[JDE](https://github.com/Titanium-Devops/jde)'s `jevJudge`, can use a Jeb in Ollama, LM Studio, llama-server,
vLLM or MLX. `--key` (or `JEBADIAH_API_KEY`) requires a bearer key; `--host 0.0.0.0` serves other machines.

## Tests

```bash
pip install -e ".[render,dev]"
pytest                                   # offline: fake runtimes on localhost, no model, no network
JEBADIAH_TEST_TOKENIZER=<a Jebadiah repo folder> pytest   # also checks the real renderer against the golden prompt
```

Apache-2.0. Made in Texas.
