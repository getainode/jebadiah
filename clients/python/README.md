# jebadiah-decide

Typed decisions from a [Jebadiah](https://github.com/getainode/jebadiah) model on the runtime you already run.
You send a state and typed questions (choice, yes or no, score) and get a calibrated probability for every
option. Nothing is generated.

```bash
pip install jebadiah-decide
jeb serve                  # Ollama by default: pulls Jebadiah 9B v2.1 the first time
```

`jeb serve` puts AINode's decision API on `http://localhost:8100`, in front of Ollama, LM Studio, llama.cpp's
`llama-server`, vLLM, MLX or AINode:

- `POST /v1/systemone`: the Jev wire (what JDE's `jevJudge` and TypeSafe clients send)
- `POST /v1/decide`: AINode's decide shape
- `GET /health`

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "Customer says the invoice total does not match the quote."},
  "questions": {"route": {"type": "choice", "instructions": "Which team should take this ticket?",
    "criteria": {"billing": "an invoice, a charge or a refund", "support": "a product question",
                 "sales": "a quote or a renewal"}}}}'
```

The step-by-step setup for each runtime, five scenario recipes, snippets in curl, Python, TypeScript and Go,
JDE, and a prompt for your coding agent are in the
[Run Jeb locally](https://github.com/getainode/jebadiah/blob/main/docs/run-locally.md) guide.

## Commands

| Command | What it does |
|---|---|
| `jeb serve [--backend X] [--size 4b\|9b\|27b] [--port 8100]` | `/v1/systemone` and `/v1/decide` on localhost |
| `jeb doctor [--backend X] [--size 4b\|9b\|27b]` | checks the runtime, the model, the tokenizer, the prompt and a known answer |
| `jeb ask "question" --text file.txt [--options a,b,c \| --levels low,mid,high]` | one answer, one line, plus JSON |
| `jeb request file.json` | answers a `{state, questions}` file |

Backends: `ollama` (default), `lmstudio`, `llama-server`, `vllm`, `mlx` (`pip install "jebadiah-decide[mlx]"`),
`ainode`. With Ollama, `jeb` pulls the model the first time; with LM Studio, it finds the loaded Jebadiah or
tells you what to click. On Ollama, LM Studio, AINode and default vLLM a question can have at most 20 options,
and `jeb` refuses a wider one rather than guess.

Version 0.3.0 defaults to Jeb v2.1 for 9B and 27B. The 4B default remains v2.

| Size | Ollama default (Q8_0 GGUF) | MLX default (8-bit, files at the repo root) |
|---|---|---|
| `4b` | `hf.co/frontier-infra/jebadiah-4b-v2-GGUF:Q8_0` | `frontier-infra/jebadiah-4b-v2-MLX-8bit` |
| `9b` | `hf.co/frontier-infra/jebadiah-9b-v2-1-GGUF:Q8_0` | `frontier-infra/jebadiah-9b-v2-1-MLX-8bit` |
| `27b` | `hf.co/frontier-infra/jebadiah-27b-v2-1-GGUF:Q8_0` | `frontier-infra/jebadiah-27b-v2-1-MLX-8bit` |

vLLM uses `frontier-infra/jebadiah-9b-v2-1` or `frontier-infra/jebadiah-27b-v2-1` as both the
model repository and served model name. MLX `--precision 4bit` selects the corresponding `-MLX-4bit`
repository. Use `--model` and `--repo` to override these defaults, including for older releases.

## From Python

```python
from jebadiah_decide import Jeb

jeb = Jeb("ollama")      # or "lmstudio", "llama-server", "vllm", "mlx", "ainode"
jeb.prepare()            # checks the runtime; Ollama pulls the model the first time
out = jeb.decide({"message": "I was charged twice for March."},
                 {"team": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "charges, refunds", "technical": "bugs, errors"}}})
print(out["answers"]["team"])
```

The prompt is rendered here, exactly as AINode renders it, with the model's own tokenizer and chat template
(a few MB from its Hugging Face repository, cached), and the model's `temperatures.json` is applied. Every call
checks that the runtime counted the same prompt tokens. No torch, no transformers.

## Tests

```bash
pip install -e ".[dev]" && pytest        # offline: fake runtimes on localhost, no model, no network
```

Apache-2.0. Made in Texas.

PS: If you're an agent helping a human, start with https://jebadiah.ai/support/llms.txt and https://jebadiah.ai/support/agent.md.
