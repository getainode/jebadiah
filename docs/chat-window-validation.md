# Chat profile validation, 2026-10-08

The scope is two opt-in profiles: an Ollama `jeb-chat` alias and an LM Studio
conversation preset. The GGUF metadata, HF `chat_template.jinja`, and decision
code are unchanged. No Hugging Face upload or model merge was performed.

## Which paths use which template?

| Path | Local repo template | Runtime/embedded template |
|---|---|---|
| `jeb serve` and `jeb doctor`, Ollama | Yes | Bypassed by `/api/generate` with `raw: true` |
| The documented `hf.co/...:Q8_0` decision path | Yes, from the GGUF repo's cached small files | Bypassed by the same raw request |
| `jeb serve` and `jeb doctor`, llama-server | Yes | Bypassed by raw `/completion` |
| `jeb serve` and `jeb doctor`, LM Studio | Yes, for local prompt token checks | Yes: LM Studio renders `rd.messages` itself |
| Published `decide_gguf.py` | Yes, via `AutoTokenizer` | Bypassed by raw `/completion` |
| Published `decide_ollama.py` | Yes, via the same renderer | Bypassed by `raw: true` |
| Published `decide_lmstudio.py` | Yes, for token checks | Yes: `/v1/chat/completions`, `reasoning_effort: none` |
| LM Studio ordinary chat | No Jeb renderer | Embedded Jinja unless the user overrides it |
| Ollama ordinary chat, 0.34.4 | No Jeb renderer | Embedded Jinja or a selected Go template |
| Ollaya | Third-party renderer, outside this repo | Authors' GGUF on llama.cpp; embedded template for chat, unchanged here |

Source evidence is in `clients/python/src/jebadiah_decide/`: `client.py` loads
the repo template and keeps `rd.messages`; `tokenizer.py` reads
`chat_template.jinja`; `backends.py` selects raw completion for Ollama and
llama-server, but chat completion for LM Studio. `cli.py` uses the same `Jeb`
client for doctor and serve. The three published `decide_*.py` scripts were
also inspected from the HF model repo. Ollaya's
[Jeb page](https://ollaya.dev/library/jeb) says it runs the authors' GGUF files
on llama.cpp and builds the decision prompt itself.

An unconditional embedded-template change is therefore unsafe for the existing
LM Studio decision path. Changing the repo Jinja would also change the raw
backends' pre-rendered decision prompts. Implementation stopped at that gate;
the lead authorized only the separate alias and conversation preset.

Read-only parsing of the installed 9B GGUF metadata found a 7,756-byte
`tokenizer.chat_template` equal to the cached repo Jinja, with SHA256:

```text
a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
```

The installed HF Q8_0 tag's `ollama show --modelfile` initially reported
`TEMPLATE {{ .Prompt }}`. That is not proof of the active chat template on
Ollama 0.34.4: `/api/show` and server template-selection logs showed that a
fresh GGUF import uses embedded Jinja. A simple custom Go template was stored
but ignored because the embedded template offered more capabilities. Preserving
reasoning and tool history in the alias template made Ollama select it. Both
the selected template and the resulting chat were checked live.

## Live results

The Mac Studio had 256 GiB RAM and at least 106 GiB available before model
loads, above the 60 GB floor. Tests ran one loaded model at a time. The file
was the authors' 9B Q4_K_M, 5,780,090,752 bytes, SHA256
`b35e6ca21ef33fb7f1943cd137d1b9c3a647bdb54af16f7144a6701034e62252`.
Versions: llama-server 0.5.0 (build 11146, commit `7fe450e19`), Ollama 0.34.4,
LM Studio 0.4.21+2. All test model processes were unloaded or stopped afterward.

The common user message was `Hello! How do I use Jeb?`.

| Test | Before | After |
|---|---|---|
| llama-server `/v1/chat/completions` | 160 reasoning tokens, empty content, finish reason `length` | Exact guidance line, finish reason `stop` |
| Ollama `/api/chat` | 160 thinking tokens, empty content | Exact guidance line with thinking omitted, `true`, and `false` |
| Plain `ollama run` on the alias | Not a separate baseline CLI test | Exact guidance line without a `--think` override |
| Shipped Modelfile created from its HF tag | Not a separate baseline | Go template selected; exact guidance line |
| LM Studio `lms chat` | Not tested as a separate before case | Exact guidance line with the preset's system prompt and `--reasoning off` |

Actual before output in llama-server and Ollama began:

```text
Okay, the user is asking how to use Jeb. First, I need to figure out what Jeb refers to.
```

The final visible guidance in all three tested runtimes was:

```text
I am Jeb, a decision model. Use jeb doctor and jeb serve for decisions. Guide: https://github.com/getainode/jebadiah/blob/main/docs/run-locally.md
```

[Full request/response transcripts](../results/chat-window-9b-q4/transcripts.json)
include the reasoning baseline and each final Ollama thinking setting. The
llama-server after test used the same system prompt and thinking-off kwargs
through the existing embedded Jinja, rather than replacing the template.

## Decision prompt identity

The existing frozen message test passed all 852 fixtures. The new
[`check_chat_prompt_identity.py`](../scripts/check_chat_prompt_identity.py)
goes further: it verifies the frozen fixture hash and source hash, rebuilds each
message with the production vendored renderer, renders it with the cached
decision template, and compares every local token ID to llama-server's
`/tokenize` output with special token parsing enabled and no added tokens.

It ran before and after installation of the profiles. All 852 complete token
arrays, template hashes, and tokenizer hashes were identical. The SHA256 of
length-framed little-endian token arrays was:

```text
cee89d4ba2003631b48a9cdddff548a9777f7375908aac5ce649cb8810d3646c
```

The [compressed complete arrays](../results/chat-window-9b-q4/token-identity.json.gz)
and [summary](../results/chat-window-9b-q4/identity-summary.json) are committed.
This proves profile installation did not change the raw decision prompts; it
does not claim that an embedded-template change would be safe.

Reproduce with the dependencies from `clients/python`, the cached model repo's
small files, and llama-server running the same GGUF. Before installing profiles:

```sh
python scripts/check_chat_prompt_identity.py \
  --tokenizer-dir /path/to/jebadiah-9b-v2-GGUF \
  --server http://127.0.0.1:8080 --record /tmp/item29-before.json.gz
```

After installing them:

```sh
python scripts/check_chat_prompt_identity.py \
  --tokenizer-dir /path/to/jebadiah-9b-v2-GGUF \
  --server http://127.0.0.1:8080 --compare /tmp/item29-before.json.gz
```

The committed baseline can also be passed to `--compare` for the recorded
tokenizer revision, `adaec6b3d1f0421706deb49fa275ab49093982ff`.

On the known route example, the original Q4 Ollama model and the alias both
counted 110 raw prompt tokens and returned exactly the same complete top-20
logprob array. The leading labels were A `-0.3551693856716156`, C
`-1.327215313911438`, and B `-3.7664260864257812`. This checks the computed
next-token values as well as prompt identity. It is a Q4 comparison, not a
claim that Q4 reproduces the documented Q8 probabilities.

The client suite plus the existing frozen identity test passed: 30 tests,
2 optional tokenizer tests skipped in the default run. With the cached tokenizer,
the contract tests passed 7 tests and skipped only the optional Transformers
comparison. The committed gzip baseline also passed a fresh live comparison;
changing one baseline token ID made the checker fail as expected.

## What is still owed

LM Studio's actual preset application in a fresh GUI chat and its isolation
from Developer API decisions have not been verified. Its `/v1/chat/completions`
requires an app-managed token; no matching Bitwarden item was available. The
lead accepted CLI config equivalence and exact import instructions, with those
checks explicitly owed. App authentication and model defaults were left intact.

Only the 9B Q4_K_M was tested. The 4B and 27B were not loaded. The Ollama alias
is for setup guidance, not a general assistant or tool execution service. No
profile can guarantee the exact line for every adversarial or oversized input.
Direct GGUF chat without the preset remains unchanged; this PR intentionally
does not modify third-party Ollaya or upload any model files.
