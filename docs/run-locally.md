# Run Jeb locally

Jebadiah (Jeb) is an open decision model. You ask it a typed question about some text or JSON (pick one of
these options, yes or no, or a score) and it answers with a calibrated probability for every option. It
never writes text, so it's fast, it's the same every time, and you can set thresholds on it.

This page gets Jeb running on your machine and answering your first real decision. On the Ollama path that
took **2 minutes 1 second** on a clean setup, most of it the 9.8 GB model download.

- [Pick your runtime](#which-runtime-are-you-using)
- [Scenarios: five recipes to copy](#scenarios)
- [Integrate it: curl, Python, TypeScript, Go](#integrate-it)
- [Set it up with your AI](#set-it-up-with-your-ai)
- [How it works, and what trips people up](#how-it-works)

## Which runtime are you using?

| You use | Go to | Options per question |
|---|---|---|
| **Ollama** (the fastest start) | [Ollama](#ollama) | 20 |
| **LM Studio** | [LM Studio](#lm-studio) | 20 |
| **vLLM** | [vLLM](#vllm) | 20 by default |
| llama.cpp's `llama-server` | [llama.cpp](#llamacpp) | no limit |
| A Mac, no runtime yet | [MLX](#mlx-on-a-mac) | no limit |
| AINode | [AINode](#ainode) | 20 |
| Nothing yet | [Ollama](#ollama) | 20 |

Every path ends the same way: `jeb serve` runs on `http://localhost:8100`. Anything that speaks AINode's
decision API or the Jev wire format can use it: JDE, your own code, a coding agent.

Everything below needs Python 3.10 or newer for the `jeb` command:

```bash
pip install "jebadiah-decide @ git+https://github.com/getainode/jebadiah#subdirectory=clients/python"
# once it is on PyPI: pip install jebadiah-decide
```

It installs `tokenizers`, `jinja2` and `huggingface-hub`. No torch, no transformers.

---

## Ollama

**1. Install Ollama** from [ollama.com/download](https://ollama.com/download) and open it once. Version
0.34 or newer.

**2. Install `jeb`** with the pip command above.

**3. Start Jeb.**

```bash
jeb serve
```

The first time, this pulls `hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0` into Ollama (9.8 GB), then:

```
Jeb is up: ollama (hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0), temperatures on, up to 20 options per question.
  POST http://localhost:8100/v1/systemone   (Jev wire: JDE's default judge, TypeSafe clients)
  POST http://localhost:8100/v1/decide      (AINode's decide shape)
  GET  http://localhost:8100/health
```

On a slow connection, start with the 4B instead (4.6 GB): `jeb serve --size 4b`.

**4. Your first decision.** In another terminal:

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "Customer says the invoice total does not match the quote."},
  "questions": {"route": {"type": "choice", "instructions": "Which team should take this ticket?",
    "criteria": {"billing": "an invoice, a charge or a refund", "support": "a product question",
                 "sales": "a quote or a renewal"}}}}'
```

```json
{"answers": {"route": {"type": "choice", "choice": "billing", "confidence": 0.461713,
  "probabilities": {"billing": 0.641142, "support": 0.036095, "sales": 0.322763}}}, ...}
```

Those are the exact numbers llama-server gives on the same file. If you want a check with a verdict,
`jeb doctor` runs the same example and tells you what's wrong if anything is.

**5. Point JDE at it** (or skip to [Integrate it](#integrate-it)). See [Use it from JDE](#use-it-from-jde).

**You're ready.** Try the [scenarios](#scenarios) with your own text.

Ollama gotchas:
- **Don't use `ollama run` or Ollama's chat window with Jeb.** They show text with thinking on, and Jeb
  isn't built to write text, so what you see there is nonsense by design. `jeb` sends the prompt raw with
  thinking off and reads the probabilities.
- 20 options at most per question: Ollama returns only the top 20 log probabilities. `jeb` refuses a wider
  question with a clear error rather than guess.

## LM Studio

**1. Install [LM Studio](https://lmstudio.ai)** (0.4 or newer). Open **Discover**, search for
`jebadiah-9b-v2`, and download the **Q8_0** from `frontier-infra/jebadiah-9b-v2-GGUF`.

**2. Load it and start the server.** Open the **Developer** tab, switch the server on, click **Select a model
to load** and pick Jebadiah. (Or in a terminal: `lms server start`, then `lms load jebadiah-9b-v2`.)

**3. Install `jeb`** with the pip command above, then:

```bash
jeb serve --backend lmstudio
```

It finds the loaded Jebadiah on its own. If there isn't one, it tells you exactly what to click. If
**Require Authentication** is on in LM Studio's server settings, create a token under **Manage Tokens** and
pass it: `jeb serve --backend lmstudio --api-key <token>`.

**4. First decision:** the same curl as the [Ollama path](#ollama), step 4. Expect billing 0.64114,
support 0.036056, sales 0.322804.

**5. Point JDE at it:** see [Use it from JDE](#use-it-from-jde).

**You're ready.**

LM Studio gotchas:
- **The chat window gives nonsense by design.** Chat runs with thinking on and shows text. `jeb` goes through
  the server with reasoning off (`"reasoning_effort": "none"`, the only switch LM Studio honours for this).
- **Ignore a "Vision" tag.** Some third-party quants ship a vision file from the Qwen base. Jeb wasn't trained
  on images, and the file isn't needed.
- 20 options at most, the same as Ollama.

## vLLM

**1. Serve the full weights.** `--language-model-only` skips the Qwen3.5 base's vision tower, which Jeb
doesn't use.

```bash
vllm serve frontier-infra/jebadiah-9b-v2 --max-model-len 4096 --language-model-only \
  --served-model-name frontier-infra/jebadiah-9b-v2
```

**2. Install `jeb`**, then:

```bash
jeb serve --backend vllm --url http://localhost:8000
```

**3. First decision:** the same curl as the [Ollama path](#ollama), step 4. On a DGX Spark in bf16 we got
billing 0.6279, support 0.0385, sales 0.3337; the small differences are the bf16 engine's rounding.

**4. Point JDE at it:** see [Use it from JDE](#use-it-from-jde). **You're ready.**

vLLM caps log probabilities at `--max-logprobs` (20 by default). Raise it and pass `--top-n` to `jeb` to go
past 20 options; we haven't run Jeb that way yet.

## llama.cpp

No option cap. This is the path every GGUF was checked on.

```bash
hf download frontier-infra/jebadiah-9b-v2-GGUF jebadiah-9b-v2-Q8_0.gguf --local-dir .
llama-server -m jebadiah-9b-v2-Q8_0.gguf -c 4096 -np 1 --port 8080
jeb serve --backend llama-server
```

llama.cpp v0.5.0 or newer (older builds don't know the `qwen35` architecture). First decision: the same curl
as step 4 of the [Ollama path](#ollama), with the same numbers. **You're ready.**

## MLX on a Mac

No other runtime needed, no option cap.

```bash
pip install "jebadiah-decide[mlx] @ git+https://github.com/getainode/jebadiah#subdirectory=clients/python"
jeb serve --backend mlx          # downloads frontier-infra/jebadiah-9b-v2-MLX (8-bit) the first time
```

First decision: the curl from the [Ollama path](#ollama). The 8-bit 9B gives billing 0.640554, support
0.037693, sales 0.321753. **You're ready.**

## AINode

[AINode](https://github.com/getainode/ainode) serves Jeb on vLLM and answers `/v1/systemone` and `/v1/decide`
itself, so you can point clients straight at the node. Load it once:

```bash
curl -s http://<node>:3000/api/models/load -H "Authorization: Bearer $AINODE_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model": "frontier-infra/jebadiah-9b-v2", "max_model_len": 4096,
  "gpu_memory_utilization": 0.3, "extra_vllm_args": ["--language-model-only"]}'
```

Then use `http://<node>:3000/v1/systemone` with `"model": "frontier-infra/jebadiah-9b-v2"` and your key. Set
`gpu_memory_utilization` for your card; 0.3 is what we run the 9B at on a 128 GB DGX Spark. `jeb doctor
--backend ainode --url http://<node>:3000 --api-key ...` checks it end to end.

---

## Use it from JDE

[JDE](https://github.com/Titanium-Devops/jde), the Jev Decision Engine, turns a judge's answers into actions
with policy bands and a ledger. Its default judge is a local Jeb at `http://localhost:8100/v1/systemone`,
which is exactly where `jeb serve` listens. So once `jeb serve` is running, JDE just works: no key, no config.

```ts
import { ask } from "@titanium/jde";

const outcome = await ask({ decision, state, questions });   // judged by the Jeb behind jeb serve
```

To point it somewhere else, set `JDE_JEB_ENDPOINT` (AINode: `http://<node>:3000/v1/systemone`, with
`JDE_JEB_API_KEY` for its key) and `JDE_JEB_MODEL`. If nothing answers, the decision takes its policy's
fallback and the error says to start a local Jeb; JDE never switches to a hosted service on its own.
TypeSafe's hosted Jev is still there if you ask for it (`"judge": "jev"` in the policy, with
`TYPESAFE_API_KEY`). JDE's shipped policy gives a judgment 2 seconds; on this Mac, Jeb 9B in Ollama answered
a whole completion check in about half a second one at a time.

Both of these were run as written against `jeb serve`:
[`examples/jde-tool-gate.mjs`](examples/jde-tool-gate.mjs) (the tool-call gate in the scenarios below) and
[`examples/jde-local-jeb.mjs`](examples/jde-local-jeb.mjs) (JDE's completion check over one of JDE's case
files; a wiring check, not a measurement).

**Judge Jeb.** A Jeb tuned on JDE's own judging questions is planned and not released yet (see
[What's next](../README.md#whats-next)). When it ships, JDE moves to it with the one `JDE_JEB_MODEL` setting.

---

## Scenarios

Five recipes. Each one is one request to `http://localhost:8100/v1/systemone`, with the answer Jeb 9B v2
gave through `jeb serve` on Ollama. The texts are made up. Change the text, keep the shape.

### 1. Route a ticket or message (choice)

```json
{"state": {"channel": "email", "message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate charge?"},
 "questions": {"team": {"type": "choice", "instructions": "Which team should handle this message?",
   "criteria": {"billing": "charges, invoices, refunds, payment methods",
                "technical": "bugs, errors, outages, how the product works",
                "account": "login, password, profile, cancelling the account",
                "sales": "pricing questions, upgrades, new purchases"}}}}
```

Answer: `billing` at 0.986 (technical 0.004, account 0.005, sales 0.005). `choice` is the pick and
`probabilities` has every option, so you can route above a threshold and send the rest to a person.

### 2. Phishing or spam (yes, no, unknown)

```json
{"state": {"from": "security@paypa1-support.com", "subject": "Urgent: your account is suspended",
           "body": "We detected unusual activity. Verify your identity within 24 hours at http://paypa1-verify.example/login or your account will be closed."},
 "questions": {"phishing": {"type": "choice", "instructions": "Is this email a phishing attempt?",
   "criteria": {"yes": "it tries to steal credentials, money or personal data by impersonation",
                "no": "a legitimate message",
                "unknown": "there is not enough in the email to tell"}}}}
```

Answer: `yes` at 0.939 (no 0.016, unknown 0.045). Giving "unknown" as an option lets the model say it can't
tell instead of forcing a guess. From the command line: `jeb ask "Is this a phishing email?" --text
email.txt --options yes,no,unknown`.

### 3. Does this reply answer the question? (yes or no, and a score)

```json
{"state": {"question": "How do I reset my password if I no longer have access to my email?",
           "reply": "You can reset your password from Settings > Security. Click 'Forgot password' and we will email you a link."},
 "questions": {"answers_it": {"type": "noul", "instructions": "The reply answers the question that was actually asked."},
               "quality": {"type": "score", "instructions": "How well does the reply answer the question?",
                 "criteria": ["does not answer it", "partly answers it", "fully answers it"]}}}
```

Answer: `answers_it` 0.067, and `quality` 0.40 on a 0 to 2 scale (does not answer 0.63, partly 0.33,
fully 0.04). The reply sends a link to the email the user just said they can't reach, so it doesn't
answer the question. A `noul` is the probability the statement is true. A `score` is the expected level,
with the probability of each level beside it.

### 4. Gate an agent's tool call (act, confirm or escalate) with JDE

The agent is about to call a tool. Jeb answers one question: does the call do only what the user asked?
JDE's bands turn the confidence into sure, lean or unsure, and code maps direction plus band to an action.
[`examples/jde-tool-gate.mjs`](examples/jde-tool-gate.mjs) is the whole thing. It printed:

```
act       P(in scope) 0.964  band sure    fs.rename for "Rename report_final.docx to report_2026.docx"
escalate  P(in scope) 0.021  band sure    fs.delete for "Rename report_final.docx to report_2026.docx"
act       P(in scope) 0.932  band sure    email.send for "Email the Q3 summary to the finance team"
confirm   P(in scope) 0.680  band unsure  flights.book for "Book me a flight to Denver next Tuesday"
```

The flight was booked in business class, which nobody asked for, so the agent asks first.

### 5. Content policy (yes or no, several rules at once)

```json
{"state": {"post": "If you show up at the meeting tomorrow I will make sure you regret it. I know where you park."},
 "questions": {"threat": {"type": "noul", "instructions": "The post threatens someone with harm."},
               "spam": {"type": "noul", "instructions": "The post is advertising or spam."}}}
```

Answer: `threat` 0.868, `spam` 0.030. One question per rule keeps each answer clean, and every question in a
request shares the same state.

---

## Integrate it

The same routing call in each language, with the same thresholds: act at 0.9 or more, confirm at 0.6 or more,
escalate below. Every snippet was run against `jeb serve`; they're in
[`examples/integrate/`](examples/integrate). For AINode, change the URL to `http://<node>:3000/v1/systemone`
and add `"model"` and your key.

<details open><summary><b>curl</b></summary>

```bash
curl -s http://localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?"},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
                          "account": "login, password, cancelling", "sales": "pricing, upgrades"}}}}'
```

The same question in AINode's `/v1/decide` shape:

```bash
curl -s http://localhost:8100/v1/decide -H 'Content-Type: application/json' -d '{
  "state": {"message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?"},
  "questions": {"team": {"question": "Which team should handle this message?",
                         "options": ["billing", "technical", "account", "sales"]}}}'
```

`/v1/decide` answers `{"decisions": {"team": {"answer": "billing", "confidence": 0.93, "distribution": {...}}}}`.
Prefer `/v1/systemone`: it's the shape Jeb was trained on, and it carries a description per option.

</details>

<details><summary><b>Python (requests)</b></summary>

```python
import requests

JEB = "http://localhost:8100/v1/systemone"
TEAMS = {"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
         "account": "login, password, cancelling", "sales": "pricing, upgrades"}


def route(message: str) -> tuple[str, str, float]:
    r = requests.post(JEB, timeout=30, json={
        "state": {"message": message},
        "questions": {"team": {"type": "choice", "instructions": "Which team should handle this message?",
                               "criteria": TEAMS}}})
    r.raise_for_status()
    answer = r.json()["answers"]["team"]
    team, p = answer["choice"], answer["probabilities"][answer["choice"]]
    if p >= 0.90:
        return "act", team, p          # route it automatically
    if p >= 0.60:
        return "confirm", team, p      # route it, and flag it for a quick human check
    return "escalate", team, p         # too close to call: a person picks the team
```

It printed `act billing 0.987` for the double charge, and `confirm technical 0.783` for "The export button does
nothing and I can't log in on the mobile app either" (two problems, two teams).

</details>

<details><summary><b>Python (the jebadiah-decide package, no server)</b></summary>

```python
from jebadiah_decide import Jeb

jeb = Jeb("ollama")      # Jeb("lmstudio"), Jeb("llama-server"), Jeb("mlx") work the same way
jeb.prepare()            # checks Ollama and pulls the model the first time
out = jeb.decide(
    {"message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?"},
    {"team": {"type": "choice", "instructions": "Which team should handle this message?",
              "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
                           "account": "login, password, cancelling", "sales": "pricing, upgrades"}}})
answer = out["answers"]["team"]
p = answer["probabilities"][answer["choice"]]
action = "act" if p >= 0.90 else "confirm" if p >= 0.60 else "escalate"
```

</details>

<details><summary><b>TypeScript / Node</b></summary>

```ts
const JEB = "http://localhost:8100/v1/systemone";

type ChoiceAnswer = { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> };
type Action = "act" | "confirm" | "escalate";

async function route(message: string): Promise<{ action: Action; team: string; p: number }> {
  const res = await fetch(JEB, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state: { message },
      questions: {
        team: {
          type: "choice",
          instructions: "Which team should handle this message?",
          criteria: { billing: "charges, invoices, refunds", technical: "bugs, errors, outages",
                      account: "login, password, cancelling", sales: "pricing, upgrades" },
        },
      },
    }),
  });
  if (!res.ok) throw new Error(`Jeb said ${res.status}: ${await res.text()}`);
  const answer = (await res.json()).answers.team as ChoiceAnswer;
  const p = answer.probabilities[answer.choice];
  const action: Action = p >= 0.9 ? "act" : p >= 0.6 ? "confirm" : "escalate";
  return { action, team: answer.choice, p };
}
```

Node 22 or newer runs the file directly (`node route.ts`). The same `fetch` works in a browser, since
`jeb serve` allows any origin on localhost.

</details>

<details><summary><b>Go</b></summary>

```go
// Options are shown to the model in the order you send them, and a Go map would sort them,
// so the criteria stay a JSON literal in a fixed order.
criteria := json.RawMessage(`{"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
	"account": "login, password, cancelling", "sales": "pricing, upgrades"}`)
body, _ := json.Marshal(map[string]any{
	"state": map[string]string{"message": message},
	"questions": map[string]any{"team": map[string]any{
		"type": "choice", "instructions": "Which team should handle this message?", "criteria": criteria}},
})
res, err := http.Post("http://localhost:8100/v1/systemone", "application/json", bytes.NewReader(body))
// decode {"answers": {"team": {"choice": ..., "probabilities": {...}}}}, then the same thresholds
```

The full program is [`examples/integrate/route.go`](examples/integrate/route.go). Option order matters: with
a Go map the options arrived sorted, and billing's probability moved from 0.987 to 0.969.

</details>

---

## Set it up with your AI

Paste this into Claude Code, Cursor, Copilot or any coding agent working in your project. It carries
everything the agent needs. We tried it on a scratch project: the agent installed `jeb`, ran doctor, started
the server, wired a help-desk urgency check into the code with thresholds, and had its tests passing against
the live server in about a minute (with the model already downloaded).

````text
Set up Jebadiah (Jeb), an open decision model, in this project, and wire it into our code.

What Jeb is: it answers typed questions about some text or JSON with a calibrated probability per option.
It never generates text. A local server, `jeb serve`, puts it on http://localhost:8100 with two endpoints:
- POST /v1/systemone (preferred), body:
  {"state": <any JSON or string>, "questions": {"<id>": {"type": "choice"|"noul"|"score", "instructions": "<question or statement>",
   "criteria": <choice: {"option": "description", ...} (2 to 20, order matters); noul: omit; score: ["level0", "level1", ...]>}}}
  Answers: choice -> {"choice", "confidence", "probabilities": {option: p}}; noul -> {"noul": P(statement is true)};
  score -> {"score": expected level index, "probabilities": {"0": p, ...}, "legend"}.
  Threshold on probabilities[choice] (for a noul, on noul for "true" or 1 - noul for "false"). The separate
  "confidence" field is chance-corrected and is not the one to threshold.
- POST /v1/decide (AINode's shape): {"state", "questions": {"<id>": {"question", "options": [...]} | {"question", "type": "boolean"}}}
  -> {"decisions": {"<id>": {"answer", "confidence", "distribution"}}}.
- GET /health answers 200 when the server is up.

Steps:
1. Which runtime do I use? If I haven't said: Ollama (default), LM Studio, llama.cpp's llama-server, vLLM,
   MLX (Mac) or AINode. If I have none, use Ollama (https://ollama.com/download).
2. Install the CLI: pip install "jebadiah-decide @ git+https://github.com/getainode/jebadiah#subdirectory=clients/python"
3. Run `jeb doctor --backend <runtime>` and fix whatever it reports (it says what to do). Then start
   `jeb serve --backend <runtime>` in the background (or its own terminal) and wait until GET /health returns 200.
   Ollama pulls the model on first run (9.8 GB; use --size 4b for 4.6 GB).
4. Find the decision in this codebase that I describe (or ask me for one), and write a small client in our
   language that calls /v1/systemone with a timeout and maps the probability to three actions, with the
   thresholds as named constants: act (>= 0.9: do it), confirm (>= 0.6: do it, but flag it or ask a person
   first) and escalate (below 0.6, or any HTTP or parse error: hand it to a person). Ask me what confirm
   and escalate should do in this app if it isn't obvious. Keep option order fixed and each question to 20
   options or fewer.
5. Add a test that stubs the HTTP call with a recorded Jeb response and checks the threshold mapping, and a
   second test, skipped unless JEB_URL is set, that sends one clear example of each outcome to the live
   server and checks the pick.
Rules: never use the runtime's chat window or /api/chat for Jeb; always go through jeb serve or the
jebadiah-decide package. Docs: https://github.com/getainode/jebadiah/blob/main/docs/run-locally.md
````

---

## How it works

Jeb reads the probability of each option label ("A", "B", ...) at the answer position. So a runtime has to
take the prompt exactly as AINode renders it, with thinking off, and return the log probabilities of the next
token. `jeb` does the rendering with the model's own tokenizer and chat template (a few MB from the model's
repository, cached), calls the runtime, reads the labels, and applies the model's `temperatures.json`.

Things that trip people up:
- **No chat window, anywhere.** Every runtime's chat UI runs with thinking on and shows text.
- **The prompt must be exact.** `jeb` checks the runtime's prompt token count against the local template on
  every call and stops if they differ.
- **20 options** on Ollama, LM Studio, AINode and default vLLM. `jeb` refuses a wider question. llama-server
  and MLX have no limit.
- **Option order matters.** Options are labelled A, B, C in the order you send them. Keep it fixed.
- **Temperatures.** Each size ships its own `temperatures.json`, fitted on held-out data; `--raw` turns them off.
- **Quantization.** Stay on Q8_0 or 8-bit unless memory forces you lower. Each model card says how many
  answers the smaller files change.
- **Long input.** On `/v1/systemone`, a state longer than 2,048 prompt tokens is cut from the end, the same as
  the published scripts, and the response says so in `warnings`.

### `jeb` commands

| Command | What it does |
|---|---|
| `jeb serve [--backend X] [--size 4b\|9b\|27b] [--port 8100]` | `/v1/systemone` and `/v1/decide` on localhost |
| `jeb doctor [--backend X]` | checks the runtime, the model, the tokenizer, the prompt and a known answer |
| `jeb ask "question" --text file.txt [--options a,b,c \| --levels low,mid,high]` | one answer, one line, plus JSON |
| `jeb request file.json` | answers a `{state, questions}` file |

Backends: `ollama` (default), `lmstudio`, `llama-server`, `vllm`, `mlx`, `ainode` / `systemone`. `--url`,
`--model` and `--api-key` override the defaults.

### Without `jeb`

Each model repository also carries one script per runtime (`scripts/decide_gguf.py`, `decide_ollama.py`,
`decide_lmstudio.py`, `decide_mlx.py`) that does the same for a single request file, and
[`server/`](../server) in this repository is a standalone server that runs the full weights with PyTorch.

### Models

| Model | GGUF | Size (Q8_0) | Same pick as bf16 on 260 held-out questions |
|---|---|---:|---:|
| Jebadiah 27B | [frontier-infra/jebadiah-27b-GGUF](https://huggingface.co/frontier-infra/jebadiah-27b-GGUF) | 29 GB | 260 |
| Jebadiah 9B v2 (default) | [frontier-infra/jebadiah-9b-v2-GGUF](https://huggingface.co/frontier-infra/jebadiah-9b-v2-GGUF) | 9.8 GB | 257 |
| Jebadiah 4B v2 | [frontier-infra/jebadiah-4b-v2-GGUF](https://huggingface.co/frontier-infra/jebadiah-4b-v2-GGUF) | 4.6 GB | 256 |

In Ollama, all three gave the same pick as llama-server on all 260. All sizes, formats and results are in the
[collection](https://huggingface.co/collections/frontier-infra/jebadiah-open-system-one-decision-models-6ab80765ddd3fa0b3eba5213)
and on [jebadiah.ai](https://jebadiah.ai).
