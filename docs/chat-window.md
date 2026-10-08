# Jeb in a chat window

Jeb scores option labels at the end of AINode's decision prompt. An ordinary chat
with thinking on can spend its entire response on reasoning and show no answer.
These optional profiles make a first chat return setup guidance. For decisions
and calibrated probabilities, follow [Run Jeb locally](run-locally.md).

The profiles are opt-in. They do not replace the GGUF's embedded template or the
model repository's `chat_template.jinja`. Keep using the documented model and
`jeb serve` / `jeb doctor` commands for decisions.

## Ollama

From this checkout:

```sh
ollama create jeb-chat -f configs/chat/jeb-chat.Modelfile
ollama run jeb-chat "Hello! How do I use Jeb?"
```

The alias uses the 9B Q4_K_M file (5.8 GB) and a separate system prompt and Go
template. It closes an empty thinking block before the answer, including when a
client asks for thinking. It also retains reasoning and tool history formatting:
Ollama 0.34.4 otherwise prefers the more capable embedded Jinja template and
silently ignores a simpler Go template. No model weights are modified.

The tested response is:

```text
I am Jeb, a decision model. Use jeb doctor and jeb serve for decisions. Guide: https://github.com/getainode/jebadiah/blob/main/docs/run-locally.md
```

Use `jeb-chat` only for this guidance. Continue using the existing
`hf.co/frontier-infra/jebadiah-9b-v2-GGUF:Q8_0` tag for the documented decisions.
Q4_K_M can change decisions and probabilities compared with Q8_0. A client that
replaces the alias's system prompt also replaces its guidance instruction.

On another Ollama version, check `ollama show jeb-chat --template`: it should
start with `{{ if .System }}`, not `{%- set image_count`. If the embedded Jinja
is selected, explicitly disable thinking:

```sh
ollama run jeb-chat --think=false "Hello! How do I use Jeb?"
```

For `/api/chat`, use top-level `"think": false`. Do not change a shared Ollama
server's environment or the GGUF metadata to enable this profile.

## LM Studio

The conversation preset contains a system prompt, thinking off, temperature 0,
and a 96-token response limit. It contains no model load settings or template
override. Use it in a fresh chat, not as a model default or Developer server
configuration.

1. Download and load Jeb in LM Studio.
2. Open a new chat and expand the Model Parameters sidebar.
3. Open the preset dropdown, choose **Import Preset**, then import from a file
   and select [`jeb-chat.preset.json`](../configs/chat/jeb-chat.preset.json).
   LM Studio 0.3.15 or newer supports preset imports.
4. Select **Jeb chat guidance** in that conversation's preset dropdown. Confirm
   thinking is off and send `Hello! How do I use Jeb?`.

Alternatively, place the file in LM Studio's preset directory, then select it
in the fresh conversation:

```sh
# macOS and Linux; do not overwrite an existing personal preset.
cp configs/chat/jeb-chat.preset.json ~/.lmstudio/config-presets/jeb-chat.preset.json
```

On Windows the directory is `%USERPROFILE%\.lmstudio\config-presets`.

The equivalent system prompt and reasoning setting were tested through `lms
chat` on LM Studio 0.4.21+2 with the 9B Q4_K_M file. Exact preset application in
the GUI and Developer API isolation still need verification: the test machine's
Developer API requires an app-managed token unavailable through Bitwarden, and
its shared desktop was left alone. The preset has not been proved to produce
identical API decisions while selected; keep it confined to a separate chat.

## Compatibility evidence

[The live report](chat-window-validation.md) records the path trace, actual
before/after transcripts, and the token identity checks. The 4B and 27B profiles
have not been tested. Ollaya runs the authors' GGUF files on llama.cpp and uses
the embedded template for chat; these opt-in profiles do not change Ollaya.

## Paragraph for the model card

Jeb is a decision model, not a general chat assistant. For typed answers and
calibrated probabilities, use `jeb doctor` and `jeb serve`, following the
[local setup guide](https://github.com/getainode/jebadiah/blob/main/docs/run-locally.md).
Optional [chat guidance profiles](https://github.com/getainode/jebadiah/blob/main/docs/chat-window.md)
provide setup instructions in Ollama and LM Studio without replacing the
decision template. The LM Studio preset still needs GUI and API validation.
