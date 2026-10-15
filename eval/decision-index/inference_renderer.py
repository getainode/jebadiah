"""Inference-only renderer options applied before the model's chat template."""


def repeat_renderer(jp, tokenizer, max_tokens, repeat_payload=False):
    """Extend the loaded model's renderer, retaining its translation and label mapping."""
    class InferenceRenderer(jp.Renderer):
        def __init__(self, tokenizer, max_tokens, repeat_payload=False):
            self.repeat_payload = repeat_payload
            super().__init__(tokenizer, max_tokens)

        def render_messages(self, state_text, question, options, letters=None):
            if not self.repeat_payload:
                return super().render_messages(state_text, question, options, letters)
            messages = jp.build_messages(state_text, None, question, options)
            if letters is not None:
                # Match the published renderer's extended-label assignment before repeating.
                head, _, rest = messages[1]["content"].partition("\nOPTIONS:\n")
                lines = rest.split("\n")
                for i in range(len(options)):
                    label = jp.option_label(i)
                    assert lines[i].startswith(f"{label}. ")
                    lines[i] = f"{letters[i]}. " + lines[i][len(label) + 2:]
                messages[1]["content"] = head + "\nOPTIONS:\n" + "\n".join(lines)
            payload = messages[1]["content"]
            messages[1]["content"] = payload + "\n\n" + payload
            return self.tok.apply_chat_template(messages, tokenize=False, **jp.CHAT_TEMPLATE_KWARGS)

    return InferenceRenderer(tokenizer, max_tokens, repeat_payload)
