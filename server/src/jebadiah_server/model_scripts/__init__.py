"""The model repository's own scripts, shipped unchanged.

ainode_prompt_verbatim.py, jebadiah_prompt.py and jebadiah_model.py are byte-for-byte the files
under scripts/ in frontier-infra/jebadiah-9b-v2 (and every v2 repo carries the same three). They
import each other by bare module name, so their directory goes on sys.path once, here, and the
rest of the server imports them through this package.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ainode_prompt_verbatim as verbatim  # noqa: E402
import jebadiah_prompt as prompt  # noqa: E402

from ainode_prompt_verbatim import (  # noqa: E402
    MAX_CRITERIA,
    MAX_OPTIONS,
    PROMPT_SOURCE_COMMIT,
    PROMPT_SOURCE_SHA256,
    TOP_LOGPROBS,
    DecideError,
    build_messages,
    option_label,
    option_labels,
    serialize_state,
    translate_questions,
)

SCRIPT_FILES = ("ainode_prompt_verbatim.py", "jebadiah_prompt.py", "jebadiah_model.py")


def model_module():
    """jebadiah_model imports torch and transformers, so it loads only when a model does."""
    import jebadiah_model
    return jebadiah_model
