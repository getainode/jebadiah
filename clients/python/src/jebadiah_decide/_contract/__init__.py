"""The prompt contract, shipped unchanged.

ainode_prompt_verbatim.py and jebadiah_prompt.py are byte-for-byte the files under scripts/ in
every Jebadiah v2 model repository (and under server/src/jebadiah_server/model_scripts/ in this
repository; tests/test_contract.py checks that). They import each other by bare module name, so
their directory goes on sys.path once, here.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ainode_prompt_verbatim import (  # noqa: E402
    MAX_CRITERIA,
    PROMPT_SOURCE_COMMIT,
    PROMPT_SOURCE_SHA256,
    DecideError,
    build_messages,
    option_label,
)
from jebadiah_prompt import (  # noqa: E402
    CHAT_TEMPLATE_KWARGS,
    Rendered,
    Renderer,
    answer_from_probs,
    wire_keys,
)

CONTRACT_FILES = ("ainode_prompt_verbatim.py", "jebadiah_prompt.py")

__all__ = [
    "CHAT_TEMPLATE_KWARGS", "CONTRACT_FILES", "DecideError", "MAX_CRITERIA", "PROMPT_SOURCE_COMMIT",
    "PROMPT_SOURCE_SHA256", "Rendered", "Renderer", "answer_from_probs", "build_messages",
    "option_label", "wire_keys",
]
