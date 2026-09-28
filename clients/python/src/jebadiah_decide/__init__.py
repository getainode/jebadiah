"""Typed decisions from a Jebadiah model on your own runtime: llama-server, Ollama, LM Studio,
vLLM, MLX, or any /v1/systemone server (AINode, jebadiah-serve). One request format for all."""
from .backends import BACKENDS, JebError
from .client import Jeb, decide

__version__ = "0.2.0"
__all__ = ["BACKENDS", "Jeb", "JebError", "decide", "__version__"]
