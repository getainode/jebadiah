"""Which model files go with which runtime, and what the known example should answer."""
from __future__ import annotations

SIZES = {"4b": "jebadiah-4b-v2", "9b": "jebadiah-9b-v2-1", "27b": "jebadiah-27b-v2-1"}
DEFAULT_SIZE = "9b"
ORG = "frontier-infra"
DEFAULT_PORT = 8100


def gguf_repo(size: str) -> str:
    return f"{ORG}/{SIZES[size]}-GGUF"


def full_repo(size: str) -> str:
    return f"{ORG}/{SIZES[size]}"


def mlx_repo(size: str, precision: str = "8bit") -> str:
    return f"{ORG}/{SIZES[size]}-MLX-{precision}"


def ollama_tag(size: str, quant: str = "Q8_0") -> str:
    return f"hf.co/{gguf_repo(size)}:{quant}"


def default_model(backend: str, size: str) -> str | None:
    """The name a runtime knows the model by, when there is one we can assume."""
    if backend == "ollama":
        return ollama_tag(size)
    if backend in ("vllm", "systemone"):
        return full_repo(size)
    if backend == "mlx":
        return mlx_repo(size)
    return None   # llama-server serves whatever it loaded; LM Studio's identifier is looked up


def default_repo(backend: str, size: str) -> str:
    """Where the tokenizer, chat template and temperatures.json come from (a few MB, cached)."""
    return full_repo(size) if backend == "vllm" else gguf_repo(size)


# example-request.json ("route": billing/support/sales, "urgent": noul) through llama-server on
# each release's Q8_0 (9B/27B v2.1, 4B v2), with the shipped temperatures. Every runtime lands within about 0.015 of these.
EXAMPLE = {
    "state": {"ticket": "Customer says the invoice total does not match the quote."},
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should take this ticket?",
                  "criteria": {"billing": "an invoice, a charge or a refund", "support": "a product question",
                               "sales": "a quote or a renewal"}},
        "urgent": {"type": "noul", "instructions": "The customer is blocked from working.",
                   "criteria": {"true": "work has stopped", "false": "it can wait"}}}}
EXPECTED = {
    "4b": {"billing": 0.598244, "urgent": 0.153519},
    "9b": {"billing": 0.565991, "urgent": 0.170974},
    "27b": {"billing": 0.761705, "urgent": 0.202962},
}
TOLERANCE = 0.03
