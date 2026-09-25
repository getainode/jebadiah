"""jebadiah-serve: one Jebadiah model behind /v1/systemone and /v1/decide, with a playground."""
from __future__ import annotations

import argparse
import logging
import os
import threading

from jebadiah_server.engine import DEFAULT_MODEL, Engine


def env(name: str, default=None):
    return os.environ.get(f"JEBADIAH_{name}", default)


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="jebadiah-serve", description=__doc__)
    ap.add_argument("--host", default=env("HOST", "127.0.0.1"),
                    help="bind address; 0.0.0.0 to serve the LAN (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=int(env("PORT", "8000")))
    ap.add_argument("--model", default=env("MODEL", DEFAULT_MODEL),
                    help=f"a Jebadiah repo id or a local directory of merged weights (default {DEFAULT_MODEL})")
    ap.add_argument("--revision", default=env("REVISION"), help="Hub revision (branch, tag or commit)")
    ap.add_argument("--device", default=env("DEVICE", "auto"),
                    help="auto (cuda, then mps, then cpu), cuda, cuda:1, mps or cpu")
    ap.add_argument("--dtype", default=env("DTYPE", "auto"),
                    choices=["auto", "bfloat16", "float16", "float32"],
                    help="auto is bfloat16 on cuda and mps, float32 on cpu")
    ap.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", "8")),
                    help="questions per forward pass (default 8)")
    ap.add_argument("--max-prompt-tokens", type=int, default=int(env("MAX_PROMPT_TOKENS", "8192")),
                    help="longest prompt accepted; longer is refused, never cut (default 8192)")
    ap.add_argument("--alias", action="append", default=None,
                    help="extra model name accepted in a request's 'model' field (repeatable; "
                         "default: jebadiah)")
    ap.add_argument("--allow-contract-mismatch", action="store_true",
                    help="serve even when the model's prompt_contract.json disagrees with the renderer")
    ap.add_argument("--log-level", default="info")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    from jebadiah_server.app import create_app

    engine = Engine(model_id=a.model, revision=a.revision, device=a.device, dtype=a.dtype,
                    batch_size=a.batch_size, max_prompt_tokens=a.max_prompt_tokens,
                    strict_contract=not a.allow_contract_mismatch)
    app = create_app(engine, aliases=tuple(a.alias or ("jebadiah",)))
    # The model loads beside the server, so /health and the playground answer while it does.
    threading.Thread(target=engine.load, name="load-model", daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level)


if __name__ == "__main__":
    main()
