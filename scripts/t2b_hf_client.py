# SPDX-License-Identifier: Apache-2.0
"""Use verified HTTPS over IPv4 for local HF control when IPv6 has no route."""
import httpx
from huggingface_hub import set_client_factory


def configure_hf_client():
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(local_address='0.0.0.0'),
        timeout=httpx.Timeout(60, connect=10), follow_redirects=True))
