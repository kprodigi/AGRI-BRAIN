"""Governance models -- re-exports from canonical modules.

The canonical Policy class lives in src.models.policy. This module
re-exports it (along with related models) so that any legacy imports
from governance_models continue to work.
"""
from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional

# Re-export the canonical Policy to avoid duplicate definitions
from src.models.policy import Policy  # noqa: F401


class ChainConfig(BaseModel):
    """Chain configuration accepted by ``POST /chain/config``.

    The ``private_key`` field was removed in 2026-04 from the request
    body. Production deployments must supply the signing key via the
    ``CHAIN_PRIVKEY`` env var instead -- POST bodies can be captured by
    misconfigured proxies, surfaced in access logs, or replayed in
    developer screenshots, and this endpoint is documented as
    plaintext (TLS termination is upstream).

    Submitting an unknown field (including ``private_key``) raises a
    422 ``ValidationError`` thanks to ``extra="forbid"``, so old
    clients that still POST the key get a clear failure pointing at
    the env var rather than a silent ingest. The 2026-05 follow-up
    keeps the schema explicit (no implicit fallthrough).
    """

    model_config = ConfigDict(extra="forbid")

    rpc: Optional[str] = None
    chain_id: int = 31337
    addresses: Dict[str, str] = {}
