"""Governance models — re-exports from canonical modules.

The canonical Policy class lives in src.models.policy.  This module
re-exports it (along with related models) so that any legacy imports
from governance_models continue to work.
"""
from pydantic import BaseModel
from typing import Dict, Optional

# Re-export the canonical Policy to avoid duplicate definitions
from src.models.policy import Policy  # noqa: F401

class ChainConfig(BaseModel):
    rpc: Optional[str] = None
    chain_id: int = 31337
    private_key: Optional[str] = None
    addresses: Dict[str, str] = {}
