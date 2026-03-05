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
    rpc: str = "http://127.0.0.1:8545"
    chain_id: int = 31337
    private_key: Optional[str] = ""
    addresses: Dict[str, str] = {"AGRIValidator": ""}

class DecisionMemo(BaseModel):
    action: str
    slca_score: float
    carbon_kg: float
    route: str
    reason: str
    tx_hash: str = "0x0"
