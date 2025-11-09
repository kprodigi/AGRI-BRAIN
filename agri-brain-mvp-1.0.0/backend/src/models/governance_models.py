
from pydantic import BaseModel, Field
from typing import Dict, Optional

class Policy(BaseModel):
    min_shelf_reroute: float = Field(0.70, ge=0, le=1)
    min_shelf_expedite: float = Field(0.50, ge=0, le=1)
    carbon_factors: Dict[str, float] = {"transport": 0.12, "cold_chain": 0.08}
    distances_km: Dict[str, float] = {"farm_to_dc": 180.0, "dc_to_retail": 220.0}
    weights: Dict[str, float] = {"carbon": 0.35, "labor_fairness": 0.25, "community_resilience": 0.20, "price_transparency": 0.20}

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
