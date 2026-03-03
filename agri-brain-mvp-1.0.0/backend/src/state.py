
from typing import Dict, Any
from dataclasses import dataclass, field
from threading import RLock
from time import time

_lock = RLock()

@dataclass
class GlobalState:
    policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_shelf_reroute": 0.70,
        "min_shelf_expedite": 0.50,
        "carbon_factors": {"transport": 0.12, "cold_chain": 0.08},
        "distances_km": {"farm_to_dc": 180, "dc_to_retail": 220},
        "weights": {"carbon": 0.35, "labor_fairness": 0.25, "community_resilience": 0.20, "price_transparency": 0.20}
    })
    chain: Dict[str, Any] = field(default_factory=lambda: {
        "rpc": "http://127.0.0.1:8545",
        "chain_id": 31337,
        "private_key": "",
        "addresses": {"AGRIValidator": ""}
    })
    scenario: str = "baseline"
    audit_logs: list = field(default_factory=list)

STATE = GlobalState()

def add_audit(entry: Dict[str, Any]):
    with _lock:
        STATE.audit_logs.append({"ts": time(), **entry})

def get_audit():
    with _lock:
        return list(STATE.audit_logs)

def get_policy():
    with _lock:
        return dict(STATE.policy)

def set_policy(p: Dict[str, Any]):
    with _lock:
        STATE.policy.update(p)

def get_chain():
    with _lock:
        return dict(STATE.chain)

def set_chain(c: Dict[str, Any]):
    with _lock:
        for k, v in c.items():
            if isinstance(v, dict) and isinstance(STATE.chain.get(k), dict):
                STATE.chain[k].update(v)
            else:
                STATE.chain[k] = v

def get_scenario() -> str:
    with _lock:
        return STATE.scenario

def set_scenario(name: str):
    with _lock:
        STATE.scenario = name
