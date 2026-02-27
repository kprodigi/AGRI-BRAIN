# backend/src/routers/governance.py
"""Governance router for policy and blockchain configuration.

GET/POST /governance/policy — read/update the canonical Policy object.
GET/POST /governance/chain — read/update on-chain connection settings.

Policy changes propagate to app.py's state["policy"] (the single source of
truth for the decision engine).
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import os, json

router = APIRouter()

# ---------------------------------------------------------------------------
# Reference to the app-level state dict (set at startup via register_app_state)
# ---------------------------------------------------------------------------
_APP_STATE: Optional[Dict[str, Any]] = None


def register_app_state(st: Dict[str, Any]) -> None:
    """Called once by app.py at startup so governance can read/write policy."""
    global _APP_STATE
    _APP_STATE = st


# ---------------------------------------------------------------------------
# CHAIN keeps on-chain connection + deployed addresses
# ---------------------------------------------------------------------------
CHAIN: Dict[str, Any] = {
    "rpc": "http://127.0.0.1:8545",
    "chain_id": 31337,
    "private_key": "",
    "addresses": {},
    "auto": True,
}

_AUTO_STATE = {"last_mtime": 0.0}


def _auto_base_dir() -> Path:
    """Where deploy.js writes addresses: backend/runtime/chain/."""
    env_dir = os.environ.get("CHAIN_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[2] / "runtime" / "chain"


def _auto_paths():
    base = _auto_base_dir()
    return [
        base / "deployed-addresses.latest.json",
        base / "deployed-addresses.localhost.json",
    ]


def _try_autoload():
    """If auto is enabled, read addresses JSON when it changes."""
    if not CHAIN.get("auto", True):
        return
    for p in _auto_paths():
        if p.exists():
            mt = p.stat().st_mtime
            if mt > _AUTO_STATE["last_mtime"]:
                try:
                    raw = json.loads(p.read_text())
                    addrs = raw.get("addresses") if isinstance(raw, dict) else None
                    CHAIN["addresses"] = addrs if addrs else raw
                    _AUTO_STATE["last_mtime"] = mt
                    CHAIN.setdefault("rpc", "http://127.0.0.1:8545")
                    CHAIN.setdefault("chain_id", 31337)
                    print(f"[governance] auto-synced addresses from {p}")
                    break
                except Exception as e:
                    print("[governance] auto-load failed:", e)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChainModel(BaseModel):
    rpc: Optional[str] = None
    chain_id: Optional[int] = None
    private_key: Optional[str] = None
    addresses: Optional[dict] = None
    addresses_json: Optional[str] = None
    auto: Optional[bool] = None


# ---------------------------------------------------------------------------
# Policy endpoints — delegate to canonical Policy object in app state
# ---------------------------------------------------------------------------
@router.get("/policy")
def get_policy():
    """Return the canonical Policy object from app state."""
    if _APP_STATE and "policy" in _APP_STATE:
        p = _APP_STATE["policy"]
        if hasattr(p, "model_dump"):
            return p.model_dump()
    # Fallback: return defaults
    from src.models.policy import Policy
    return Policy().model_dump()


@router.post("/policy")
def set_policy(payload: Dict[str, Any]):
    """Update the canonical Policy object with provided fields.

    Accepts partial updates: only provided fields are changed.
    Supports legacy shapes (carbon_factors nested dict).
    """
    from src.models.policy import Policy

    if _APP_STATE and "policy" in _APP_STATE:
        current = _APP_STATE["policy"]
        current_dict = current.model_dump() if hasattr(current, "model_dump") else {}
    else:
        current_dict = Policy().model_dump()

    # Handle legacy carbon_factors nested shape
    cf = (payload or {}).get("carbon_factors")
    if isinstance(cf, dict):
        if "transport" in cf:
            payload["carbon_per_km"] = float(cf["transport"])
        if "cold_chain" in cf:
            pass  # no direct mapping needed

    # Merge: current values + provided overrides
    merged = {**current_dict, **{k: v for k, v in (payload or {}).items()
                                  if k in current_dict}}

    try:
        new_policy = Policy(**merged)
    except Exception:
        # If validation fails, return current policy
        return current_dict

    if _APP_STATE is not None:
        _APP_STATE["policy"] = new_policy

    return new_policy.model_dump()


# ---------------------------------------------------------------------------
# Chain endpoints
# ---------------------------------------------------------------------------
@router.get("/chain")
def get_chain():
    """Return blockchain configuration with auto-synced addresses."""
    _try_autoload()
    # Also sync from app state if available
    if _APP_STATE and "chain" in _APP_STATE:
        app_chain = _APP_STATE["chain"]
        for k in ("rpc", "chain_id", "private_key"):
            if app_chain.get(k):
                CHAIN.setdefault(k, app_chain[k])

    addrs = CHAIN.get("addresses") or {}
    addresses_json = json.dumps(addrs, indent=2, ensure_ascii=False)
    return {
        "rpc": CHAIN.get("rpc"),
        "chain_id": CHAIN.get("chain_id"),
        "private_key": CHAIN.get("private_key") or "",
        "auto": CHAIN.get("auto", True),
        "addresses": addrs,
        "addresses_json": addresses_json,
    }


@router.post("/chain")
def set_chain(c: ChainModel):
    """Update blockchain configuration."""
    data = {k: v for k, v in c.model_dump().items() if v is not None}
    if "addresses_json" in data and isinstance(data["addresses_json"], str):
        try:
            data["addresses"] = json.loads(data["addresses_json"])
        except Exception:
            pass
    for k in ("rpc", "chain_id", "private_key", "auto"):
        if k in data:
            CHAIN[k] = data[k]
    if "addresses" in data and isinstance(data["addresses"], dict):
        CHAIN["addresses"] = data["addresses"]

    # Propagate to app state
    if _APP_STATE is not None:
        _APP_STATE["chain"] = {
            "rpc": CHAIN.get("rpc"),
            "chain_id": CHAIN.get("chain_id"),
            "private_key": CHAIN.get("private_key"),
            "addresses": CHAIN.get("addresses", {}),
        }

    return get_chain()
