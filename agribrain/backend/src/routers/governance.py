# backend/src/routers/governance.py
"""Governance router for policy and blockchain configuration.

GET/POST /governance/policy — read/update the canonical Policy object.
GET/POST /governance/chain — read/update on-chain connection settings.

Policy changes propagate to app.py's state["policy"] (the single source of
truth for the decision engine).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import logging
import os
import json

logger = logging.getLogger(__name__)

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
    """If auto is enabled, read addresses JSON when it changes.

    Side effect: also surfaces the ``CHAIN_PRIVKEY`` env var into
    ``CHAIN["private_key"]`` so that downstream chain bridge functions
    (``src.chain.contracts._get_contract`` -> ``eth.Account.from_key``)
    can sign transactions. The env-var read happens unconditionally so a
    user who exports CHAIN_PRIVKEY mid-process picks up the new key on
    the next governance call without restarting the backend.

    Pre-2026-05 the env-var was documented in the route docstring and in
    ``.env.example`` but never actually plumbed into the in-process
    ``CHAIN`` dict; the bridge then silently returned ``None`` from
    ``_get_contract``, the DAO router returned ``ok=false`` with no
    surfaced error, and a live demo of propose -> vote -> queue ->
    execute would degenerate to a series of mysterious 200 OK no-ops.
    Surfaced here so the bridge actually has a signer to work with when
    the operator has set the env-var; if it is unset, the bridge still
    returns None and the routers' existing ``ok=false`` path covers the
    no-signer case visibly.
    """
    # Always pull CHAIN_PRIVKEY from env on every call -- this is cheap and
    # lets an operator export the key after the backend started without
    # forcing a restart.
    _env_key = os.environ.get("CHAIN_PRIVKEY", "")
    if _env_key:
        CHAIN["private_key"] = _env_key
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
                    # Propagate to app state so decide.py sees the same config
                    if _APP_STATE is not None:
                        _APP_STATE["chain"] = dict(CHAIN)
                    logger.info("auto-synced addresses from %s", p)
                    break
                except Exception as e:
                    logger.warning("auto-load failed: %s", e)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChainModel(BaseModel):
    """Chain configuration body for POST /chain.

    The `private_key` field was removed in 2026-04. Production keys must
    be supplied via the CHAIN_PRIVKEY env var, not the HTTP POST body —
    POST bodies can be captured by misconfigured proxies, and this
    endpoint is documented as plaintext (TLS termination is upstream).
    """
    rpc: Optional[str] = None
    chain_id: Optional[int] = None
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
    except (TypeError, ValueError) as exc:
        from fastapi import HTTPException
        raise HTTPException(400, f"Invalid policy parameters: {exc}")

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
    # Sync from app state: app-state values always win so that runtime
    # updates (e.g. from decide.py or lifespan) are reflected here.
    # Uses `is not None` instead of truthiness so that explicit empty
    # values (e.g. clearing addresses to {}) propagate correctly.
    if _APP_STATE and "chain" in _APP_STATE:
        app_chain = _APP_STATE["chain"]
        for k in ("rpc", "chain_id", "private_key", "addresses"):
            val = app_chain.get(k)
            if val is not None:
                CHAIN[k] = val

    addrs = CHAIN.get("addresses") or {}
    addresses_json = json.dumps(addrs, indent=2, ensure_ascii=False)
    pk = CHAIN.get("private_key") or ""
    return {
        "rpc": CHAIN.get("rpc"),
        "chain_id": CHAIN.get("chain_id"),
        "private_key_set": bool(pk),
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
        except (json.JSONDecodeError, ValueError):
            from fastapi import HTTPException
            raise HTTPException(400, "Invalid addresses_json: must be valid JSON")
    for k in ("rpc", "chain_id", "auto"):
        if k in data:
            CHAIN[k] = data[k]
    # private_key intentionally NOT accepted from the POST body. To
    # configure the signing key, set CHAIN_PRIVKEY in the runtime env
    # before starting the server. This change is part of the 2026-04
    # security hardening; see docs/REVIEWER_2_FIXES.md.
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


# ---------------------------------------------------------------------------
# Contract integration endpoints (SLCARewards, AgriDAO, PolicyStore, AgentRegistry)
# ---------------------------------------------------------------------------

class RewardSlashRequest(BaseModel):
    address: str
    amount: int

class ProposalRequest(BaseModel):
    description: str
    policy_key: str = ""
    policy_value: int = 0

class VoteRequest(BaseModel):
    proposal_id: int
    support: bool

class ExecuteRequest(BaseModel):
    proposal_id: int

class PolicyStoreSetRequest(BaseModel):
    key: str
    value: int

class PolicyStoreGetRequest(BaseModel):
    key: str

class AgentRegisterRequest(BaseModel):
    # ``account`` is the on-chain address of the agent being registered.
    # The Solidity contract's sponsorRegister / ownerRegister paths
    # both take this as the first arg; without it the wrapper had no
    # way to admit anyone other than the calling EOA. Added 2026-05
    # alongside the AgentRegistry ABI fix that aligned the backend
    # wrapper with the actual on-chain function signatures.
    account: str
    agent_id: str
    role: str
    meta: str = ""
    # method = "sponsorRegister" (production) or "ownerRegister"
    # (bootstrap-only, reverts unless caller is contract owner).
    method: str = "sponsorRegister"

class AgentActiveRequest(BaseModel):
    active: bool


def _parse_policy_key(value: str | None) -> bytes:
    """Accept bytes32 hex keys or hash plain-text keys for DAO proposals."""
    if not value:
        return b"\x00" * 32
    raw = value.strip()
    if raw.startswith("0x"):
        raw = raw[2:]
    if raw and all(c in "0123456789abcdefABCDEF" for c in raw):
        if len(raw) > 64:
            raise HTTPException(400, "policy_key hex too long (max 32 bytes)")
        try:
            return bytes.fromhex(raw.ljust(64, "0"))
        except ValueError as exc:
            raise HTTPException(400, f"Invalid policy_key hex: {exc}")
    try:
        from web3 import Web3
        return bytes(Web3.keccak(text=value))
    except Exception as exc:
        raise HTTPException(400, f"Invalid policy_key: {exc}")


@router.post("/contracts/slca-rewards/reward")
def contract_slca_reward(req: RewardSlashRequest):
    """Reward an agent with SLCA tokens on-chain."""
    from src.chain.contracts import slca_reward
    _try_autoload()
    txh = slca_reward(req.address, req.amount, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/slca-rewards/slash")
def contract_slca_slash(req: RewardSlashRequest):
    """Slash SLCA tokens from an agent on-chain."""
    from src.chain.contracts import slca_slash
    _try_autoload()
    txh = slca_slash(req.address, req.amount, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/dao/propose")
def contract_dao_propose(req: ProposalRequest):
    """Submit a governance proposal on-chain."""
    from src.chain.contracts import dao_propose
    _try_autoload()
    key_bytes = _parse_policy_key(req.policy_key)
    txh = dao_propose(req.description, key_bytes, req.policy_value, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/dao/vote")
def contract_dao_vote(req: VoteRequest):
    """Vote on a governance proposal on-chain."""
    from src.chain.contracts import dao_vote
    _try_autoload()
    txh = dao_vote(req.proposal_id, req.support, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/dao/execute")
def contract_dao_execute(req: ExecuteRequest):
    """Execute an approved governance proposal on-chain."""
    from src.chain.contracts import dao_execute
    _try_autoload()
    txh = dao_execute(req.proposal_id, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}


@router.post("/contracts/dao/finalize")
def contract_dao_finalize(req: ExecuteRequest):
    """Finalize voting outcome for a governance proposal on-chain."""
    from src.chain.contracts import dao_finalize
    _try_autoload()
    txh = dao_finalize(req.proposal_id, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}


@router.post("/contracts/dao/queue")
def contract_dao_queue(req: ExecuteRequest):
    """Queue a succeeded governance proposal for timelocked execution."""
    from src.chain.contracts import dao_queue
    _try_autoload()
    txh = dao_queue(req.proposal_id, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/policy-store/set")
def contract_policy_store_set(req: PolicyStoreSetRequest):
    """Store a policy parameter on-chain."""
    from src.chain.contracts import policy_store_set
    _try_autoload()
    txh = policy_store_set(req.key, req.value, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/policy-store/get")
def contract_policy_store_get(req: PolicyStoreGetRequest):
    """Read a policy parameter from on-chain storage."""
    from src.chain.contracts import policy_store_get
    _try_autoload()
    value = policy_store_get(req.key, CHAIN)
    return {"ok": value is not None, "key": req.key, "value": value}

@router.post("/contracts/agent-registry/register")
def contract_agent_register(req: AgentRegisterRequest):
    """Register an agent on-chain via sponsor or owner path.

    See ``agent_register`` for the sponsor/owner semantics.
    """
    from src.chain.contracts import agent_register
    _try_autoload()
    txh = agent_register(
        req.account, req.agent_id, req.role, req.meta, CHAIN,
        method=req.method,
    )
    return {"ok": txh is not None, "tx_hash": txh}

@router.post("/contracts/agent-registry/set-active")
def contract_agent_set_active(req: AgentActiveRequest):
    """Toggle an agent's active status on-chain."""
    from src.chain.contracts import agent_set_active
    _try_autoload()
    txh = agent_set_active(req.active, CHAIN)
    return {"ok": txh is not None, "tx_hash": txh}
