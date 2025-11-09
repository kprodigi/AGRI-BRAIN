# backend/src/routers/decide.py
from fastapi import APIRouter
from pydantic import BaseModel
from time import time
import random

# --- Optional shared state for PDF and others
try:
    from src.routers.case import STATE as CASE_STATE
except Exception:
    CASE_STATE = {}

# --- Optional scenario state (Admin → Scenarios)
try:
    from src.routers.scenarios import ACTIVE as SCENARIO_ACTIVE
except Exception:
    SCENARIO_ACTIVE = {"name": None, "intensity": 1.0}

# --- Optional chain config (Admin → Blockchain)
try:
    from src.routers.governance import CHAIN as CHAIN_CFG   # should be a dict
except Exception:
    CHAIN_CFG = {}

router = APIRouter()


# ---------- Request / Response models ----------
class DecideRequest(BaseModel):
    agent: str = "farm"
    role: str = ""  # farm / processor / distributor / retail ...

    # Optional knobs used by the QuickDecision panel (safe to ignore)
    inventory_units: float | None = None
    demand_units: float | None = None
    temp_c: float | None = None
    volatility: float | None = None


class DecisionMemo(BaseModel):
    # Core
    agent: str
    role: str = ""
    action: str
    slca_score: float
    carbon_kg: float
    reason: str = ""
    tx_hash: str = "0x0"
    ts: int

    # Extras often shown in the PDF
    shelf_left: float = 0.0
    volatility: float = 0.0
    km: float = 0.0
    unit_price: float = 0.0


# ---------- In-memory log ----------
DECISIONS: list[DecisionMemo] = []
LAST: DecisionMemo | None = None


def _to_dict(m: DecisionMemo) -> dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


def _compat_from_memo(m: DecisionMemo) -> dict:
    """Map to the exact keys the PDF expects."""
    d = _to_dict(m)
    return {
        "time":       d.get("ts"),
        "agent":      d.get("agent") or "",
        "role":       d.get("role") or "",
        "decision":   d.get("action") or d.get("decision") or "",
        "shelf_left": d.get("shelf_left") or 0.0,
        "volatility": d.get("volatility") or 0.0,
        "km":         d.get("km") or 0.0,
        "carbon_kg":  d.get("carbon_kg") or d.get("carbon") or 0.0,
        "unit_price": d.get("unit_price") or 0.0,
        "slca":       d.get("slca_score") or d.get("slca") or 0.0,
        "tx":         d.get("tx_hash") or d.get("tx") or "",
        "note":       d.get("reason") or d.get("note") or "",
    }


def _persist_last(memo: DecisionMemo) -> None:
    DECISIONS.append(memo)
    globals()["LAST"] = memo
    try:
        CASE_STATE["last_decision"] = _compat_from_memo(memo)
    except Exception:
        pass


# ---------- Core decision logic ----------
def _base_scores(req: DecideRequest) -> tuple[float, float, float, float]:
    """Produce baseline SLCA/carbon and extras, then adjust with optional inputs."""
    # Baseline demo
    slca = 0.6 + 0.4 * random.random()           # 0.60–1.00
    carbon = 0.5 + 10.0 * random.random()        # 0.5–10.5 kg
    shelf_left = 0.5 + 0.5 * random.random()     # 50–100%
    volatility = 0.0 + 0.3 * random.random()     # 0–0.3

    # Optional user inputs can nudge the baseline a bit
    if req.temp_c is not None:
        # Warmer temp → slightly lower SLCA, slightly higher carbon
        delta = max(0.0, req.temp_c - 4.0) * 0.02
        slca = max(0.0, slca - delta)
        carbon += delta * 3

    if req.volatility is not None:
        v = max(0.0, min(1.0, req.volatility))
        slca = max(0.0, slca - 0.15 * v)
        volatility = max(volatility, v)

    if req.inventory_units is not None and req.demand_units is not None:
        # If inventory >> demand, routing toward redistribution may increase SLCA
        ratio = (req.demand_units + 1e-6) / (req.inventory_units + 1e-6)
        if ratio < 0.9:
            slca = min(1.0, slca + 0.05)
        else:
            slca = max(0.0, slca - 0.02)

    return round(slca, 3), round(carbon, 2), round(shelf_left, 2), round(volatility, 2)


def _apply_scenario(action: str,
                    slca: float,
                    carbon: float,
                    shelf_left: float,
                    vol: float) -> tuple[str, float, float, float, float, str]:
    """Adjust outputs given the active scenario."""
    scn = SCENARIO_ACTIVE or {}
    name = scn.get("name")
    intensity = float(scn.get("intensity") or 1.0)

    reason_bits: list[str] = []
    if name == "climate_shock":
        # Heatwave → spoilage risk; reroute nearer DC; more volatility & carbon
        slca = max(0.0, round(slca - 0.15 * intensity, 3))
        carbon = round(carbon + 1.2 * intensity, 2)
        shelf_left = max(0.0, round(shelf_left - 0.10 * intensity, 2))
        vol = min(1.0, round(vol + 0.15 * intensity, 2))
        action = "reroute_to_near_dc"
        reason_bits.append("climate_shock")

    elif name == "reverse_logistics":
        # Promote redistribution/recovery before thresholds are crossed
        slca = min(1.0, round(slca + 0.06 * intensity, 3))
        action = "redistribute_or_recover"
        reason_bits.append("reverse_logistics")

    elif name == "cyber_outage":
        # Node outage → local redistribution; slight SLCA drop
        slca = max(0.0, round(slca - 0.10 * intensity, 3))
        action = "local_redistribution"
        reason_bits.append("cyber_outage")

    elif name == "adaptive_pricing":
        # Learned pricing → equity-aware redistribution when saturated
        slca = min(1.0, round(slca + 0.08 * intensity, 3))
        action = "price_adjusted_route"
        reason_bits.append("adaptive_pricing")

    reason = "Policy+SLCA demo"
    if reason_bits:
        reason += " | " + ",".join(reason_bits)

    return action, slca, carbon, shelf_left, vol, reason


# ---------- Routes ----------
@router.post("/decide", response_model=DecisionMemo)
def decide(req: DecideRequest):
    # 1) Baseline scores (with optional QuickDecision nudges)
    slca, carbon, shelf_left, vol = _base_scores(req)

    # 2) Default action from baseline
    action = "reroute_to_near_dc" if slca < 0.7 else "standard_cold_chain"

    # 3) Scenario adjustments (if any active)
    action, slca, carbon, shelf_left, vol, reason = _apply_scenario(
        action, slca, carbon, shelf_left, vol
    )

    # Distance & price for the PDF (demo)
    km = round(5 + 100 * random.random(), 1)
    unit_price = round(1.0 + 4.0 * random.random(), 2)

    memo = DecisionMemo(
        agent=req.agent,
        role=req.role,
        action=action,
        slca_score=slca,
        carbon_kg=carbon,
        reason=reason,
        tx_hash="0x0",
        ts=int(time()),
        shelf_left=shelf_left,
        volatility=vol,
        km=km,
        unit_price=unit_price,
    )

    # Persist in memory + compat state
    _persist_last(memo)

    # 4) Optional: log to blockchain (no-op if not configured)
    try:
        from src.chain.eth import log_decision_onchain  # lazy import to keep deps optional
        txh = log_decision_onchain(_to_dict(memo), CHAIN_CFG or {})
        if txh:
            memo.tx_hash = txh
            # also update compat dict for PDF
            try:
                CASE_STATE["last_decision"]["tx"] = txh
            except Exception:
                pass
    except Exception as e:
        # Keep failures silent so the main flow never breaks
        print("[chain] log_decision_onchain failed:", e)

    return memo


# GET alias for quick testing in a browser
@router.get("/decide", response_model=DecisionMemo)
def decide_get(agent: str = "farm", role: str = ""):
    return decide(DecideRequest(agent=agent, role=role))


# Legacy/compat endpoint some frontends call
@router.api_route("/decision/take", methods=["GET", "POST"])
def decision_take(agent: str = "farm", role: str = ""):
    return decide(DecideRequest(agent=agent, role=role))


# Public read used by the PDF (compatibility dict)
@router.get("/last-decision", response_model=dict)
def last_decision():
    try:
        data = CASE_STATE.get("last_decision")
        if data:
            return data
    except Exception:
        pass
    if LAST:
        return _compat_from_memo(LAST)
    return {}


# Optional: recent memos feed
@router.get("/decisions", response_model=list[dict])
def decisions_feed():
    return [_to_dict(m) for m in reversed(DECISIONS[-50:])]
