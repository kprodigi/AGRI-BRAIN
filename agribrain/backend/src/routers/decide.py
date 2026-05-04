# backend/src/routers/decide.py
"""
Decision router -- compatibility shim around the canonical implementation
in :mod:`src.app`.

The 2026-05 consolidation removed a parallel standalone implementation
that re-derived feature vectors, SLCA composites, the waste model and
the reward decomposition. Maintaining two policy paths in lock-step
across feature-vector changes (the phi_6..phi_9 supply/demand
extension, the SLCA stress-attenuation refit, the route_rho_factor
update) was a perpetual source of subtle behavioural drift; the
duplicated path now delegates to :func:`src.app.decide` so both REST
entry points share one source of truth.

The thin Pydantic request model is still exported so the legacy
``compat`` router can keep coercing query/body payloads into the
canonical request shape, but the request is forwarded to ``app.decide``
verbatim. If you find yourself writing logic *here* instead of in
``app.py``, the policy path needs to grow there -- this module should
stay a translation layer.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------- Request / Response models ----------
class DecideRequest(BaseModel):
    agent: str = "farm"
    role: str = ""
    step: int | None = None
    deterministic: bool = True
    mode: Literal["static", "hybrid_rl", "no_pinn", "no_slca", "agribrain", "no_context", "mcp_only", "pirag_only"] = "agribrain"

    # Optional knobs accepted from legacy QuickDecision payloads. The
    # canonical /decide handler does not consume them (it reads from the
    # loaded telemetry dataframe), but they are carried through so that
    # diagnostic clients can pass them without 422-ing on schema mismatch.
    inventory_units: float | None = None
    demand_units: float | None = None
    temp_c: float | None = None
    volatility: float | None = None
    y_hat: float | None = None
    supply_hat: float | None = None
    supply_std: float | None = None
    demand_std: float | None = None
    price_signal: float | None = None


def _to_canonical(req: DecideRequest) -> Optional[Dict[str, Any]]:
    """Forward to :func:`src.app.decide`. Returns ``None`` on import error."""
    try:
        from src.app import decide as _app_decide, DecideIn
    except ImportError as exc:
        logger.warning("canonical /decide handler unavailable: %s", exc)
        return None
    try:
        return _app_decide(DecideIn(
            agent_id=req.agent,
            role=req.role,
            step=req.step,
            deterministic=req.deterministic,
            mode=req.mode,
        ))
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("canonical /decide handler failed: %s", exc)
        return None


@router.post("/decide")
def decide(req: DecideRequest):
    """Forward to ``src.app.decide``; 503 if the canonical handler is missing.

    Earlier revisions of this router carried a ~250-line standalone
    fallback that re-implemented feature extraction, SLCA, waste, and
    reward computation. The fallback was effectively dead code in any
    real deployment (``src.app`` is always loaded) and routinely drifted
    out of sync with the canonical handler. Consolidating here removes
    the drift surface.
    """
    result = _to_canonical(req)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="decision handler unavailable",
        )
    return result


# Note: GET /decide and /decision/take aliases live in compat.py and
# /last-decision and /decisions live in app.py.
