"""Deployment-phase router.

Implements the three deployment phases described in §1 and §4.13 of the
AGRI-BRAIN manuscript:

* ``monitoring``  -- decisions are computed and surfaced to the operator,
                      but they do not mutate ledger state and they are
                      not anchored on-chain. The dashboard sees the
                      recommendation; the supply chain does not.
* ``advisory``    -- decisions are computed and queued. An operator must
                      explicitly approve (or reject) each entry through
                      ``POST /phase/advisory/{decision_id}/approve`` (or
                      ``/reject``) before the decision is finalised
                      (logged on-chain, broadcast, mirrored to case state).
                      Each pending entry expires after a TTL so that an
                      unattended operator does not silently block the
                      pipeline.
* ``autonomous``  -- decisions are finalised immediately, matching the
                      pre-existing behaviour and what the simulator and
                      benchmark suites expect.

Backend integration
-------------------
``app.py``'s primary ``/decide`` endpoint reads the active phase via
``get_active_phase()`` and calls ``finalize_or_queue_decision()``.

The phase can be set at launch via ``DEPLOYMENT_PHASE=monitoring``, or
changed at runtime via ``POST /phase`` (which the Admin panel uses).
"""
from __future__ import annotations

import logging
import threading
import time as _time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.settings import SETTINGS, VALID_DEPLOYMENT_PHASES

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory state -- intentionally process-local. The advisory queue is
# operator-facing, not persistence; in production it should be backed by a
# durable store (Redis, Postgres). Restarting the process drops the queue.
# ---------------------------------------------------------------------------
_PHASE_LOCK = threading.RLock()
_ACTIVE_PHASE: str = SETTINGS.deployment_phase
_DEFAULT_TTL_S: float = 600.0  # 10 minutes; tuneable via PHASE_ADVISORY_TTL_S
_ADVISORY_QUEUE: "Deque[Dict[str, Any]]" = deque(maxlen=512)
_ADVISORY_HISTORY: "Deque[Dict[str, Any]]" = deque(maxlen=512)


def get_active_phase() -> str:
    with _PHASE_LOCK:
        return _ACTIVE_PHASE


def set_active_phase(phase: str) -> str:
    if phase not in VALID_DEPLOYMENT_PHASES:
        raise ValueError(f"unknown phase: {phase}")
    with _PHASE_LOCK:
        global _ACTIVE_PHASE
        _ACTIVE_PHASE = phase
    return phase


# ---------------------------------------------------------------------------
# Public API used by app.py /decide
# ---------------------------------------------------------------------------
def finalize_or_queue_decision(
    memo: Dict[str, Any],
    *,
    finalize: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply phase semantics to a freshly-computed decision memo.

    ``finalize`` performs the side-effects of an autonomous decision
    (chain logging, websocket broadcast, mirroring to case state). It is
    passed in so this module does not need to import them and create a
    cycle.

    Returns the (possibly tagged) memo. The caller is responsible for
    returning the wrapper response to the HTTP client.
    """
    phase = get_active_phase()
    memo["deployment_phase"] = phase

    if phase == "autonomous":
        return finalize(memo)

    if phase == "monitoring":
        # Recommendation only -- no side effects, no on-chain log.
        memo["tx"] = None
        memo["tx_hash"] = None
        memo["phase_status"] = "monitoring_preview"
        return memo

    # advisory
    decision_id = str(uuid.uuid4())
    expiry = _time.time() + _ttl_seconds()
    memo["tx"] = None
    memo["tx_hash"] = None
    memo["phase_status"] = "advisory_pending"
    memo["advisory_decision_id"] = decision_id
    memo["advisory_expires_at"] = expiry

    queued = {
        "decision_id": decision_id,
        "queued_at": _time.time(),
        "expires_at": expiry,
        "memo": memo,
        "_finalize": finalize,
    }
    with _PHASE_LOCK:
        _ADVISORY_QUEUE.append(queued)
    return memo


def _ttl_seconds() -> float:
    import os as _os
    raw = _os.environ.get("PHASE_ADVISORY_TTL_S")
    if not raw:
        return _DEFAULT_TTL_S
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_TTL_S
    except ValueError:
        return _DEFAULT_TTL_S


def _expire_old() -> None:
    now = _time.time()
    with _PHASE_LOCK:
        kept: List[Dict[str, Any]] = []
        for entry in list(_ADVISORY_QUEUE):
            if entry["expires_at"] < now:
                rec = {
                    "decision_id": entry["decision_id"],
                    "queued_at": entry["queued_at"],
                    "resolved_at": now,
                    "outcome": "expired",
                    "memo": entry["memo"],
                }
                _ADVISORY_HISTORY.append(rec)
            else:
                kept.append(entry)
        _ADVISORY_QUEUE.clear()
        _ADVISORY_QUEUE.extend(kept)


def _serialise(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision_id": entry["decision_id"],
        "queued_at": entry["queued_at"],
        "expires_at": entry["expires_at"],
        "memo": entry.get("memo"),
    }


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
class PhaseModel(BaseModel):
    phase: Literal["monitoring", "advisory", "autonomous"]


@router.get("")
def get_phase() -> Dict[str, Any]:
    return {
        "phase": get_active_phase(),
        "valid_phases": list(VALID_DEPLOYMENT_PHASES),
        "advisory_ttl_s": _ttl_seconds(),
        "queue_depth": len(_ADVISORY_QUEUE),
    }


@router.post("")
def post_phase(payload: PhaseModel) -> Dict[str, Any]:
    set_active_phase(payload.phase)
    logger.info("deployment phase set to %s", payload.phase)
    return {"phase": get_active_phase()}


@router.get("/advisory/pending")
def list_pending() -> Dict[str, Any]:
    _expire_old()
    with _PHASE_LOCK:
        items = [_serialise(e) for e in _ADVISORY_QUEUE]
    return {"pending": items, "count": len(items)}


@router.get("/advisory/history")
def list_history(limit: int = 50) -> Dict[str, Any]:
    with _PHASE_LOCK:
        items = list(_ADVISORY_HISTORY)[-int(max(1, min(limit, 512))) :]
    return {"history": items, "count": len(items)}


def _resolve_entry(decision_id: str, outcome: str) -> Dict[str, Any]:
    _expire_old()
    with _PHASE_LOCK:
        target: Optional[Dict[str, Any]] = None
        kept: List[Dict[str, Any]] = []
        for entry in _ADVISORY_QUEUE:
            if entry["decision_id"] == decision_id and target is None:
                target = entry
            else:
                kept.append(entry)
        if target is None:
            raise HTTPException(404, f"unknown advisory decision_id: {decision_id}")
        _ADVISORY_QUEUE.clear()
        _ADVISORY_QUEUE.extend(kept)

        memo = target["memo"]
        finalize = target.get("_finalize")
        if outcome == "approve" and finalize is not None:
            try:
                memo = finalize(memo)
            except Exception as exc:  # noqa: BLE001
                logger.warning("advisory finalize failed: %s", exc)
                memo["phase_status"] = "advisory_finalize_error"
                memo["advisory_error"] = str(exc)
            else:
                memo["phase_status"] = "advisory_approved"
        elif outcome == "reject":
            memo["phase_status"] = "advisory_rejected"
            memo["tx"] = None
            memo["tx_hash"] = None

        rec = {
            "decision_id": target["decision_id"],
            "queued_at": target["queued_at"],
            "resolved_at": _time.time(),
            "outcome": outcome,
            "memo": memo,
        }
        _ADVISORY_HISTORY.append(rec)
        return rec


@router.post("/advisory/{decision_id}/approve")
def approve(decision_id: str) -> Dict[str, Any]:
    return _resolve_entry(decision_id, "approve")


@router.post("/advisory/{decision_id}/reject")
def reject(decision_id: str) -> Dict[str, Any]:
    return _resolve_entry(decision_id, "reject")
