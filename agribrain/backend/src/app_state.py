"""Centralized accessors for the FastAPI app's shared state.

The pre-2026-05 codebase scattered process-local state across several
modules: ``app.py:state``, ``routers/case.py:STATE``,
``routers/scenarios.py:ACTIVE``, ``routers/governance.py:CHAIN``,
``routers/phase.py:_ACTIVE_PHASE``, plus a removed ``routers/decide.py``
parallel log. Each container had its own write path, and consumers
reached across module boundaries to read whichever copy was current,
which made the data-flow hard to follow and let synchronisation bugs
hide between updates.

This module is the single import surface for read-side consumers:

* :func:`get_app_state`     -- the canonical decision-policy container
                               (telemetry df, policy, chain config,
                               in-memory decision log).
* :func:`get_case_state`    -- the case-router CSV + summary + last
                               decision mirror used by the PDF and
                               audit endpoints.
* :func:`get_active_scenario` -- the scenarios router's current
                               selection.
* :func:`get_chain_config`  -- whichever chain config is currently
                               authoritative (app state wins; the
                               governance mirror is a sync target).
* :func:`get_active_phase`  -- the deployment-phase router's current
                               phase value.

Writers still mutate the legacy containers in place (so existing
tests and the lifespan startup keep working unchanged). Read-side
consumers should call these helpers instead of reaching into module
globals so future consolidation can swap the storage backend without
touching every caller. The accessor surface deliberately returns
*copies* of dictionary state so callers do not accidentally mutate
the shared container.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def get_app_state() -> Dict[str, Any]:
    """Return the FastAPI app state dict (lazy import to avoid cycles)."""
    try:
        from src.app import state
        return state
    except ImportError:
        return {}


def get_case_state() -> Dict[str, Any]:
    """Return the case router's STATE dict (CSV rows + last decision)."""
    try:
        from src.routers.case import STATE
        return STATE
    except ImportError:
        return {}


def get_active_scenario() -> Dict[str, Any]:
    """Return a snapshot of the active scenario.

    Returns a copy so callers cannot mutate the shared container by
    accident. The "name" field is None when no scenario has been run
    since startup or since the last reset.
    """
    try:
        from src.routers.scenarios import ACTIVE
    except ImportError:
        return {"name": None, "intensity": 1.0}
    return {
        "name": ACTIVE.get("name"),
        "intensity": float(ACTIVE.get("intensity") or 1.0),
    }


def get_chain_config() -> Dict[str, Any]:
    """Return the authoritative chain config (app state wins)."""
    app_state = get_app_state()
    cfg = app_state.get("chain") if app_state else None
    if isinstance(cfg, dict):
        # Defensive copy: callers should not mutate the live container.
        return dict(cfg)
    try:
        from src.routers.governance import CHAIN
    except ImportError:
        return {}
    return dict(CHAIN)


def get_active_phase() -> str:
    """Return the deployment-phase router's current phase value."""
    try:
        from src.routers.phase import get_active_phase as _gap
    except ImportError:
        return "autonomous"
    return _gap()


def get_decision_log() -> list[Dict[str, Any]]:
    """Return the in-memory decision log (newest last).

    Returns a list copy so iteration is stable even when a parallel
    request appends a new entry.
    """
    log = get_app_state().get("log") or []
    return list(log)


def get_last_decision() -> Optional[Dict[str, Any]]:
    """Return the most recent decision memo, or None when none exist."""
    log = get_app_state().get("log") or []
    if log:
        return log[-1]
    case = get_case_state()
    last = case.get("last_decision")
    return last if isinstance(last, dict) else None
