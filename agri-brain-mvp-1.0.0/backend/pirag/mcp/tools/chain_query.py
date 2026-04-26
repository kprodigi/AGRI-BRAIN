"""Blockchain query tool for the MCP server.

Queries recent decision records from the on-chain audit trail.

Returns a structured payload that distinguishes three states so the
protocol-recorder counter (and the MCP Tool Reliability figure) can tell
them apart instead of treating all empty results as success:

  _status="ok"        -> live state reachable, returned records.
  _status="empty"     -> live state reachable but the log is genuinely empty
                         (start of run, no decisions yet). Not an error.
  _status="error"     -> live state itself is unreachable from this process
                         (typically the simulator subprocess where the FastAPI
                         app's state dict is never populated). Used to be a
                         silent ``return []``.
"""
from __future__ import annotations

from typing import Any, Dict, List


def query_recent_decisions(n: int = 10) -> Dict[str, Any]:
    """Query recent routing decisions from the live FastAPI app state.

    Parameters
    ----------
    n : number of recent records to return.

    Returns
    -------
    Dict with ``_status``, ``records`` (list, possibly empty), and on
    failure paths ``_error_kind`` plus ``_message``.
    """
    try:
        from src.app import state as app_state
    except ImportError as exc:
        return {
            "_status": "error",
            "_error_kind": "state_import_failed",
            "_message": f"src.app not importable: {exc}",
            "records": [],
        }

    if not isinstance(app_state, dict) or "log" not in app_state:
        return {
            "_status": "error",
            "_error_kind": "state_unavailable",
            "_message": (
                "live FastAPI state has no 'log' key in this process; chain "
                "audit trail cannot be read here (this happens when the "
                "simulator subprocess never populates the REST app's state)"
            ),
            "records": [],
        }

    logs = app_state.get("log", [])
    records: List[Dict[str, Any]] = []
    for entry in logs[-n:]:
        records.append({
            "timestamp": entry.get("time", ""),
            "action": entry.get("action", "unknown"),
            "agent": entry.get("agent", ""),
            "role": entry.get("role", ""),
            "slca_score": entry.get("slca", 0.0),
            "carbon_kg": entry.get("carbon_kg", 0.0),
            "waste": entry.get("waste", 0.0),
            "tx_hash": entry.get("tx_hash", "0x0"),
            "mode": entry.get("mode", "agribrain"),
        })

    return {
        "_status": "ok" if records else "empty",
        "records": records,
    }
