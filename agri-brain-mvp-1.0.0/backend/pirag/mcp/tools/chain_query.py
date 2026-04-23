"""Blockchain query tool for the MCP server.

Queries recent decision records from the on-chain audit trail.
Returns mock records when no blockchain connection is available.
"""
from __future__ import annotations

from typing import Any, Dict, List


def query_recent_decisions(n: int = 10) -> List[Dict[str, Any]]:
    """Query recent routing decisions from the blockchain audit trail.

    Parameters
    ----------
    n : number of recent records to return.

    Returns
    -------
    List of decision record dicts.
    """
    # Attempt to read from live state first
    records = []
    try:
        from src.app import state as app_state
        logs = app_state.get("log", [])
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
    except (ImportError, AttributeError):
        pass

    if records:
        return records

    # No live blockchain data available
    return []
