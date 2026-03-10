"""Blockchain query tool for the MCP server.

Queries recent decision records from the on-chain audit trail.
Returns mock records when no blockchain connection is available.
"""
from __future__ import annotations

import time
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

    # Return mock records if no live data available
    base_time = int(time.time())
    actions = ["cold_chain", "local_redistribute", "recovery"]
    mock = []
    for i in range(min(n, 5)):
        mock.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(base_time - i * 900)),
            "action": actions[i % 3],
            "agent": f"agent_{i}",
            "role": ["farm", "processor", "cooperative", "distributor", "recovery"][i % 5],
            "slca_score": round(0.65 + i * 0.03, 3),
            "carbon_kg": round(8.5 - i * 0.5, 2),
            "waste": round(0.05 + i * 0.01, 3),
            "tx_hash": f"0x{'ab' * 16}{i:04x}",
            "mode": "agribrain",
            "note": "mock record (no live blockchain connection)",
        })
    return mock
