"""Blockchain query tool for the MCP server.

Queries recent decision records. Two read paths:

1. **Live FastAPI state** — when the server is running inside the
   FastAPI process (`src.app.state["log"]`), this is the canonical
   source.
2. **Decision-ledger JSONL fallback** — when the server is running
   inside the simulator subprocess (`mvp/simulation/generate_results.py`)
   the FastAPI process is not started here, so `state["log"]` is
   never populated. Earlier revisions of this tool returned
   `_status="error"` in this case, which created spurious "MCP tool
   error" entries in every simulation run. This version falls back to
   reading the most recently written `decision_ledger/*.jsonl`
   produced by the simulator's `DecisionLedger.write_jsonl` (see
   `agri-brain-mvp-1.0.0/backend/src/chain/decision_ledger.py`). The
   path is configurable via the `DECISION_LEDGER_DIR` env var, same
   as the simulator uses.

Status codes:

  _status="ok"        -> records returned (from app state OR JSONL).
  _status="empty"     -> source reachable but no records yet
                         (start of run).
  _status="error"     -> neither source reachable. Genuine error.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_app_state(n: int) -> Optional[Dict[str, Any]]:
    """Try to read from the live FastAPI app state; return None on absence."""
    try:
        from src.app import state as app_state
    except ImportError:
        return None
    if not isinstance(app_state, dict) or "log" not in app_state:
        return None

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
    return {"_status": "ok" if records else "empty",
            "_source": "app_state",
            "records": records}


def _read_ledger_jsonl(n: int) -> Optional[Dict[str, Any]]:
    """Read the most-recent decision_ledger JSONL produced by the simulator.

    Returns None if no ledger files are present (not an error — just
    no fallback available).
    """
    candidate_dirs: List[Path] = []
    env_dir = os.environ.get("DECISION_LEDGER_DIR")
    if env_dir:
        candidate_dirs.append(Path(env_dir))
    # Default search path mirrors generate_results.RESULTS_DIR /
    # decision_ledger; we resolve from the import location of this
    # module to the repo root rather than guessing CWD.
    here = Path(__file__).resolve()
    repo_default = here.parent.parent.parent.parent.parent.parent / "mvp" / "simulation" / "results" / "decision_ledger"
    if repo_default.exists():
        candidate_dirs.append(repo_default)

    for d in candidate_dirs:
        if not d.exists() or not d.is_dir():
            continue
        files = sorted(
            (p for p in d.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            continue
        latest = files[0]
        records: List[Dict[str, Any]] = []
        try:
            with latest.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()[-n:]
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    entry = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                records.append({
                    "timestamp": entry.get("ts", entry.get("hour", 0)),
                    "action": entry.get("action", "unknown"),
                    "agent": entry.get("agent", ""),
                    "role": entry.get("role", ""),
                    "slca_score": entry.get("slca", 0.0),
                    "carbon_kg": entry.get("carbon_kg", 0.0),
                    "waste": entry.get("waste", 0.0),
                    "tx_hash": entry.get("tx_hash", "0x0"),
                    "mode": entry.get("mode", "agribrain"),
                })
        except OSError:
            continue
        return {"_status": "ok" if records else "empty",
                "_source": f"jsonl:{latest.name}",
                "records": records}

    return None


def query_recent_decisions(n: int = 10) -> Dict[str, Any]:
    """Query recent routing decisions.

    Tries the live FastAPI ``state["log"]`` first, then falls back to
    the most-recent ``decision_ledger/*.jsonl`` produced by the
    simulator. Returns ``_status="error"`` only when both sources are
    unavailable.

    Parameters
    ----------
    n : number of recent records to return.
    """
    via_app = _read_app_state(n)
    if via_app is not None:
        return via_app

    via_ledger = _read_ledger_jsonl(n)
    if via_ledger is not None:
        return via_ledger

    return {
        "_status": "error",
        "_error_kind": "no_source_reachable",
        "_message": (
            "Neither the FastAPI app state nor a decision_ledger JSONL "
            "is reachable from this process. Run inside the FastAPI "
            "server, point DECISION_LEDGER_DIR at a populated ledger "
            "directory, or wait for the first scenario episode to "
            "write a ledger."
        ),
        "records": [],
    }
