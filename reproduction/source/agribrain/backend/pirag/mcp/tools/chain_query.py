"""Local audit-ledger decision-history tool for the MCP server.

Despite its legacy registry name, ``chain_query`` does not query a blockchain
or establish on-chain state. It reads recent routing decisions from three
local sources, in priority order:

1. **Active same-episode ledger** — a ContextVar-bound in-memory
   :class:`DecisionLedger`. It contains only decisions already made in the
   current episode. An empty active ledger is authoritative and shadows every
   file, preventing experimental arms from importing stale history.
2. **Live FastAPI audit state** — when the server runs inside the FastAPI
   process, ``src.app.state["log"]`` is the local runtime source.
3. **Local JSONL audit-ledger fallback** — outside an active episode, the tool
   can read the most recently written ``decision_ledger/*.jsonl`` produced by
   :meth:`DecisionLedger.write_jsonl`. ``DECISION_LEDGER_DIR`` selects the
   directory; an explicit directory is exclusive and never falls through to
   the repository default.

Status codes:

  _status="ok"        -> records returned from the active episode, app state,
                         or an explicitly selected JSONL ledger.
  _status="empty"     -> source reachable but no records yet
                         (for example, the first decision of an episode).
  _status="error"     -> no local decision-history source is reachable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalise_records(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map ledger/app records to the stable MCP response schema."""
    return [
        {
            "timestamp": entry.get("ts", entry.get("hour", entry.get("time", 0))),
            "action": entry.get("action", "unknown"),
            "agent": entry.get("agent", ""),
            "role": entry.get("role", ""),
            "slca_score": entry.get("slca", 0.0),
            "carbon_kg": entry.get("carbon_kg", 0.0),
            "waste": entry.get("waste", 0.0),
            "tx_hash": entry.get("tx_hash", "0x0"),
            "mode": entry.get("mode", "agribrain"),
        }
        for entry in entries
    ]


def _read_active_episode_ledger(n: int) -> Optional[Dict[str, Any]]:
    """Read the current simulator episode, shadowing every external source."""
    try:
        from src.chain.decision_ledger import get_active_episode_ledger
    except ImportError:
        return None
    ledger = get_active_episode_ledger()
    if ledger is None:
        return None
    records = _normalise_records(ledger.recent_records(n))
    return {
        "_status": "ok" if records else "empty",
        "_source": "active_episode_ledger",
        "records": records,
    }


def _read_app_state(n: int) -> Optional[Dict[str, Any]]:
    """Try to read from the live FastAPI app state; return None on absence."""
    n = max(0, int(n))
    try:
        from src.app import state as app_state
    except ImportError:
        return None
    if not isinstance(app_state, dict) or "log" not in app_state:
        return None

    logs = app_state.get("log", [])
    records = _normalise_records(list(logs[-n:]) if n else [])
    return {"_status": "ok" if records else "empty",
            "_source": "app_state",
            "records": records}


def _read_ledger_jsonl(n: int) -> Optional[Dict[str, Any]]:
    """Read the most-recent decision_ledger JSONL produced by the simulator.

    An explicit empty scope returns ``_status="empty"`` and is authoritative.
    Without an explicit scope, absence of a default ledger returns ``None`` so
    the caller can report that no local source is reachable.
    """
    n = max(0, int(n))
    candidate_dirs: List[Path] = []
    scoped_dir = None
    try:
        from src.chain.decision_ledger import get_active_decision_ledger_output_dir
        scoped_dir = get_active_decision_ledger_output_dir()
    except ImportError:
        pass
    env_dir = os.environ.get("DECISION_LEDGER_DIR")
    explicit_scope = scoped_dir is not None or bool(env_dir)
    if scoped_dir is not None:
        candidate_dirs.append(Path(scoped_dir))
    elif env_dir:
        candidate_dirs.append(Path(env_dir))
    else:
        # An explicit scope is exclusive: if it is empty at the beginning of
        # an experimental arm, falling through to the repository-wide default
        # would import decisions from a different mode, scenario, seed, or
        # parallel Slurm task.  Use the default only when no scope was declared.
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
            if explicit_scope:
                return {
                    "_status": "empty",
                    "_source": f"jsonl_scope:{d}",
                    "records": [],
                }
            continue
        latest = files[0]
        records: List[Dict[str, Any]] = []
        try:
            with latest.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()[-n:] if n else []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    entry = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if entry.get("_header") is True:
                    continue
                records.extend(_normalise_records([entry]))
        except OSError:
            continue
        return {"_status": "ok" if records else "empty",
                "_source": f"jsonl:{latest.name}",
                "records": records}

    return None


def query_recent_decisions(n: int = 10) -> Dict[str, Any]:
    """Query recent routing decisions.

    Uses an active simulator episode first, then live FastAPI ``state["log"]``,
    then the most-recent explicitly scoped/default JSONL ledger. Returns
    ``_status="error"`` only when no source is available.

    Parameters
    ----------
    n : number of recent records to return.
    """
    n = max(0, int(n))
    # A simulator episode is authoritative even when it is still empty.  Do
    # not fall through: doing so would import stale decisions from another arm.
    via_episode = _read_active_episode_ledger(n)
    if via_episode is not None:
        return via_episode

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
