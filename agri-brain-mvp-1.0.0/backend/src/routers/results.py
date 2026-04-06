# backend/src/routers/results.py
"""
API endpoints for running simulations and serving generated figures.

POST /results/generate  — kicks off simulation in background, returns job ID
GET  /results/status     — poll for completion
GET  /results/summary    — fetch last completed summary
GET  /results/figures/{filename} — serves a generated figure file
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# Ensure the simulation module is importable
# ---------------------------------------------------------------------------
# Locate simulation dir: env var override, or walk up to repo root / mvp / simulation
_SIM_DIR = Path(os.environ.get("SIM_DIR", "")) if os.environ.get("SIM_DIR") else (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "mvp" / "simulation"
)
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

_RESULTS_DIR = _SIM_DIR / "results"

# Background job state (guarded by _JOB_LOCK for thread safety)
_JOB_LOCK = threading.Lock()
_JOB = {"running": False, "started_at": None, "finished_at": None,
        "error": None, "summary": None}


def _run_in_background(seed: int | None = None):
    """Worker: run simulation and save tables."""
    try:
        from generate_results import run_all, save_tables, get_summary_json
        data = run_all(seed=seed) if seed is not None else run_all()
        save_tables(data["table1"], data["table2"])
        with _JOB_LOCK:
            _JOB["summary"] = get_summary_json(data)
            _JOB["error"] = None
    except Exception as exc:
        with _JOB_LOCK:
            _JOB["error"] = str(exc)
            _JOB["summary"] = None
    finally:
        with _JOB_LOCK:
            _JOB["running"] = False
            _JOB["finished_at"] = time.time()


# ---------------------------------------------------------------------------
# POST /results/generate — non-blocking: kicks off background job
# ---------------------------------------------------------------------------
@router.post("/generate")
def generate_results(seed: int | None = None):
    """Start full simulation (5 scenarios x 8 modes) in the background.

    Returns immediately with a job status. Poll GET /results/status for
    completion. This avoids HTTP timeouts for long-running simulations.
    """
    with _JOB_LOCK:
        if _JOB["running"]:
            elapsed = time.time() - (_JOB["started_at"] or time.time())
            return {"ok": True, "status": "running", "elapsed_s": round(elapsed, 1)}

    try:
        from generate_results import run_all  # noqa: F401 — verify import
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot import simulation module: {exc}. "
                   f"Ensure mvp/simulation/ exists relative to the backend.",
        )

    with _JOB_LOCK:
        if _JOB["running"]:
            return {"ok": True, "status": "running"}
        _JOB["running"] = True
        _JOB["started_at"] = time.time()
        _JOB["finished_at"] = None
        _JOB["error"] = None
        _JOB["summary"] = None

    t = threading.Thread(target=_run_in_background, kwargs={"seed": seed}, daemon=True)
    t.start()

    return {"ok": True, "status": "started", "seed": seed}


@router.get("/status")
def results_status():
    """Poll simulation job status."""
    with _JOB_LOCK:
        if _JOB["running"]:
            elapsed = time.time() - (_JOB["started_at"] or time.time())
            return {"status": "running", "elapsed_s": round(elapsed, 1)}
        if _JOB["finished_at"]:
            duration = round((_JOB["finished_at"] - (_JOB["started_at"] or _JOB["finished_at"])), 1)
            if _JOB["error"]:
                return {"status": "error", "error": _JOB["error"], "duration_s": duration}
            return {"status": "complete", "duration_s": duration}
        return {"status": "idle"}


@router.get("/summary")
def results_summary():
    """Return the last completed simulation summary."""
    if _JOB["summary"]:
        return {"ok": True, "summary": _JOB["summary"],
                "tables": {"table1": str(_RESULTS_DIR / "table1_summary.csv"),
                           "table2": str(_RESULTS_DIR / "table2_ablation.csv")}}
    # Fallback: try loading from disk
    t1 = _RESULTS_DIR / "table1_summary.csv"
    if t1.exists():
        return {"ok": True, "summary": None, "tables_on_disk": True,
                "tables": {"table1": str(t1), "table2": str(_RESULTS_DIR / "table2_ablation.csv")}}
    return {"ok": False, "error": "No simulation results available. Run POST /results/generate first."}


# ---------------------------------------------------------------------------
# GET /results/figures/{filename}
# ---------------------------------------------------------------------------
@router.get("/figures/{filename}")
def get_figure(filename: str):
    """Serve a generated figure file (PNG or PDF)."""
    # Sanitise: only allow filenames, no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = _RESULTS_DIR / filename
    # Verify resolved path stays inside the results directory
    if not path.resolve().is_relative_to(_RESULTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Figure not found: {filename}")

    _MIME = {
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }
    media = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(path), media_type=media)
