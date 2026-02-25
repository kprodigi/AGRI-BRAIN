# backend/src/routers/results.py
"""
API endpoints for running simulations and serving generated figures.

POST /results/generate  — triggers the full simulation run, returns summary JSON
GET  /results/figures/{filename} — serves a generated figure file
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# Ensure the simulation module is importable
# ---------------------------------------------------------------------------
_SIM_DIR = Path(__file__).resolve().parent.parent.parent.parent / "mvp" / "simulation"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

_RESULTS_DIR = _SIM_DIR / "results"


# ---------------------------------------------------------------------------
# POST /results/generate
# ---------------------------------------------------------------------------
@router.post("/generate")
def generate_results():
    """Run all 5 scenarios x 5 modes and return a summary JSON.

    Also saves CSV tables and (optionally) figures to disk.
    """
    try:
        from generate_results import run_all, save_tables, get_summary_json
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot import simulation module: {exc}. "
                   f"Ensure mvp/simulation/ exists relative to the backend.",
        )

    data = run_all()
    save_tables(data["table1"], data["table2"])
    summary = get_summary_json(data)

    return {
        "ok": True,
        "summary": summary,
        "tables": {
            "table1": str(_RESULTS_DIR / "table1_summary.csv"),
            "table2": str(_RESULTS_DIR / "table2_ablation.csv"),
        },
    }


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
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Figure not found: {filename}")

    media = "image/png" if path.suffix == ".png" else "application/pdf"
    return FileResponse(str(path), media_type=media)
