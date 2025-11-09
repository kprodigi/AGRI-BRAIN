# backend/src/routers/case.py
from fastapi import APIRouter
from typing import Any, Dict, List, Optional
import os, csv, statistics, time

router = APIRouter()

# In-memory store read by other routers (PDF, decide, etc.)
STATE: Dict[str, Any] = {
    "rows": [],
    "summary": {
        "records": 0,
        "avg_tempC": None,          # NOTE: capital C matches the PDF
        "anomaly_points": 0,
        "waste_rate_baseline": 0.0,
        "waste_rate_agri": 0.0,
    },
    "loaded_at": None,
    "path": None,
    "last_decision": None,          # decide.py will set this
}

def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "timestamp": row.get("timestamp"),
                "tempC": _to_float(row.get("tempC")),
                "RH": _to_float(row.get("RH")),
                "shockG": _to_float(row.get("shockG")),
                "ambientC": _to_float(row.get("ambientC")),
                "inventory_units": _to_float(row.get("inventory_units")),
                "demand_units": _to_float(row.get("demand_units")),
            })
    return rows

def _compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    temps = [r["tempC"] for r in rows if isinstance(r.get("tempC"), (int, float, float))]
    shocks = [r["shockG"] for r in rows if isinstance(r.get("shockG"), (int, float, float))]
    records = len(rows)
    avg_temp = round(statistics.fmean(temps), 2) if temps else None

    # simple demo anomaly rule: temp > 4°C or shock > 1.5G
    anomalies = sum(
        1 for r in rows
        if (isinstance(r.get("tempC"), (int, float)) and r["tempC"] > 4.0)
        or (isinstance(r.get("shockG"), (int, float)) and r["shockG"] > 1.5)
    )

    # demo waste values (replace with your real model later)
    waste_baseline = 0.0
    waste_agri = 0.0

    return {
        "records": records,
        "avg_tempC": avg_temp,               # <-- exact name used by PDF
        "anomaly_points": anomalies,
        "waste_rate_baseline": waste_baseline,
        "waste_rate_agri": waste_agri,
    }

def _default_csv_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "data", "data_spinach.csv"))

# ---------- Routes (mounted under prefix="/case") ----------

@router.post("/load")
def load_case(path: Optional[str] = None):
    """
    Loads a CSV of demo spinach data:
      - if 'path' is provided, use it
      - else try env DATA_CSV
      - else fall back to backend/src/data/data_spinach.csv
    """
    csv_path = path or os.environ.get("DATA_CSV") or _default_csv_path()
    if not os.path.exists(csv_path):
        return {"ok": False, "error": f"CSV not found: {csv_path}"}

    rows = _load_csv(csv_path)
    summary = _compute_summary(rows)

    STATE.update({
        "rows": rows,
        "summary": summary,
        "loaded_at": int(time.time()),
        "path": csv_path,
    })
    return {"ok": True, "path": csv_path, "rows": len(rows), "summary": summary}

@router.get("/kpis")
def kpis():
    """
    Returns the summary used by the PDF.
    If nothing loaded yet, try auto-load the default CSV once.
    """
    if not STATE["summary"]["records"]:
        p = _default_csv_path()
        if os.path.exists(p):
            rows = _load_csv(p)
            STATE.update({
                "rows": rows,
                "summary": _compute_summary(rows),
                "loaded_at": int(time.time()),
                "path": p,
            })
    return STATE["summary"]

@router.get("/last_decision")
def case_last_decision():
    return STATE.get("last_decision") or {}
