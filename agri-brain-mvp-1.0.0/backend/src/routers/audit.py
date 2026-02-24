# backend/src/routers/audit.py
from fastapi import APIRouter, Response
from io import BytesIO
from typing import Any, Dict, List

router = APIRouter()

# ---------------- Optional imports (robust to missing modules) ----------------
try:
    from ..state import get_audit  # type: ignore
except Exception:  # pragma: no cover
    def get_audit() -> List[Dict[str, Any]]:
        return []

try:
    from src.routers.decide import LAST as _LAST, DECISIONS as _DECISIONS  # type: ignore
except Exception:  # pragma: no cover
    _LAST, _DECISIONS = None, []

try:
    from src.routers.case import STATE as _CASE_STATE  # type: ignore
except Exception:  # pragma: no cover
    _CASE_STATE = {}

# --------------------------------- Helpers -----------------------------------
def _as_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if hasattr(x, "model_dump"):   # pydantic v2
        return x.model_dump()
    if hasattr(x, "dict"):         # pydantic v1
        return x.dict()
    return x if isinstance(x, dict) else {}

def _get_last_decision_raw() -> Dict[str, Any]:
    """Best-effort: app.state → LAST → DECISIONS → case.STATE['last_decision']."""
    # Prefer the app-level log (has full softmax memo)
    try:
        from src.app import state as _app_state  # lazy to avoid circular at import
        log = _app_state.get("log")
        if log:
            return log[-1]
    except Exception:
        pass
    if _LAST:
        return _as_dict(_LAST)
    if _DECISIONS:
        return _as_dict(_DECISIONS[-1])
    try:
        ld = _CASE_STATE.get("last_decision")
        if isinstance(ld, dict):
            return ld
    except Exception:
        pass
    return {}

def _map_for_pdf(memo: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize last-decision keys to the labels used in the PDF."""
    base = {
        "time":       memo.get("ts") or memo.get("time") or "",
        "agent":      memo.get("agent") or "",
        "role":       memo.get("role") or "",
        "decision":   memo.get("action") or memo.get("decision") or "",
        "shelf_left": memo.get("shelf_left") or 0,
        "volatility": memo.get("volatility") or 0,
        "km":         memo.get("km") or 0,
        "carbon_kg":  memo.get("carbon_kg") or memo.get("carbon") or 0,
        "unit_price": memo.get("unit_price") or 0,
        "slca":       memo.get("slca_score") or memo.get("slca") or 0,
        "tx":         memo.get("tx_hash") or memo.get("tx") or "",
        "note":       memo.get("reason") or memo.get("note") or "",
    }
    # Extended fields from regime-aware softmax policy
    if memo.get("action_probabilities"):
        base["action_probabilities"] = memo["action_probabilities"]
    if memo.get("slca_components"):
        base["slca_components"] = memo["slca_components"]
    if memo.get("footprint"):
        base["footprint"] = memo["footprint"]
    if memo.get("regime"):
        base["regime"] = memo["regime"]
    if memo.get("reward_decomposition"):
        base["reward_decomposition"] = memo["reward_decomposition"]
    if memo.get("spoilage_risk") is not None:
        base["spoilage_risk"] = memo["spoilage_risk"]
    return base

def _ensure_state_has_kpis() -> None:
    """
    If KPIs were never computed, force-load the default spinach CSV so the
    PDF header has non-zero numbers even when /case/load wasn't called.
    """
    try:
        if _CASE_STATE.get("metrics"):
            return
        from src.routers import case as _case  # type: ignore
        import os, time as _t
        csv_path = _case._default_csv_path()                 # noqa: SLF001
        if os.path.exists(csv_path):
            rows = _case._load_csv(csv_path)                 # noqa: SLF001
            metrics = _case._compute_kpis(rows)              # noqa: SLF001
            _CASE_STATE.update({
                "rows": rows,
                "metrics": metrics,
                "loaded_at": int(_t.time()),
                "path": csv_path,
            })
    except Exception:
        # If anything fails, we'll just show zeros; keep audit routes resilient.
        pass

def _fetch_kpis() -> Dict[str, Any]:
    """
    Prefer the app-level /kpis (includes extended metrics); fall back to case.STATE.
    """
    # Try the app-level kpis first (includes mean_ari, mean_rle, etc.)
    try:
        from src.app import kpis as _app_kpis
        return _app_kpis()
    except Exception:
        pass

    _ensure_state_has_kpis()

    metrics: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    try:
        metrics = _CASE_STATE.get("metrics") or {}
        rows = _CASE_STATE.get("rows") or []
    except Exception:
        pass

    records = int(metrics.get("records") or (len(rows) if rows else 0))
    avg_tempC = (
        metrics.get("avg_tempC")
        or metrics.get("avg_temp_c")
        or metrics.get("avg_temp")
        or 0
    )
    anomaly_points = int(metrics.get("anomaly_points") or 0)

    # case.py uses *_pct; support both shapes
    waste_rate_baseline = (
        metrics.get("waste_rate_baseline")
        or metrics.get("waste_baseline_pct")
        or 0
    )
    waste_rate_agri = (
        metrics.get("waste_rate_agri")
        or metrics.get("waste_agri_pct")
        or 0
    )

    return {
        "records": records,
        "avg_tempC": avg_tempC,
        "anomaly_points": anomaly_points,
        "waste_rate_baseline": waste_rate_baseline,
        "waste_rate_agri": waste_rate_agri,
    }

def _render_pdf(kpis: Dict[str, Any], last: Dict[str, Any]) -> bytes:
    """Render a simple PDF using reportlab."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import LETTER

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    x, y = 72, height - 72  # 1-inch margins

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "AGRI-BRAIN Spinach — Decision Memo")
    y -= 28

    # KPIs
    c.setFont("Helvetica", 10)
    for line in [
        f"records: {kpis.get('records', 0)}",
        f"avg_tempC: {kpis.get('avg_tempC', 0)}",
        f"anomaly_points: {kpis.get('anomaly_points', 0)}",
        f"waste_rate_baseline: {kpis.get('waste_rate_baseline', 0)}",
        f"waste_rate_agri: {kpis.get('waste_rate_agri', 0)}",
    ]:
        c.drawString(x, y, line); y -= 14

    # Last Decision
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Last Decision")
    y -= 18
    c.setFont("Helvetica", 10)
    for label in [
        "time", "agent", "role", "decision", "shelf_left", "spoilage_risk",
        "volatility", "km", "carbon_kg", "unit_price", "slca", "tx", "note",
    ]:
        c.drawString(x, y, f"{label}: {last.get(label, '')}")
        y -= 14

    # SLCA components
    sc = last.get("slca_components")
    if isinstance(sc, dict):
        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "SLCA Components"); y -= 14
        c.setFont("Helvetica", 10)
        for k in ("carbon", "labor", "resilience", "transparency", "composite"):
            c.drawString(x + 10, y, f"{k}: {sc.get(k, '')}"); y -= 14

    # Reward decomposition
    rd = last.get("reward_decomposition")
    if isinstance(rd, dict):
        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Reward Decomposition"); y -= 14
        c.setFont("Helvetica", 10)
        for k in ("slca", "energy_penalty", "water_penalty", "waste_penalty", "total"):
            c.drawString(x + 10, y, f"{k}: {rd.get(k, '')}"); y -= 14

    # Action probabilities
    ap = last.get("action_probabilities")
    if isinstance(ap, dict):
        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Action Probabilities"); y -= 14
        c.setFont("Helvetica", 10)
        for k, v in ap.items():
            c.drawString(x + 10, y, f"{k}: {v}"); y -= 14

    # Footprint & regime
    fp = last.get("footprint")
    if isinstance(fp, dict):
        c.drawString(x, y, f"footprint: energy_J={fp.get('energy_J','')} water_L={fp.get('water_L','')}"); y -= 14
    rg = last.get("regime")
    if isinstance(rg, dict):
        c.drawString(x, y, f"regime: tau={rg.get('tau','')} bollinger_z={rg.get('bollinger_z','')}"); y -= 14

    c.showPage()
    c.save()
    return buf.getvalue()

# ----------------------------------- Routes ----------------------------------
@router.get("/logs")
def audit_logs():
    # Merge the global audit log with the app-level decision log
    items = list(get_audit())
    try:
        from src.app import state as _app_state  # lazy import avoids circular at module level
        for m in _app_state.get("log", []):
            items.append(_map_for_pdf(m))
    except Exception:
        pass
    return {"items": items}

@router.get("/memo.json")
def audit_memo_json():
    return {
        "kpis": _fetch_kpis(),
        "last_decision": _map_for_pdf(_get_last_decision_raw()),
    }

@router.get("/memo.pdf")
def audit_memo_pdf():
    kpis = _fetch_kpis()
    last = _map_for_pdf(_get_last_decision_raw())
    try:
        return Response(content=_render_pdf(kpis, last), media_type="application/pdf")
    except Exception:
        msg = (
            "PDF generator not available.\n"
            "Install with: pip install reportlab\n\n"
            "Preview data at /audit/memo.json\n"
        )
        return Response(content=msg, media_type="text/plain")

# Friendly aliases if something links here:
@router.get("/report/memo", include_in_schema=False)
def audit_memo_pdf_alias1():
    return audit_memo_pdf()

@router.get("/memo", include_in_schema=False)
def audit_memo_pdf_alias2():
    return audit_memo_pdf()

# Optional on-chain fetch for Admin → Audit
@router.get("/chain")
def audit_chain():
    try:
        from src.routers.governance import CHAIN as CHAIN_CFG  # type: ignore
        from src.chain.eth import fetch_recent_decisions          # type: ignore
        items = fetch_recent_decisions(CHAIN_CFG or {}, 0, "latest")
        return {"items": list(reversed(items))}
    except Exception as e:
        return {"items": [], "error": str(e)}
