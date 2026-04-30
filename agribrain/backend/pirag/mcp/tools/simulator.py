
from typing import Dict, Any
from src.settings import SETTINGS
try:
    import requests
except Exception:
    requests = None


class SimulatorUnavailable(RuntimeError):
    """Raised when the internal sim API is not reachable.

    Surfacing as an exception lets the MCP server's tools/call handler set
    ``result.isError = True`` so the protocol recorder counts the call as
    a tool failure instead of a silent ``None``.
    """


def _internal_headers() -> Dict[str, str]:
    """Build auth headers for internal API calls."""
    h: Dict[str, str] = {}
    if SETTINGS.require_api_key and SETTINGS.api_key:
        h["x-api-key"] = SETTINGS.api_key
    return h


def simulate(endpoint: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """Run a forward simulation via the internal sim API.

    Returns a structured ``{"_status": "error", ...}`` payload when the
    sim API is not configured (e.g., simulator subprocess), instead of
    raising. The MCP tools/call handler now flips ``result.isError =
    True`` when ``_status == "error"`` so the failure is still visible
    as a tool failure in protocol traces — but it does not crash the
    dispatcher with an exception, and the recorder records a
    well-formed response either way.
    """
    base = SETTINGS.sim_api_base.rstrip("/")
    if not base:
        return {
            "_status": "error",
            "_error_kind": "sim_api_not_configured",
            "_message": (
                "SETTINGS.sim_api_base is empty; no internal sim API "
                "configured. Set SIM_API_BASE in the runtime env to "
                "enable forward-simulation MCP calls."
            ),
        }
    if requests is None:
        return {
            "_status": "error",
            "_error_kind": "requests_not_installed",
            "_message": "requests not installed; cannot reach the internal sim API",
        }
    url = f"{base}/{endpoint.lstrip('/')}"
    try:
        r = requests.post(url, json=payload, headers=_internal_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {
            "_status": "error",
            "_error_kind": "request_failed",
            "_message": f"{type(exc).__name__}: {exc}",
        }
