
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
    base = SETTINGS.sim_api_base.rstrip("/")
    if not base:
        raise SimulatorUnavailable("SETTINGS.sim_api_base is empty; no internal sim API configured")
    if requests is None:
        raise SimulatorUnavailable("requests not installed; cannot reach the internal sim API")
    url = f"{base}/{endpoint.lstrip('/')}"
    r = requests.post(url, json=payload, headers=_internal_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()
