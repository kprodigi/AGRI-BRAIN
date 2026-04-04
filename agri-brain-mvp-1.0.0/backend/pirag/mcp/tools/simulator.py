
import os
from typing import Dict, Any, Optional
from src.settings import SETTINGS
try:
    import requests
except Exception:
    requests = None

def _internal_headers() -> Dict[str, str]:
    """Build auth headers for internal API calls."""
    h: Dict[str, str] = {}
    if SETTINGS.require_api_key and SETTINGS.api_key:
        h["x-api-key"] = SETTINGS.api_key
    return h

def simulate(endpoint: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    base = SETTINGS.sim_api_base.rstrip("/")
    if not base or requests is None:
        return None
    url = f"{base}/{endpoint.lstrip('/')}"
    try:
        r = requests.post(url, json=payload, headers=_internal_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
