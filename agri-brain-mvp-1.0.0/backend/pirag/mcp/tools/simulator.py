
import os
from typing import Dict, Any, Optional
from src.settings import SETTINGS
try:
    import requests
except Exception:
    requests = None
def simulate(endpoint: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    base = SETTINGS.sim_api_base.rstrip("/")
    if not base or requests is None:
        return None
    url = f"{base}/{endpoint.lstrip('/')}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
