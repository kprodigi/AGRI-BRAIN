
import os, json, re
from typing import Dict, Any
from src.settings import SETTINGS
try:
    import requests
except Exception:
    requests = None

def parse_numbers(answer: str):
    return [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", answer)]

def within_ranges(answer: str, constraints: Dict[str, Any]) -> bool:
    xs = parse_numbers(answer)
    if not xs:
        return True
    mn = constraints.get("min", float("-inf"))
    mx = constraints.get("max", float("inf"))
    return all(mn <= v <= mx for v in xs)

def verify_with_sim(answer: str, context: Dict[str, Any], timeout: int = 30) -> bool:
    base = SETTINGS.sim_api_base.rstrip("/")
    if not base or requests is None:
        return True
    url = f"{base}/sim/validate"
    headers: Dict[str, str] = {}
    if SETTINGS.require_api_key and SETTINGS.api_key:
        headers["x-api-key"] = SETTINGS.api_key
    try:
        r = requests.post(url, json={"answer": answer, "context": context},
                          headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("feasible", True))
    except Exception:
        return False
