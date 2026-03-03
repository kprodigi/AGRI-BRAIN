
import os, json, re
from typing import Dict, Any
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
    base = os.getenv("SIM_API_BASE", "").rstrip("/")
    if not base or requests is None:
        return True
    url = f"{base}/sim/validate"
    try:
        r = requests.post(url, json={"answer": answer, "context": context}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("feasible", True))
    except Exception:
        return False
