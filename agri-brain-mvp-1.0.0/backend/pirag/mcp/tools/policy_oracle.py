
import os, time, yaml
from typing import Dict, Any
_POLICY = None
_POLICY_MTIME = 0.0
def _load_policy():
    global _POLICY, _POLICY_MTIME
    path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "policy.yaml")
    path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(path)
        if mtime != _POLICY_MTIME:
            with open(path, "r", encoding="utf-8") as f:
                _POLICY = yaml.safe_load(f) or {}
            _POLICY_MTIME = mtime
    except Exception:
        _POLICY = {"allowlist": [], "retention_days": 30, "rate_limits": {}}
def check_access(user_id: str, tool_name: str) -> bool:
    _load_policy()
    al = set(_POLICY.get("allowlist", []))
    if al and user_id not in al:
        return False
    return True
