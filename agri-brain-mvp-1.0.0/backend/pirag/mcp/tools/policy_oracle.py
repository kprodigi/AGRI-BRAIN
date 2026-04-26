
import logging
import os
import yaml

_log = logging.getLogger(__name__)

_POLICY = None
_POLICY_MTIME = 0.0


class PolicyLoadError(RuntimeError):
    """Raised when the access-policy YAML cannot be loaded.

    The previous code silently fell back to an empty policy on any IO or
    parse error, which disabled allowlist enforcement without surfacing
    a signal. Surfacing as an exception lets the MCP server's tools/call
    handler set ``result.isError = True`` so the failure is visible to
    auditors and to the MCP Tool Reliability counter.
    """


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
    except FileNotFoundError as exc:
        raise PolicyLoadError(f"policy.yaml not found at {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyLoadError(f"failed to load policy.yaml ({path}): {exc}") from exc


def check_access(user_id: str, tool_name: str) -> bool:
    _load_policy()
    al = set(_POLICY.get("allowlist", []))
    if al and user_id not in al:
        return False
    return True
