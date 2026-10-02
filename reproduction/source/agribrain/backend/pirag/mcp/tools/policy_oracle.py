
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


def check_access(user_id: str, tool_name: str) -> dict:
    """Return ``{"allowed": bool, "reason": str}``.

    Returning a dict (rather than a bare bool) keeps the MCP tool
    contract uniform across tools — every tool returns a dict that
    the protocol layer JSON-encodes inside ``result.content[0].text``.
    The previous bare-bool return was an inconsistency relative to the
    rest of the MCP tool contract.
    """
    _load_policy()
    al = set(_POLICY.get("allowlist", []))
    if al and user_id not in al:
        return {
            "allowed": False,
            "reason": f"user {user_id!r} not in allowlist",
            "tool": tool_name,
        }
    return {"allowed": True, "reason": "ok", "tool": tool_name}
