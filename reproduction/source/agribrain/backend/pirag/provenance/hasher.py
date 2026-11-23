"""Provenance hashing utilities for evidence and MCP tool invocations."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def hash_artifact(obj: Any) -> str:
    """SHA-256 hash of a JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def hash_text(s: str) -> str:
    """SHA-256 hash of a text string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_mcp_call(tool_name: str, args: Dict[str, Any], result: Any) -> str:
    """SHA-256 hash of an MCP tool invocation for provenance.

    Parameters
    ----------
    tool_name : name of the invoked tool.
    args : arguments passed to the tool.
    result : result returned by the tool.

    Returns
    -------
    Hex-encoded SHA-256 hash.
    """
    payload = {
        "tool": tool_name,
        "args": args,
        "result": result,
    }
    return hash_artifact(payload)
