"""Record actual MCP JSON-RPC interactions during simulation.

Wraps the MCPServer's handle_message method to capture every request
and response pair. These are genuine protocol interactions, not
synthetic reconstructions.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from .protocol import MCPMessage, MCPServer


class ProtocolRecorder:
    """Records actual MCP protocol interactions."""

    def __init__(self, server: MCPServer, max_records: int = 200) -> None:
        self._server = server
        self._original_handler = server.handle_message
        self._records: List[Dict[str, Any]] = []
        self.max_records = max_records
        self._enabled = True

        # Intercept the server's handle_message
        server.handle_message = self._recording_handler  # type: ignore[method-assign]

    def _recording_handler(self, msg: MCPMessage) -> MCPMessage:
        """Intercept and record every MCP message."""
        t0 = time.time()
        response = self._original_handler(msg)
        elapsed_ms = (time.time() - t0) * 1000.0

        if self._enabled and len(self._records) < self.max_records:
            record: Dict[str, Any] = {
                "timestamp": time.time(),
                "request": {
                    "jsonrpc": msg.jsonrpc,
                    "id": msg.id,
                    "method": msg.method,
                    "params": msg.params,
                },
                "response": {
                    "jsonrpc": response.jsonrpc,
                    "id": response.id,
                },
                "latency_ms": round(elapsed_ms, 3),
            }
            if response.result is not None:
                record["response"]["result"] = _truncate(response.result, max_depth=3)
            if response.error is not None:
                record["response"]["error"] = response.error
            self._records.append(record)

        return response

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def get_records_for_method(self, method: str) -> List[Dict]:
        return [r for r in self._records if r["request"]["method"] == method]

    def export_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self._records, f, indent=2, default=str)

    def summary(self) -> Dict[str, Any]:
        methods: Dict[str, int] = {}
        for r in self._records:
            m = r["request"]["method"]
            methods[m] = methods.get(m, 0) + 1
        return {
            "total_interactions": len(self._records),
            "methods": methods,
            "has_errors": any(r["response"].get("error") for r in self._records),
        }

    def reset(self) -> None:
        self._records.clear()

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True


def _truncate(obj: Any, max_depth: int = 3, max_str_len: int = 200) -> Any:
    """Truncate nested dicts/lists for storage."""
    if max_depth <= 0:
        return "..." if isinstance(obj, (dict, list)) else obj
    if isinstance(obj, str) and len(obj) > max_str_len:
        return obj[:max_str_len] + "..."
    if isinstance(obj, dict):
        return {k: _truncate(v, max_depth - 1, max_str_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(v, max_depth - 1, max_str_len) for v in obj[:10]]
    return obj
