"""Record MCP dispatcher traffic during simulation.

Wraps the ``MCPServer.handle_message`` method to capture every
``(request, response)`` pair that flows through the in-process
dispatcher. The recorded records are *in-process dispatch traces*: the
``MCPMessage`` dataclasses are real, the JSON-RPC method/params are
real, the dispatched return values are real — but they were never
serialized to a network socket. The previous version of this module
(and its docstring) called this "genuine protocol traffic over the
wire", which was inaccurate. The accurate framing is "real MCP
dispatcher invocations recorded in-process". When the simulator wants
serialization round-trip behaviour, it should drive
``MCPClient(InProcessTransport(server))``, which JSON-roundtrips
inside ``InProcessTransport.send`` (see ``transport.py``). The
recorder still provides honest evidence of which methods were called,
in what order, with what params, and how long they took.

Counts ``isError`` tool responses as errors in ``summary()``; the
2024-11-05 MCP spec routes tool failures through ``result.isError``
rather than the JSON-RPC ``error`` field, and the previous summary
missed those.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List

from .protocol import MCPMessage, MCPServer


_log = logging.getLogger(__name__)


class ProtocolRecorder:
    """Records MCP dispatcher invocations in-process."""

    def __init__(self, server: MCPServer, max_records: int = 200) -> None:
        self._server = server
        self._original_handler = server.handle_message
        self._records: List[Dict[str, Any]] = []
        self.max_records = max_records
        self._enabled = True
        self._lock = threading.Lock()
        self._dropped = 0
        self._next_local_id = 0

        # Intercept the server's handle_message
        server.handle_message = self._recording_handler  # type: ignore[method-assign]

    def _recording_handler(self, msg: MCPMessage) -> MCPMessage:
        """Intercept and record every MCP message."""
        t0 = time.time()
        response = self._original_handler(msg)
        elapsed_ms = (time.time() - t0) * 1000.0

        with self._lock:
            if not self._enabled:
                return response
            if len(self._records) >= self.max_records:
                if self._dropped == 0:
                    _log.warning(
                        "ProtocolRecorder reached max_records=%d; further "
                        "records will be dropped silently. Increase "
                        "max_records or rotate to disk.",
                        self.max_records,
                    )
                self._dropped += 1
                return response

            # Assign a monotonic local id when the caller forgot to set
            # one (notably the simulator's tool_dispatch, which used to
            # hard-code id=0 on every dispatched request). The wire id
            # remains whatever the caller sent; this `_recorder_seq`
            # field gives reviewers a per-record correlation key that is
            # always unique even when the upstream caller does not
            # multiplex.
            self._next_local_id += 1
            seq = self._next_local_id

            record: Dict[str, Any] = {
                "timestamp": time.time(),
                "_recorder_seq": seq,
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
        with self._lock:
            return list(self._records)

    def get_records_for_method(self, method: str) -> List[Dict]:
        with self._lock:
            return [r for r in self._records if r["request"]["method"] == method]

    def export_json(self, filepath: str) -> None:
        with self._lock:
            data = list(self._records)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            methods: Dict[str, int] = {}
            jsonrpc_errors = 0
            tool_iserror = 0
            for r in self._records:
                m = r["request"]["method"]
                methods[m] = methods.get(m, 0) + 1
                resp = r["response"]
                if resp.get("error"):
                    jsonrpc_errors += 1
                # 2024-11-05 spec: tool failures appear as
                # result.isError == True with structured content; the
                # JSON-RPC error envelope is reserved for protocol-level
                # failures (unknown method, invalid params, etc).
                result = resp.get("result")
                if isinstance(result, dict) and result.get("isError") is True:
                    tool_iserror += 1
            return {
                "total_interactions": len(self._records),
                "dropped_interactions": self._dropped,
                "methods": methods,
                "jsonrpc_errors": jsonrpc_errors,
                "tool_iserror_responses": tool_iserror,
                "has_errors": jsonrpc_errors > 0 or tool_iserror > 0,
            }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._dropped = 0
            self._next_local_id = 0

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
