"""Record MCP dispatcher traffic during simulation.

Wraps the ``MCPServer.handle_message`` method to capture every
``(request, response)`` pair that flows through the in-process
dispatcher. The recorded records are *in-process dispatch traces*: the
project ``MCPMessage`` dataclasses are instantiated, and the JSON-RPC
method/params and dispatched return values are recorded — but they were never
serialized to a network socket. The previous version of this module
(and its docstring) called this "genuine protocol traffic over the
wire", which was inaccurate. The accurate framing is "project MCP-style
dispatcher invocations recorded in-process". This custom subset has not
undergone official MCP-conformance or client-interoperability testing. When the simulator wants
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

    def __init__(self, server: MCPServer, max_records: int = 4096) -> None:
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
        """Intercept and record every MCP message.

        Safe under the JSON-RPC notification path: when the wrapped
        ``handle_message`` returns ``None`` (per spec, notifications
        receive no response), this method records the request side
        only and returns ``None`` without dereferencing the missing
        response.
        """
        t0 = time.time()
        response = self._original_handler(msg)
        elapsed_ms = (time.time() - t0) * 1000.0

        with self._lock:
            if not self._enabled:
                return response

            if len(self._records) >= self.max_records:
                if self._dropped == 0:
                    _log.warning(
                        "ProtocolRecorder reached max_records=%d; subsequent "
                        "interactions will be absent from the exported trace.",
                        self.max_records,
                    )
                self._dropped += 1
                return response

            # JSON-RPC 2.0 notification: server returned None. Record
            # the request as a notification and return None so the
            # transport / caller honors the spec.
            if response is None:
                self._next_local_id += 1
                seq = self._next_local_id
                self._records.append({
                    "timestamp": time.time(),
                    "_recorder_seq": seq,
                    "_notification": True,
                    "request": {
                        "jsonrpc": msg.jsonrpc,
                        "id": msg.id,
                        "method": msg.method,
                        "params": msg.params,
                    },
                    "response": None,
                    "latency_ms": round(elapsed_ms, 3),
                })
                return None

            # Assign a monotonic local id when the caller forgot to set
            # one (notably the simulator's tool_dispatch, which used to
            # hard-code id=0 on every dispatched request). The wire id
            # remains whatever the caller sent; this `_recorder_seq`
            # field provides a per-record correlation key that is
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

    # Compatibility hook for an explicitly documented non-failure that a tool
    # must encode as ``isError``. There are currently no such cases. In
    # particular, chain_query reads the active same-episode audit ledger during
    # simulation; failure to reach any local audit source is a real error.
    _BY_DESIGN_TOOL_ERRORS: tuple[tuple[str, str], ...] = ()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            methods: Dict[str, int] = {}
            jsonrpc_errors = 0
            tool_iserror = 0
            tool_iserror_by_design = 0
            tool_iserror_breakdown: Dict[str, int] = {}
            notifications = 0
            for r in self._records:
                m = r["request"]["method"]
                methods[m] = methods.get(m, 0) + 1
                resp = r["response"]
                # Notifications carry response=None (per JSON-RPC 2.0
                # §4.1, notifications get no response). Skip them when
                # tallying error counters.
                if resp is None:
                    notifications += 1
                    continue
                if resp.get("error"):
                    jsonrpc_errors += 1
                # 2024-11-05 spec: tool failures appear as
                # result.isError == True with structured content; the
                # JSON-RPC error envelope is reserved for protocol-level
                # failures (unknown method, invalid params, etc).
                result = resp.get("result")
                if isinstance(result, dict) and result.get("isError") is True:
                    tool_iserror += 1
                    # Recover the tool name + error kind so any explicitly
                    # documented non-failure can be separated without hiding
                    # ordinary tool errors.
                    tool_name = ""
                    params = r["request"].get("params") or {}
                    if isinstance(params, dict):
                        tool_name = str(params.get("name") or "")
                    content = result.get("content") or []
                    error_text = ""
                    if (
                        isinstance(content, list)
                        and content
                        and isinstance(content[0], dict)
                    ):
                        error_text = str(content[0].get("text") or "")
                    breakdown_key = tool_name or "<unknown>"
                    tool_iserror_breakdown[breakdown_key] = (
                        tool_iserror_breakdown.get(breakdown_key, 0) + 1
                    )
                    for known_tool, known_kind in self._BY_DESIGN_TOOL_ERRORS:
                        if tool_name == known_tool and known_kind in error_text:
                            tool_iserror_by_design += 1
                            break
            tool_iserror_real = tool_iserror - tool_iserror_by_design
            real_error_responses = jsonrpc_errors + tool_iserror_real
            return {
                "total_interactions": len(self._records),
                "dropped_interactions": self._dropped,
                "methods": methods,
                "jsonrpc_errors": jsonrpc_errors,
                "tool_iserror_responses": tool_iserror,
                # Retain the real/by-design split for output-schema backward
                # compatibility. With no documented exclusions, *_real equals
                # the raw count and *_by_design is zero.
                "tool_iserror_responses_real": tool_iserror_real,
                "tool_iserror_responses_by_design": tool_iserror_by_design,
                "tool_iserror_breakdown": tool_iserror_breakdown,
                "notifications": notifications,
                "has_errors": jsonrpc_errors > 0 or tool_iserror > 0,
                "real_error_responses": real_error_responses,
                "has_real_errors": real_error_responses > 0,
            }

    def finalize_episode(
        self,
        *,
        strict_validation: bool = False,
        episode_label: str = "",
    ) -> Dict[str, Any]:
        """Return the episode summary and fail closed in strict runs.

        JSON-RPC errors and project MCP-style ``result.isError`` tool responses are
        treatment-changing failures: the dispatcher falls back to missing
        context after them.  A strict publication run therefore cannot retain
        an episode containing either.  Recorder truncation is also fatal in
        strict mode because a truncated trace cannot establish a zero-error
        count.  Deliberate H3 result drops occur *after* a successful protocol
        response and consequently do not increment any of these counters.
        """
        summary = self.summary()
        if not strict_validation:
            return summary

        label = f" for {episode_label}" if episode_label else ""
        dropped = int(summary.get("dropped_interactions", 0))
        if dropped:
            raise RuntimeError(
                "incomplete MCP protocol record"
                f"{label}: {dropped} interaction(s) exceeded recorder capacity"
            )

        jsonrpc_errors = int(summary.get("jsonrpc_errors", 0))
        real_tool_errors = int(summary.get("tool_iserror_responses_real", 0))
        if jsonrpc_errors or real_tool_errors:
            raise RuntimeError(
                "MCP protocol/tool failure"
                f"{label}: jsonrpc_errors={jsonrpc_errors}, "
                f"real_tool_isError_responses={real_tool_errors}, "
                f"tool_breakdown={summary.get('tool_iserror_breakdown', {})}"
            )
        return summary

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
