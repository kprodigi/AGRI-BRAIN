"""Stdio MCP server entry point: ``python -m pirag.mcp.serve``.

Reads newline-delimited JSON-RPC requests from stdin, dispatches each
through the canonical MCPServer, and writes the JSON response to stdout.
Pairs with ``StdioTransport`` on the client side to honor the standard
MCP local-client pattern (one server process per agent, connected via
pipes).

Lifecycle is line-oriented: one request per line in, one response per
line out. EOF on stdin terminates the loop. Notifications (JSON-RPC
requests with no ``id`` field) are dispatched but no response is
written, per JSON-RPC 2.0 §4.

This file makes the README's "three transports" claim non-vacuous: a
real stdio entry point is now shippable, not just a class with no
process behind it.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

from .protocol import MCPMessage, MCPServer
from .registry import get_default_registry


def _serve(stdin: Any = None, stdout: Any = None, stderr: Any = None) -> int:
    """Run the stdio dispatch loop. Returns process exit code."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    server = MCPServer(server_name="agribrain-mcp", registry=get_default_registry())

    while True:
        try:
            line = stdin.readline()
        except KeyboardInterrupt:
            return 0
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue

        try:
            payload: Dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            stdout.write(json.dumps(err) + "\n")
            stdout.flush()
            continue

        msg = MCPMessage(
            jsonrpc=payload.get("jsonrpc", "2.0"),
            id=payload.get("id"),
            method=payload.get("method"),
            params=payload.get("params", {}),
        )

        # Notifications (no id) get no response, per JSON-RPC 2.0 §4.
        is_notification = "id" not in payload

        response = server.handle_message(msg)
        if is_notification:
            continue

        resp_dict: Dict[str, Any] = {
            "jsonrpc": response.jsonrpc,
            "id": response.id,
        }
        if response.result is not None:
            resp_dict["result"] = response.result
        if response.error is not None:
            resp_dict["error"] = response.error

        stdout.write(json.dumps(resp_dict, default=str) + "\n")
        stdout.flush()


def main() -> None:
    sys.exit(_serve())


if __name__ == "__main__":
    main()
