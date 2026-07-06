"""ProtocolRecorder.summary() reliability-counter split.

Fig 9(b) of the paper reports MCP tool reliability across a benchmark
run. The pre-2026-05 recorder counted every ``isError=True`` response
into ``tool_iserror_responses``, including chain_query's deliberate
``state_unavailable`` errors when the simulator runs without a live
FastAPI app. The 2026-05 fix adds ``tool_iserror_responses_real``
and ``tool_iserror_responses_by_design`` so the figure can use the
clean count without losing the raw total.
"""
from __future__ import annotations

import pytest

from pirag.mcp.protocol_recorder import ProtocolRecorder


class _StubServer:
    """Minimal stand-in for MCPServer that the recorder can wrap."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def handle_message(self, msg):
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


def _request(method: str, tool_name: str | None = None) -> dict:
    params = {"name": tool_name} if tool_name else {}
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


def _tool_error_response(text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": True,
        },
    }


def _tool_success_response() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "{\"ok\": true}"}]},
    }


@pytest.fixture
def recorder():
    server = _StubServer([])
    return ProtocolRecorder(server)


def _record(recorder, request, response):
    """Push a (request, response) pair into the recorder's internal log."""
    with recorder._lock:
        recorder._records.append({"request": request, "response": response})


def test_chain_query_state_unavailable_counted_as_by_design(recorder):
    _record(
        recorder,
        _request("tools/call", "chain_query"),
        _tool_error_response('{"_error_kind": "state_unavailable", "_status": "error"}'),
    )
    summary = recorder.summary()
    assert summary["tool_iserror_responses"] == 1
    assert summary["tool_iserror_responses_by_design"] == 1
    assert summary["tool_iserror_responses_real"] == 0
    assert summary["tool_iserror_breakdown"] == {"chain_query": 1}


def test_other_tool_error_counted_as_real(recorder):
    _record(
        recorder,
        _request("tools/call", "calculator"),
        _tool_error_response('{"error": "division by zero"}'),
    )
    summary = recorder.summary()
    assert summary["tool_iserror_responses"] == 1
    assert summary["tool_iserror_responses_by_design"] == 0
    assert summary["tool_iserror_responses_real"] == 1
    assert summary["tool_iserror_breakdown"] == {"calculator": 1}


def test_chain_query_non_state_error_counted_as_real(recorder):
    """A genuine chain_query failure (not state_unavailable) counts as real."""
    _record(
        recorder,
        _request("tools/call", "chain_query"),
        _tool_error_response('{"error": "RPC timeout"}'),
    )
    summary = recorder.summary()
    assert summary["tool_iserror_responses_by_design"] == 0
    assert summary["tool_iserror_responses_real"] == 1


def test_mixed_records_split_correctly(recorder):
    _record(recorder, _request("tools/call", "chain_query"),
            _tool_error_response('"_error_kind": "state_unavailable"'))
    _record(recorder, _request("tools/call", "calculator"),
            _tool_error_response('{"error": "boom"}'))
    _record(recorder, _request("tools/call", "pirag_query"),
            _tool_success_response())
    _record(recorder, _request("tools/call", "chain_query"),
            _tool_error_response('{"_error_kind": "state_unavailable"}'))
    summary = recorder.summary()
    assert summary["tool_iserror_responses"] == 3
    assert summary["tool_iserror_responses_by_design"] == 2
    assert summary["tool_iserror_responses_real"] == 1
    assert summary["tool_iserror_breakdown"] == {"chain_query": 2, "calculator": 1}


def test_summary_returns_zero_when_no_errors(recorder):
    _record(recorder, _request("tools/call", "calculator"), _tool_success_response())
    summary = recorder.summary()
    assert summary["tool_iserror_responses"] == 0
    assert summary["tool_iserror_responses_real"] == 0
    assert summary["tool_iserror_responses_by_design"] == 0
    assert summary["tool_iserror_breakdown"] == {}
    assert summary["has_errors"] is False
