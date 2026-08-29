"""Publication MCP dispatch must not depend on wall time or call order."""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# The focused test exercises the disabled path, which never parses YAML. Keep
# it runnable in a minimal stdlib test interpreter while using real PyYAML in
# the locked publication environment.
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - only local minimal interpreter
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.YAMLError = Exception
    yaml_stub.safe_load = lambda stream: {}
    sys.modules["yaml"] = yaml_stub

from pirag.mcp.protocol import MCPMessage, MCPServer  # noqa: E402
from pirag.mcp.protocol_recorder import ProtocolRecorder  # noqa: E402
from pirag.mcp.registry import ToolRegistry, ToolSpec  # noqa: E402
import pirag.mcp.rate_limiter as rate_limiter  # noqa: E402
import pirag.mcp.tool_dispatch as tool_dispatch  # noqa: E402


class PublicationRateLimitTests(unittest.TestCase):
    def tearDown(self):
        rate_limiter._DEFAULT_LIMITER = None

    def test_disabled_publication_limiter_never_reads_wall_clock(self):
        with patch.dict(os.environ, {"MCP_RATE_LIMITS": "disabled"}):
            limiter = rate_limiter.RateLimiter(policy_path="unused-policy.yaml")
            with patch.object(
                rate_limiter.time,
                "monotonic",
                side_effect=AssertionError("wall clock must not be read"),
            ):
                for tool_name in (["calculator", "convert_units"] * 500):
                    limiter.check(tool_name, source="transport")
                    limiter.check(tool_name, source="registry")

    def test_protocol_bursts_are_order_independent_when_disabled(self):
        registry = ToolRegistry()
        for name in ("calculator", "convert_units"):
            registry.register(ToolSpec(
                name=name,
                description=name,
                capabilities=["test"],
                fn=lambda value, _name=name: {"tool": _name, "value": value},
                schema={"value": "integer"},
            ))
        server = MCPServer(registry=registry)
        orders = [
            ["calculator", "convert_units"] * 150,
            ["convert_units", "calculator"] * 150,
        ]
        with patch.dict(os.environ, {"MCP_RATE_LIMITS": "disabled"}):
            rate_limiter._DEFAULT_LIMITER = None
            with patch.object(
                rate_limiter.time,
                "monotonic",
                side_effect=AssertionError("wall clock must not be read"),
            ):
                for order in orders:
                    for call_id, tool_name in enumerate(order):
                        response = server.handle_message(MCPMessage(
                            id=call_id,
                            method="tools/call",
                            params={"name": tool_name, "arguments": {"value": call_id}},
                        ))
                        self.assertIsNone(response.error)
                        self.assertFalse(response.result.get("isError", False))
                        payload = json.loads(response.result["content"][0]["text"])
                        self.assertEqual(payload["tool"], tool_name)

    def test_strict_episode_summary_rejects_jsonrpc_and_real_tool_errors(self):
        protocol_server = MCPServer(registry=ToolRegistry())
        protocol_recorder = ProtocolRecorder(protocol_server)
        response = protocol_server.handle_message(MCPMessage(
            id=1, method="not/a/real/method",
        ))
        self.assertIsNotNone(response.error)
        with self.assertRaisesRegex(RuntimeError, "jsonrpc_errors=1"):
            protocol_recorder.finalize_episode(
                strict_validation=True, episode_label="protocol-error",
            )

        tool_registry = ToolRegistry()
        tool_registry.register(ToolSpec(
            name="broken_tool",
            description="raises for strict-gate test",
            capabilities=["test"],
            fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            schema={},
        ))
        tool_server = MCPServer(registry=tool_registry)
        tool_recorder = ProtocolRecorder(tool_server)
        response = tool_server.handle_message(MCPMessage(
            id=2,
            method="tools/call",
            params={"name": "broken_tool", "arguments": {}},
        ))
        self.assertTrue(response.result["isError"])
        with self.assertRaisesRegex(RuntimeError, "real_tool_isError_responses=1"):
            tool_recorder.finalize_episode(
                strict_validation=True, episode_label="tool-error",
            )

    def test_post_call_h3_drop_is_not_a_protocol_error(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="calculator",
            description="successful call",
            capabilities=["test"],
            fn=lambda value: {"value": value},
            schema={"value": "integer"},
        ))
        server = MCPServer(registry=registry)
        recorder = ProtocolRecorder(server)
        response = server.handle_message(MCPMessage(
            id=1,
            method="tools/call",
            params={"name": "calculator", "arguments": {"value": 7}},
        ))
        self.assertFalse(response.result.get("isError", False))

        # Mirrors H3: mutate the already-returned context payload after a
        # successful protocol call. This is the declared stress dose, not a
        # dispatcher/tool failure.
        post_call_context = {"calculator": {"value": 7}}
        post_call_context["calculator"] = None
        summary = recorder.finalize_episode(
            strict_validation=True, episode_label="h3-post-call-drop",
        )
        self.assertEqual(summary["jsonrpc_errors"], 0)
        self.assertEqual(summary["tool_iserror_responses_real"], 0)
        self.assertEqual(summary["real_error_responses"], 0)

    def test_strict_episode_summary_rejects_recorder_truncation(self):
        recorder = ProtocolRecorder(MCPServer(registry=ToolRegistry()))
        recorder._dropped = 1
        with self.assertRaisesRegex(RuntimeError, "exceeded recorder capacity"):
            recorder.finalize_episode(strict_validation=True)

    def test_dispatcher_distinguishes_structural_skip_from_trigger_failure(self):
        structural_workflow = [
            ("optional_tool", lambda obs, prior, shared: False,
             lambda obs, prior, shared: {}),
        ]
        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_structural": structural_workflow},
        ), patch.dict(
            os.environ,
            {"STRICT_VALIDATION": "1", "MCP_RATE_LIMITS": "disabled"},
        ):
            result = tool_dispatch.dispatch_tools(
                "unit_structural", object(), ToolRegistry()
            )
        self.assertEqual(result["_tools_skipped"], ["optional_tool"])
        self.assertEqual(result["_tools_failed"], [])

        broken_workflow = [
            ("optional_tool",
             lambda obs, prior, shared: (_ for _ in ()).throw(ValueError("bad trigger")),
             lambda obs, prior, shared: {}),
        ]
        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_trigger_failure": broken_workflow},
        ), patch.dict(
            os.environ,
            {"STRICT_VALIDATION": "0", "MCP_RATE_LIMITS": "disabled"},
        ):
            result = tool_dispatch.dispatch_tools(
                "unit_trigger_failure", object(), ToolRegistry()
            )
        self.assertEqual(result["_tools_failed"], ["optional_tool"])
        self.assertEqual(result["_tool_failure_details"][0]["stage"], "trigger")

        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_trigger_failure": broken_workflow},
        ), patch.dict(
            os.environ,
            {"STRICT_VALIDATION": "1", "MCP_RATE_LIMITS": "disabled"},
        ):
            with self.assertRaisesRegex(RuntimeError, r"optional_tool\[trigger\]"):
                tool_dispatch.dispatch_tools(
                    "unit_trigger_failure", object(), ToolRegistry()
                )

    def test_strict_dispatcher_rejects_argument_invoke_and_react_failures(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="test_tool", description="test", capabilities=["test"],
            fn=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("invoke failed")),
            schema={},
        ))
        argument_workflow = [
            ("test_tool", lambda obs, prior, shared: True,
             lambda obs, prior, shared: (_ for _ in ()).throw(ValueError("bad args"))),
        ]
        invoke_workflow = [
            ("test_tool", lambda obs, prior, shared: True,
             lambda obs, prior, shared: {}),
        ]
        strict_env = {"STRICT_VALIDATION": "1", "MCP_RATE_LIMITS": "disabled"}
        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_args": argument_workflow},
        ), patch.dict(os.environ, strict_env):
            with self.assertRaisesRegex(RuntimeError, r"test_tool\[arguments\]"):
                tool_dispatch.dispatch_tools("unit_args", object(), registry)

        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_invoke": invoke_workflow},
        ), patch.dict(os.environ, strict_env):
            with self.assertRaisesRegex(RuntimeError, r"test_tool\[invoke\]"):
                tool_dispatch.dispatch_tools("unit_invoke", object(), registry)

        react_registry = ToolRegistry()
        react_registry.register(ToolSpec(
            name="check_compliance", description="critical", capabilities=["test"],
            fn=lambda: {
                "compliant": False,
                "violations": [{"severity": "critical"}],
            },
            schema={},
        ))
        react_registry.register(ToolSpec(
            name="spoilage_forecast", description="broken", capabilities=["test"],
            fn=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("react failed")),
            schema={},
        ))
        react_workflow = [
            ("check_compliance", lambda obs, prior, shared: True,
             lambda obs, prior, shared: {}),
        ]

        class _Obs:
            rho = 0.4
            temp = 12.0
            rh = 90.0
            hour = 5.0

        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_react": react_workflow},
        ), patch.dict(os.environ, strict_env):
            with self.assertRaisesRegex(
                RuntimeError,
                r"spoilage_forecast\[conditional_followup_invoke\]",
            ):
                tool_dispatch.dispatch_tools(
                    "unit_react", _Obs(), react_registry
                )

        missing_followup_registry = ToolRegistry()
        missing_followup_registry.register(ToolSpec(
            name="check_compliance", description="critical", capabilities=["test"],
            fn=lambda: {
                "compliant": False,
                "violations": [{"severity": "critical"}],
            },
            schema={},
        ))
        with patch.dict(
            tool_dispatch.ROLE_WORKFLOWS,
            {"unit_react_missing": react_workflow},
        ), patch.dict(os.environ, strict_env):
            with self.assertRaisesRegex(
                RuntimeError,
                r"spoilage_forecast\[conditional_followup_registry_lookup\]",
            ):
                tool_dispatch.dispatch_tools(
                    "unit_react_missing", _Obs(), missing_followup_registry
                )


if __name__ == "__main__":
    unittest.main()
