"""Regression tests for the policy_oracle MCP tool's dict contract.

Reviewer-2 flagged that ``check_access`` previously returned a bare
bool while consumers (``context_builder.py``, ``tool_dispatch.py``)
defended against a dict shape. The bool→dict migration in 2026-04
made the contract uniform across MCP tools but introduced the risk
that a future refactor reverts to bool, breaking the consumers.

This module pins:

1. ``check_access`` returns a dict with the documented keys
   (``allowed``, ``reason``, ``tool``).
2. The two known consumers (``context_builder``,
   ``tool_dispatch._processor_calculator_trigger``) use defensive
   ``.get("allowed", True)`` access — a future change to ``is True``
   or ``== True`` would silently break the trigger when the oracle
   is absent.
3. A missing-from-allowlist user produces ``allowed=False``.
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure the policy YAML resolves; the loader is path-anchored relative
# to the policy_oracle module itself, so no cwd manipulation is needed.


def test_check_access_returns_dict_with_documented_keys():
    from pirag.mcp.tools.policy_oracle import check_access

    result = check_access(user_id="system", tool_name="surplus_management")
    assert isinstance(result, dict), (
        f"check_access must return a dict (the MCP tool contract); "
        f"got {type(result).__name__}"
    )
    # The three documented keys must all be present.
    for key in ("allowed", "reason", "tool"):
        assert key in result, (
            f"check_access result missing documented key {key!r}; "
            f"keys present: {sorted(result.keys())}"
        )
    assert isinstance(result["allowed"], bool), (
        f"`allowed` must be a bool, got {type(result['allowed']).__name__}"
    )
    assert isinstance(result["reason"], str)
    assert result["tool"] == "surplus_management"


def test_check_access_blocks_user_outside_allowlist(monkeypatch):
    """When the loaded policy has a non-empty allowlist that excludes
    the caller, ``allowed`` flips to False and the reason names the
    user. We patch the cached policy directly to avoid coupling the
    test to the YAML on disk.
    """
    from pirag.mcp.tools import policy_oracle

    # Stash and override the module-level policy cache.
    monkeypatch.setattr(policy_oracle, "_POLICY", {"allowlist": ["alice"]})
    # Bump mtime so _load_policy() does not re-read the YAML and undo
    # our patch. We patch the loader to a no-op for the duration.
    monkeypatch.setattr(policy_oracle, "_load_policy", lambda: None)

    result = policy_oracle.check_access(
        user_id="bob", tool_name="surplus_management"
    )
    assert result["allowed"] is False
    assert "bob" in result["reason"]
    assert result["tool"] == "surplus_management"

    # Allowlisted user → allowed.
    result_alice = policy_oracle.check_access(
        user_id="alice", tool_name="surplus_management"
    )
    assert result_alice["allowed"] is True


def test_consumers_use_defensive_dict_access():
    """The two known consumers must read ``allowed`` via ``.get`` so
    an absent oracle (oracle not yet invoked) defaults to permissive.
    A future refactor that flips to ``is True`` / ``== True`` /
    bare-truthy on the dict itself would defeat this contract — the
    truthiness of a dict is always True regardless of ``allowed``.
    """
    # context_builder.ROLE_QUERY_TEMPLATES["processor"]: trigger
    # expression must evaluate False when oracle is missing (so the
    # augmentation does not fire).
    from pirag.context_builder import ROLE_QUERY_TEMPLATES

    processor_conditions = ROLE_QUERY_TEMPLATES.get("processor", {}).get(
        "conditions", []
    )
    governance_rule = next(
        (r for r in processor_conditions if "governance" in r.get("append", "")),
        None,
    )
    assert governance_rule is not None, (
        "processor template should include a governance-policy condition; "
        "see context_builder.ROLE_QUERY_TEMPLATES['processor']['conditions']"
    )

    # Mock obs and an empty mcp dict — the trigger must NOT fire.
    class _Obs:
        surplus_ratio = 0.0

    # The trigger uses ``not (mcp.get(...) or {}).get("allowed", True)`` —
    # absent key defaults to True, so the trigger evaluates to False.
    assert governance_rule["trigger"](_Obs(), {}) is False, (
        "missing policy_oracle in mcp dict must default to permissive "
        "(allowed=True), so the governance augmentation does not fire"
    )

    # Explicit allowed=False fires the augmentation.
    mcp_blocked = {"policy_oracle": {"allowed": False, "reason": "denied"}}
    assert governance_rule["trigger"](_Obs(), mcp_blocked) is True

    # Explicit allowed=True does not fire.
    mcp_ok = {"policy_oracle": {"allowed": True, "reason": "ok"}}
    assert governance_rule["trigger"](_Obs(), mcp_ok) is False

    # tool_dispatch: _processor_calculator_trigger must coerce via .get.
    from pirag.mcp.tool_dispatch import _processor_calculator_trigger

    class _Obs2:
        surplus_ratio = 0.6

    # Empty prior dict → trigger returns True (permissive default).
    assert _processor_calculator_trigger(_Obs2(), {}, None) is True
    # Explicit allowed=False → blocked.
    assert _processor_calculator_trigger(
        _Obs2(),
        {"policy_oracle": {"allowed": False, "reason": "denied"}},
        None,
    ) is False
    # Below the surplus threshold → blocked regardless of oracle.
    class _Obs3:
        surplus_ratio = 0.1

    assert _processor_calculator_trigger(
        _Obs3(),
        {"policy_oracle": {"allowed": True, "reason": "ok"}},
        None,
    ) is False
