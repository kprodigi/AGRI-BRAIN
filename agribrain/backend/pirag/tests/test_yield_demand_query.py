"""Unit tests for the yield_query and demand_query MCP tools.

The previous test suite covered neither tool directly; integration
checks were limited to "the registry exposes the names". These tests
exercise both code paths (cached short-circuit and computed) and pin
the contract that the protocol traces expose.
"""
from __future__ import annotations

import pytest


# --- yield_query ----------------------------------------------------


def test_query_yield_cached_returns_cached_payload():
    from pirag.mcp.tools.yield_query import query_yield
    out = query_yield(
        cached_uncertainty=0.42,
        cached_forecast=[100.0, 101.0],
        cached_std=2.5,
    )
    assert out["source"] == "cached"
    assert out["uncertainty"] == 0.42
    assert out["forecast"] == [100.0, 101.0]
    assert out["std"] == 2.5


def test_query_yield_cached_clamps_to_unit_interval():
    from pirag.mcp.tools.yield_query import query_yield
    out = query_yield(cached_uncertainty=1.5)
    assert 0.0 <= out["uncertainty"] <= 1.0
    out_neg = query_yield(cached_uncertainty=-0.2)
    assert out_neg["uncertainty"] >= 0.0


def test_query_yield_no_history_returns_empty_computed():
    from pirag.mcp.tools.yield_query import query_yield
    out = query_yield(inventory_history=[])
    assert out["source"] == "computed"
    assert out["forecast"] == []
    assert out["uncertainty"] == 0.0


def test_query_yield_with_history_runs_holts_linear():
    from pirag.mcp.tools.yield_query import query_yield
    history = [100.0, 102.0, 104.0, 106.0, 108.0]
    out = query_yield(inventory_history=history, horizon=3)
    assert out["source"] == "computed"
    assert len(out["forecast"]) == 3
    # Trend is +2 per step, so the forecast should be monotone-ish
    assert out["forecast"][0] <= out["forecast"][-1] + 0.5
    assert 0.0 <= out["uncertainty"] <= 1.0
    assert out["std"] >= 0.0


# --- demand_query ---------------------------------------------------


def test_query_demand_cached_returns_cached_payload():
    from pirag.mcp.tools.demand_query import query_demand
    out = query_demand(
        cached_uncertainty=0.33,
        cached_forecast=[80.0],
        cached_std=4.0,
    )
    assert out["source"] == "cached"
    assert out["uncertainty"] == 0.33


def test_query_demand_no_history_returns_empty():
    from pirag.mcp.tools.demand_query import query_demand
    out = query_demand(demand_history=[])
    assert out["source"] == "computed"
    assert out["forecast"] == []


def test_query_demand_lstm_default():
    from pirag.mcp.tools.demand_query import query_demand
    history = [80.0, 90.0, 110.0, 95.0, 100.0, 105.0, 108.0, 95.0]
    out = query_demand(demand_history=history, horizon=2)
    assert out["source"] == "computed"
    assert len(out["forecast"]) == 2
    assert out["uncertainty"] >= 0.0


def test_query_demand_holts_alias_runs_holts_linear():
    from pirag.mcp.tools.demand_query import query_demand
    history = [80.0, 90.0, 110.0, 95.0, 100.0, 105.0]
    # The legacy `holt_winters` alias selects Holt's linear (level+trend);
    # the implementation is not seasonal Holt-Winters.
    out = query_demand(demand_history=history, horizon=1, method="holt_winters")
    assert out["source"] == "computed"
    assert len(out["forecast"]) == 1


# --- registry registration -----------------------------------------


def test_both_tools_are_registered():
    from pirag.mcp.registry import get_default_registry
    reg = get_default_registry()
    names = {t["name"] for t in reg.list_tools()}
    assert "yield_query" in names
    assert "demand_query" in names


def test_registry_status_reports_no_failures():
    from pirag.mcp.registry import get_default_registry, mcp_registration_status
    get_default_registry()  # ensure populated
    status = mcp_registration_status()
    assert status["registered_count"] >= 14
    # On a clean install no optional tools should fail to import.
    assert status["failed_count"] == 0
