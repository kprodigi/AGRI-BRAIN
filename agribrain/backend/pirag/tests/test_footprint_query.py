"""The MCP footprint query must not masquerade proxies as telemetry."""
from __future__ import annotations

import pytest

from pirag.mcp.tools.footprint_query import query_footprint


def test_footprint_query_labels_fixed_step_coefficients_as_proxies():
    result = query_footprint(steps_completed=288)

    assert result["estimate_basis"] == "fixed_per_step_proxy_not_elapsed_time"
    assert result["estimation_status"] == "proxy only; not hardware telemetry"
    assert result["time_based_estimate_available"] is False
    assert result["proxy_step_unit"] == (
        "simulation/inference step count supplied by caller"
    )
    assert result["efficiency_flag_basis"] == (
        "declared_energy_per_step_proxy_threshold"
    )
    assert "per_step" not in result
    assert "cumulative" not in result
    assert result["per_step_proxy"]["energy_j"] == pytest.approx(0.05)
    assert result["per_step_proxy"]["water_l"] == pytest.approx(1.8e-6)
    assert result["cumulative_step_count_proxy"]["energy_j"] == pytest.approx(
        14.4
    )
    assert result["cumulative_step_count_proxy"]["water_l"] == pytest.approx(
        288 * 1.8e-6
    )


def test_footprint_query_carbon_is_explicitly_proxy_derived():
    result = query_footprint(steps_completed=10, energy_per_step_j=1.0)
    expected_kwh = 10.0 / 3_600_000.0
    expected_kg = expected_kwh * result[
        "grid_carbon_intensity_proxy_kg_per_kwh"
    ]
    assert result["cumulative_step_count_proxy"]["co2_kg"] == pytest.approx(
        expected_kg,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"steps_completed": -1},
        {"steps_completed": 1.5},
        {"steps_completed": 1, "energy_per_step_j": -0.1},
        {"steps_completed": 1, "water_per_step_l": float("nan")},
    ),
)
def test_footprint_query_rejects_invalid_proxy_inputs(kwargs):
    with pytest.raises(ValueError):
        query_footprint(**kwargs)
