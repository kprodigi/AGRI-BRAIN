"""Regression tests for measured-time footprint estimates and proxy labels."""
from __future__ import annotations

import pytest

from src.models.footprint import FootprintMeter


def test_time_based_estimates_use_elapsed_seconds_and_declared_rates():
    meter = FootprintMeter(
        assumed_active_power_W=12.0,
        water_per_server_second_L=2.0e-6,
        measurement_scope="unit-test timed action selection",
        proxy_step_unit="unit-test decision",
        energy_per_step_proxy_J=0.05,
        water_per_step_proxy_L=1.8e-6,
    )
    result = meter.compute_footprint(steps=2, elapsed_seconds=0.25)

    assert result["energy_J"] == pytest.approx(3.0)
    assert result["water_L"] == pytest.approx(0.5e-6)
    assert result["cumulative_elapsed_seconds"] == pytest.approx(0.25)
    assert result["estimate_basis"] == (
        "measured_elapsed_seconds_x_declared_rates"
    )
    assert result["estimation_status"] == (
        "activity-based estimate; not hardware telemetry"
    )
    assert result["measurement_scope"] == "unit-test timed action selection"
    assert result["proxy_step_unit"] == "unit-test decision"

    # Historical constants are retained only under unambiguous proxy labels.
    assert result["energy_per_step_proxy_J"] == pytest.approx(0.05)
    assert result["water_per_step_proxy_L"] == pytest.approx(1.8e-6)
    assert result["step_count_energy_proxy_J"] == pytest.approx(0.10)
    assert result["step_count_water_proxy_L"] == pytest.approx(3.6e-6)

    summary = meter.summary()
    assert summary["measurement_scope"] == "unit-test timed action selection"
    assert summary["proxy_step_unit"] == "unit-test decision"
    assert summary["cumulative_energy_J"] == pytest.approx(
        summary["assumed_active_power_W"]
        * summary["cumulative_elapsed_seconds"]
    )
    assert summary["cumulative_water_L"] == pytest.approx(
        summary["water_rate_L_per_server_second"]
        * summary["cumulative_elapsed_seconds"]
    )


def test_missing_timing_never_masquerades_as_time_based_measurement():
    meter = FootprintMeter()
    result = meter.compute_footprint(steps=3)

    assert result["time_based_estimate_available"] is False
    assert result["energy_J"] is None
    assert result["water_L"] is None
    assert result["cumulative_energy_J"] == 0.0
    assert result["cumulative_water_L"] == 0.0
    assert result["energy_per_step_proxy_J"] == pytest.approx(0.05)
    assert result["water_per_step_proxy_L"] == pytest.approx(1.8e-6)
    assert result["step_count_energy_proxy_J"] == pytest.approx(0.15)
    assert result["step_count_water_proxy_L"] == pytest.approx(5.4e-6)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"steps": -1, "elapsed_seconds": 0.1},
        {"steps": 1, "elapsed_seconds": -0.1},
        {"steps": 1, "elapsed_seconds": float("nan")},
        {"steps": 1, "elapsed_seconds": 0.1, "active_power_override_W": -1},
        {
            "steps": 1,
            "elapsed_seconds": 0.1,
            "water_rate_override_L_per_second": -1,
        },
    ),
)
def test_invalid_activity_inputs_fail_closed(kwargs):
    with pytest.raises(ValueError):
        FootprintMeter().compute_footprint(**kwargs)
