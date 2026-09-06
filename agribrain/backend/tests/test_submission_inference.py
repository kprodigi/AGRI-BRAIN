"""Regression tests for the submission-ready scientific posture."""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd


def test_mechanistic_spoilage_is_bounded_and_monotone() -> None:
    from src.models.spoilage import compute_spoilage

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=12, freq="15min"),
        "tempC": np.linspace(4.0, 12.0, 12),
        "RH": np.linspace(85.0, 94.0, 12),
    })
    mechanistic = compute_spoilage(frame)
    risk = mechanistic["spoilage_risk"].to_numpy()
    shelf = mechanistic["shelf_left"].to_numpy()
    assert np.all((0.0 <= risk) & (risk <= 1.0))
    assert np.all(np.diff(risk) >= 0.0)
    np.testing.assert_allclose(shelf, 1.0 - risk, rtol=0.0, atol=0.0)


def test_forward_spoilage_forecast_continues_lag_clock() -> None:
    from pirag.mcp.tools.spoilage_forecast import forecast_spoilage

    fresh_clock = forecast_spoilage(0.2, 10.0, 90.0, hours_ahead=6, age_hours=0.0)
    mature_clock = forecast_spoilage(0.2, 10.0, 90.0, hours_ahead=6, age_hours=24.0)
    assert mature_clock["forecast_rho"] > fresh_clock["forecast_rho"]
    assert mature_clock["age_hours"] == 24.0


def _stress_module():
    sim = Path(__file__).resolve().parents[3] / "mvp" / "simulation"
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))
    from benchmarks import run_stress_suite
    return run_stress_suite


def test_h3_tost_accepts_small_paired_seed_changes() -> None:
    result = _stress_module()._equivalence_tost(
        [-0.002, -0.001, 0.0, 0.001, 0.002] * 4,
        margin=0.01,
    )
    assert result["n"] == 20
    assert result["equivalent_alpha_0p05"] is True
    assert result["ci90_low"] > -0.01
    assert result["ci90_high"] < 0.01


def test_h3_tost_rejects_changes_outside_declared_margin() -> None:
    result = _stress_module()._equivalence_tost(
        [0.018, 0.019, 0.020, 0.021, 0.022] * 4,
        margin=0.01,
    )
    assert result["n"] == 20
    assert result["equivalent_alpha_0p05"] is False
    assert result["ci90_low"] > 0.01


def test_h3_json_sanitizer_maps_absent_dataframe_cells_to_null() -> None:
    cleaned = _stress_module()._json_safe({
        "missing": float("nan"),
        "positive_inf": float("inf"),
        "finite": np.float64(0.25),
        "nested": [np.int64(2), -math.inf],
    })
    assert cleaned == {
        "missing": None,
        "positive_inf": None,
        "finite": 0.25,
        "nested": [2, None],
    }
    assert json.loads(json.dumps(cleaned, allow_nan=False)) == cleaned


def test_without_context_reconstruction_preserves_mode_and_temperature() -> None:
    """The policy trace must remove context without changing other inputs."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "agents" / "coordinator.py"
    ).read_text(encoding="utf-8")
    assert 'mode=self._step_mode' in source
    assert 'float(self._step_policy_temperature)' in source
    assert 'context_modifier=None' in source
