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


def test_without_context_reconstruction_preserves_mode_and_temperature() -> None:
    """The policy trace must remove context without changing other inputs."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "agents" / "coordinator.py"
    ).read_text(encoding="utf-8")
    assert 'mode=self._step_mode' in source
    assert 'float(self._step_policy_temperature)' in source
    assert 'context_modifier=None' in source
