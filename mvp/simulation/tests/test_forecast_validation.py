from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from mvp.simulation.validation.validate_forecasts import (
    CANONICAL_RESULTS_DIR,
    DEFAULT_DATA,
    REPO_ROOT,
    _metrics,
    build_validation,
    main,
    rolling_origin_records,
)


def test_rolling_origin_forecasts_never_see_the_target() -> None:
    series = np.linspace(10.0, 29.0, 20)

    def persistence(frame: pd.DataFrame) -> dict:
        value = float(frame["x"].iloc[-1])
        return {
            "forecast": [value],
            "ci_lower": [value - 1.0],
            "ci_upper": [value + 1.0],
        }

    rows = rolling_origin_records(
        series, model_name="fixture", forecast_fn=persistence,
        column="x", lookback=6,
    )
    assert rows
    assert all(row["history_end_index"] < row["target_index"] for row in rows)
    assert all(row["history_count"] <= 6 for row in rows)
    assert {row["split"] for row in rows} == {"validation", "test"}


def test_forecast_metrics_compare_against_persistence_and_coverage() -> None:
    rows = [
        {
            "target": 10.0,
            "prediction": 9.0,
            "persistence_prediction": 8.0,
            "interval_lower": 8.5,
            "interval_upper": 10.5,
        },
        {
            "target": 12.0,
            "prediction": 12.0,
            "persistence_prediction": 10.0,
            "interval_lower": 11.0,
            "interval_upper": 13.0,
        },
    ]
    metrics = _metrics(rows)
    assert metrics["n"] == 2
    assert metrics["mae"] == 0.5
    assert metrics["persistence_mae"] == 2.0
    assert metrics["interval_coverage"] == 1.0
    assert metrics["mae_improvement_vs_persistence_fraction"] == 0.75


def test_canonical_forecast_receipt_requires_commit_and_run_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGRIBRAIN_GIT_COMMIT", raising=False)
    monkeypatch.delenv("RUN_TAG", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_forecasts.py", "--output-dir", str(CANONICAL_RESULTS_DIR)],
    )
    with pytest.raises(RuntimeError, match="source commit"):
        main()


def test_publication_forecast_replay_requires_validator_authority(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGRIBRAIN_PUBLICATION_REPLAY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_forecasts.py",
            "--output-dir", str(tmp_path),
            "--source-commit", "a" * 40,
            "--run-tag", "aaaaaaa_20260828_120000",
            "--publication-replay",
        ],
    )
    with pytest.raises(RuntimeError, match="restricted"):
        main()


@pytest.mark.slow
def test_locked_forecast_selection_and_metrics_match_protocol() -> None:
    summary, rows = build_validation(DEFAULT_DATA)
    protocol = json.loads(
        (REPO_ROOT / "mvp" / "simulation" / "experiment_protocol.json")
        .read_text(encoding="utf-8")
    )["forecast_protocol"]

    assert len(rows) == 580
    assert summary["selection_rule"] == {
        "criterion": "minimum validation-segment RMSE",
        "test_segment_used_for_selection": False,
        "selected_demand_method": "holt_linear",
        "selected_demand_model_id": "holt_linear_demand_candidate",
        "selected_supply_proxy_method": "persistence",
        "selected_supply_proxy_model_id": "persistence_supply_proxy",
    }
    assert summary["dataset"]["sha256"] == protocol["dataset_sha256"]
    bindings = {
        "demand_holt_linear": "holt_linear_demand_candidate",
        "demand_persistence": "persistence_demand",
        "demand_lstm": "lstm_demand",
        "supply_holt_linear": "holt_linear_supply_proxy",
        "supply_persistence": "persistence_supply_proxy",
    }
    for locked_name, model in bindings.items():
        assert summary["metrics"][model]["validation"]["rmse"] == pytest.approx(
            protocol["validation_rmse"][locked_name], rel=1e-12, abs=1e-12,
        )
        assert summary["metrics"][model]["test"]["rmse"] == pytest.approx(
            protocol["test_rmse_report_only"][locked_name], rel=1e-12, abs=1e-12,
        )
