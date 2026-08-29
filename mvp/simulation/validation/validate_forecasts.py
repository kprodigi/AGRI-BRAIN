#!/usr/bin/env python3
"""Leakage-free rolling-origin validation of the two benchmark forecasters.

The repository contains a synthetic spinach time series, not an independent
field dataset.  This script therefore evaluates internal predictive behavior
only and labels it accordingly.  It never upgrades the benchmark to external
validation.

The diagnostic compares LSTM, non-seasonal Holt-linear, and persistence demand
forecasts, and compares Holt-linear with persistence for inventory (the
declared supply/yield proxy). Each target is predicted from observations
strictly earlier than that target. The first 60 percent of targets are
development history, the next 20 percent select the minimum-RMSE candidate,
and the final 20 percent form a report-only locked test segment. The selected
confirmatory defaults are Holt-linear demand and persistence supply; the test
segment is never used for selection and this internal synthetic exercise is
not external predictive validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "agribrain" / "backend"
import sys

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.models.lstm_demand import lstm_demand_forecast  # noqa: E402
from src.models.persistence_forecast import persistence_forecast  # noqa: E402
from src.models.yield_forecast import yield_supply_forecast  # noqa: E402


DEFAULT_DATA = BACKEND_ROOT / "src" / "data_spinach.csv"
CANONICAL_RESULTS_DIR = REPO_ROOT / "mvp" / "simulation" / "results"
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_for_target(target_index: int, n: int) -> str:
    validation_start = int(math.floor(0.60 * n))
    test_start = int(math.floor(0.80 * n))
    if target_index < validation_start:
        return "development"
    if target_index < test_start:
        return "validation"
    return "test"


def rolling_origin_records(
    series: np.ndarray,
    *,
    model_name: str,
    forecast_fn: Callable[[pd.DataFrame], dict[str, Any]],
    column: str,
    lookback: int = 48,
    origin_stride: int = 1,
) -> list[dict[str, Any]]:
    """Return one-step forecasts with an auditable no-lookahead boundary."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or len(values) < max(6, lookback // 2):
        raise ValueError("forecast validation series is too short")
    if not np.all(np.isfinite(values)):
        raise ValueError("forecast validation series contains non-finite values")
    if origin_stride < 1:
        raise ValueError("origin_stride must be positive")

    first_origin = max(2, int(math.floor(0.60 * len(values))) - 1)
    rows: list[dict[str, Any]] = []
    for origin in range(first_origin, len(values) - 1, origin_stride):
        history_start = max(0, origin + 1 - lookback)
        history = values[history_start:origin + 1]
        result = forecast_fn(pd.DataFrame({column: history}))
        forecast = list(result.get("forecast", []))
        lower = list(result.get("ci_lower", []))
        upper = list(result.get("ci_upper", []))
        if not forecast:
            raise RuntimeError(f"{model_name} returned no one-step forecast")
        target_index = origin + 1
        target = float(values[target_index])
        prediction = float(forecast[0])
        lo = float(lower[0]) if lower else float("nan")
        hi = float(upper[0]) if upper else float("nan")
        rows.append({
            "model": model_name,
            "series": column,
            "split": _split_for_target(target_index, len(values)),
            "origin_index": int(origin),
            "target_index": int(target_index),
            "history_start_index": int(history_start),
            "history_end_index": int(origin),
            "history_count": int(len(history)),
            "target": target,
            "prediction": prediction,
            "persistence_prediction": float(history[-1]),
            "interval_lower": lo,
            "interval_upper": hi,
            "interval_nominal_coverage": 0.95,
            "no_lookahead": bool(origin < target_index),
        })
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    target = np.asarray([row["target"] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    persistence = np.asarray(
        [row["persistence_prediction"] for row in rows], dtype=float,
    )
    lower = np.asarray([row["interval_lower"] for row in rows], dtype=float)
    upper = np.asarray([row["interval_upper"] for row in rows], dtype=float)
    error = pred - target
    persistence_error = persistence - target
    finite_interval = np.isfinite(lower) & np.isfinite(upper)
    coverage = (
        float(np.mean((target[finite_interval] >= lower[finite_interval])
                      & (target[finite_interval] <= upper[finite_interval])))
        if finite_interval.any() else None
    )
    width = (
        float(np.mean(upper[finite_interval] - lower[finite_interval]))
        if finite_interval.any() else None
    )
    mae = float(np.mean(np.abs(error)))
    persistence_mae = float(np.mean(np.abs(persistence_error)))
    return {
        "n": int(len(rows)),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_error": float(np.mean(error)),
        "persistence_mae": persistence_mae,
        "persistence_rmse": float(np.sqrt(np.mean(persistence_error ** 2))),
        "mae_improvement_vs_persistence_fraction": (
            float((persistence_mae - mae) / persistence_mae)
            if persistence_mae > 0.0 else None
        ),
        "interval_coverage": coverage,
        "mean_interval_width": width,
    }


def build_validation(
    data_path: Path,
    *,
    lstm_epochs: int = 80,
    origin_stride: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(data_path)
    required = {"demand_units", "inventory_units"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"forecast dataset lacks columns {sorted(missing)}")

    demand_rows = rolling_origin_records(
        frame["demand_units"].to_numpy(dtype=float),
        model_name="lstm_demand",
        column="demand_units",
        lookback=48,
        origin_stride=origin_stride,
        forecast_fn=lambda history: lstm_demand_forecast(
            history, horizon=1, lookback=48, hidden_size=16,
            epochs=lstm_epochs, seed=42,
        ),
    )
    demand_holt_rows = rolling_origin_records(
        frame["demand_units"].to_numpy(dtype=float),
        model_name="holt_linear_demand_candidate",
        column="demand_units",
        lookback=48,
        origin_stride=origin_stride,
        forecast_fn=lambda history: yield_supply_forecast(
            history, horizon=1, lookback=48, ema_alpha=0.5,
            trend_beta=0.2, series_col="demand_units",
        ),
    )
    demand_persistence_rows = rolling_origin_records(
        frame["demand_units"].to_numpy(dtype=float),
        model_name="persistence_demand",
        column="demand_units",
        lookback=48,
        origin_stride=origin_stride,
        forecast_fn=lambda history: persistence_forecast(
            history, horizon=1, lookback=48, residual_tail=8,
            series_col="demand_units",
        ),
    )
    supply_rows = rolling_origin_records(
        frame["inventory_units"].to_numpy(dtype=float),
        model_name="holt_linear_supply_proxy",
        column="inventory_units",
        lookback=48,
        origin_stride=origin_stride,
        forecast_fn=lambda history: yield_supply_forecast(
            history, horizon=1, lookback=48, ema_alpha=0.5,
            trend_beta=0.2,
        ),
    )
    supply_persistence_rows = rolling_origin_records(
        frame["inventory_units"].to_numpy(dtype=float),
        model_name="persistence_supply_proxy",
        column="inventory_units",
        lookback=48,
        origin_stride=origin_stride,
        forecast_fn=lambda history: persistence_forecast(
            history, horizon=1, lookback=48, residual_tail=8,
            series_col="inventory_units",
        ),
    )
    records = (
        demand_rows + demand_holt_rows + demand_persistence_rows
        + supply_rows + supply_persistence_rows
    )
    if not all(row["no_lookahead"] for row in records):
        raise RuntimeError("forecast validation contains look-ahead leakage")

    metrics: dict[str, Any] = {}
    for model in sorted({row["model"] for row in records}):
        metrics[model] = {}
        for split in ("validation", "test"):
            metrics[model][split] = _metrics([
                row for row in records
                if row["model"] == model and row["split"] == split
            ])

    demand_candidates = (
        "lstm_demand", "holt_linear_demand_candidate", "persistence_demand",
    )
    supply_candidates = (
        "holt_linear_supply_proxy", "persistence_supply_proxy",
    )
    selected_demand_model = min(
        demand_candidates,
        key=lambda name: metrics[name]["validation"]["rmse"],
    )
    selected_supply_model = min(
        supply_candidates,
        key=lambda name: metrics[name]["validation"]["rmse"],
    )
    demand_runtime_methods = {
        "lstm_demand": "lstm",
        "holt_linear_demand_candidate": "holt_linear",
        "persistence_demand": "persistence",
    }
    supply_runtime_methods = {
        "holt_linear_supply_proxy": "holt_linear",
        "persistence_supply_proxy": "persistence",
    }
    try:
        dataset_path = data_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        dataset_path = data_path.resolve().as_posix()

    summary = {
        "schema_version": 1,
        "validation_scope": "internal synthetic benchmark only",
        "external_validation": False,
        "dataset": {
            "path": dataset_path,
            "sha256": _sha256(data_path),
            "n_rows": int(len(frame)),
            "temporal_split": {
                "development": "first 60 percent",
                "validation": "next 20 percent",
                "test": "final 20 percent",
            },
        },
        "rolling_origin": {
            "horizon_steps": 1,
            "lookback_steps": 48,
            "origin_stride": int(origin_stride),
            "retrained_at_each_origin": True,
            "targets_seen_during_fit": False,
        },
        "models": {
            "lstm_demand": {
                "hidden_units": 16,
                "optimizer": "full-sequence gradient descent",
                "learning_rate": 0.005,
                "epochs": int(lstm_epochs),
                "batch_size": "one rolling sequence",
                "early_stopping": False,
                "initialization_seed": 42,
            },
            "holt_linear_supply_proxy": {
                "series": "inventory_units",
                "ema_alpha": 0.5,
                "trend_beta": 0.2,
                "seasonal_component": False,
            },
            "holt_linear_demand_candidate": {
                "series": "demand_units",
                "ema_alpha": 0.5,
                "trend_beta": 0.2,
                "seasonal_component": False,
            },
            "persistence": {
                "point_forecast": "last observed value",
                "uncertainty": "standard deviation of last eight one-step differences",
                "multi_step_scaling": "square root of horizon",
            },
        },
        "baseline": "one-step persistence",
        "selection_rule": {
            "criterion": "minimum validation-segment RMSE",
            "test_segment_used_for_selection": False,
            "selected_demand_method": demand_runtime_methods[selected_demand_model],
            "selected_demand_model_id": selected_demand_model,
            "selected_supply_proxy_method": supply_runtime_methods[selected_supply_model],
            "selected_supply_proxy_model_id": selected_supply_model,
        },
        "metrics": metrics,
    }
    return summary, pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lstm-epochs", type=int, default=80)
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument(
        "--source-commit",
        default=os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip(),
        help="Exact simulation-source Git commit (required for canonical output).",
    )
    parser.add_argument(
        "--run-tag",
        default=os.environ.get("RUN_TAG", "").strip(),
        help="Run tag shared by the benchmark publication pipeline.",
    )
    parser.add_argument(
        "--publication-replay",
        action="store_true",
        help=(
            "Recreate publication-scoped bytes in a noncanonical temporary "
            "directory. Restricted to the final validator."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    canonical_output = output_dir == CANONICAL_RESULTS_DIR.resolve()
    source_commit = str(args.source_commit).strip()
    run_tag = str(args.run_tag).strip()
    publication_scope = canonical_output or args.publication_replay
    if args.publication_replay and (
        canonical_output
        or os.environ.get("AGRIBRAIN_PUBLICATION_REPLAY", "") != "1"
    ):
        raise RuntimeError(
            "--publication-replay is restricted to an isolated final-validator run"
        )
    if publication_scope:
        if _FULL_SHA1.fullmatch(source_commit) is None:
            raise RuntimeError(
                "canonical forecast receipt requires a full lowercase source commit"
            )
        if not run_tag:
            raise RuntimeError("canonical forecast receipt requires RUN_TAG")
    elif source_commit and _FULL_SHA1.fullmatch(source_commit) is None:
        raise RuntimeError("--source-commit must be a full lowercase Git SHA-1")

    summary, rows = build_validation(
        args.data, lstm_epochs=args.lstm_epochs,
        origin_stride=args.origin_stride,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "forecast_validation_summary.json"
    rows_path = args.output_dir / "forecast_validation_predictions.csv"
    rows.to_csv(rows_path, index=False)
    summary["provenance"] = {
        "scope": "publication" if publication_scope else "development",
        "source_commit": source_commit or None,
        "run_tag": run_tag or None,
    }
    summary["predictions_artifact"] = {
        "file": rows_path.name,
        "row_count": int(len(rows)),
        "bytes": rows_path.stat().st_size,
        "sha256": _sha256(rows_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] wrote {summary_path} and {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
