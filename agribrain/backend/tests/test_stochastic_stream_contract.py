"""Focused regression tests for the publication environmental streams."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (
    REPO_ROOT,
    REPO_ROOT / "mvp" / "simulation",
    REPO_ROOT / "agribrain" / "backend",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from mvp.simulation import generate_results as gr
from mvp.simulation.stochastic import make_stochastic_layer


def _disable_unrelated_sources(monkeypatch) -> None:
    monkeypatch.setenv("DETERMINISTIC_MODE", "false")
    monkeypatch.setenv("STOCH_TEMP_STD_C", "0")
    monkeypatch.setenv("STOCH_RH_STD", "0")
    monkeypatch.setenv("STOCH_INVENTORY_FRAC_STD", "0")
    monkeypatch.setenv("STOCH_TRANSPORT_KM_STD", "0")
    monkeypatch.setenv("STOCH_K_REF_STD", "0")
    monkeypatch.setenv("STOCH_EA_R_STD", "0")
    monkeypatch.setenv("STOCH_ONSET_JITTER_H", "0")
    monkeypatch.setenv("STOCH_THETA_NOISE_STD", "0")
    monkeypatch.setenv("STOCH_POLICY_TEMP_STD", "0")
    monkeypatch.setenv("STOCH_DELAY_PROB", "0")


def test_adaptive_pricing_uses_explicit_per_seed_rng():
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(40)
    policy = gr.Policy()

    first = gr.apply_scenario(
        frame, "adaptive_pricing", policy, np.random.default_rng(101),
    )
    repeat = gr.apply_scenario(
        frame, "adaptive_pricing", policy, np.random.default_rng(101),
    )
    other_seed = gr.apply_scenario(
        frame, "adaptive_pricing", policy, np.random.default_rng(202),
    )

    assert np.array_equal(first["demand_units"], repeat["demand_units"])
    assert not np.array_equal(first["demand_units"], other_seed["demand_units"])


def test_counter_keyed_draws_do_not_depend_on_intervening_calls(monkeypatch):
    _disable_unrelated_sources(monkeypatch)
    monkeypatch.setenv("STOCH_DEMAND_FRAC_STD", "0.25")
    layer = make_stochastic_layer(
        np.random.default_rng(999), stream_seed=123456,
    )

    expected = layer.perturb_demand(100.0, counter=17)
    layer.perturb_temperature(5.0, counter=4)
    layer.perturb_inventory(1000.0, counter=91)
    layer.should_delay(counter=12)
    actual = layer.perturb_demand(100.0, counter=17)

    assert actual == expected
    other_stream = make_stochastic_layer(
        np.random.default_rng(999), stream_seed=654321,
    )
    assert other_stream.perturb_demand(100.0, counter=17) != expected


def test_demand_noise_precedes_forecast_bollinger_and_price(
    tmp_path, monkeypatch,
):
    _disable_unrelated_sources(monkeypatch)
    monkeypatch.setenv("STOCH_DEMAND_FRAC_STD", "0.25")
    monkeypatch.setattr(gr, "FORECAST_METHOD", "holt_winters")
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(24)
    stream_seed = 424242
    layer = make_stochastic_layer(
        np.random.default_rng(stream_seed), stream_seed=stream_seed,
    )

    with gr.decision_ledger_scope(tmp_path / "ledger", reset=True):
        episode = gr.run_episode(
            frame,
            "static",
            gr.Policy(),
            np.random.default_rng(7),
            stoch=layer,
            seed=42,
            benchmark_seed=42,
            episode_index=3,
            environment_stream_id=gr._stream_id(42, "baseline", 3, "environment"),
            policy_stream_id=gr._stream_id(42, "baseline", 3, "policy"),
            stochastic_stream_id=gr._stream_id(42, "baseline", 3, "environment"),
        )

    latent = [float(value) for value in frame["demand_units"]]
    observed = episode["demand_policy_observed_trace"]
    expected_observed = [
        layer.perturb_demand(value, counter=index)
        for index, value in enumerate(latent)
    ]
    assert observed == expected_observed
    assert observed != latent

    policy = gr.Policy()
    for index in range(len(observed)):
        history = observed[max(0, index - 47):index + 1]
        expected_forecast = gr.query_demand(
            demand_history=history,
            horizon=1,
            method="holt_winters",
        )["forecast"][0]
        assert episode["demand_forecast_policy_observed_trace"][index] == (
            expected_forecast
        )

        series = pd.Series(history, dtype=float)
        rolling_mean = series.rolling(
            policy.boll_window, min_periods=1,
        ).mean().iloc[-1]
        rolling_std = series.rolling(
            policy.boll_window, min_periods=1,
        ).std().fillna(0.0).iloc[-1]
        z_score = (
            (float(series.iloc[-1]) - float(rolling_mean))
            / max(float(rolling_std), 1e-6)
        )
        assert episode["price_signal_trace"][index] == float(
            np.clip(z_score, -1.0, 1.0)
        )
        assert episode["demand_regime_flag_trace"][index] == float(
            abs(z_score) > policy.boll_k
        )
