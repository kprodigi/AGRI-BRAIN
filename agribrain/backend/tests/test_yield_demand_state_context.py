"""Regression and unit tests for yield/demand forecasts and the
state-vs-context separation.

State + context shapes:

- The state vector phi(s) has shape (10,). Indices 6-8 carry supply
  and demand forecast information (point + uncertainty); index 9
  carries a demand-volatility Bollinger z-score used as a
  price-pressure proxy.
- The context vector psi has shape (5,) — the original size before
  the brief experimental run that smuggled forecast features into
  psi. Supply and demand forecast information now lives strictly on
  the state side, where it belongs.
- Theta has shape (3, 10); Theta_context has shape (3, 5).
- yield_query is a useful MCP tool that supplies the Holt's linear
  (level + trend) forecast and its residual-std uncertainty; its
  output is consumed by the state vector via ``build_feature_vector``,
  not by the context modifier.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. yield_query MCP tool (unchanged behaviour)
# ---------------------------------------------------------------------------
class TestYieldQuery:
    """Unit tests for pirag.mcp.tools.yield_query.query_yield."""

    def test_query_with_history_returns_uncertainty_in_unit_interval(self):
        from pirag.mcp.tools.yield_query import query_yield
        result = query_yield(
            inventory_history=[100, 102, 99, 101, 103, 100, 98, 102, 105, 99],
            horizon=3,
        )
        assert "forecast" in result
        assert "uncertainty" in result
        assert 0.0 <= result["uncertainty"] <= 1.0
        assert len(result["forecast"]) == 3
        assert result["source"] == "computed"

    def test_query_with_empty_history_returns_zero(self):
        from pirag.mcp.tools.yield_query import query_yield
        result = query_yield(inventory_history=[], horizon=6)
        assert result["uncertainty"] == 0.0
        assert result["forecast"] == []

    def test_query_with_none_history_returns_zero(self):
        from pirag.mcp.tools.yield_query import query_yield
        result = query_yield(inventory_history=None, horizon=6)
        assert result["uncertainty"] == 0.0

    def test_uncertainty_grows_with_volatility(self):
        from pirag.mcp.tools.yield_query import query_yield
        stable = query_yield(
            inventory_history=[100, 100, 100, 100, 100, 100, 100, 100],
            horizon=3,
        )
        volatile = query_yield(
            inventory_history=[100, 80, 120, 70, 130, 60, 140, 50],
            horizon=3,
        )
        assert volatile["uncertainty"] > stable["uncertainty"]

    def test_cached_short_circuit(self):
        from pirag.mcp.tools.yield_query import query_yield
        result = query_yield(
            cached_uncertainty=0.42,
            cached_forecast=[100.0, 101.5],
            cached_std=4.2,
        )
        assert result["uncertainty"] == pytest.approx(0.42)
        assert result["forecast"] == [100.0, 101.5]
        assert result["std"] == 4.2
        assert result["source"] == "cached"

    def test_cached_uncertainty_clamped(self):
        from pirag.mcp.tools.yield_query import query_yield
        r_high = query_yield(cached_uncertainty=1.7)
        r_neg = query_yield(cached_uncertainty=-0.3)
        assert r_high["uncertainty"] == 1.0
        assert r_neg["uncertainty"] == 0.0

    def test_registry_round_trip(self):
        from pirag.mcp.registry import get_default_registry
        reg = get_default_registry()
        spec = reg.get("yield_query")
        assert spec is not None, "yield_query not registered"
        result = reg.invoke(
            "yield_query",
            inventory_history=[100, 100, 99, 101, 103],
            horizon=2,
        )
        assert isinstance(result, dict)
        assert "uncertainty" in result


# ---------------------------------------------------------------------------
# 2. Context vector psi is 5-dim, institutional only
# ---------------------------------------------------------------------------
class TestContextFiveFeatures:
    """Psi lives in R^5 now; no supply- or demand-forecast features here."""

    def test_extract_context_features_returns_5d(self):
        from pirag.context_to_logits import extract_context_features

        class _Obs:
            raw = {}

        psi = extract_context_features({}, {}, _Obs())
        assert psi.shape == (5,)

    def test_theta_context_is_3x5(self):
        from pirag.context_to_logits import THETA_CONTEXT
        assert THETA_CONTEXT.shape == (3, 5)

    def test_context_modifier_rules_has_five_entries(self):
        from pirag.context_to_logits import MODIFIER_RULES
        assert len(MODIFIER_RULES) == 5
        names = [r["name"] for r in MODIFIER_RULES]
        assert names == [
            "compliance_severity",
            "forecast_urgency",
            "retrieval_confidence",
            "regulatory_pressure",
            "recovery_saturation",
        ]

    def test_masks_are_5d(self):
        from pirag.context_to_logits import (
            _MCP_FEATURE_MASK, _PIRAG_FEATURE_MASK,
        )
        assert _MCP_FEATURE_MASK.shape == (5,)
        assert _PIRAG_FEATURE_MASK.shape == (5,)
        # MCP features: compliance, forecast urgency, recovery saturation.
        assert np.allclose(_MCP_FEATURE_MASK, [1.0, 1.0, 0.0, 0.0, 1.0])
        # piRAG features: retrieval confidence, regulatory pressure.
        assert np.allclose(_PIRAG_FEATURE_MASK, [0.0, 0.0, 1.0, 1.0, 0.0])

    def test_context_mode_accepts_three_values(self):
        from pirag.context_to_logits import compute_context_modifier

        class _Obs:
            raw = {}

        kwargs = dict(
            mcp_results={"check_compliance": {"compliant": False,
                                               "violations": [{"severity": "critical"}]}},
            rag_context={"guards_passed": True, "top_citation_score": 0.5,
                         "top_doc_id": "fda_regulatory.txt"},
            obs=_Obs(),
        )
        full = compute_context_modifier(context_mode="full", **kwargs)
        mcp = compute_context_modifier(context_mode="mcp_only", **kwargs)
        pirag = compute_context_modifier(context_mode="pirag_only", **kwargs)
        # mcp_only zeros the piRAG features; pirag_only zeros the MCP features.
        # They should not be identical to full (compliance is on, retrieval is on).
        assert not np.allclose(mcp, full)
        assert not np.allclose(pirag, full)


# ---------------------------------------------------------------------------
# 3. State vector phi is 10-dim (supply/demand forecast + price signal)
# ---------------------------------------------------------------------------
class TestStateNineFeatures:
    """Phi has 10 dimensions; columns 6-8 carry supply/demand info,
    column 9 carries the price-pressure proxy."""

    def test_build_feature_vector_returns_10d(self):
        from src.models.action_selection import build_feature_vector
        phi = build_feature_vector(rho=0.1, inv=12000, y_hat=20, temp=4.0)
        assert phi.shape == (10,)

    def test_theta_is_3x10(self):
        from src.models.action_selection import THETA
        assert THETA.shape == (3, 10)

    def test_price_signal_is_zero_by_default(self):
        from src.models.action_selection import build_feature_vector
        phi = build_feature_vector(rho=0.1, inv=12000, y_hat=20, temp=4.0)
        assert phi[9] == 0.0

    def test_price_signal_clipped_to_unit_interval(self):
        from src.models.action_selection import build_feature_vector
        phi_pos = build_feature_vector(
            rho=0.1, inv=12000, y_hat=20, temp=4.0, price_signal=5.0,
        )
        phi_neg = build_feature_vector(
            rho=0.1, inv=12000, y_hat=20, temp=4.0, price_signal=-5.0,
        )
        assert phi_pos[9] == 1.0
        assert phi_neg[9] == -1.0

    def test_supply_point_centered_at_zero_for_baseline(self):
        """phi_6 must be 0 when supply equals baseline inventory."""
        from src.models.action_selection import build_feature_vector, INV_BASELINE
        phi = build_feature_vector(
            rho=0.1, inv=12000, y_hat=20, temp=4.0,
            supply_hat=INV_BASELINE, supply_std=0.0, demand_std=0.0,
        )
        assert phi[6] == pytest.approx(0.0)

    def test_supply_point_positive_under_surplus(self):
        from src.models.action_selection import build_feature_vector, INV_BASELINE
        phi = build_feature_vector(
            rho=0.1, inv=12000, y_hat=20, temp=4.0,
            supply_hat=INV_BASELINE * 2.5, supply_std=100.0, demand_std=2.0,
        )
        assert phi[6] > 0  # surplus
        # Clipped to +0.5 upper bound
        assert phi[6] <= 0.5 + 1e-9

    def test_supply_point_negative_under_shortage(self):
        from src.models.action_selection import build_feature_vector, INV_BASELINE
        phi = build_feature_vector(
            rho=0.1, inv=12000, y_hat=20, temp=4.0,
            supply_hat=INV_BASELINE * 0.3, supply_std=50.0, demand_std=2.0,
        )
        assert phi[6] < 0  # shortage
        assert phi[6] >= -0.5 - 1e-9

    def test_uncertainty_columns_in_unit_interval(self):
        from src.models.action_selection import build_feature_vector
        phi = build_feature_vector(
            rho=0.1, inv=12000, y_hat=20, temp=4.0,
            supply_hat=12000.0, supply_std=5000.0, demand_std=100.0,
        )
        assert 0.0 <= phi[7] <= 1.0
        assert 0.0 <= phi[8] <= 1.0

    def test_missing_forecasts_yield_zero_on_new_columns(self):
        from src.models.action_selection import build_feature_vector
        phi = build_feature_vector(rho=0.1, inv=12000, y_hat=20, temp=4.0)
        # Defaults: supply_hat=None, supply_std=None, demand_std=None.
        assert phi[6] == 0.0
        assert phi[7] == 0.0
        assert phi[8] == 0.0

    def test_theta_sign_justifications_supply_point_column(self):
        from src.models.action_selection import THETA
        # phi_6 supply_point: cold chain negative (surplus disfavours CC),
        # redistribution strongly positive, recovery mildly positive.
        assert THETA[0, 6] < 0, f"THETA[0,6]={THETA[0,6]}, expected negative"
        assert THETA[1, 6] > 0, f"THETA[1,6]={THETA[1,6]}, expected positive"
        assert THETA[2, 6] >= 0, f"THETA[2,6]={THETA[2,6]}, expected non-negative"

    def test_theta_sign_justifications_supply_uncertainty_column(self):
        from src.models.action_selection import THETA
        # phi_7 supply_uncertainty: real-options logic. Cold chain positive
        # (preserve optionality), recovery negative (avoid irreversible).
        assert THETA[0, 7] > 0
        assert THETA[2, 7] < 0

    def test_theta_sign_justifications_demand_uncertainty_column(self):
        from src.models.action_selection import THETA
        # phi_8 demand_uncertainty: cold chain positive (inventory
        # positioning under bullwhip), recovery negative (demand may still
        # materialise).
        assert THETA[0, 8] > 0
        assert THETA[2, 8] < 0

    def test_baseline_new_columns_contribute_less_than_005(self):
        """At nominal conditions (supply_point ~ 0, CVs ~ 0.05), the three
        new columns must contribute less than 0.05 per logit in absolute
        value so the calibration of the original six columns is preserved."""
        from src.models.action_selection import THETA
        new_cols = THETA[:, 6:9]
        baseline = np.array([0.0, 0.05, 0.05])
        contribution = new_cols @ baseline
        assert np.all(np.abs(contribution) < 0.05)


# ---------------------------------------------------------------------------
# 4. Residual-std uncertainty in the forecasters
# ---------------------------------------------------------------------------
class TestForecasterResidualStd:
    """Both forecasters now emit residual-std under the ``std`` key."""

    def test_lstm_residual_std_nonnegative(self):
        from src.models.lstm_demand import lstm_demand_forecast
        import pandas as pd
        df = pd.DataFrame(
            {"demand_units": [20 + 2 * i % 5 for i in range(30)]}
        )
        out = lstm_demand_forecast(df, horizon=1, epochs=10)
        assert "std" in out
        assert "series_std" in out
        assert out["std"] >= 0.0

    def test_holt_winters_residual_std_nonnegative(self):
        from src.models.yield_forecast import yield_supply_forecast
        import pandas as pd
        df = pd.DataFrame({
            "inventory_units": [12000 + 200 * np.sin(i / 3) for i in range(30)],
        })
        out = yield_supply_forecast(df, horizon=1)
        assert "std" in out
        assert "series_std" in out
        assert out["std"] >= 0.0

    def test_holt_winters_std_grows_with_volatility(self):
        from src.models.yield_forecast import yield_supply_forecast
        import pandas as pd
        stable = pd.DataFrame({"inventory_units": [12000.0] * 30})
        volatile = pd.DataFrame({
            "inventory_units": [12000 + 2000 * ((-1) ** i) for i in range(30)]
        })
        s_std = yield_supply_forecast(stable, horizon=1)["std"]
        v_std = yield_supply_forecast(volatile, horizon=1)["std"]
        assert v_std > s_std


# ---------------------------------------------------------------------------
# 5. ContextMatrixLearner with 3x5 Theta
# ---------------------------------------------------------------------------
class TestContextLearnerFiveFeatures:
    """Verify the REINFORCE learner handles the 3x5 THETA_CONTEXT."""

    _INITIAL = np.array([
        [-0.80, -0.60, -0.15, -0.30, +0.25],
        [+0.50, +0.40, +0.20, +0.25, +0.10],
        [+0.30, +0.20, -0.05, +0.05, -0.35],
    ])

    def test_learner_accepts_3x5_initial_theta(self):
        from pirag.context_learner import ContextMatrixLearner
        learner = ContextMatrixLearner(initial_theta=self._INITIAL)
        assert learner.theta.shape == (3, 5)
        assert learner.sign_mask.shape == (3, 5)

    def test_reinforce_update_preserves_signs_in_3x5(self):
        from pirag.context_learner import ContextMatrixLearner
        learner = ContextMatrixLearner(
            initial_theta=self._INITIAL,
            learning_rate=0.05,
            sign_constrained=True,
        )
        rng = np.random.default_rng(42)
        for _ in range(200):
            psi = rng.uniform(0.0, 1.0, size=5)
            action = int(rng.integers(0, 3))
            probs = np.array([0.33, 0.34, 0.33])
            reward = float(rng.normal(0.0, 1.0))
            learner.update(psi=psi, action=action, probs=probs, reward=reward)

        sign_match = (np.sign(learner.theta) * learner.sign_mask) >= 0
        zero_init = (learner.sign_mask == 0)
        assert bool(np.all(sign_match | zero_init))

    def test_summary_reports_five_feature_geometry(self):
        from pirag.context_learner import ContextMatrixLearner
        learner = ContextMatrixLearner(initial_theta=np.eye(3, 5))
        s = learner.summary()
        assert len(s["initial_theta"]) == 3
        assert len(s["initial_theta"][0]) == 5
