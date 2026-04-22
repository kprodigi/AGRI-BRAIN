"""Path B regression and unit tests.

Three classes covering: yield_query MCP tool, six-feature psi context
vector, ContextMatrixLearner with 3x6 Theta.
"""
from __future__ import annotations

import numpy as np
import pytest


# ============================================================================
# 1.  yield_query MCP tool
# ============================================================================
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

    def test_uncertainty_clamped_to_one(self):
        from pirag.mcp.tools.yield_query import query_yield
        result = query_yield(
            inventory_history=[1, 50, 1, 50, 1, 50, 1, 50],
            horizon=3,
        )
        assert 0.0 <= result["uncertainty"] <= 1.0

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
        assert "supply" in spec.capabilities
        result = reg.invoke(
            "yield_query",
            inventory_history=[100, 100, 99, 101, 103],
            horizon=2,
        )
        assert isinstance(result, dict)
        assert "uncertainty" in result


# ============================================================================
# 2.  Six-feature psi context vector
# ============================================================================
class TestContextSixFeatures:
    """Unit tests for the extended psi in R^6 context feature vector."""

    def test_extract_context_features_returns_6d(self):
        from pirag.context_to_logits import extract_context_features

        class _Obs:
            raw = {}

        psi = extract_context_features(
            mcp_results={},
            rag_context={},
            obs=_Obs(),
        )
        assert psi.shape == (6,)

    def test_psi_5_populated_from_yield_query(self):
        from pirag.context_to_logits import extract_context_features

        class _Obs:
            raw = {}

        psi = extract_context_features(
            mcp_results={"yield_query": {"uncertainty": 0.42}},
            rag_context={},
            obs=_Obs(),
        )
        assert psi[5] == pytest.approx(0.42)

    def test_psi_5_defaults_to_zero_when_yield_query_absent(self):
        from pirag.context_to_logits import extract_context_features

        class _Obs:
            raw = {}

        psi = extract_context_features(
            mcp_results={"check_compliance": {"compliant": True}},
            rag_context={},
            obs=_Obs(),
        )
        assert psi[5] == 0.0

    def test_theta_context_3x6_shape(self):
        from pirag.context_to_logits import THETA_CONTEXT
        assert THETA_CONTEXT.shape == (3, 6)

    def test_theta_context_signs(self):
        from pirag.context_to_logits import THETA_CONTEXT
        assert THETA_CONTEXT[0, 5] > 0   # cold chain
        assert THETA_CONTEXT[1, 5] > 0   # local redistribute
        assert THETA_CONTEXT[2, 5] < 0   # recovery

    def test_mcp_mask_includes_psi_5(self):
        from pirag.context_to_logits import _MCP_FEATURE_MASK
        assert _MCP_FEATURE_MASK.shape == (6,)
        assert _MCP_FEATURE_MASK[5] == 1.0

    def test_pirag_mask_excludes_psi_5(self):
        from pirag.context_to_logits import _PIRAG_FEATURE_MASK
        assert _PIRAG_FEATURE_MASK.shape == (6,)
        assert _PIRAG_FEATURE_MASK[5] == 0.0

    def test_no_yield_ablation_mode_zeros_psi_5(self):
        from pirag.context_to_logits import compute_context_modifier

        class _Obs:
            raw = {}

        full = compute_context_modifier(
            mcp_results={"yield_query": {"uncertainty": 0.8}},
            rag_context={"guards_passed": True},
            obs=_Obs(),
            context_mode="full",
        )
        no_yield = compute_context_modifier(
            mcp_results={"yield_query": {"uncertainty": 0.8}},
            rag_context={"guards_passed": True},
            obs=_Obs(),
            context_mode="no_yield",
        )
        assert not np.allclose(full, no_yield)


# ============================================================================
# 3.  ContextMatrixLearner with 3x6 Theta
# ============================================================================
class TestContextLearnerSixFeatures:
    """Verify the REINFORCE learner handles 3x6 THETA_CONTEXT correctly."""

    def test_learner_accepts_3x6_initial_theta(self):
        from pirag.context_learner import ContextMatrixLearner
        initial = np.array([
            [-0.80, -0.60, -0.15, -0.30, +0.25, +0.20],
            [+0.50, +0.40, +0.20, +0.25, +0.10, +0.05],
            [+0.30, +0.20, -0.05, +0.05, -0.35, -0.15],
        ])
        learner = ContextMatrixLearner(initial_theta=initial)
        assert learner.theta.shape == (3, 6)
        assert learner.sign_mask.shape == (3, 6)

    def test_reinforce_update_preserves_signs_in_3x6(self):
        from pirag.context_learner import ContextMatrixLearner
        initial = np.array([
            [-0.80, -0.60, -0.15, -0.30, +0.25, +0.20],
            [+0.50, +0.40, +0.20, +0.25, +0.10, +0.05],
            [+0.30, +0.20, -0.05, +0.05, -0.35, -0.15],
        ])
        learner = ContextMatrixLearner(
            initial_theta=initial,
            learning_rate=0.05,
            sign_constrained=True,
        )
        rng = np.random.default_rng(42)
        for _ in range(200):
            psi = rng.uniform(0.0, 1.0, size=6)
            action = int(rng.integers(0, 3))
            probs = np.array([0.33, 0.34, 0.33])
            reward = float(rng.normal(0.0, 1.0))
            learner.update(psi=psi, action=action, probs=probs, reward=reward)

        sign_match = (np.sign(learner.theta) * learner.sign_mask) >= 0
        zero_init = (learner.sign_mask == 0)
        assert bool(np.all(sign_match | zero_init)), (
            f"Sign constraint violated:\n"
            f"theta=\n{learner.theta}\nsign_mask=\n{learner.sign_mask}"
        )

    def test_magnitude_constraint_applied_to_psi_5_column(self):
        from pirag.context_learner import ContextMatrixLearner
        initial = np.array([
            [-0.80, -0.60, -0.15, -0.30, +0.25, +0.20],
            [+0.50, +0.40, +0.20, +0.25, +0.10, +0.05],
            [+0.30, +0.20, -0.05, +0.05, -0.35, -0.15],
        ])
        learner = ContextMatrixLearner(
            initial_theta=initial,
            learning_rate=0.5,
            sign_constrained=True,
        )
        rng = np.random.default_rng(7)
        for _ in range(500):
            psi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            action = 0
            probs = np.array([0.10, 0.45, 0.45])
            reward = 5.0
            learner.update(psi=psi, action=action, probs=probs, reward=reward)

        max_mag = np.maximum(np.abs(initial) * 2.0, 0.10)
        assert bool(np.all(np.abs(learner.theta) <= max_mag + 1e-9))

    def test_summary_reports_six_feature_geometry(self):
        from pirag.context_learner import ContextMatrixLearner
        initial = np.eye(3, 6)
        learner = ContextMatrixLearner(initial_theta=initial)
        s = learner.summary()
        assert len(s["initial_theta"]) == 3
        assert len(s["initial_theta"][0]) == 6
