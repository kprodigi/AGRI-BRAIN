"""Tests for calibration-derived governance override thresholds.

The governance override used to be hard-coded to logit-space magic
numbers (``logit[0] < -2.0`` and ``logit[1] > logit[0] + 3``). It was
rewritten to fire on policy probabilities with calibration-derived
ceilings and advantage floors, so the condition is auditable without
reference to the raw logit scale. These tests lock in the new semantics
and the calibration helper that derives the constants.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.action_selection import (
    GOVERNANCE_CC_PROB_CEILING,
    GOVERNANCE_LOCAL_ADVANTAGE_MIN,
    calibrate_governance_thresholds,
    select_action,
    ACTIONS,
)


class _DummyPolicy:
    gamma_coldchain = 0.0
    gamma_local = 0.0
    gamma_recovery = 0.0


def test_default_thresholds_are_valid_probabilities():
    assert 0.0 < GOVERNANCE_CC_PROB_CEILING < 1.0
    assert 0.0 < GOVERNANCE_LOCAL_ADVANTAGE_MIN < 1.0


def test_calibration_returns_requested_quantiles():
    rng = np.random.default_rng(0)
    probs = np.column_stack([
        rng.uniform(0.0, 0.3, size=1000),   # cold_chain
        rng.uniform(0.4, 0.8, size=1000),   # local_redistribute
        rng.uniform(0.0, 0.2, size=1000),   # recovery
    ])
    probs = probs / probs.sum(axis=1, keepdims=True)
    out = calibrate_governance_thresholds(probs, cc_quantile=0.05, local_quantile=0.50)
    expected_cc = float(np.quantile(probs[:, 0], 0.05))
    expected_gap = float(np.quantile(probs[:, 1] - probs[:, 0], 0.50))
    assert out["cc_prob_ceiling"] == pytest.approx(expected_cc)
    assert out["local_advantage_min"] == pytest.approx(expected_gap)


def test_calibration_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"prob_rollouts must be shape \(N, 3\)"):
        calibrate_governance_thresholds(np.zeros((10, 2)))
    with pytest.raises(ValueError, match=r"prob_rollouts must be shape \(N, 3\)"):
        calibrate_governance_thresholds(np.zeros(5))


def test_calibration_rejects_out_of_range_quantile():
    probs = np.array([[0.2, 0.6, 0.2]] * 10)
    with pytest.raises(ValueError, match=r"quantile"):
        calibrate_governance_thresholds(probs, cc_quantile=1.5)
    with pytest.raises(ValueError, match=r"quantile"):
        calibrate_governance_thresholds(probs, local_quantile=-0.1)


def test_override_fires_when_context_pushes_cold_chain_down():
    """With a strong local-favouring context modifier and cold-chain-
    disfavouring logits, the governance override fires and returns a
    one-hot on local_redistribute. Tested at rho=0.20 — inside the
    at-risk band (>0.10) but below the Recovery knee (0.30), so the
    LR triage is the right call and the override is not in tension
    with food-safety routing."""
    rng = np.random.default_rng(0)
    action, probs = select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
        context_modifier=np.array([-5.0, 5.0, 0.0]),
    )
    assert ACTIONS[action] == "local_redistribute"
    # Override returns a one-hot distribution, not the softmax probs.
    np.testing.assert_array_equal(probs, np.array([0.0, 1.0, 0.0]))


def test_recovery_knee_overrides_lr_governance_at_high_rho():
    """At rho well above the Recovery knee (0.30), Recovery's food-
    safety triage logit boost dominates over the LR-favouring context
    modifier. The governance override deliberately does *not* force
    LR in this regime — the safety hierarchy says non-marketable
    produce must go to Recovery regardless of context. This is the
    intended behaviour of the boosted knee gain (5.0 / 3.0)."""
    rng = np.random.default_rng(0)
    action, _ = select_action(
        mode="agribrain",
        rho=0.95, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
        context_modifier=np.array([-5.0, 5.0, 0.0]),
    )
    assert ACTIONS[action] == "recovery"


def test_override_does_not_fire_without_context_modifier():
    """Non-context modes (no context_modifier) should never trigger the
    override even if logits would satisfy the probability condition."""
    rng = np.random.default_rng(0)
    action, probs = select_action(
        mode="hybrid_rl",
        rho=0.95, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    # Probs should be a real softmax, never the one-hot override result.
    assert 0.0 < probs[0] < 1.0
    assert probs.sum() == pytest.approx(1.0)


def test_override_does_not_fire_on_cold_chain_favouring_context():
    """With a zero context modifier on a cold-chain-favouring state
    (low rho, low temp), the softmax probability of cold-chain is well
    above the ceiling so the override must not fire."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.1, inv=5000, y_hat=50, temp=2.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
        context_modifier=np.zeros(3),
    )
    # A non-firing call returns real softmax probs that sum to 1, not the
    # one-hot [0, 1, 0] the override would produce.
    assert probs.sum() == pytest.approx(1.0)
    assert not (probs[0] == 0.0 and probs[1] == 1.0 and probs[2] == 0.0)
    # pi(cold_chain) must be above the ceiling for the override to have
    # been skipped; this is the documented semantic of the new threshold.
    assert probs[0] >= GOVERNANCE_CC_PROB_CEILING
