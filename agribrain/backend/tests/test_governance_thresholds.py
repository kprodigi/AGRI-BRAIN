"""Tests for the declared probability-gap action-substitution rule.

The compatibility implementation used to be hard-coded to logit-space magic
numbers (``logit[0] < -2.0`` and ``logit[1] > logit[0] + 3``). It was
rewritten to fire on policy probabilities with declared ceilings and
advantage floors, so the condition is auditable without reference to the raw
logit scale. These tests lock in that two-predicate rule and separately test
the optional exploratory calibration helper.
"""
from __future__ import annotations

import copy
import numpy as np
import pytest

from src.models.action_selection import (
    GOVERNANCE_CC_PROB_CEILING,
    GOVERNANCE_LOCAL_ADVANTAGE_MIN,
    calibrate_governance_thresholds,
    governance_override_applies,
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


def test_override_predicate_has_exact_strict_probability_boundaries():
    """Only the declared cold-chain ceiling and local gap govern firing."""
    eps = 1e-9
    cc = GOVERNANCE_CC_PROB_CEILING
    gap = GOVERNANCE_LOCAL_ADVANTAGE_MIN

    # Strictly inside both bounds: fires.
    p0 = cc - eps
    p1 = p0 + gap + eps
    passing = np.array([p0, p1, 1.0 - p0 - p1])
    assert governance_override_applies(passing)

    # Equality at either boundary does not fire because both comparisons are
    # intentionally strict in the executable policy and manuscript equation.
    p0 = cc
    p1 = p0 + gap + eps
    at_cc_ceiling = np.array([p0, p1, 1.0 - p0 - p1])
    assert not governance_override_applies(at_cc_ceiling)
    p0 = cc - eps
    p1 = p0 + gap
    at_gap_floor = np.array([p0, p1, 1.0 - p0 - p1])
    assert not governance_override_applies(at_gap_floor)


def test_override_predicate_validates_probability_vector_shape():
    with pytest.raises(ValueError, match="length-3"):
        governance_override_applies(np.array([0.1, 0.9]))


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
    disfavouring logits, the probability-gap rule activates and returns a
    one-hot on local_redistribute. Tested at rho=0.20 — inside the
    at-risk band (>0.10) but below the Recovery knee (0.30), so the
    LR is the declared preferred action and the probability-gap rule is
    directionally consistent with that synthetic policy band."""
    rng = np.random.default_rng(0)
    action, probs = select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
        context_modifier=np.array([-5.0, 5.0, 0.0]),
    )
    assert ACTIONS[action] == "local_redistribute"
    # The rule returns a one-hot distribution, not the softmax probabilities.
    np.testing.assert_array_equal(probs, np.array([0.0, 1.0, 0.0]))


def test_recovery_knee_overrides_lr_governance_at_high_rho():
    """At rho well above the Recovery knee (0.30), the author-declared
    Recovery logit boost dominates the LR-favouring context modifier. The
    probability-gap rule deliberately does *not* force LR in this synthetic
    high-risk band. This is the intended behaviour of the declared knee gain
    (5.0 / 3.0), not a real marketability or food-safety classification."""
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


def test_stochastic_override_consumes_one_policy_draw_like_non_override():
    """An override must not shift later common-random-number draws."""
    override_rng = np.random.default_rng(724)
    ordinary_rng = np.random.default_rng(724)

    override_action, override_probs = select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=override_rng,
        context_modifier=np.array([-5.0, 5.0, 0.0]),
    )
    select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=ordinary_rng,
        context_modifier=None,
    )

    assert override_action == 1
    np.testing.assert_array_equal(
        override_probs, np.array([0.0, 1.0, 0.0]),
    )
    assert override_rng.bit_generator.state == ordinary_rng.bit_generator.state


def test_override_and_context_ablation_reuse_same_saved_policy_draw():
    """The live override discards, but still consumes, the paired draw."""
    live_rng = np.random.default_rng(819)
    saved_state = copy.deepcopy(live_rng.bit_generator.state)
    live_action, _ = select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=live_rng,
        context_modifier=np.array([-5.0, 5.0, 0.0]),
    )

    ablated_rng = np.random.default_rng()
    ablated_rng.bit_generator.state = saved_state
    ablated_action, ablated_probs = select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=ablated_rng,
        context_modifier=None,
    )

    reference_rng = np.random.default_rng()
    reference_rng.bit_generator.state = saved_state
    expected_ablated_action = int(
        reference_rng.choice(len(ACTIONS), p=ablated_probs)
    )
    assert live_action == 1
    assert ablated_action == expected_ablated_action
    assert live_rng.bit_generator.state == ablated_rng.bit_generator.state
    assert ablated_rng.bit_generator.state == reference_rng.bit_generator.state


def test_deterministic_override_remains_draw_free():
    """Explicit deterministic policy evaluation must not consume RNG."""
    rng = np.random.default_rng(910)
    state_before = copy.deepcopy(rng.bit_generator.state)
    action, _ = select_action(
        mode="agribrain",
        rho=0.20, inv=5000, y_hat=50, temp=12.0, tau=1.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
        context_modifier=np.array([-5.0, 5.0, 0.0]),
    )
    assert action == 1
    assert rng.bit_generator.state == state_before
