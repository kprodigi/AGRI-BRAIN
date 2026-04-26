"""Unit tests for :class:`pirag.context_learner.PolicyDeltaLearner`."""
from __future__ import annotations

import numpy as np
import pytest

from pirag.context_learner import PolicyDeltaLearner
from src.models.action_selection import THETA


def _phi(forecast=(0.2, 0.3, 0.1), price_signal: float = 0.0):
    """Build a 10-dim phi with forecast channels and price channel set."""
    phi = np.zeros(10, dtype=np.float64)
    phi[6:9] = np.asarray(forecast, dtype=np.float64)
    phi[9] = float(price_signal)
    return phi


def test_initial_delta_is_zero():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    np.testing.assert_array_equal(learner.get_theta_delta(), np.zeros_like(THETA))


def test_effective_theta_equals_initial_at_start():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    np.testing.assert_array_equal(learner.get_effective_theta(), THETA)


def test_init_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"initial_theta must be shape \(3, 10\)"):
        PolicyDeltaLearner(initial_theta=np.zeros((3, 9)))


def test_update_raises_on_wrong_phi_shape():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    with pytest.raises(ValueError, match=r"phi must be shape \(10,\)"):
        learner.update(phi=np.zeros(5), action=0,
                       probs=np.array([0.33, 0.33, 0.34]), reward=0.5)


def test_update_raises_on_wrong_probs_shape():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    with pytest.raises(ValueError, match=r"probs must be shape \(3,\)"):
        learner.update(phi=_phi(), action=0, probs=np.array([0.5, 0.5]), reward=0.5)


def test_zero_initial_entries_stay_at_zero_under_learning():
    """Entries the hand-calibration set to zero must remain zero under
    the magnitude cap regardless of how hard the learner pushes."""
    learner = PolicyDeltaLearner(
        initial_theta=THETA, learning_rate=1.0, prior_precision=0.0,
        grad_clip=100.0,
    )
    zero_mask = THETA == 0.0
    for _ in range(100):
        learner.update(
            phi=_phi(forecast=(1.0, 1.0, 1.0), price_signal=1.0),
            action=1, probs=np.array([0.01, 0.98, 0.01]),
            reward=100.0,
        )
    delta = learner.get_theta_delta()
    assert np.all(delta[zero_mask] == 0.0)


def test_magnitude_cap_bounds_nonzero_entries():
    """No entry of the delta exceeds cap_fraction * |initial|."""
    learner = PolicyDeltaLearner(
        initial_theta=THETA, learning_rate=1.0, prior_precision=0.0,
        grad_clip=100.0, magnitude_cap_fraction=0.25,
    )
    for _ in range(100):
        learner.update(
            phi=_phi(forecast=(1.0, 1.0, 1.0), price_signal=1.0),
            action=1, probs=np.array([0.01, 0.98, 0.01]),
            reward=100.0,
        )
    delta = learner.get_theta_delta()
    bound = np.abs(THETA) * 0.25
    assert np.all(np.abs(delta) <= bound + 1e-9)


def test_sign_constraint_preserves_initial_sign_even_under_larger_cap():
    """Under a looser magnitude cap, the sign constraint still prevents
    effective-theta entries from flipping sign from the initial."""
    learner = PolicyDeltaLearner(
        initial_theta=THETA, learning_rate=1.0, prior_precision=0.0,
        grad_clip=100.0, magnitude_cap_fraction=2.0, sign_constrained=True,
    )
    for _ in range(50):
        learner.update(
            phi=_phi(forecast=(1.0, 1.0, 1.0), price_signal=1.0),
            action=0, probs=np.array([0.98, 0.01, 0.01]),
            reward=-100.0,
        )
    effective = learner.get_effective_theta()
    initial_signs = np.sign(THETA)
    effective_signs = np.sign(effective)
    nonzero = initial_signs != 0
    preserved = (effective_signs[nonzero] == initial_signs[nonzero]) | (effective[nonzero] == 0.0)
    assert np.all(preserved)


def test_shrinkage_decays_delta_without_signal():
    """With zero advantage and a strong prior, delta decays toward zero."""
    learner = PolicyDeltaLearner(
        initial_theta=THETA, learning_rate=0.5, prior_precision=1.0,
        baseline_decay=0.0,
    )
    learner.theta_delta = 0.1 * np.abs(THETA) * np.sign(THETA)
    for _ in range(30):
        learner.update(
            phi=_phi(), action=0,
            probs=np.array([0.33, 0.33, 0.34]), reward=0.0,
        )
    assert np.linalg.norm(learner.get_theta_delta()) < 0.05


def test_reset_returns_to_initial_state():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    learner.update(
        phi=_phi(), action=0,
        probs=np.array([0.4, 0.3, 0.3]), reward=0.7,
    )
    assert learner.n_updates == 1
    learner.reset()
    assert learner.n_updates == 0
    np.testing.assert_array_equal(learner.get_theta_delta(), np.zeros_like(THETA))


def test_summary_reports_drift_statistics():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    for _ in range(5):
        learner.update(
            phi=_phi(forecast=(0.5, 0.5, 0.5), price_signal=0.3),
            action=1, probs=np.array([0.2, 0.6, 0.2]), reward=1.0,
        )
    s = learner.summary()
    for key in ("n_updates", "final_theta_delta", "effective_theta",
                "delta_frobenius_norm", "max_delta_entry",
                "max_fractional_drift", "magnitude_cap_fraction",
                "sign_constrained"):
        assert key in s
    assert s["n_updates"] == 5
    assert s["max_fractional_drift"] <= s["magnitude_cap_fraction"] + 1e-9


def test_save_load_round_trip_preserves_delta_and_baseline():
    trained = PolicyDeltaLearner(initial_theta=THETA)
    for _ in range(5):
        trained.update(
            phi=_phi(forecast=(0.3, 0.4, 0.5), price_signal=0.2),
            action=1, probs=np.array([0.2, 0.5, 0.3]), reward=0.7,
        )
    snapshot = trained.save_state()

    restored = PolicyDeltaLearner(initial_theta=THETA)
    restored.load_state(snapshot)
    np.testing.assert_array_equal(
        restored.get_theta_delta(), trained.get_theta_delta()
    )
    assert restored.reward_baseline == pytest.approx(trained.reward_baseline)
    assert restored.n_updates == trained.n_updates


def test_load_state_rejects_wrong_delta_shape():
    learner = PolicyDeltaLearner(initial_theta=THETA)
    with pytest.raises(ValueError, match=r"theta_delta shape"):
        learner.load_state({"theta_delta": np.zeros((3, 9)).tolist()})
