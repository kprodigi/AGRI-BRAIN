"""Unit tests for :class:`pirag.context_learner.ForecastWeightsLearner`."""
from __future__ import annotations

import numpy as np
import pytest

from pirag.context_learner import ForecastWeightsLearner


def _phi(forecast=(0.2, 0.3, 0.1), price_signal: float = 0.0):
    """Build a 10-dim phi with forecast channels set to the given tuple
    and the price channel at the given scalar."""
    phi = np.zeros(10, dtype=np.float64)
    phi[6:9] = np.asarray(forecast, dtype=np.float64)
    phi[9] = float(price_signal)
    return phi


def test_initial_delta_is_zero():
    learner = ForecastWeightsLearner()
    np.testing.assert_array_equal(learner.get_theta_delta(), np.zeros((3, 3)))


def test_update_raises_on_wrong_phi_shape():
    learner = ForecastWeightsLearner()
    bad = np.zeros(5)
    with pytest.raises(ValueError, match=r"phi must be shape \(10,\)"):
        learner.update(phi=bad, action=0, probs=np.array([0.33, 0.33, 0.34]), reward=0.5)


def test_update_raises_on_wrong_probs_shape():
    learner = ForecastWeightsLearner()
    with pytest.raises(ValueError, match=r"probs must be shape \(3,\)"):
        learner.update(phi=_phi(), action=0, probs=np.array([0.5, 0.5]), reward=0.5)


def test_positive_advantage_pushes_delta_toward_action_row():
    """When action 1 gets a positive advantage, the gradient pushes delta
    row 1 up on the forecast channels that are active."""
    learner = ForecastWeightsLearner(learning_rate=0.1, prior_precision=0.0)
    phi = _phi(forecast=(0.5, 0.5, 0.5))
    probs = np.array([0.33, 0.34, 0.33])
    learner.update(phi=phi, action=1, probs=probs, reward=1.0)
    d = learner.get_theta_delta()
    # Row 1 (chosen action) receives positive updates on its forecast channels
    assert np.all(d[1, :] > 0.0)
    # Rows 0 and 2 receive negative updates (e_a - probs < 0 for them)
    assert np.all(d[0, :] < 0.0)
    assert np.all(d[2, :] < 0.0)


def test_shrinkage_pulls_delta_toward_zero_when_no_signal():
    """With a strong prior and zero advantage, delta decays toward zero."""
    learner = ForecastWeightsLearner(
        learning_rate=0.5, prior_precision=1.0, baseline_decay=0.0,
    )
    learner.theta_delta = np.ones((3, 3)) * 0.5
    # Reward matches baseline from step 1 onward, so advantage stays zero.
    for _ in range(20):
        learner.update(
            phi=_phi(), action=0,
            probs=np.array([0.33, 0.33, 0.34]), reward=0.0,
        )
    # After many shrinkage steps, delta should be near zero.
    assert np.linalg.norm(learner.get_theta_delta()) < 0.1


def test_zero_forecast_phi_produces_zero_gradient():
    """If phi[6:9] is all zero, the forecast columns get no gradient,
    and only the shrinkage pulls the delta toward zero."""
    learner = ForecastWeightsLearner(learning_rate=0.1, prior_precision=0.0)
    phi = _phi(forecast=(0.0, 0.0, 0.0))
    learner.update(phi=phi, action=1, probs=np.array([0.2, 0.6, 0.2]), reward=1.0)
    np.testing.assert_array_equal(learner.get_theta_delta(), np.zeros((3, 3)))


def test_delta_clip_bounds_individual_entries():
    """Each entry of the delta should stay within [-delta_clip, +delta_clip]
    even under aggressive updates."""
    learner = ForecastWeightsLearner(
        learning_rate=10.0, prior_precision=0.0, grad_clip=100.0, delta_clip=0.5,
    )
    for _ in range(50):
        learner.update(
            phi=_phi(forecast=(1.0, 1.0, 1.0)),
            action=1, probs=np.array([0.01, 0.98, 0.01]),
            reward=100.0,
        )
    d = learner.get_theta_delta()
    assert np.abs(d).max() <= 0.5 + 1e-9


def test_reset_returns_to_initial_state():
    learner = ForecastWeightsLearner()
    learner.update(
        phi=_phi(), action=0,
        probs=np.array([0.5, 0.3, 0.2]), reward=1.0,
    )
    assert learner.n_updates == 1
    learner.reset()
    assert learner.n_updates == 0
    np.testing.assert_array_equal(learner.get_theta_delta(), np.zeros((3, 3)))
    assert learner.reward_baseline == 0.0


def test_summary_fields_present_after_update():
    learner = ForecastWeightsLearner()
    learner.update(
        phi=_phi(), action=0,
        probs=np.array([0.4, 0.3, 0.3]), reward=0.5,
    )
    s = learner.summary()
    for key in ("n_updates", "final_theta_delta", "delta_frobenius_norm",
                "max_delta_entry", "reward_baseline", "mean_advantage",
                "learning_rate", "prior_precision"):
        assert key in s
    assert s["n_updates"] == 1


def test_save_load_round_trip_preserves_delta_and_baseline():
    trained = ForecastWeightsLearner()
    for _ in range(5):
        trained.update(
            phi=_phi(forecast=(0.3, 0.4, 0.5)),
            action=1,
            probs=np.array([0.2, 0.5, 0.3]),
            reward=0.7,
        )
    snapshot = trained.save_state()

    restored = ForecastWeightsLearner()
    restored.load_state(snapshot)
    np.testing.assert_array_equal(
        restored.get_theta_delta(), trained.get_theta_delta()
    )
    assert restored.reward_baseline == pytest.approx(trained.reward_baseline)
    assert restored.n_updates == trained.n_updates


def test_load_state_rejects_wrong_delta_shape():
    learner = ForecastWeightsLearner()
    with pytest.raises(ValueError, match=r"theta_delta shape"):
        learner.load_state({"theta_delta": np.zeros((2, 3)).tolist()})
