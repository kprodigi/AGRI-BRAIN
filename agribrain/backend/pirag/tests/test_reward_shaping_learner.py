"""Unit tests for :class:`pirag.context_learner.RewardShapingLearner`."""
from __future__ import annotations

import numpy as np
import pytest

from pirag.context_learner import RewardShapingLearner
from src.models.action_selection import SLCA_BONUS, SLCA_RHO_BONUS, NO_SLCA_OFFSET


def _new_learner(**overrides):
    kwargs = dict(
        initial_slca_bonus=SLCA_BONUS,
        initial_slca_rho_bonus=SLCA_RHO_BONUS,
        initial_no_slca_offset=NO_SLCA_OFFSET,
    )
    kwargs.update(overrides)
    return RewardShapingLearner(**kwargs)


def test_initial_deltas_are_zero():
    learner = _new_learner()
    np.testing.assert_array_equal(learner.get_slca_bonus_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_slca_rho_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_no_slca_offset_delta(), np.zeros(3))


def test_init_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"initial_slca_bonus must be shape \(3,\)"):
        RewardShapingLearner(
            initial_slca_bonus=np.zeros(2),
            initial_slca_rho_bonus=SLCA_RHO_BONUS,
            initial_no_slca_offset=NO_SLCA_OFFSET,
        )


def test_agribrain_mode_updates_slca_vectors_only():
    """In an SLCA_BONUS-eligible mode, both SLCA deltas move but the
    offset delta only shrinks (never gains signal)."""
    learner = _new_learner(learning_rate=0.1, prior_precision=0.0)
    learner.update(
        action=1,
        probs=np.array([0.3, 0.5, 0.2]),
        reward=1.0,
        mode="agribrain",
        rho=0.3,
    )
    assert np.any(learner.get_slca_bonus_delta() != 0.0)
    assert np.any(learner.get_slca_rho_delta() != 0.0)
    # Offset got only shrinkage (which is a no-op on a zero vector), so
    # it is still exactly zero.
    np.testing.assert_array_equal(learner.get_no_slca_offset_delta(), np.zeros(3))


def test_no_slca_mode_updates_offset_only():
    learner = _new_learner(learning_rate=0.1, prior_precision=0.0)
    learner.update(
        action=0,
        probs=np.array([0.6, 0.2, 0.2]),
        reward=1.0,
        mode="no_slca",
        rho=0.3,
    )
    assert np.any(learner.get_no_slca_offset_delta() != 0.0)
    np.testing.assert_array_equal(learner.get_slca_bonus_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_slca_rho_delta(), np.zeros(3))


def test_hybrid_rl_mode_is_noop_on_all_deltas():
    """hybrid_rl does not use any reward-shaping vector; its updates
    only shrink the deltas (no gradient)."""
    learner = _new_learner(learning_rate=0.1, prior_precision=0.0)
    learner.update(
        action=1, probs=np.array([0.3, 0.5, 0.2]),
        reward=1.0, mode="hybrid_rl", rho=0.3,
    )
    np.testing.assert_array_equal(learner.get_slca_bonus_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_slca_rho_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_no_slca_offset_delta(), np.zeros(3))


def test_magnitude_cap_bounds_each_vector():
    learner = _new_learner(
        learning_rate=1.0, prior_precision=0.0,
        grad_clip=100.0, magnitude_cap_fraction=0.25,
    )
    for _ in range(80):
        learner.update(
            action=1, probs=np.array([0.01, 0.98, 0.01]),
            reward=100.0, mode="agribrain", rho=0.5,
        )
    bonus_bound = np.abs(SLCA_BONUS) * 0.25
    rho_bound = np.abs(SLCA_RHO_BONUS) * 0.25
    assert np.all(np.abs(learner.get_slca_bonus_delta()) <= bonus_bound + 1e-9)
    assert np.all(np.abs(learner.get_slca_rho_delta()) <= rho_bound + 1e-9)


def test_sign_constraint_preserves_initial_signs_under_looser_cap():
    learner = _new_learner(
        learning_rate=1.0, prior_precision=0.0,
        grad_clip=100.0, magnitude_cap_fraction=3.0, sign_constrained=True,
    )
    for _ in range(50):
        learner.update(
            action=0, probs=np.array([0.98, 0.01, 0.01]),
            reward=-100.0, mode="no_slca", rho=0.3,
        )
    effective = NO_SLCA_OFFSET + learner.get_no_slca_offset_delta()
    initial_signs = np.sign(NO_SLCA_OFFSET)
    effective_signs = np.sign(effective)
    nonzero = initial_signs != 0
    preserved = (effective_signs[nonzero] == initial_signs[nonzero]) | (effective[nonzero] == 0.0)
    assert np.all(preserved)


def test_save_load_round_trip():
    trained = _new_learner()
    for _ in range(5):
        trained.update(
            action=1, probs=np.array([0.2, 0.6, 0.2]),
            reward=0.7, mode="agribrain", rho=0.4,
        )
    snapshot = trained.save_state()

    restored = _new_learner()
    restored.load_state(snapshot)
    np.testing.assert_array_equal(
        restored.get_slca_bonus_delta(), trained.get_slca_bonus_delta()
    )
    np.testing.assert_array_equal(
        restored.get_slca_rho_delta(), trained.get_slca_rho_delta()
    )
    assert restored.n_updates == trained.n_updates


def test_reset_clears_state():
    learner = _new_learner()
    learner.update(
        action=1, probs=np.array([0.2, 0.6, 0.2]),
        reward=1.0, mode="agribrain", rho=0.3,
    )
    assert learner.n_updates == 1
    learner.reset()
    assert learner.n_updates == 0
    np.testing.assert_array_equal(learner.get_slca_bonus_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_slca_rho_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_no_slca_offset_delta(), np.zeros(3))


def test_summary_contains_effective_vectors():
    learner = _new_learner()
    learner.update(
        action=1, probs=np.array([0.3, 0.5, 0.2]),
        reward=1.0, mode="agribrain", rho=0.3,
    )
    s = learner.summary()
    for key in ("slca_bonus_delta", "slca_rho_delta", "no_slca_offset_delta",
                "effective_slca_bonus", "effective_slca_rho_bonus",
                "effective_no_slca_offset", "magnitude_cap_fraction",
                "sign_constrained"):
        assert key in s
