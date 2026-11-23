"""PolicyLearner shape-assertion guard.

Before the 2026-05 fix, ``record()`` silently accepted any 1-D
ndarray, so a caller using the legacy 6-dim phi(s) into a 10-dim
learner produced misaligned theta gradients on ``update()`` without
raising. The new assertion fails at record-time. These tests pin
that contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.policy_learner import PolicyLearner


def test_record_accepts_canonical_shape():
    learner = PolicyLearner(n_actions=3, n_features=10)
    learner.record(np.zeros(10), 1, 0.5)
    assert len(learner._buffer) == 1


def test_record_rejects_wrong_feature_dim():
    learner = PolicyLearner(n_actions=3, n_features=10)
    with pytest.raises(ValueError, match="shape"):
        learner.record(np.zeros(6), 1, 0.5)


def test_record_rejects_two_d_array():
    learner = PolicyLearner(n_actions=3, n_features=10)
    with pytest.raises(ValueError, match="shape"):
        learner.record(np.zeros((10, 1)), 1, 0.5)


def test_record_rejects_action_out_of_range():
    learner = PolicyLearner(n_actions=3, n_features=10)
    with pytest.raises(ValueError, match="action"):
        learner.record(np.zeros(10), 3, 0.5)


def test_record_rejects_negative_action():
    learner = PolicyLearner(n_actions=3, n_features=10)
    with pytest.raises(ValueError, match="action"):
        learner.record(np.zeros(10), -1, 0.5)


def test_record_supports_custom_dimensions():
    """Construction parameters should still work end-to-end."""
    learner = PolicyLearner(n_actions=4, n_features=6)
    learner.record(np.zeros(6), 3, 0.0)
    learner.record(np.zeros(6), 0, 1.0)
    assert len(learner._buffer) == 2
