"""Regression tests for the action-specific volatility-regime modifier."""
from __future__ import annotations

import numpy as np
import pytest


def test_regime_flag_changes_relative_logits_and_probabilities():
    """The binary flag must not be a softmax-invariant common shift."""
    from src.models.action_selection import select_action
    from src.models.policy import Policy

    kwargs = {
        "mode": "hybrid_rl",
        "rho": 0.10,
        "inv": 12_000.0,
        "y_hat": 20.0,
        "temp": 4.0,
        "policy": Policy(),
        "deterministic": True,
    }
    _, quiet = select_action(
        tau=0.0, rng=np.random.default_rng(7), **kwargs,
    )
    _, volatile = select_action(
        tau=1.0, rng=np.random.default_rng(7), **kwargs,
    )

    assert not np.allclose(quiet, volatile)
    # Declared b_tau=[+0.25,+0.05,-0.25] shifts probability toward
    # cold-chain and away from recovery when the regime flag is active.
    assert volatile[0] > quiet[0]
    assert volatile[2] < quiet[2]


def test_default_regime_bias_is_action_specific():
    from src.models.policy import Policy

    policy = Policy()
    values = np.array([
        policy.gamma_coldchain,
        policy.gamma_local,
        policy.gamma_recovery,
    ])
    assert np.allclose(values, [0.25, 0.05, -0.25])
    assert np.ptp(values) > 0.0


def test_regime_term_is_recorded_and_nonbinary_flag_fails_closed():
    from src.models.action_selection import select_action
    from src.models.policy import Policy

    kwargs = {
        "mode": "hybrid_rl", "rho": 0.2, "inv": 12_000.0,
        "y_hat": 20.0, "temp": 7.0, "policy": Policy(),
        "rng": np.random.default_rng(23), "deterministic": True,
    }
    diagnostics = {}
    select_action(tau=1.0, out=diagnostics, **kwargs)
    np.testing.assert_allclose(
        diagnostics["regime_logit_bias"], [0.25, 0.05, -0.25],
        rtol=0.0, atol=0.0,
    )

    with pytest.raises(ValueError, match="binary flag 0 or 1"):
        select_action(tau=0.5, **kwargs)
