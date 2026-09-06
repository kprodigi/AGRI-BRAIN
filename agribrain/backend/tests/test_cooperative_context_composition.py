"""Executable lock for the disclosed cooperative-context composition."""
from __future__ import annotations

import numpy as np

from src.agents.coordinator import _compose_context_attribution


def _trace(modifier: np.ndarray) -> dict:
    features = np.zeros((3, 5), dtype=float)
    features[:, 0] = modifier
    return {
        "feature_contributions": features,
        "nonfeature_residual": np.zeros(3, dtype=float),
        "modifier_theta_jacobian": np.ones((3, 5), dtype=float),
    }


def test_ordinary_overlay_uses_disclosed_70_30_composition() -> None:
    primary = np.array([0.20, 0.40, 0.60])
    cooperative = np.array([0.60, 0.20, -0.20])
    modifier, features, residual, scope, jacobian, composition = (
        _compose_context_attribution(
            primary,
            _trace(primary),
            cooperative,
            _trace(cooperative),
        )
    )

    expected = 0.70 * primary + 0.30 * cooperative
    np.testing.assert_allclose(modifier, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(features.sum(axis=1) + residual, expected)
    np.testing.assert_allclose(jacobian, np.ones((3, 5)))
    assert scope == "cooperative_blend"
    assert composition["scope"] == "cooperative_blend"


def test_critical_envelope_branch_uses_disclosed_bias_and_final_clip() -> None:
    primary = np.array([-0.90, -0.90, -0.90])
    cooperative = np.array([-0.95, 0.95, 0.25])
    bias = np.array([-0.20, 0.20, 0.00])
    modifier, features, residual, scope, jacobian, composition = (
        _compose_context_attribution(
            primary,
            _trace(primary),
            cooperative,
            _trace(cooperative),
            bias,
        )
    )

    expected = np.clip(cooperative + bias, -1.0, 1.0)
    np.testing.assert_allclose(modifier, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(features.sum(axis=1) + residual, expected)
    # The first two rows saturate, so the declared zero subgradient applies.
    np.testing.assert_allclose(jacobian[:2], np.zeros((2, 5)))
    np.testing.assert_allclose(jacobian[2], np.ones(5))
    assert scope == "cooperative_veto"  # legacy trace key
    assert composition["scope"] == "cooperative_veto"
