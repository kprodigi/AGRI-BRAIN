"""Tests for route-conditioned reward shaping with temperature-
conditional factors.

The reward used to penalise ``rho`` directly with no reference to the
chosen action - which meant the policy gradient on the rho channel was
identical regardless of whether the agent chose cold_chain or
local_redistribute. The reward is now route-conditioned with a
*temperature-conditional* cold-chain factor: at nominal ambient
(T < 30 degC) cold chain has the smallest factor (0.15) because
real refrigerated trucks maintain ~85% temperature integrity; the
factor steps to 0.40 (stressed) at 30-35 degC and to 1.00
(overwhelmed) above 35 degC. LR is fixed at 0.45, Recovery at 0.00.

These tests pin the new semantics:

  - at nominal T (4 degC), CC pays the smallest rho penalty (0.15)
  - LR is fixed across temperatures (0.45)
  - Recovery always pays zero rho penalty
  - the per-action ranking flips at the 30 / 35 degC breakpoints
  - backward compatibility: omitting route_factor yields the
    unfactored reward (legacy callers are bit-identical)
"""
from __future__ import annotations

import pytest

from src.models.resilience import (
    NOMINAL_ROUTE_RHO_FACTOR,
    route_rho_factor,
)
from src.models.reward import compute_reward, compute_reward_extended


# ---------------------------------------------------------------------------
# compute_reward: route-conditioned semantics
# ---------------------------------------------------------------------------

def test_cold_chain_pays_smallest_rho_penalty_at_nominal():
    """At nominal ambient (T < 30 degC), cold chain has the smallest
    route factor (0.15) so it pays the smallest rho penalty - matching
    the real-world fact that refrigerated trucks insulate produce."""
    T = 4.0
    cc = compute_reward(0.70, 0.05, rho=0.40,
                        route_factor=route_rho_factor("cold_chain", T))
    lr = compute_reward(0.70, 0.05, rho=0.40,
                        route_factor=route_rho_factor("local_redistribute", T))
    rec = compute_reward(0.70, 0.05, rho=0.40,
                         route_factor=route_rho_factor("recovery", T))
    # Recovery pays zero rho penalty -> highest reward; CC pays the
    # next-smallest; LR pays the most among the in-pool actions.
    assert rec > cc > lr


def test_route_factor_advantage_equals_eta_rho_times_factor_diff():
    """The reward gap between two actions is exactly
    eta_rho * rho * (factor_a - factor_b)."""
    T = 4.0
    rho = 0.40
    eta_rho = 0.50
    cc_factor = route_rho_factor("cold_chain", T)
    lr_factor = route_rho_factor("local_redistribute", T)
    cc = compute_reward(0.70, 0.05, rho=rho, eta_rho=eta_rho,
                        route_factor=cc_factor)
    lr = compute_reward(0.70, 0.05, rho=rho, eta_rho=eta_rho,
                        route_factor=lr_factor)
    expected_gap = eta_rho * rho * (lr_factor - cc_factor)
    assert (lr - cc) == pytest.approx(-expected_gap, abs=1e-6)


def test_recovery_pays_zero_rho_penalty():
    rec = compute_reward(0.70, 0.05, rho=0.80,
                         route_factor=route_rho_factor("recovery", 30.0))
    no_rho = compute_reward(0.70, 0.05, rho=0.0,
                            route_factor=route_rho_factor("cold_chain", 4.0))
    assert rec > no_rho - 1e-9


def test_route_factor_omitted_equals_legacy_form():
    """Legacy callers that don't pass route_factor must produce
    bit-identical rewards to the previous unfactored form."""
    legacy = compute_reward(0.70, 0.05, rho=0.40, eta=0.50, eta_rho=0.50)
    full_pen = compute_reward(0.70, 0.05, rho=0.40, eta=0.50, eta_rho=0.50,
                              route_factor=1.00)
    assert legacy == pytest.approx(full_pen)


def test_per_action_ranking_below_30c_cold_chain_preferred():
    """At T < 30 degC, the policy gradient on rho prefers Recovery > CC > LR.
    Cold chain BEATS local-redistribute on the rho channel because
    refrigerated trucks have higher temperature integrity."""
    T = 25.0
    rho = 0.40
    rewards = {
        a: compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor(a, T))
        for a in ("cold_chain", "local_redistribute", "recovery")
    }
    assert rewards["recovery"] > rewards["cold_chain"] > rewards["local_redistribute"]


def test_per_action_ranking_in_stress_band_close():
    """At 30-35 degC the CC and LR factors are 0.40 vs 0.45 - close,
    so the rho-channel reward gap is small. Recovery still dominates
    on the rho channel."""
    T = 32.0
    rho = 0.40
    cc = compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor("cold_chain", T))
    lr = compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor("local_redistribute", T))
    rec = compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor("recovery", T))
    # CC and LR within ~0.05 reward of each other; Recovery clearly wins.
    assert abs(cc - lr) < 0.05
    assert rec > max(cc, lr)


def test_per_action_ranking_above_35c_lr_beats_cc():
    """Above 35 degC, cold chain is overwhelmed (factor 1.00) and LR
    (0.45) provides better thermal protection. This is the only
    regime where LR genuinely beats CC on the rho channel."""
    T = 38.0
    rho = 0.40
    cc = compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor("cold_chain", T))
    lr = compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor("local_redistribute", T))
    rec = compute_reward(0.70, 0.05, rho, route_factor=route_rho_factor("recovery", T))
    assert rec > lr > cc, (
        f"at T={T} degC expected Rec > LR > CC; got "
        f"Rec={rec:.4f} LR={lr:.4f} CC={cc:.4f}"
    )


def test_per_action_reward_ranking_neutral_at_low_spoilage():
    """At rho ~ 0, the route_factor multiplies a near-zero quantity, so
    the per-action reward differences from the rho channel vanish.
    The policy then routes purely on SLCA / waste / state features."""
    rho = 0.01
    rewards = [
        compute_reward(0.70, 0.05, rho,
                       route_factor=route_rho_factor(a, 4.0))
        for a in ("cold_chain", "local_redistribute", "recovery")
    ]
    spread = max(rewards) - min(rewards)
    assert spread < 0.01, (
        f"at rho={rho:.2f} reward spread should be ~zero; got {spread:.5f}"
    )


# ---------------------------------------------------------------------------
# compute_reward_extended
# ---------------------------------------------------------------------------

def test_extended_reward_route_factor_stored_in_decomposition():
    out = compute_reward_extended(
        slca_composite=0.70, waste=0.05, rho=0.40,
        energy_J=0.0, water_L=0.0,
        route_factor=NOMINAL_ROUTE_RHO_FACTOR["local_redistribute"],
    )
    assert out["route_factor"] == pytest.approx(0.45)


def test_extended_reward_rho_penalty_uses_route_factor_at_nominal_T():
    """At nominal ambient (T < 30 degC), CC pays the SMALLER rho
    penalty than LR. This corrects the previous wrong-physics test
    that asserted the opposite."""
    cc = compute_reward_extended(
        slca_composite=0.70, waste=0.05, rho=0.40,
        route_factor=route_rho_factor("cold_chain", 4.0),
    )
    lr = compute_reward_extended(
        slca_composite=0.70, waste=0.05, rho=0.40,
        route_factor=route_rho_factor("local_redistribute", 4.0),
    )
    assert cc["rho_penalty"] < lr["rho_penalty"]
    assert cc["total"] > lr["total"]


def test_extended_reward_rho_penalty_inverts_when_cc_overwhelmed():
    """Above 35 degC, cold chain pays the LARGER rho penalty than LR
    because the truck cooling has failed."""
    cc = compute_reward_extended(
        slca_composite=0.70, waste=0.05, rho=0.40,
        route_factor=route_rho_factor("cold_chain", 38.0),
    )
    lr = compute_reward_extended(
        slca_composite=0.70, waste=0.05, rho=0.40,
        route_factor=route_rho_factor("local_redistribute", 38.0),
    )
    assert cc["rho_penalty"] > lr["rho_penalty"]
    assert lr["total"] > cc["total"]
