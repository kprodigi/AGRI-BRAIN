"""Tests for the policy-responsive effective-rho model and the
Recovery knee triage transition introduced for Figure 2 honesty.

These exercises pin the operating-point claims the figure relies on:

  - effective rho is monotonically smaller for an LR-leaning policy
    than for a CC-locked policy at the same environmental rho trace;
  - effective rho on a Recovery-only policy goes to zero in the limit;
  - the Recovery knee inverts LR / Recovery dominance above
    rho = RHO_RECOVERY_KNEE under otherwise neutral conditions;
  - at rho near zero, agribrain is cold-chain dominant (the operational
    baseline that Figure 2 panel (c) needs to show pre-heatwave);
  - hybrid_rl is not subject to the Recovery knee (no rho-shaping by
    design).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.action_selection import (
    ACTIONS,
    RHO_RECOVERY_KNEE,
    select_action,
)
from src.models.resilience import (
    CC_FACTOR_NOMINAL,
    CC_FACTOR_OVERWHELMED,
    CC_FACTOR_STRESSED,
    LR_FACTOR_COOL,
    LR_FACTOR_CONSTANT,
    LR_FACTOR_HOT,
    LR_FACTOR_NOMINAL,
    LR_FACTOR_STRESSED,
    NOMINAL_ROUTE_RHO_FACTOR,
    compute_effective_rho,
    route_rho_factor,
)


class _DummyPolicy:
    gamma_coldchain = 0.0
    gamma_local = 0.0
    gamma_recovery = 0.0


# ---------------------------------------------------------------------------
# compute_effective_rho
# ---------------------------------------------------------------------------

def test_effective_rho_cold_chain_only_under_nominal_ambient():
    """At nominal ambient (T < 30 degC), CC carries only 0.15 of
    env_rho, so the retail-pool eff_rho is well below env_rho. This is
    the realistic-physics correction: cold chain protects produce."""
    env_rho = np.linspace(0.0, 0.6, 100)
    probs = np.tile(np.array([1.0, 0.0, 0.0]), (100, 1))
    T = np.full(100, 4.0)  # nominal cold storage
    eff = compute_effective_rho(env_rho, probs,
                                turnover_halflife_hours=12.0, dt_hours=0.25,
                                ambient_temp_c=T)
    # Should be roughly factor * env_rho (CC factor = 0.15) with decay
    # smoothing.
    assert np.all(eff <= env_rho * CC_FACTOR_NOMINAL + 0.05)


def test_effective_rho_recovery_only_goes_to_zero():
    """A Recovery-only policy never adds rho to retail inventory, so
    after the first step the effective rho decays toward zero."""
    env_rho = np.linspace(0.0, 0.8, 200)
    probs = np.tile(np.array([0.0, 0.0, 1.0]), (200, 1))
    eff = compute_effective_rho(env_rho, probs,
                                turnover_halflife_hours=12.0, dt_hours=0.25)
    assert eff[-1] == pytest.approx(0.0, abs=1e-3)


def test_effective_rho_cc_below_lr_at_moderate_ambient():
    """At T in the LR nominal band (15-30 degC), CC's 0.15 factor is
    well below LR's 0.45, so a CC-only policy produces LOWER retail
    rho than an LR-only policy. Cold chain is genuinely the better
    refrigerated route at moderate ambient."""
    env_rho = np.linspace(0.0, 0.7, 288)
    T = np.full(288, 20.0)  # LR nominal band
    cc_only = np.tile([1.0, 0.0, 0.0], (288, 1))
    lr_only = np.tile([0.0, 1.0, 0.0], (288, 1))
    eff_cc = compute_effective_rho(env_rho, cc_only, ambient_temp_c=T)
    eff_lr = compute_effective_rho(env_rho, lr_only, ambient_temp_c=T)
    assert eff_cc.mean() < eff_lr.mean(), (
        f"at moderate ambient expected CC < LR; got "
        f"CC={eff_cc.mean():.4f} LR={eff_lr.mean():.4f}"
    )
    # At equilibrium the ratio should approach factor_cc/factor_lr.
    nominal_ratio = CC_FACTOR_NOMINAL / LR_FACTOR_NOMINAL
    assert eff_cc[-1] / max(eff_lr[-1], 1e-9) == pytest.approx(
        nominal_ratio, abs=0.15
    )


def test_effective_rho_cc_close_to_lr_at_cool_ambient():
    """In the LR cool band (T < 15 degC), LR factor drops to 0.20 —
    near-CC performance because food-bank walk-in coolers operate at
    4-7 degC. CC still wins on retail rho but the gap is narrow."""
    env_rho = np.linspace(0.0, 0.5, 288)
    T = np.full(288, 4.0)  # cool band
    cc_only = np.tile([1.0, 0.0, 0.0], (288, 1))
    lr_only = np.tile([0.0, 1.0, 0.0], (288, 1))
    eff_cc = compute_effective_rho(env_rho, cc_only, ambient_temp_c=T)
    eff_lr = compute_effective_rho(env_rho, lr_only, ambient_temp_c=T)
    assert eff_cc.mean() < eff_lr.mean()
    # Equilibrium ratio: 0.15 / 0.20 = 0.75 — much closer than the
    # 0.33 ratio of the previous temperature-independent LR factor.
    cool_ratio = CC_FACTOR_NOMINAL / LR_FACTOR_COOL
    assert eff_cc[-1] / max(eff_lr[-1], 1e-9) == pytest.approx(
        cool_ratio, abs=0.15
    )


def test_effective_rho_lr_below_cc_when_overwhelmed():
    """Above 35 degC the cold chain is overwhelmed (factor 1.00) while
    LR holds at 0.85 (hot band), so the ordering flips: LR-only < CC-only."""
    env_rho = np.linspace(0.0, 0.7, 288)
    T = np.full(288, 38.0)  # cold chain overwhelmed, LR hot band
    cc_only = np.tile([1.0, 0.0, 0.0], (288, 1))
    lr_only = np.tile([0.0, 1.0, 0.0], (288, 1))
    eff_cc = compute_effective_rho(env_rho, cc_only, ambient_temp_c=T)
    eff_lr = compute_effective_rho(env_rho, lr_only, ambient_temp_c=T)
    assert eff_lr.mean() < eff_cc.mean()


def test_route_rho_factor_lr_piecewise():
    """LR factor is piecewise-constant on ambient temperature with
    breakpoints at 15, 30, 35 degC. Pin the four bands."""
    assert route_rho_factor("local_redistribute", 4.0) == LR_FACTOR_COOL
    assert route_rho_factor("local_redistribute", 14.99) == LR_FACTOR_COOL
    assert route_rho_factor("local_redistribute", 15.0) == LR_FACTOR_NOMINAL
    assert route_rho_factor("local_redistribute", 25.0) == LR_FACTOR_NOMINAL
    assert route_rho_factor("local_redistribute", 30.0) == LR_FACTOR_STRESSED
    assert route_rho_factor("local_redistribute", 33.0) == LR_FACTOR_STRESSED
    assert route_rho_factor("local_redistribute", 35.01) == LR_FACTOR_HOT
    assert route_rho_factor("local_redistribute", 40.0) == LR_FACTOR_HOT


def test_route_rho_factor_lr_monotone_in_temperature():
    """LR factor is monotonically non-decreasing in ambient temperature
    — warmer staging gives less protection."""
    temps = np.linspace(0.0, 45.0, 100)
    factors = np.array([route_rho_factor("local_redistribute", float(t))
                        for t in temps])
    assert np.all(np.diff(factors) >= 0.0)


def test_lr_factor_constant_alias_matches_nominal():
    """Backward-compatible alias: the old LR_FACTOR_CONSTANT should
    equal the nominal-band factor so any un-migrated caller still gets
    the previous numerics."""
    assert LR_FACTOR_CONSTANT == LR_FACTOR_NOMINAL


def test_effective_rho_clipped_to_unit():
    env_rho = np.full(50, 1.5)
    probs = np.tile([1.0, 0.0, 0.0], (50, 1))
    eff = compute_effective_rho(env_rho, probs)
    assert np.all(eff <= 1.0)
    assert np.all(eff >= 0.0)


def test_effective_rho_rejects_misshaped_probs():
    env_rho = np.linspace(0.0, 0.5, 20)
    bad = np.zeros((20, 2))
    with pytest.raises(ValueError, match="action_probs must be shape"):
        compute_effective_rho(env_rho, bad)


# ---------------------------------------------------------------------------
# Recovery knee
# ---------------------------------------------------------------------------

def test_low_rho_agribrain_is_cold_chain_dominant():
    """At fresh-produce / safe-temperature operating points, the policy
    must prefer cold chain - the operational baseline Figure 2 needs."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.02, inv=10000, y_hat=20, temp=4.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    cc, lr, rec = probs
    assert cc > lr, f"expected CC > LR at rho~0; got CC={cc:.3f} LR={lr:.3f}"
    assert cc > rec
    # CC plurality at the operational baseline; not necessarily a
    # majority because LR carries a small constant SLCA_BONUS and at
    # nominal demand the inv_pressure / demand_pt columns lift LR too.
    assert cc > 0.45


def test_high_rho_agribrain_prefers_recovery_above_knee():
    """Above the Recovery knee (rho > 0.50), the triage logic must
    flip the policy to Recovery dominance, not LR dominance. With the
    boosted knee gain (5.0/3.0) Recovery should be the clear majority,
    not just plurality."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.85, inv=10000, y_hat=20, temp=20.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    cc, lr, rec = probs
    assert rec > cc, f"expected Rec > CC at rho=0.85; got Rec={rec:.3f} CC={cc:.3f}"
    assert rec > lr, f"expected Rec > LR above knee; got Rec={rec:.3f} LR={lr:.3f}"
    # With KNEE_GAIN=5.0 + LR_PENALTY=3.0, Recovery should be the
    # clear majority at rho=0.85 (the previous 2.50/1.50 magnitudes
    # produced only a plurality with Recovery share around 0.4-0.5).
    assert rec > 0.55, (
        f"expected Rec > 0.55 with boosted knee at rho=0.85; got Rec={rec:.3f}"
    )


def test_recovery_dominant_at_food_safety_cutoff():
    """At the food-safety cutoff (rho = 0.65, the threshold above
    which BatchInventory's hard override fires), the policy should
    already prefer Recovery over LR via the soft knee, so the policy
    and the override agree on the routing direction. With the
    boosted knee gains (5.00 / 3.00) Recovery should be above LR
    here — the previous (2.50 / 1.50) magnitudes only flipped the
    ordering at rho > 0.75."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.65, inv=10000, y_hat=20, temp=25.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    cc, lr, rec = probs
    assert rec > lr, (
        f"expected Rec > LR at food-safety cutoff rho=0.65; got "
        f"Rec={rec:.3f} LR={lr:.3f}"
    )


def test_mid_rho_agribrain_prefers_lr_below_knee():
    """In the at-risk band (RLE < rho < knee), the policy should still
    prefer LR - this is the marketable-but-degrading triage band.
    Tested at rho=0.20 which sits inside the at-risk band (>0.10) and
    below the new knee threshold of 0.30."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.20, inv=10000, y_hat=20, temp=15.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    cc, lr, rec = probs
    assert lr > cc, f"expected LR > CC in at-risk band; got LR={lr:.3f} CC={cc:.3f}"
    assert lr > rec, f"expected LR > Rec below knee; got LR={lr:.3f} Rec={rec:.3f}"


def test_hybrid_rl_not_subject_to_recovery_knee():
    """hybrid_rl has no rho-shaping by design, so the knee must not
    apply to it. At high rho the distribution should still come from
    plain THETA @ phi without any added Recovery boost."""
    rng = np.random.default_rng(0)
    _, probs_low = select_action(
        mode="hybrid_rl",
        rho=0.40, inv=10000, y_hat=20, temp=15.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    _, probs_high = select_action(
        mode="hybrid_rl",
        rho=0.90, inv=10000, y_hat=20, temp=15.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    # Without a knee, the absolute Recovery share between rho=0.40 and
    # rho=0.90 is driven only by THETA's spoilage column (+1.5 on Rec)
    # vs LR's +2.0; LR should still grow faster than Rec, so the
    # Recovery share at high rho stays below LR's share.
    assert probs_high[2] < probs_high[1] + 1e-6


def test_knee_threshold_constant_is_in_realistic_range():
    """The knee should sit between the at-risk threshold (0.10) and
    the food-safety hard cutoff (0.65)."""
    assert 0.10 < RHO_RECOVERY_KNEE < 0.65
