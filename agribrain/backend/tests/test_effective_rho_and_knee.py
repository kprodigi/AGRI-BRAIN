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
    LR_FACTOR_CONSTANT,
    NOMINAL_ROUTE_RHO_FACTOR,
    CC_FACTOR_NOMINAL,
    compute_effective_rho,
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


def test_effective_rho_cc_below_lr_at_nominal_ambient():
    """At nominal ambient, CC has the smaller factor (0.15 vs 0.45 LR),
    so a CC-only policy produces LOWER retail rho than an LR-only
    policy. This corrects the previous wrong-physics test that said
    LR has lower retail rho - in reality cold chain is the better
    refrigerated route at nominal ambient."""
    env_rho = np.linspace(0.0, 0.7, 288)
    T = np.full(288, 4.0)
    cc_only = np.tile([1.0, 0.0, 0.0], (288, 1))
    lr_only = np.tile([0.0, 1.0, 0.0], (288, 1))
    eff_cc = compute_effective_rho(env_rho, cc_only, ambient_temp_c=T)
    eff_lr = compute_effective_rho(env_rho, lr_only, ambient_temp_c=T)
    assert eff_cc.mean() < eff_lr.mean(), (
        f"at nominal ambient expected CC < LR; got "
        f"CC={eff_cc.mean():.4f} LR={eff_lr.mean():.4f}"
    )
    # At equilibrium the ratio should approach factor_cc/factor_lr.
    nominal_ratio = CC_FACTOR_NOMINAL / LR_FACTOR_CONSTANT
    assert eff_cc[-1] / max(eff_lr[-1], 1e-9) == pytest.approx(
        nominal_ratio, abs=0.15
    )


def test_effective_rho_lr_below_cc_when_overwhelmed():
    """Above 35 degC the cold chain is overwhelmed (factor 1.00) while
    LR holds at 0.45, so the ordering flips: LR-only < CC-only."""
    env_rho = np.linspace(0.0, 0.7, 288)
    T = np.full(288, 38.0)  # cold chain overwhelmed
    cc_only = np.tile([1.0, 0.0, 0.0], (288, 1))
    lr_only = np.tile([0.0, 1.0, 0.0], (288, 1))
    eff_cc = compute_effective_rho(env_rho, cc_only, ambient_temp_c=T)
    eff_lr = compute_effective_rho(env_rho, lr_only, ambient_temp_c=T)
    assert eff_lr.mean() < eff_cc.mean()


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
    flip the policy to Recovery dominance, not LR dominance."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.85, inv=10000, y_hat=20, temp=20.0, tau=0.0,
        policy=_DummyPolicy(), rng=rng, deterministic=True,
    )
    cc, lr, rec = probs
    assert rec > cc, f"expected Rec > CC at rho=0.85; got Rec={rec:.3f} CC={cc:.3f}"
    assert rec > lr, f"expected Rec > LR above knee; got Rec={rec:.3f} LR={lr:.3f}"


def test_mid_rho_agribrain_prefers_lr_below_knee():
    """In the at-risk band (RLE < rho < knee), the policy should still
    prefer LR - this is the marketable-but-degrading triage band."""
    rng = np.random.default_rng(0)
    _, probs = select_action(
        mode="agribrain",
        rho=0.30, inv=10000, y_hat=20, temp=15.0, tau=0.0,
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
    the upper marketability boundary (~0.65)."""
    assert 0.30 < RHO_RECOVERY_KNEE < 0.70
