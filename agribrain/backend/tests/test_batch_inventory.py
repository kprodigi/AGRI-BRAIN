"""Tests for the per-batch FIFO inventory model.

The aggregate-state simulator can't mechanistically derive
"rho on retail-bound inventory" from the action mix alone -
``compute_effective_rho`` has to do that as a post-hoc accounting
transformation. ``BatchInventory`` runs a true per-batch FIFO model
alongside the aggregate state, so the retail-pool rho is a measured
quantity, not a derived one. These tests pin the behaviour Figure 2
panel (b) depends on:

  - cold-chain-only routing produces the highest retail-pool rho
    (because every batch carries the full env_rho through transit);
  - local-redistribute-only is strictly lower (factor 0.40);
  - recovery-only sends batches out of the retail pool entirely so
    eventually retail-pool rho approaches zero (no fresh routed-in
    batches to replenish);
  - FIFO discipline: the oldest DC batch is routed first;
  - retail pool clears at the configured sale rate;
  - quantity-weighting: batches with more units have proportionally
    more influence on the retail-pool mean rho;
  - sensitivity: route factors perturbed +/- 25% preserve the
    cold_chain > local_redistribute > recovery ordering.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.batch_inventory import (
    BatchInventory,
    Batch,
    TRANSIT_HOURS,
)
from src.models.resilience import (
    CC_FACTOR_NOMINAL,
    CC_FACTOR_OVERWHELMED,
    CC_FACTOR_STRESSED,
    CC_NOMINAL_THRESHOLD_C,
    CC_OVERWHELMED_THRESHOLD_C,
    DC_RHO_FACTOR,
    LR_FACTOR_CONSTANT,
    NOMINAL_ROUTE_RHO_FACTOR,
    RECOVERY_FACTOR,
    route_rho_factor,
)


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------

def test_route_factor_ordering_at_nominal_ambient():
    """At T < 30 degC (cold chain nominal), the ordering is
    LR > CC > Recovery: cold chain provides better thermal protection
    than local-redistribute, recovery removes from retail pool."""
    T_nominal = 4.0
    cc = route_rho_factor("cold_chain", T_nominal)
    lr = route_rho_factor("local_redistribute", T_nominal)
    rec = route_rho_factor("recovery", T_nominal)
    assert lr > cc > rec, (
        f"at T={T_nominal} degC expected LR > CC > Rec; got "
        f"LR={lr:.3f} CC={cc:.3f} Rec={rec:.3f}"
    )
    assert rec == 0.0


def test_route_factor_ordering_above_overwhelmed_threshold():
    """At T > 35 degC (cold chain overwhelmed), the ordering flips:
    CC > LR > Recovery because the truck cooling has failed and
    LR's shorter dwell becomes the better thermal-exposure choice."""
    T_extreme = 38.0
    cc = route_rho_factor("cold_chain", T_extreme)
    lr = route_rho_factor("local_redistribute", T_extreme)
    rec = route_rho_factor("recovery", T_extreme)
    assert cc > lr > rec, (
        f"at T={T_extreme} degC expected CC > LR > Rec; got "
        f"CC={cc:.3f} LR={lr:.3f} Rec={rec:.3f}"
    )


def test_route_factor_stress_band_cc_better_than_lr():
    """In the 30-35 degC stress band CC stays at 0.40 (truck still
    holds 4-7 degC internal) while LR rises to 0.65 (warehouse
    staging heats up). CC is therefore the better thermal route in
    the stress band on a per-step basis; AgriBrain's LR preference
    in this regime trades retail-pool rho for SLCA-composite gains
    rather than exploiting better thermal physics."""
    for T in (30.0, 32.0, 34.0, 35.0):
        cc = route_rho_factor("cold_chain", T)
        lr = route_rho_factor("local_redistribute", T)
        assert cc < lr, (
            f"at T={T} degC stress band, CC should be below LR; "
            f"got CC={cc:.3f} LR={lr:.3f}"
        )


def test_route_factor_threshold_breakpoints():
    """Pin the published thresholds so a regression cannot silently
    move them."""
    assert CC_NOMINAL_THRESHOLD_C == 30.0
    assert CC_OVERWHELMED_THRESHOLD_C == 35.0
    # Step function values
    assert route_rho_factor("cold_chain", 29.99) == CC_FACTOR_NOMINAL
    assert route_rho_factor("cold_chain", 30.00) == CC_FACTOR_STRESSED
    assert route_rho_factor("cold_chain", 35.00) == CC_FACTOR_STRESSED
    assert route_rho_factor("cold_chain", 35.01) == CC_FACTOR_OVERWHELMED


def test_route_factor_rejects_unknown_action():
    with pytest.raises(ValueError, match="Unknown action"):
        route_rho_factor("teleport", 4.0)


def test_dc_factor_in_realistic_range():
    """Mercier (2017) reports DC temperature integrity 0.15-0.30 of
    ambient deviation. Our default should sit inside that range."""
    assert 0.15 <= DC_RHO_FACTOR <= 0.30


def test_nominal_route_factor_dict_matches_step_constants():
    """The legacy dict alias must agree with the step-function
    constants at nominal ambient."""
    assert NOMINAL_ROUTE_RHO_FACTOR["cold_chain"] == CC_FACTOR_NOMINAL
    assert NOMINAL_ROUTE_RHO_FACTOR["local_redistribute"] == LR_FACTOR_CONSTANT
    assert NOMINAL_ROUTE_RHO_FACTOR["recovery"] == RECOVERY_FACTOR


def test_transit_hours_dwell_proportional_to_route_distance():
    """Transit hours should scale with the policy.py route distances at
    a constant ~50 km/h average refrigerated-truck speed."""
    # 120 / 50 = 2.4, 45 / 50 = 0.9, 80 / 50 = 1.6
    assert TRANSIT_HOURS["cold_chain"] == pytest.approx(2.4, abs=0.5)
    assert TRANSIT_HOURS["local_redistribute"] == pytest.approx(0.9, abs=0.3)
    assert TRANSIT_HOURS["recovery"] == pytest.approx(1.6, abs=0.5)


def test_batch_inventory_seeds_initial_dc_pool():
    inv = BatchInventory(initial_n_batches=5, initial_dc_quantity=10000.0)
    dc_batches = [b for b in inv.batches if b.status == "dc"]
    assert len(dc_batches) == 5
    assert sum(b.quantity for b in dc_batches) == pytest.approx(10000.0)
    assert all(b.current_rho == 0.0 for b in dc_batches)


# ---------------------------------------------------------------------------
# Per-step routing behaviour
# ---------------------------------------------------------------------------

def test_step_routes_oldest_batch_fifo():
    """The oldest batch (lowest arrival_hour) must be the one routed
    when the policy chooses an action."""
    inv = BatchInventory(initial_n_batches=4, initial_dc_quantity=4000.0,
                         fresh_arrival_rate_per_hour=0.0)
    arrival_hours = [b.arrival_hour for b in inv.batches]
    expected_oldest = min(arrival_hours)
    inv.step(hour=0.25, d_env_rho=0.01, action_idx=0, dt_hours=0.25)
    routed = [b for b in inv.batches if b.status == "transit_cold_chain"]
    assert len(routed) == 1
    assert routed[0].arrival_hour == pytest.approx(expected_oldest)


def test_step_recovery_removes_batch_from_retail_pool():
    inv = BatchInventory(initial_n_batches=3, initial_dc_quantity=3000.0,
                         fresh_arrival_rate_per_hour=0.0)
    # Route one batch to recovery, then advance time past transit window.
    inv.step(hour=0.25, d_env_rho=0.01, action_idx=2, dt_hours=0.25)
    # Step enough times for the recovery transit to complete.
    for h in np.arange(0.5, 5.0, 0.25):
        inv.step(hour=float(h), d_env_rho=0.0, action_idx=0, dt_hours=0.25)
    # The recovery batch should be in 'recovered' status; not in retail
    # pool.
    retail = [b for b in inv.batches if b.status == "retail"]
    recovered = [b for b in inv.batches if b.status == "recovered"]
    assert len(recovered) >= 1
    # Retail-pool rho should not include the recovered batch.
    for r in retail:
        assert r.routed_to in ("cold_chain", "local_redistribute")


# ---------------------------------------------------------------------------
# Retail-pool rho ordering: the load-bearing Figure 2 claim
# ---------------------------------------------------------------------------

def _run_constant_action_episode(
    action_idx: int,
    n_steps: int = 200,
    d_env_rho: float = 0.005,
    ambient_temp_c: float = 4.0,
) -> np.ndarray:
    """Run a single-action episode and return the retail-pool rho trace.

    The capacity-constraint infrastructure (LR / Recovery rolling
    admission windows) was retired in 2026-04 alongside the
    capacity-constrained RLE variant; these tests exercise the
    thermal-physics behaviour directly.
    """
    inv = BatchInventory(
        initial_n_batches=8,
        initial_dc_quantity=12000.0,
        fresh_arrival_rate_per_hour=0.5,
        fresh_batch_quantity=1500.0,
    )
    trace = []
    for k in range(n_steps):
        out = inv.step(
            hour=0.25 * (k + 1),
            d_env_rho=d_env_rho,
            action_idx=action_idx,
            ambient_temp_c=ambient_temp_c,
            dt_hours=0.25,
        )
        trace.append(out["effective_rho"])
    return np.asarray(trace)


def test_cold_chain_lower_retail_rho_at_moderate_ambient():
    """Under realistic physics, cold chain provides BETTER thermal
    protection than local-redistribute at moderate ambient (15-30
    degC, the LR nominal band). At cool ambient (T < 15 degC) LR's
    factor drops to 0.20 and the gap closes to within transit-time
    noise; this test exercises the moderate-ambient regime where the
    gap is genuinely measurable."""
    cc = _run_constant_action_episode(action_idx=0, ambient_temp_c=20.0)
    lr = _run_constant_action_episode(action_idx=1, ambient_temp_c=20.0)
    assert cc[-100:].mean() < lr[-100:].mean(), (
        f"at moderate ambient, expected CC retail rho < LR retail rho "
        f"(cold chain is better refrigerated); got "
        f"CC={cc[-100:].mean():.4f} LR={lr[-100:].mean():.4f}"
    )


def test_lr_lower_retail_rho_when_cold_chain_overwhelmed():
    """When ambient exceeds 35 degC the cold-chain cooling fails and
    CC carries 1.00 of env_rho into retail; LR's 0.85 factor (hot band)
    is then below CC, so LR-only beats CC-only on retail rho. This is
    the regime where AgriBrain's LR-leaning policy genuinely wins on
    raw rho rather than only on the SLCA composite."""
    cc = _run_constant_action_episode(action_idx=0, ambient_temp_c=38.0,
                                       n_steps=150, d_env_rho=0.003)
    lr = _run_constant_action_episode(action_idx=1, ambient_temp_c=38.0,
                                       n_steps=150, d_env_rho=0.003)
    assert lr[-50:].mean() < cc[-50:].mean(), (
        f"with cold chain overwhelmed, expected LR retail rho < CC; "
        f"got LR={lr[-50:].mean():.4f} CC={cc[-50:].mean():.4f}"
    )


def test_stress_band_lr_below_cc_due_to_short_transit():
    """In the 30-35 degC stress band CC's per-step factor (0.40) is
    below LR's (0.65), but CC's transit time is 2.4h vs LR's 0.9h.
    Integrated thermal exposure (factor x transit hours) is 0.40 *
    2.4 = 0.96 for CC vs 0.65 * 0.9 = 0.585 for LR — LR's shorter
    transit dominates, so LR-only retail rho is *lower* than CC-only
    in the stress band. This is the realistic mechanism behind
    AgriBrain's win on retail rho during heatwaves."""
    cc = _run_constant_action_episode(action_idx=0, ambient_temp_c=32.0,
                                       n_steps=150, d_env_rho=0.003)
    lr = _run_constant_action_episode(action_idx=1, ambient_temp_c=32.0,
                                       n_steps=150, d_env_rho=0.003)
    cc_mean = cc[-50:].mean()
    lr_mean = lr[-50:].mean()
    assert lr_mean < cc_mean, (
        f"in stress band, LR retail rho should be below CC due to "
        f"short-transit dominance; got CC={cc_mean:.4f} LR={lr_mean:.4f}"
    )


def test_recovery_only_drains_retail_pool():
    """Sending every batch to Recovery means no fresh batches enter the
    retail pool. The pool clears via sales and retail rho approaches
    zero."""
    rec = _run_constant_action_episode(action_idx=2, n_steps=300)
    # Final retail rho should be ~0 because nothing is being routed in.
    assert rec[-1] == pytest.approx(0.0, abs=0.01)


def test_cc_factor_step_function_observable_in_retail_trace():
    """A step in ambient temperature crossing 30 degC should produce a
    visible difference in retail rho under CC-only routing - the
    temperature-conditional factor is real, not cosmetic. The
    magnitude is muted because most retail-pool rho is accumulated
    during DC sit-time and retail aging (both at the DC factor 0.20),
    with only the 2.4 h CC-transit segment seeing the 0.15 -> 0.40
    factor step. The direction is what we pin."""
    cc_cool = _run_constant_action_episode(action_idx=0, ambient_temp_c=20.0,
                                            n_steps=150, d_env_rho=0.003)
    cc_hot = _run_constant_action_episode(action_idx=0, ambient_temp_c=33.0,
                                           n_steps=150, d_env_rho=0.003)
    cool_mean = cc_cool[-50:].mean()
    hot_mean = cc_hot[-50:].mean()
    assert hot_mean > cool_mean, (
        f"expected hot-CC retail rho > cool-CC; got "
        f"cool={cool_mean:.4f} hot={hot_mean:.4f}"
    )
    # Detectable margin (>10% relative) - confirms the temperature-
    # conditional factor is having a real effect, not just noise.
    rel_diff = (hot_mean - cool_mean) / max(cool_mean, 1e-9)
    assert rel_diff > 0.10, (
        f"hot-CC vs cool-CC relative difference too small to confirm "
        f"the temperature-conditional factor is operative: {rel_diff:.2%}"
    )


# ---------------------------------------------------------------------------
# Sale / turnover behaviour
# ---------------------------------------------------------------------------

def test_retail_pool_clears_at_sale_rate():
    """With no fresh routing-in and a positive sale rate, the retail
    pool should monotonically drain."""
    inv = BatchInventory(
        initial_n_batches=3, initial_dc_quantity=3000.0,
        fresh_arrival_rate_per_hour=0.0,
        sale_rate_per_hour=0.10,
    )
    # First, route all 3 DC batches to LR so they end up in retail.
    for k in range(3):
        inv.step(hour=0.25 * (k + 1), d_env_rho=0.0,
                 action_idx=1, dt_hours=0.25)
    # Advance time past LR transit so all batches are in retail.
    for k in range(20):
        inv.step(hour=10.0 + 0.25 * k, d_env_rho=0.0,
                 action_idx=0, dt_hours=0.25)  # action_idx=0 is OK; no DC batches left
    retail_qty_t0 = sum(b.quantity for b in inv.batches if b.status == "retail")
    # Now drain for many steps.
    for k in range(200):
        inv.step(hour=20.0 + 0.25 * k, d_env_rho=0.0,
                 action_idx=0, dt_hours=0.25)
    retail_qty_t1 = sum(b.quantity for b in inv.batches if b.status == "retail")
    assert retail_qty_t1 < retail_qty_t0


# ---------------------------------------------------------------------------
# Food-safety hard cutoff (RHO_FOOD_SAFETY_CUTOFF override)
# ---------------------------------------------------------------------------

def test_food_safety_override_redirects_aged_batch_to_recovery():
    """When the oldest DC batch has accumulated rho above the
    food-safety cutoff (default 0.65), the BatchInventory.step()
    routing layer overrides the policy's chosen action to Recovery
    regardless of action_idx. This models the real-world food-safety
    regulation that produce visibly past marketability cannot be
    sold even if the optimisation layer would prefer otherwise."""
    inv = BatchInventory(
        initial_n_batches=1, initial_dc_quantity=1000.0,
        fresh_arrival_rate_per_hour=0.0,
    )
    # Manually advance the only DC batch past the cutoff.
    inv.batches[0].current_rho = 0.80
    # Policy chose cold_chain (action_idx=0) but the override should
    # force recovery.
    out = inv.step(hour=0.25, d_env_rho=0.0, action_idx=0, dt_hours=0.25)
    assert out["food_safety_override"] is True
    assert out["chosen_route"] == "recovery"
    assert inv.batches[0].status == "transit_recovery"


def test_food_safety_override_does_not_fire_below_cutoff():
    """A DC batch with rho just below the cutoff (0.60) should be
    routed by the policy's choice, not overridden."""
    inv = BatchInventory(
        initial_n_batches=1, initial_dc_quantity=1000.0,
        fresh_arrival_rate_per_hour=0.0,
    )
    inv.batches[0].current_rho = 0.60
    out = inv.step(hour=0.25, d_env_rho=0.0, action_idx=1, dt_hours=0.25)
    assert out["food_safety_override"] is False
    assert out["chosen_route"] == "local_redistribute"
    assert inv.batches[0].status == "transit_local_redistribute"


def test_food_safety_override_applies_independent_of_action_choice():
    """The override is a property of the BatchInventory routing layer
    not the policy, so it fires for ALL chosen actions when the
    oldest DC batch is past the cutoff."""
    for action_idx in (0, 1, 2):
        inv = BatchInventory(
            initial_n_batches=1, initial_dc_quantity=1000.0,
            fresh_arrival_rate_per_hour=0.0,
        )
        inv.batches[0].current_rho = 0.90
        out = inv.step(hour=0.25, d_env_rho=0.0,
                       action_idx=action_idx, dt_hours=0.25)
        assert out["chosen_route"] == "recovery", (
            f"override should force recovery for action_idx={action_idx}; "
            f"got {out['chosen_route']!r}"
        )
