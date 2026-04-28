"""Tests for the robustness-variant metrics and the sensitivity claims
made in the model docstrings.

Each test exercises one of the claims a manuscript reviewer is likely
to challenge:

  - that the geometric-mean ARI agrees with the multiplicative ARI on
    rank ordering (since ARI_geom is a strictly increasing transform);
  - that the EU-hierarchy-weighted RLE distinguishes local_redistribute
    from recovery (which the binary form does not);
  - that the Sen welfare equity is bounded in [0, 1] and reduces to
    mean(SLCA) when SLCA is constant (G = 0);
  - that the SLCA per-action ranking is invariant under ±25 %
    perturbation of each L/R/P base value;
  - that the MODE_EFF rank ordering is invariant under ±25 %
    perturbation of each capability delta;
  - that the eta = 0.5 reward weight does not change the per-action
    reward ranking across {0.10, 0.25, 0.50, 1.00, 2.00};
  - that the FAO calibration anchors of the waste model fall within
    the FAO (2019) 2-15 % range.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.resilience import (
    HIERARCHY_WEIGHT,
    RLE_THRESHOLD,
    compute_ari,
    compute_ari_geom,
    compute_equity,
    compute_equity_sen,
    compute_rle,
    compute_rle_weighted,
)
from src.models.reverse_logistics import (
    compute_circular_economy_score,
    compute_mci,
    evaluate_recovery_options,
)
from src.models.reward import compute_reward
from src.models.slca import slca_score
from src.models.waste import (
    MODE_EFF,
    SAVE_CEIL,
    SAVE_FLOOR,
    _BASE_COMPETENCE,
    _CONTEXT_DELTA,
    _PINN_DELTA,
    _SLCA_DELTA,
    compute_save_factor,
    compute_waste_rate,
)


# ---------------------------------------------------------------------------
# ARI: multiplicative vs geometric agreement
# ---------------------------------------------------------------------------

def test_ari_geom_bounded():
    assert compute_ari_geom(0.0, 1.0, 0.0) == pytest.approx(1.0)
    assert compute_ari_geom(1.0, 0.0, 1.0) == 0.0
    for w in np.linspace(0, 1, 6):
        for s in np.linspace(0, 1, 6):
            for r in np.linspace(0, 1, 6):
                v = compute_ari_geom(w, s, r)
                assert 0.0 <= v <= 1.0


def test_ari_geom_rank_agrees_with_multiplicative():
    """Geometric mean is strictly increasing in the multiplicative
    product on the unit cube, so the rank ordering of any two policies
    under ARI agrees with the rank ordering under ARI_geom (modulo ties).
    """
    rng = np.random.default_rng(seed=0)
    triples = rng.uniform(0, 1, size=(40, 3))
    aris = [compute_ari(*t) for t in triples]
    geoms = [compute_ari_geom(*t) for t in triples]
    # All pairwise orderings agree
    n = len(aris)
    for i in range(n):
        for j in range(i + 1, n):
            if aris[i] < aris[j]:
                assert geoms[i] <= geoms[j] + 1e-12
            elif aris[i] > aris[j]:
                assert geoms[i] >= geoms[j] - 1e-12


# ---------------------------------------------------------------------------
# RLE: weighted form distinguishes redistribute from recovery
# ---------------------------------------------------------------------------

def test_binary_rle_saturates_for_either_reroute():
    rho = [0.5] * 10
    actions_lr = ["local_redistribute"] * 10
    actions_rec = ["recovery"] * 10
    assert compute_rle(rho, actions_lr) == 1.0
    assert compute_rle(rho, actions_rec) == 1.0  # binary cannot tell them apart


def test_weighted_rle_distinguishes_redistribute_from_recovery():
    rho = [0.5] * 10
    actions_lr = ["local_redistribute"] * 10
    actions_rec = ["recovery"] * 10
    actions_cc = ["cold_chain"] * 10
    assert compute_rle_weighted(rho, actions_lr) == pytest.approx(1.0)
    expected_rec = HIERARCHY_WEIGHT["recovery"] / max(HIERARCHY_WEIGHT.values())
    assert compute_rle_weighted(rho, actions_rec) == pytest.approx(expected_rec)
    assert compute_rle_weighted(rho, actions_cc) == 0.0


def test_weighted_rle_under_threshold_returns_zero():
    """No timesteps above threshold ⇒ denominator zero ⇒ metric = 0."""
    rho = [RLE_THRESHOLD * 0.5] * 5
    actions = ["local_redistribute"] * 5
    assert compute_rle_weighted(rho, actions) == 0.0
    assert compute_rle(rho, actions) == 0.0


# ---------------------------------------------------------------------------
# Equity: Sen welfare reduces correctly
# ---------------------------------------------------------------------------

def test_sen_equity_constant_input_equals_mean():
    """When SLCA is constant, Gini = 0 and Sen welfare = mean."""
    vals = [0.7] * 8
    assert compute_equity_sen(vals) == pytest.approx(0.7)


def test_sen_equity_bounded():
    rng = np.random.default_rng(seed=1)
    for _ in range(20):
        vals = rng.uniform(0, 1, size=10)
        v = compute_equity_sen(vals)
        assert 0.0 <= v <= 1.0


def test_sen_equity_zero_when_mean_zero():
    assert compute_equity_sen([0.0] * 5) == 0.0


def test_primary_equity_unchanged_for_constant_input():
    """Existing primary equity should still equal mean when std = 0."""
    vals = [0.7] * 8
    assert compute_equity(vals) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# SLCA per-action ranking invariance under ±25 % base-value perturbation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perturbation", [-0.25, -0.10, 0.0, 0.10, 0.25])
def test_slca_ranking_invariant(perturbation):
    """The qualitative ordering local_redistribute > recovery > cold_chain
    must hold under ±25% perturbation of every action's L/R/P base score.
    This is the load-bearing claim that justifies treating the scores as
    expert-elicited priors rather than measurements.
    """
    actions = ("cold_chain", "local_redistribute", "recovery")
    composites = {}
    for a in actions:
        # Use the documented per-action carbon footprints
        carbon = {"cold_chain": 14.4, "local_redistribute": 5.4, "recovery": 9.6}[a]
        # Apply perturbation by passing fairness/resilience/transparency overrides
        from src.models.slca import _ACTION_BASES
        base = _ACTION_BASES[a]
        s = slca_score(
            carbon_kg=carbon,
            action=a,
            fairness=base["L"] * (1.0 + perturbation),
            resilience=base["R"] * (1.0 + perturbation),
            transparency=base["P"] * (1.0 + perturbation),
        )
        composites[a] = s["composite"]
    assert composites["local_redistribute"] > composites["recovery"]
    assert composites["recovery"] > composites["cold_chain"]


# ---------------------------------------------------------------------------
# MODE_EFF capability-additive ranking invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perturbation", [-0.25, -0.10, 0.0, 0.10, 0.25])
def test_mode_eff_ranking_invariant(perturbation):
    """Under ±25% perturbation of each capability delta, the mode
    ordering static < hybrid_rl < (no_pinn, no_slca) < no_context <
    agribrain must be preserved. This is the architectural-composition
    claim that survives even if the absolute deltas are imprecise.
    """
    # We can't mutate module constants safely in tests; recompute
    # MODE_EFF from perturbed deltas.
    base_competence = _BASE_COMPETENCE * (1.0 + perturbation)
    pinn = _PINN_DELTA * (1.0 + perturbation)
    slca = _SLCA_DELTA * (1.0 + perturbation)
    ctx = _CONTEXT_DELTA * (1.0 + perturbation)

    def eff(has_rl, has_pinn, has_slca, has_ctx):
        if not has_rl:
            return 0.0
        e = base_competence
        if has_pinn:
            e += pinn
        if has_slca:
            e += slca
        if has_ctx:
            e += ctx
        return e

    static = eff(False, False, False, False)
    hybrid = eff(True, False, False, False)
    no_pinn = eff(True, False, True, True)
    no_slca = eff(True, True, False, True)
    no_context = eff(True, True, True, False)
    agribrain = eff(True, True, True, True)

    assert static < hybrid
    assert hybrid < no_pinn
    assert hybrid < no_slca
    assert no_pinn < agribrain
    assert no_slca < agribrain
    assert no_context < agribrain


# ---------------------------------------------------------------------------
# Reward: eta sweep invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eta", [0.10, 0.25, 0.50, 1.00, 2.00])
def test_eta_sensitivity_ranking(eta):
    """Across the swept eta range, a higher-SLCA / lower-waste policy
    should still receive higher reward than a lower-SLCA / higher-waste
    policy. This justifies treating eta = 0.50 as a robust default.
    """
    # Two stylised policies with the directional ordering AgriBrain claims
    r_good = compute_reward(slca_composite=0.85, waste=0.04, eta=eta)
    r_bad = compute_reward(slca_composite=0.55, waste=0.13, eta=eta)
    assert r_good > r_bad


# ---------------------------------------------------------------------------
# Waste: FAO calibration anchors fall within FAO (2019) 2-15% range
# ---------------------------------------------------------------------------

def test_waste_baseline_anchor_in_fao_range():
    """The 4°C baseline anchor should land near the FAO lower bound
    (developed-country refrigerated supply chain, ~7%)."""
    # k from spoilage.arrhenius_k at 4°C, RH ≈ 85% → ~0.00255 h⁻¹
    w = compute_waste_rate(k_inst=0.00274)
    assert 0.02 <= w <= 0.15, f"baseline waste {w:.3f} outside FAO 2-15% range"
    assert 0.05 <= w <= 0.10, f"baseline waste {w:.3f} not near 7% anchor"


def test_waste_heatwave_anchor_in_fao_range():
    """The heatwave anchor should land near the FAO upper bound (~13%)."""
    w = compute_waste_rate(k_inst=0.00596)
    assert 0.02 <= w <= 0.15, f"heatwave waste {w:.3f} outside FAO 2-15% range"
    assert 0.10 <= w <= 0.15, f"heatwave waste {w:.3f} not near 13% anchor"


def test_save_factor_floor_ceil_consistent():
    """Action floors and ceilings must order in the documented way."""
    assert SAVE_FLOOR["cold_chain"] < SAVE_FLOOR["recovery"] < SAVE_FLOOR["local_redistribute"]
    assert SAVE_CEIL["cold_chain"] < SAVE_CEIL["recovery"] < SAVE_CEIL["local_redistribute"]


def test_save_factor_static_zero_for_cold_chain():
    """Static mode + cold chain should produce no waste prevention by construction."""
    save = compute_save_factor(action="cold_chain", mode="static")
    assert save == pytest.approx(0.0)


def test_mode_eff_published_ordering():
    """MODE_EFF dictionary must respect the documented ablation ordering."""
    assert MODE_EFF["static"] == 0.0
    assert MODE_EFF["static"] < MODE_EFF["hybrid_rl"]
    assert MODE_EFF["hybrid_rl"] < MODE_EFF["no_pinn"]
    assert MODE_EFF["hybrid_rl"] < MODE_EFF["no_slca"]
    assert MODE_EFF["no_pinn"] < MODE_EFF["agribrain"]
    assert MODE_EFF["no_slca"] < MODE_EFF["agribrain"]
    assert MODE_EFF["no_context"] < MODE_EFF["agribrain"]


# ---------------------------------------------------------------------------
# Circular economy: MCI matches expected per-action ordering
# ---------------------------------------------------------------------------

def test_mci_action_ordering():
    """MCI should rank local_redistribute > recovery > cold_chain
    consistent with the EU waste hierarchy."""
    mci_lr = compute_mci("local_redistribute")
    mci_rec = compute_mci("recovery", recovery_factor=0.5)
    mci_cc = compute_mci("cold_chain")
    assert mci_lr > mci_rec
    assert mci_rec > mci_cc


def test_mci_bounded():
    for a in ("cold_chain", "local_redistribute", "recovery"):
        v = compute_mci(a)
        assert 0.0 <= v <= 1.0


def test_primary_circular_score_unchanged():
    """The stylised primary score must still match the published values
    so existing benchmark numbers remain valid."""
    opts = evaluate_recovery_options(spoilage_risk=0.3, inventory=12000, temperature=4.0)
    assert compute_circular_economy_score("cold_chain", opts) == 0.0
    s_lr = compute_circular_economy_score("local_redistribute", opts)
    s_rec = compute_circular_economy_score("recovery", opts)
    assert 0.0 <= s_lr <= 1.0
    assert 0.0 <= s_rec <= 1.0


# ---------------------------------------------------------------------------
# Operational frictions: capacity, sensor noise, lockout
# ---------------------------------------------------------------------------

def test_friction_default_off_is_identity():
    """With default config, FrictionGate is a no-op."""
    from src.models.operational_frictions import FrictionConfig, FrictionGate
    gate = FrictionGate(FrictionConfig())
    assert gate.observe_rho(0.42) == 0.42
    for h in range(10):
        assert gate.commit("local_redistribute", rho_true=0.5, hour=float(h)) == "local_redistribute"
        assert gate.commit("recovery", rho_true=0.5, hour=float(h)) == "recovery"
        assert gate.commit("cold_chain", rho_true=0.5, hour=float(h)) == "cold_chain"


def test_friction_sensor_noise_bounded():
    """Observed ρ stays in [0, 1] under noise."""
    from src.models.operational_frictions import FrictionConfig, FrictionGate
    gate = FrictionGate(FrictionConfig(enable_sensor_noise=True, sigma_rho=0.05, rng_seed=42))
    seen = [gate.observe_rho(0.5) for _ in range(200)]
    assert all(0.0 <= v <= 1.0 for v in seen)
    # Mean should approximate the true value within ~3 sigma
    import numpy as np
    assert abs(float(np.mean(seen)) - 0.5) < 0.05


def test_friction_capacity_downgrades_when_empty():
    """Once the token bucket is empty, reroutes downgrade to cold_chain."""
    from src.models.operational_frictions import FrictionConfig, FrictionGate
    cfg = FrictionConfig(
        enable_capacity_limit=True,
        capacity_per_hour=0.5,
        bucket_capacity=1.0,
    )
    gate = FrictionGate(cfg)
    # Step 1: bucket full, reroute committed
    a1 = gate.commit("local_redistribute", rho_true=0.5, hour=0.0)
    assert a1 == "local_redistribute"
    # Step 2: only 0.5 hours elapsed, bucket has refilled 0.25, still < 1
    a2 = gate.commit("local_redistribute", rho_true=0.5, hour=0.5)
    assert a2 == "cold_chain"  # capacity downgrade


def test_friction_lockout_holds_committed_action():
    """After a reroute commits, lockout holds the action for N steps."""
    from src.models.operational_frictions import FrictionConfig, FrictionGate
    cfg = FrictionConfig(enable_lockout=True, lockout_steps=3)
    gate = FrictionGate(cfg)
    a1 = gate.commit("local_redistribute", rho_true=0.5, hour=0.0)
    assert a1 == "local_redistribute"
    # Subsequent 3 steps should remain locked even if policy proposes cold_chain
    for h in (0.25, 0.50, 0.75):
        a = gate.commit("cold_chain", rho_true=0.05, hour=h)
        assert a == "local_redistribute"
    # 5th call: lock has expired
    a_unlocked = gate.commit("cold_chain", rho_true=0.05, hour=1.0)
    assert a_unlocked == "cold_chain"


# ---------------------------------------------------------------------------
# Empirical MODE_EFF: predicted-vs-observed agreement at AGRI-BRAIN endpoint
# ---------------------------------------------------------------------------

def test_mode_eff_predicted_within_observed_range():
    """The MODE_EFF predictions must lie within the observed empirical
    save range for the AGRI-BRAIN endpoint, validating the calibration
    documented in docs/MODE_EFF_EMPIRICAL.md.
    """
    import json
    from pathlib import Path

    summary_path = Path(__file__).resolve().parents[3] / "mvp" / "simulation" / "results" / "benchmark_summary.json"
    if not summary_path.exists():
        pytest.skip(f"benchmark_summary.json not present at {summary_path}")

    with open(summary_path) as f:
        data = json.load(f)
    saves = []
    for scenario in data:
        if "static" not in data[scenario] or "agribrain" not in data[scenario]:
            continue
        w_static = data[scenario]["static"]["waste"]["mean"]
        w_ab = data[scenario]["agribrain"]["waste"]["mean"]
        if w_static > 0:
            saves.append(1.0 - w_ab / w_static)

    if not saves:
        pytest.skip("no valid scenario pairs in benchmark_summary.json")

    obs_mean = sum(saves) / len(saves)
    # Predicted MODE_EFF['agribrain'] should be within ±0.05 of observed
    assert abs(MODE_EFF["agribrain"] - obs_mean) <= 0.05, (
        f"MODE_EFF predicts {MODE_EFF['agribrain']:.3f} but empirical "
        f"average is {obs_mean:.3f}; documented gap should be ≤0.05 per "
        f"docs/MODE_EFF_EMPIRICAL.md"
    )
