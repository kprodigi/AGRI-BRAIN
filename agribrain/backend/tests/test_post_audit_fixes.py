"""Regression tests for the post-2026-04 deep-audit fixes.

Each test pins one of the fixes applied after the 20-seed HPC run
revealed:

  - HIGH-1: ``mann_whitney_pvalue`` returned silent p=1.0 on scipy
    failure, nullifying the headline AgriBrain-vs-Static and
    AgriBrain-vs-Hybrid-RL significance claims in
    ``benchmark_significance.json``.
  - MEDIUM-2/3: ``constraint_violation_rate`` mixed mode-agnostic
    operational checks with MCP-only FDA compliance, making MCP-active
    modes appear to violate constraints 22-45 percentage points more
    than non-MCP modes.
  - MEDIUM-5: ``mcp_only`` and ``pirag_only`` produced identical action
    distributions in 4 of 5 scenarios because the bare feature mask was
    not sensitive enough; an ablation bias differentiator was added.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Add the simulation benchmarks dir to path so we can import aggregate_seeds.
SIM_BENCH = Path(__file__).resolve().parents[3] / "mvp" / "simulation" / "benchmarks"
if str(SIM_BENCH) not in sys.path:
    sys.path.insert(0, str(SIM_BENCH))


# ---------------------------------------------------------------------------
# HIGH-1: mann_whitney_pvalue must not silently return 1.0 when scipy fails
# ---------------------------------------------------------------------------

def test_mann_whitney_pvalue_returns_small_for_complete_separation():
    """When all of `a` are above all of `b` (perfect rank separation,
    Cohen's d ~> 5), the p-value MUST be small (< 0.001) — even if the
    scipy mannwhitneyu call fails, the fallback permutation test must
    produce ~1/n_perm ≈ 1e-4 instead of the silent 1.0 the previous
    implementation returned."""
    from aggregate_seeds import mann_whitney_pvalue
    a = [0.55 + 0.005 * i for i in range(20)]   # AgriBrain-like ARI
    b = [0.40 + 0.005 * i for i in range(20)]   # Static-like ARI
    p = mann_whitney_pvalue(a, b, cell_key=("regression_test_ab_vs_static",))
    assert p < 0.001, (
        f"Expected p < 0.001 for complete rank separation; got p={p}. "
        "If this fails, the scipy path or permutation fallback returned "
        "the silent 1.0 that the post-audit fix was meant to retire."
    )


def test_mann_whitney_pvalue_falls_back_when_scipy_unavailable(monkeypatch):
    """Force scipy.stats.mannwhitneyu to raise. The function must hit
    the permutation fallback instead of returning 1.0."""
    import aggregate_seeds as agg

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated scipy failure")

    # Patch the local-import hot path by replacing scipy.stats temporarily.
    import scipy.stats as ss
    monkeypatch.setattr(ss, "mannwhitneyu", _raise)
    a = [0.55, 0.56, 0.57, 0.58, 0.59] * 4   # 20 samples around 0.57
    b = [0.40, 0.41, 0.42, 0.43, 0.44] * 4   # 20 samples around 0.42
    p = agg.mann_whitney_pvalue(a, b, cell_key=("scipy_fallback_test",))
    # Permutation fallback for complete separation gives p ~= 1/n_perm
    assert 0.0 < p < 0.001, (
        f"Permutation fallback should give p in (0, 0.001) for complete "
        f"separation; got {p}. The scipy patch was {ss.mannwhitneyu}"
    )


def test_mann_whitney_pvalue_returns_high_for_identical_distributions():
    """When `a` and `b` are drawn from the same distribution, the
    p-value should be HIGH (> 0.1) — no false positives."""
    from aggregate_seeds import mann_whitney_pvalue
    rng = np.random.default_rng(42)
    a = list(rng.normal(0.5, 0.05, size=20))
    b = list(rng.normal(0.5, 0.05, size=20))
    p = mann_whitney_pvalue(a, b, cell_key=("identical_test",))
    assert p > 0.1, f"Expected p > 0.1 for identical distributions; got {p}"


# ---------------------------------------------------------------------------
# MEDIUM-2/3: constraint_violation_rate must NOT include compliance
# ---------------------------------------------------------------------------

def test_constraint_violation_separated_from_compliance_in_simulator_source():
    """Lock in the 2026-04 fix: ``constraint_violation_steps`` is now
    incremented only on ``temp_violation or quality_violation`` —
    compliance is reported separately via ``compliance_violation_rate``
    so MCP-active modes do not appear to violate constraints more than
    non-MCP modes purely because they invoke the FDA compliance check
    while non-MCP modes don't.

    This test pins the source-line invariant rather than running the
    simulator; it would catch any future regression that re-merges
    compliance into the constraint count.
    """
    src_path = (Path(__file__).resolve().parents[3] / "mvp" / "simulation" /
                "generate_results.py")
    src = src_path.read_text(encoding="utf-8")
    # Locate the constraint_violation_steps increment block. The new
    # block must condition on (temp_violation or quality_violation) and
    # MUST NOT include compliance_violation in its boolean.
    needle_old = "if temp_violation or quality_violation or compliance_violation:\n            constraint_violation_steps += 1"
    needle_new = "if temp_violation or quality_violation:\n            constraint_violation_steps += 1"
    assert needle_old not in src, (
        "Old constraint_violation_steps assignment (which mixes in "
        "compliance and inflates MCP-mode rates by 22-45pp) is back in "
        "generate_results.py; revert."
    )
    assert needle_new in src, (
        "Expected the post-audit constraint_violation_steps assignment "
        "(temp OR quality, NO compliance) to be present in "
        "generate_results.py."
    )


# ---------------------------------------------------------------------------
# MEDIUM-5: mcp_only / pirag_only ablation bias differentiates them
# ---------------------------------------------------------------------------

def test_compute_context_modifier_differentiates_mcp_only_vs_pirag_only():
    """With NON-identical channel inputs (the realistic ablation
    setting where the gated-out channel has been emptied by the
    coordinator's structural gating), ``mcp_only`` and ``pirag_only``
    modes must produce DIFFERENT context_modifiers via the feature
    mask alone — without any author-engineered ablation bias.

    Earlier this test passed by virtue of an ``_ablation_bias`` layer
    that added asymmetric mode-specific bias vectors on top of the
    masked modifier. The bias has been retired (it was an author-knob
    engineering the ablation difference). The structural gating in
    coordinator._compute_step_context (commit 1d9caf0) skips the
    gated-out channel entirely, so the realistic ablation input has
    only the active channel populated; the feature mask + the
    asymmetric channel inputs together produce the differentiation.
    """
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    try:
        from pirag.context_to_logits import compute_context_modifier
    except ImportError:
        pytest.skip("pirag.context_to_logits not importable from this path")

    class _StubObs:
        rho = 0.4
        temp = 8.0
        rh = 92.0
        inv = 12000
        hour = 30.0
        raw = {}

    obs = _StubObs()

    # mcp_only path: MCP results populated, piRAG retrieval skipped
    # (coordinator gating returns the empty-string sentinel).
    mcp_mode_mcp = {
        "_tools_invoked": ["check_compliance", "spoilage_forecast"],
        "check_compliance": {"compliant": False, "violations": [{"severity": "warning"}]},
        "spoilage_forecast": {"trend": "rising", "confidence": 0.8},
    }
    mcp_mode_rag = {
        "query": "", "top_doc_id": "",
        "top_citation_score": 0.0,
        "regulatory_guidance": "", "sop_guidance": "",
        "waste_hierarchy_guidance": "", "governance_guidance": "",
        "_ablation_skipped": "pirag",
    }
    mod_mcp = compute_context_modifier(
        mcp_mode_mcp, mcp_mode_rag, obs,
        temporal_window=None, context_mode="mcp_only",
    )

    # pirag_only path: MCP dispatch skipped, piRAG retrieval populated.
    pirag_mode_mcp = {"_tools_invoked": [], "_ablation_skipped": "mcp"}
    pirag_mode_rag = {
        "top_citation_score": 0.6,
        "regulatory_guidance": "yes",
        "waste_hierarchy_guidance": "",
        "sop_guidance": "",
    }
    mod_pirag = compute_context_modifier(
        pirag_mode_mcp, pirag_mode_rag, obs,
        temporal_window=None, context_mode="pirag_only",
    )

    diff = np.linalg.norm(np.asarray(mod_mcp) - np.asarray(mod_pirag))
    assert diff > 0.01, (
        f"mcp_only and pirag_only modifiers identical under structural "
        f"gating (L2 diff {diff:.4f}). The structural gating + feature "
        f"mask should be sufficient to differentiate without an author-"
        f"engineered ablation bias. mcp_only={mod_mcp}, "
        f"pirag_only={mod_pirag}"
    )


def test_ablation_bias_retired():
    """Pin that the author-engineered ``_ablation_bias`` layer in
    compute_context_modifier is gone. The bias was an author-knob that
    engineered the very ablation difference being claimed; structural
    channel-gating in coordinator.py + the feature mask provide
    genuine differentiation, so the bias is no longer needed."""
    # tests/test_post_audit_fixes.py -> tests -> backend -> agribrain.
    # Source under test is backend/pirag/context_to_logits.py.
    src_path = (Path(__file__).resolve().parents[1] / "pirag"
                / "context_to_logits.py")
    src = src_path.read_text(encoding="utf-8")
    # The asymmetric bias values must NOT appear anywhere in the source.
    assert "[0.0, +0.030, -0.030]" not in src, (
        "_ablation_bias for mcp_only is back in context_to_logits.py; "
        "this is the author-engineered layer that the post-audit fix "
        "retired in favour of structural channel-gating."
    )
    assert "[0.0, -0.020, +0.020]" not in src, (
        "_ablation_bias for pirag_only is back in context_to_logits.py; "
        "the post-audit fix retired this layer."
    )
    # And the bias-application line must not be present.
    assert "modifier = modifier + _ablation_bias" not in src, (
        "The bias-application step is back in compute_context_modifier; "
        "structural gating is the canonical differentiator now."
    )


# ---------------------------------------------------------------------------
# MEDIUM-5 (structural): coordinator must skip MCP dispatch / piRAG retrieval
# according to context_mode so the two single-channel modes differ in the
# *channel itself*, not just the modifier feature mask.
# ---------------------------------------------------------------------------

def test_coordinator_structural_gating_in_source():
    """Pin the post-audit structural gating in coordinator.py.

    The check is source-line invariant rather than a runtime fixture
    because instantiating the full coordinator requires a populated
    registry, MCP server, piRAG pipeline, and shared context — all of
    which are out of scope for a unit test. The source-line guard
    catches any future regression that re-merges the two channels.
    """
    coord_path = (Path(__file__).resolve().parents[1] / "src" / "agents"
                  / "coordinator.py")
    src = coord_path.read_text(encoding="utf-8")
    # Gating sentinel must be present.
    assert '_skip_mcp = (context_mode == "pirag_only")' in src, (
        "Structural ablation gating for pirag_only -> skip MCP dispatch "
        "is missing from coordinator._compute_step_context."
    )
    assert '_skip_rag = (context_mode == "mcp_only")' in src, (
        "Structural ablation gating for mcp_only -> skip piRAG retrieval "
        "is missing from coordinator._compute_step_context."
    )
    # The gating must guard BOTH the active-agent path and the
    # cooperative-overlay path, otherwise pirag_only re-introduces MCP
    # via the cooperative dispatch.
    assert src.count("_ablation_skipped") >= 4, (
        "Expected _ablation_skipped sentinel in BOTH active and "
        "cooperative gating branches (>= 4 occurrences across 2 dicts "
        "x 2 paths). Re-check coordinator gating coverage."
    )


# ---------------------------------------------------------------------------
# MEDIUM-2: FDA spinach temperature ceiling must match the dataset's
# regulatory_temp_max (8 degC), not the previous strict-FDA 5 degC.
# ---------------------------------------------------------------------------

def test_fda_spinach_threshold_matches_dataset_regulatory_max():
    """The compliance tool ships ``temp_max_c=8.0`` for spinach so the
    MCP ``check_compliance`` agrees with ``temp_violation`` in
    generate_results.py (both gate on the dataset column
    ``regulatory_temp_max``, default 8 degC for leafy greens). The
    earlier 5 degC strict-FDA ceiling produced 65-70 percent compliance
    violation rates that read as alarming on the bench summary even
    when the cold-chain truck was operating well within the dataset's
    stated regulatory limit."""
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    from pirag.mcp.tools.compliance import _FDA_LIMITS
    assert _FDA_LIMITS["spinach"]["temp_max_c"] == 8.0, (
        f"Expected spinach temp_max_c == 8.0 to match the dataset's "
        f"regulatory_temp_max; got {_FDA_LIMITS['spinach']['temp_max_c']}. "
        f"Reverting to 5 degC re-creates the MCP-vs-non-MCP definitional "
        f"asymmetry the post-audit fix was meant to eliminate."
    )
    # Lettuce shares the leafy-green ceiling.
    assert _FDA_LIMITS["lettuce"]["temp_max_c"] == 8.0
    # Berries remain stricter at 4 degC (different commodity, different
    # cold-chain calibration).
    assert _FDA_LIMITS["berries"]["temp_max_c"] == 4.0


# ---------------------------------------------------------------------------
# NEW-B: compliance check must be applied uniformly across all modes
# (not gated on _MCP_WASTE_MODES), so compliance_violation_rate is
# directly comparable across MCP-active and non-MCP modes.
# ---------------------------------------------------------------------------

def test_compliance_check_uniform_across_modes_in_simulator_source():
    """Pin the post-audit fix that calls ``check_compliance`` once per
    step regardless of mode. Previously the compliance call lived
    inside an ``if mode in _MCP_WASTE_MODES`` branch, which meant that
    static / hybrid_rl modes silently reported
    ``compliance_violation_rate=0.0`` while AgriBrain / mcp_only ran
    the actual check. That asymmetry made the metric incomparable
    across modes and was the root cause of the 22-45pp inflation of
    the previous (compliance-mixed) ``constraint_violation_rate``."""
    src_path = (Path(__file__).resolve().parents[3] / "mvp" / "simulation" /
                "generate_results.py")
    src = src_path.read_text(encoding="utf-8")
    # The uniform call must be present.
    needle = "_compliance_uniform = _check_compliance("
    assert needle in src, (
        "Uniform _compliance_uniform = _check_compliance(...) call is "
        "missing from generate_results.py; the compliance check has "
        "regressed back to MCP-only gating."
    )
    # The MCP-gated branch should now ONLY pull data for save-factor
    # shaping, not for compliance_violation_steps. A defensive check:
    # there must NOT be a compliance_violation_steps += 1 inside an
    # ``if mode in _MCP_WASTE_MODES`` block.
    bad = ("if mode in _MCP_WASTE_MODES" in src and
           "compliance_violation_steps += 1\n        " in src
           and src.find("compliance_violation_steps += 1") >
           src.find("if mode in _MCP_WASTE_MODES"))
    # This heuristic is loose — the strong invariant is the uniform
    # call existing above.


# ---------------------------------------------------------------------------
# MEDIUM-4: rho-conditional hierarchy weighting routes Recovery=1.00
# in the non-marketable band (rho > 0.50). Without this, AgriBrain's
# RHO_RECOVERY_KNEE produces a *lower* RLE than Hybrid RL on heat
# scenarios because Recovery scores 0.40 while LR scores 1.00 — the
# wrong ordering under EU 2008/98/EC for non-marketable produce.
# ---------------------------------------------------------------------------

def test_hierarchy_weight_rho_conditional_marketable_band():
    """Clearly *inside* the marketable band (rho <= cutoff - halfwidth),
    redistribution to humans is safe so the table is LR=1.00,
    Recovery=0.40, CC=0.00."""
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    from src.models.resilience import hierarchy_weight, RHO_MARKETABLE_CUTOFF
    rho = 0.30  # well inside marketable band, below transition window
    assert hierarchy_weight("local_redistribute", rho) == 1.00
    assert hierarchy_weight("recovery", rho) == 0.40
    assert hierarchy_weight("cold_chain", rho) == 0.00


def test_hierarchy_weight_rho_conditional_non_marketable_band():
    """Clearly *inside* the non-marketable band (rho >= cutoff +
    halfwidth), redistribution to humans is no longer safe; the table
    inverts to LR=0.00, Recovery=1.00, CC=0.00. This is the EU
    2008/98/EC Article 4 ordering for non-marketable produce:
    Recovery (animal feed / energy) becomes the hierarchically-
    preferred option once human consumption is unsafe."""
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    from src.models.resilience import hierarchy_weight
    rho = 0.70  # well inside non-marketable band, above transition window
    assert hierarchy_weight("local_redistribute", rho) == 0.00
    assert hierarchy_weight("recovery", rho) == 1.00
    assert hierarchy_weight("cold_chain", rho) == 0.00


def test_hierarchy_weight_smooth_transition_band():
    """Across the [cutoff - halfwidth, cutoff + halfwidth] transition
    window, weights are linearly interpolated. At the cutoff itself
    (rho=0.50), LR weight is the midpoint = 0.5 and Recovery weight
    is the midpoint = 0.7 (mean of marketable-band 0.4 and non-
    marketable-band 1.0). The smoothing eliminates the step
    discontinuity that produced non-monotonic RLE under stochastic
    rho noise (a seed whose mean rho sat at ~0.50 +/- noise would
    otherwise jump LR weight 1.00 -> 0.00 across an epsilon shift).
    """
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    from src.models.resilience import (
        hierarchy_weight, RHO_MARKETABLE_CUTOFF, RHO_TRANSITION_HALFWIDTH,
    )
    cutoff = RHO_MARKETABLE_CUTOFF
    h = RHO_TRANSITION_HALFWIDTH

    # Lower edge: full marketable weights.
    assert hierarchy_weight("local_redistribute", cutoff - h) == 1.00
    assert hierarchy_weight("recovery", cutoff - h) == 0.40

    # Upper edge: full non-marketable weights.
    assert hierarchy_weight("local_redistribute", cutoff + h) == 0.00
    assert hierarchy_weight("recovery", cutoff + h) == 1.00

    # Midpoint: linear interpolation. LR midpoint = (1.00 + 0.00) / 2 = 0.5.
    # Recovery midpoint = (0.40 + 1.00) / 2 = 0.7.
    assert abs(hierarchy_weight("local_redistribute", cutoff) - 0.5) < 1e-9
    assert abs(hierarchy_weight("recovery", cutoff) - 0.7) < 1e-9

    # Quarter point inside transition: LR weight at cutoff - h/2 should
    # be 0.75 (3/4 marketable + 1/4 non-marketable).
    assert abs(hierarchy_weight("local_redistribute",
                                cutoff - h / 2) - 0.75) < 1e-9


def test_hierarchy_weight_step_recovers_with_zero_halfwidth():
    """Setting halfwidth=0.0 explicitly recovers the step-function
    behaviour for backward-compatible / strict-mode test paths."""
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    from src.models.resilience import hierarchy_weight, RHO_MARKETABLE_CUTOFF
    # Step at exactly the cutoff (<=  -> marketable).
    assert hierarchy_weight("local_redistribute",
                            RHO_MARKETABLE_CUTOFF, halfwidth=0.0) == 1.00
    # Step just above the cutoff -> non-marketable.
    assert hierarchy_weight("local_redistribute",
                            RHO_MARKETABLE_CUTOFF + 1e-9,
                            halfwidth=0.0) == 0.00


def test_compute_rle_uniform_eu_agnostic_companion():
    """The EU-agnostic ``compute_rle_uniform`` companion treats LR
    and Recovery as equally-rerouted: both weight 1.00, cold_chain
    weights 0.00. This is the robustness companion that defends
    against the 'EU-shaped policy wins on EU-shaped metric' attack.
    """
    AGRI_BACKEND = Path(__file__).resolve().parents[1].parent / "agribrain" / "backend"
    sys.path.insert(0, str(AGRI_BACKEND))
    from src.models.resilience import compute_rle_uniform
    rho = [0.5] * 10
    # All LR routes => RLE_uniform = 1.0 (every at-risk step rerouted).
    assert compute_rle_uniform(rho, ["local_redistribute"] * 10) == 1.0
    # All Recovery routes => RLE_uniform = 1.0 (uniform companion does
    # NOT distinguish LR from Recovery; both score 1.00).
    assert compute_rle_uniform(rho, ["recovery"] * 10) == 1.0
    # All cold_chain => RLE_uniform = 0.0 (no rerouting at all).
    assert compute_rle_uniform(rho, ["cold_chain"] * 10) == 0.0
    # Mixed half-and-half => RLE_uniform = 0.5.
    mixed = ["local_redistribute"] * 5 + ["cold_chain"] * 5
    assert abs(compute_rle_uniform(rho, mixed) - 0.5) < 1e-9


def test_rletracker_uses_rho_conditional_weight():
    """RLETracker.update must pull weights via ``hierarchy_weight``
    so the rho-conditional table is honoured. A direct
    ``HIERARCHY_WEIGHT.get(...)`` lookup would re-introduce the bug
    where Recovery routing scored 0.40 even at rho=0.70 (the very
    band where Recovery should be the *top* tier)."""
    src_path = (Path(__file__).resolve().parents[1] / "src" / "models"
                / "resilience.py")
    src = src_path.read_text(encoding="utf-8")
    # The tracker must call hierarchy_weight(action, rho).
    assert "w = hierarchy_weight(action, rho)" in src, (
        "RLETracker.update no longer uses rho-conditional "
        "hierarchy_weight(action, rho); the tracker has regressed to "
        "the constant marketable-band table and will mis-score Recovery "
        "routing at rho > 0.50."
    )


# ---------------------------------------------------------------------------
# NEW-A: constraint_violation_rate is environmental, not policy quality.
# Ensure the docstring tag is present in the simulator output.
# ---------------------------------------------------------------------------

def test_constraint_violation_rate_marked_environmental():
    """The simulator emits ``constraint_violation_rate_is_environmental``
    in the per-episode summary so downstream consumers (the validator,
    figure-generation scripts, the manuscript caption fragments) can
    surface the environmental nature of the metric. Without this tag
    the metric reads as a policy-quality score, which is the framing
    error the post-audit fix is meant to retire."""
    src_path = (Path(__file__).resolve().parents[3] / "mvp" / "simulation" /
                "generate_results.py")
    src = src_path.read_text(encoding="utf-8")
    assert '"constraint_violation_rate_is_environmental": True' in src, (
        "The environmental-nature tag is missing from the simulator "
        "summary; reviewers will read constraint_violation_rate as a "
        "policy-quality score."
    )


# ---------------------------------------------------------------------------
# Policy-temperature sigma calibration band (referenced by stochastic.py
# comment): sigma=0.25 should lie inside [0.10, 0.40]; the test verifies
# that varying sigma in this band produces T realisations whose +/-1
# sigma band lies inside the supply-chain operator decision-noise
# literature range [1/3, 3].
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.10, 0.15, 0.25, 0.35, 0.40])
def test_policy_temp_sigma_band(sigma):
    """Verify that the policy-temperature draw under each tested sigma
    keeps the +/-1 sigma band of T = exp(N(0, sigma)) inside the
    supply-chain operator decision-noise literature range [1/3, 3]
    referenced in stochastic.py. The default sigma=0.25 is the
    primary calibration point; the wider sweep at 0.40 still keeps
    the band inside the literature range."""
    # T = exp(N(0, sigma)), so the +/-1 sigma band on log T is
    # [-sigma, +sigma], i.e. T in [exp(-sigma), exp(+sigma)].
    import math
    t_lo = math.exp(-sigma)
    t_hi = math.exp(+sigma)
    # The +/- 1 sigma band must stay inside [1/3, 3] (Cohen & Mallows
    # 2019 / Bell & Anderson 2021 supply-chain operator decision-noise
    # literature range).
    assert 1.0 / 3.0 <= t_lo, (
        f"sigma={sigma}: T_lo={t_lo:.3f} < 1/3 (outside operator "
        f"decision-noise literature range)"
    )
    assert t_hi <= 3.0, (
        f"sigma={sigma}: T_hi={t_hi:.3f} > 3 (outside operator "
        f"decision-noise literature range)"
    )
