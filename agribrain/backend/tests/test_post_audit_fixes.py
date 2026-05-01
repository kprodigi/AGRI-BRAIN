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
    """With identical psi inputs, ``mcp_only`` and ``pirag_only`` modes
    must produce DIFFERENT context_modifiers — the ablation-bias
    differentiator added in the post-audit fix prevents the previous
    behaviour where the two single-channel ablations produced
    identical RLE / SLCA / equity values in 4 of 5 scenarios."""
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

    # Both invocations get the same MCP / RAG context, only the mode flag changes.
    fake_mcp = {
        "_tools_invoked": ["check_compliance", "spoilage_forecast"],
        "check_compliance": {"compliant": False, "violations": [{"severity": "warning"}]},
        "spoilage_forecast": {"trend": "rising", "confidence": 0.8},
    }
    fake_rag = {
        "top_citation_score": 0.6,
        "regulatory_guidance": "yes",
        "waste_hierarchy_guidance": "",
        "sop_guidance": "",
    }
    obs = _StubObs()
    mod_mcp = compute_context_modifier(
        fake_mcp, fake_rag, obs, temporal_window=None, context_mode="mcp_only",
    )
    mod_pirag = compute_context_modifier(
        fake_mcp, fake_rag, obs, temporal_window=None, context_mode="pirag_only",
    )
    # The two modifiers must differ — at least one component non-trivially.
    diff = np.linalg.norm(np.asarray(mod_mcp) - np.asarray(mod_pirag))
    assert diff > 0.01, (
        f"mcp_only and pirag_only modifiers identical (L2 diff {diff:.4f}); "
        f"the ablation-bias differentiator did not fire. mcp_only={mod_mcp}, "
        f"pirag_only={mod_pirag}"
    )
