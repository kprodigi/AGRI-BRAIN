"""Cyber-outage reroute-probability calibration drift guard (fast).

The slow-test ``test_cyber_outage_reports_bernoulli_policy`` runs the
full simulator to verify the prob_trace is the Bernoulli reroute
distribution. This unit test catches the same class of regression
without touching the simulator: it asserts that
``CYBER_REROUTE_PROB[mode]`` is consistent with the
capability-additive composition ``_cyber_reroute_prob_from_capabilities(*caps)``
for every mode in ``_CYBER_CAPABILITIES``, and that the cross-mode
ordering documented in commit b0f7d9a holds at the canonical deltas.

If a future edit changes a per-capability delta or a mode's
capability flags but forgets to update the dict comprehension that
populates ``CYBER_REROUTE_PROB``, this fails in <50 ms instead of
waiting for the slow simulator path.
"""
from __future__ import annotations

import pytest

from src.models.action_selection import (
    CYBER_REROUTE_PROB,
    _CYBER_BASE_RL_COMPETENCE,
    _CYBER_CAPABILITIES,
    _CYBER_CONTEXT_DELTA,
    _CYBER_MCP_ONLY_FRACTION,
    _CYBER_PINN_DELTA,
    _CYBER_PIRAG_ONLY_FRACTION,
    _CYBER_SLCA_DELTA,
    _cyber_reroute_prob_from_capabilities,
)


def test_dict_matches_capability_formula():
    """Every entry in CYBER_REROUTE_PROB must equal the formula output."""
    for mode, caps in _CYBER_CAPABILITIES.items():
        expected = _cyber_reroute_prob_from_capabilities(*caps)
        assert CYBER_REROUTE_PROB[mode] == pytest.approx(expected), (
            f"{mode}: dict has {CYBER_REROUTE_PROB[mode]!r}, "
            f"formula gives {expected!r}. The dict comprehension that "
            f"builds CYBER_REROUTE_PROB has drifted from "
            f"_cyber_reroute_prob_from_capabilities."
        )


def test_agribrain_at_canonical_deltas():
    """Pin the canonical agribrain value so a delta change is loud.

    Calibration: 0.55 (base) + 0.05 (PINN) + 0.03 (SLCA) + 0.10 (full
    context) = 0.73. Any change to the four canonical deltas without
    a corresponding test update fails here. A *deliberate* recalibration
    bumps both the deltas and this assertion in one commit.
    """
    assert _CYBER_BASE_RL_COMPETENCE == 0.55
    assert _CYBER_PINN_DELTA == 0.05
    assert _CYBER_SLCA_DELTA == 0.03
    assert _CYBER_CONTEXT_DELTA == 0.10
    assert _CYBER_MCP_ONLY_FRACTION == 0.70
    assert _CYBER_PIRAG_ONLY_FRACTION == 0.60
    assert CYBER_REROUTE_PROB["agribrain"] == pytest.approx(0.73)


def test_cross_mode_ordering_documented_in_b0f7d9a():
    """The canonical-deltas ordering documented in commit b0f7d9a:

        hybrid_rl < no_context < no_pinn < pirag_only
                  < mcp_only <= no_slca < agribrain

    (`no_slca` and `mcp_only` both compute to 0.70 because both add
    PINN+context but no SLCA/full-context-only respectively, so they
    tie at the canonical deltas; their relative ordering is exercised
    in test_cyber_reroute_ranking_invariant under perturbation.)
    """
    p = CYBER_REROUTE_PROB
    assert p["hybrid_rl"] < p["no_context"]
    assert p["no_context"] < p["no_pinn"]
    assert p["no_pinn"] < p["pirag_only"]
    assert p["pirag_only"] < p["mcp_only"] or p["pirag_only"] < p["no_slca"]
    assert p["mcp_only"] == p["no_slca"] == pytest.approx(0.70)
    assert p["no_slca"] < p["agribrain"]


def test_static_excluded():
    """static returns ColdChain before reaching cyber logic; no entry needed."""
    assert "static" not in CYBER_REROUTE_PROB


def test_all_probabilities_in_unit_interval():
    for mode, p in CYBER_REROUTE_PROB.items():
        assert 0.0 <= p <= 1.0, f"{mode}: p = {p} outside [0, 1]"
