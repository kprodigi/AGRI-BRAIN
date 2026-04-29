"""Tests for the realistic match-quality RLE that replaces the
saturating binary RLE in fig3 panel (c).

The previous binary form ``routed / at_risk`` saturated at 1.0 for any
policy that always reroutes - which is why the original figure showed
AgriBrain RLE pinned at 1.0 across the whole at-risk window. The new
match-quality form scores tier-vs-severity match per timestep and
cannot saturate trivially.

These tests pin the operating-point claims:

  - cold_chain at any rho: match score 0.0
  - LR at marketable rho (< 0.30): match score 1.0
  - LR at non-marketable rho (>= 0.60): match score 0.0
  - Recovery at marketable rho (< 0.30): match score 0.40 (under-utilising)
  - Recovery at non-marketable rho (>= 0.60): match score 1.0
  - linear transitions across the 0.30-0.60 zone
  - always-LR policy cannot reach rle_realistic = 1.0 at high rho
  - always-Recovery policy cannot reach rle_realistic = 1.0 at low rho
  - properly-triaging policy (LR for low, Recovery for high) approaches 1.0
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.resilience import (
    RLE_MATCH_LR_FULL_BAND_END,
    RLE_MATCH_LR_ZERO_BAND_START,
    RLE_MATCH_RECOVERY_BASE,
    RLE_THRESHOLD,
    RLETracker,
    compute_rle,
    compute_rle_realistic,
    compute_rle_weighted,
    match_quality,
)


# ---------------------------------------------------------------------------
# match_quality — point checks
# ---------------------------------------------------------------------------

def test_cold_chain_match_zero_everywhere():
    for rho in (0.05, 0.20, 0.50, 0.80, 0.99):
        assert match_quality("cold_chain", rho) == 0.0


def test_lr_full_band_perfect_match():
    """Below 0.30, LR is the optimal triage tier."""
    for rho in (0.10, 0.15, 0.20, 0.29):
        assert match_quality("local_redistribute", rho) == pytest.approx(1.0)


def test_lr_zero_band_above_food_bank_rejection():
    """Above 0.60, food banks reject heavily-spoiled produce."""
    for rho in (0.60, 0.70, 0.85, 1.00):
        assert match_quality("local_redistribute", rho) == pytest.approx(0.0)


def test_lr_transition_zone_linear():
    """LR score drops linearly from 1.0 to 0.0 across [0.30, 0.60]."""
    mid = (RLE_MATCH_LR_FULL_BAND_END + RLE_MATCH_LR_ZERO_BAND_START) / 2
    assert match_quality("local_redistribute", mid) == pytest.approx(0.5)


def test_recovery_marketable_band_under_utilising():
    """Below 0.30, Recovery scores 0.40 (sending salvageable produce
    to compost is wasteful — EU hierarchy weight for recovery)."""
    for rho in (0.10, 0.20, 0.29):
        assert match_quality("recovery", rho) == pytest.approx(RLE_MATCH_RECOVERY_BASE)


def test_recovery_non_marketable_band_perfect_match():
    """Above 0.60, Recovery is the only valid tier."""
    for rho in (0.60, 0.75, 0.90, 1.00):
        assert match_quality("recovery", rho) == pytest.approx(1.0)


def test_recovery_transition_zone_linear():
    """Recovery score rises linearly from 0.40 to 1.0 across [0.30, 0.60]."""
    mid = (RLE_MATCH_LR_FULL_BAND_END + RLE_MATCH_LR_ZERO_BAND_START) / 2
    expected = RLE_MATCH_RECOVERY_BASE + (1.0 - RLE_MATCH_RECOVERY_BASE) * 0.5
    assert match_quality("recovery", mid) == pytest.approx(expected)


def test_lr_recovery_crossover_in_transition_zone():
    """LR and Recovery score equally at one point in [0.30, 0.60].
    With LR: 1 - (rho-0.30)/0.30 and Recovery: 0.40 + (rho-0.30)/0.30 * 0.60,
    they intersect at rho where 1.0 - x = 0.40 + 0.60*x  →  x = 0.375
    so rho_cross = 0.30 + 0.375 * 0.30 = 0.4125"""
    rho_cross = 0.4125
    lr = match_quality("local_redistribute", rho_cross)
    rec = match_quality("recovery", rho_cross)
    assert lr == pytest.approx(rec, abs=0.02)


# ---------------------------------------------------------------------------
# rle_realistic — saturation-prevention claims
# ---------------------------------------------------------------------------

def _episode_with_constant_action(rho_trace, action):
    actions = [action] * len(rho_trace)
    return rho_trace, actions


def test_always_lr_cannot_saturate_when_high_rho_steps_present():
    """An always-LR policy fails on the high-rho steps (food-bank
    rejection) so its rle_realistic must be strictly less than 1.0
    when the rho trace contains non-marketable timesteps."""
    rho = [0.20] * 50 + [0.80] * 50  # half marketable, half non-marketable
    rho, acts = _episode_with_constant_action(rho, "local_redistribute")
    rle = compute_rle_realistic(rho, acts)
    assert 0.0 < rle < 1.0, (
        f"always-LR should be capped well below 1.0 with non-marketable "
        f"steps; got {rle:.4f}"
    )
    # Specifically less than 0.7 because half the timesteps score zero.
    assert rle < 0.70


def test_always_recovery_cannot_saturate_at_low_rho():
    """An always-Recovery policy under-uses marketable produce (score 0.40)
    so its rle_realistic is bounded near 0.40 when all rho is marketable."""
    rho = [0.15] * 100  # all marketable
    rho, acts = _episode_with_constant_action(rho, "recovery")
    rle = compute_rle_realistic(rho, acts)
    assert rle == pytest.approx(RLE_MATCH_RECOVERY_BASE, abs=0.01)


def test_always_cold_chain_scores_zero():
    rho = [0.20] * 50 + [0.70] * 50
    rho, acts = _episode_with_constant_action(rho, "cold_chain")
    rle = compute_rle_realistic(rho, acts)
    assert rle == pytest.approx(0.0)


def test_perfect_triage_policy_approaches_one():
    """A policy that picks LR for marketable rho and Recovery for
    non-marketable rho should score very close to 1.0."""
    rho_low = [0.15, 0.20, 0.25, 0.28]
    rho_high = [0.65, 0.75, 0.85, 0.95]
    rho = rho_low + rho_high
    actions = ["local_redistribute"] * 4 + ["recovery"] * 4
    rle = compute_rle_realistic(rho, actions)
    assert rle == pytest.approx(1.0, abs=0.01)


def test_inverted_triage_policy_scores_poorly():
    """A policy that picks Recovery for marketable and LR for
    non-marketable is the worst-case (other than cold_chain).
    Should score below 0.40."""
    rho_low = [0.15, 0.20, 0.25, 0.28]
    rho_high = [0.65, 0.75, 0.85, 0.95]
    rho = rho_low + rho_high
    actions = ["recovery"] * 4 + ["local_redistribute"] * 4  # inverted
    rle = compute_rle_realistic(rho, actions)
    assert rle < 0.40


def test_realistic_rle_does_not_saturate_when_binary_does():
    """The decisive saturation test: the same trace where the binary
    RLE saturates at 1.0 should produce a strictly-less-than-1.0
    realistic RLE, demonstrating the metric improvement."""
    rho = [0.15] * 30 + [0.45] * 30 + [0.80] * 30
    actions_always_lr = ["local_redistribute"] * len(rho)
    binary = compute_rle(rho, actions_always_lr)
    realistic = compute_rle_realistic(rho, actions_always_lr)
    assert binary == pytest.approx(1.0)  # saturates
    assert realistic < 0.85  # does not saturate


# ---------------------------------------------------------------------------
# Severity weighting
# ---------------------------------------------------------------------------

def test_severity_weighting_higher_rho_dominates():
    """A single high-rho mismatch should pull the realistic RLE down
    more than a single low-rho mismatch."""
    rho_lo = [0.15] * 9 + [0.20]  # all matched
    acts_lo = ["local_redistribute"] * 10  # perfect match
    rho_hi_miss = [0.15] * 9 + [0.90]  # last is non-marketable
    acts_hi_miss = ["local_redistribute"] * 10  # mismatch on the high one
    rle_perfect = compute_rle_realistic(rho_lo, acts_lo)
    rle_mismatch = compute_rle_realistic(rho_hi_miss, acts_hi_miss)
    assert rle_perfect == pytest.approx(1.0)
    # The mismatched high-severity step pulls down the score MORE than
    # an equivalent number of mismatched low-severity steps would.
    assert rle_mismatch < 0.70


# ---------------------------------------------------------------------------
# RLETracker integration
# ---------------------------------------------------------------------------

def test_tracker_emits_all_three_variants_in_one_pass():
    rho = [0.15, 0.20, 0.45, 0.50, 0.80, 0.85]
    actions = [
        "local_redistribute", "local_redistribute",
        "recovery", "recovery",
        "recovery", "recovery",
    ]
    tracker = RLETracker()
    for r, a in zip(rho, actions):
        tracker.update(r, a)
    assert 0.0 < tracker.rle <= 1.0
    assert 0.0 < tracker.rle_weighted <= 1.0
    assert 0.0 < tracker.rle_realistic <= 1.0
    # The realistic one for this near-perfect triage trace should be high.
    assert tracker.rle_realistic > 0.85


def test_tracker_returns_zero_when_no_at_risk():
    tracker = RLETracker()
    for r in (0.01, 0.05, 0.08):  # all below threshold
        tracker.update(r, "cold_chain")
    assert tracker.rle == 0.0
    assert tracker.rle_weighted == 0.0
    assert tracker.rle_realistic == 0.0
