"""Unit tests for ``compute_violation_disposition`` in resilience.py.

The function records what the policy chose to do on each timestep
where the environment fired a safety-window violation (temperature
ceiling exceeded OR shelf-fraction below expedite floor — the same
predicate the simulator uses for ``constraint_violation_rate``).

These tests pin the contract that:

  - the three rates (downstream / redistribute / contained) sum to
    1.0 on any episode that has at least one violation event;
  - all three rates are 0.0 by convention when no violation events
    fired (rather than NaN or a divide-by-zero error);
  - cold_chain on a violation increments downstream;
    local_redistribute increments redistribute; recovery increments
    contained; non-violation timesteps are ignored regardless of
    action;
  - either ``temp_violation`` or ``quality_violation`` (logical OR)
    is sufficient to trigger an event count;
  - canonical-action aliases resolved by ``action_aliases.resolve_action``
    are accepted by the disposition counter.

The metric is the policy-quality answer to ConstraintViolationRate
(which is environmental and method-invariant by construction); the
ordering claim ``Static.downstream >> AgriBrain.downstream`` is
asserted in the simulator-level integration test rather than here.
"""
from __future__ import annotations

import pytest

from src.models.resilience import compute_violation_disposition


# ---------------------------------------------------------------------------
# Empty / trivial episodes
# ---------------------------------------------------------------------------

def test_zero_violations_returns_all_zero_rates():
    """An episode with no violation events should report 0.0 for every
    rate and event_count=0, NOT raise or return NaN."""
    out = compute_violation_disposition(
        temp_violations=[False, False, False],
        quality_violations=[False, False, False],
        actions=["cold_chain", "cold_chain", "cold_chain"],
    )
    assert out["downstream_violation_rate"] == 0.0
    assert out["redistribute_violation_rate"] == 0.0
    assert out["contained_violation_rate"] == 0.0
    assert out["violation_event_count"] == 0


def test_empty_traces_returns_all_zero_rates():
    out = compute_violation_disposition(
        temp_violations=[], quality_violations=[], actions=[],
    )
    assert out == {
        "downstream_violation_rate":   0.0,
        "redistribute_violation_rate": 0.0,
        "contained_violation_rate":    0.0,
        "violation_event_count":       0,
    }


# ---------------------------------------------------------------------------
# Pure-policy boundary cases (the metric SHOULD discriminate these)
# ---------------------------------------------------------------------------

def test_static_like_policy_lands_at_downstream_one():
    """When the policy always picks cold_chain on violation events,
    DownstreamViolationRate must be 1.0 and the other two 0.0."""
    out = compute_violation_disposition(
        temp_violations=[True, True, True, False],
        quality_violations=[False, True, False, False],
        actions=["cold_chain", "cold_chain", "cold_chain", "cold_chain"],
    )
    assert out["downstream_violation_rate"] == pytest.approx(1.0)
    assert out["redistribute_violation_rate"] == pytest.approx(0.0)
    assert out["contained_violation_rate"] == pytest.approx(0.0)
    assert out["violation_event_count"] == 3


def test_recovery_only_policy_lands_at_contained_one():
    """When the policy always picks recovery on violation events,
    ContainedViolationRate must be 1.0 and the other two 0.0."""
    out = compute_violation_disposition(
        temp_violations=[True, False, True, False],
        quality_violations=[False, True, False, False],
        actions=["recovery", "recovery", "recovery", "cold_chain"],
    )
    assert out["downstream_violation_rate"] == pytest.approx(0.0)
    assert out["redistribute_violation_rate"] == pytest.approx(0.0)
    assert out["contained_violation_rate"] == pytest.approx(1.0)
    assert out["violation_event_count"] == 3


def test_local_only_policy_lands_at_redistribute_one():
    out = compute_violation_disposition(
        temp_violations=[True, True],
        quality_violations=[False, False],
        actions=["local_redistribute", "local_redistribute"],
    )
    assert out["downstream_violation_rate"] == pytest.approx(0.0)
    assert out["redistribute_violation_rate"] == pytest.approx(1.0)
    assert out["contained_violation_rate"] == pytest.approx(0.0)
    assert out["violation_event_count"] == 2


# ---------------------------------------------------------------------------
# Mixed-action episodes
# ---------------------------------------------------------------------------

def test_mixed_actions_split_proportionally():
    """4 violation events: 1 cold_chain, 1 local_redistribute, 2 recovery
    -> downstream=0.25, redistribute=0.25, contained=0.50."""
    out = compute_violation_disposition(
        temp_violations=[True, True, True, True, False],
        quality_violations=[False, False, False, False, False],
        actions=[
            "cold_chain", "local_redistribute", "recovery", "recovery",
            "cold_chain",  # this last one is NOT a violation, must be ignored
        ],
    )
    assert out["downstream_violation_rate"] == pytest.approx(0.25)
    assert out["redistribute_violation_rate"] == pytest.approx(0.25)
    assert out["contained_violation_rate"] == pytest.approx(0.50)
    assert out["violation_event_count"] == 4
    # The three rates must sum to 1.0 by construction whenever there's
    # at least one event.
    total = (
        out["downstream_violation_rate"]
        + out["redistribute_violation_rate"]
        + out["contained_violation_rate"]
    )
    assert total == pytest.approx(1.0)


def test_quality_violation_alone_triggers_event_count():
    """``quality_violation`` (shelf below floor) must trigger the event
    counter even when ``temp_violation`` is False — the predicate is
    logical OR per the simulator definition."""
    out = compute_violation_disposition(
        temp_violations=[False, False],
        quality_violations=[True, True],
        actions=["recovery", "cold_chain"],
    )
    assert out["violation_event_count"] == 2
    assert out["contained_violation_rate"] == pytest.approx(0.5)
    assert out["downstream_violation_rate"] == pytest.approx(0.5)


def test_both_violations_simultaneously_count_as_one_event():
    """A step with both temp AND quality violations counts as one event,
    not two — this is the same convention the simulator uses for
    ``operational_violation_steps`` and ``constraint_violation_steps``."""
    out = compute_violation_disposition(
        temp_violations=[True, True],
        quality_violations=[True, False],
        actions=["recovery", "recovery"],
    )
    # Two events (steps 0 and 1), both routed to recovery -> contained=1.0
    assert out["violation_event_count"] == 2
    assert out["contained_violation_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Non-violation steps must NOT influence the rates
# ---------------------------------------------------------------------------

def test_non_violation_steps_are_ignored():
    """A step where neither temp_violation nor quality_violation fires
    must contribute nothing to any counter, regardless of action."""
    out = compute_violation_disposition(
        temp_violations=[True, False, False, False, True],
        quality_violations=[False, False, False, False, False],
        actions=[
            "cold_chain",         # violation -> downstream
            "cold_chain",         # no violation, must be ignored
            "local_redistribute", # no violation, must be ignored
            "recovery",           # no violation, must be ignored
            "recovery",           # violation -> contained
        ],
    )
    assert out["violation_event_count"] == 2
    assert out["downstream_violation_rate"] == pytest.approx(0.5)
    assert out["contained_violation_rate"] == pytest.approx(0.5)
    assert out["redistribute_violation_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Action-alias resolution (uses action_aliases.resolve_action)
# ---------------------------------------------------------------------------

def test_aliased_action_names_are_resolved():
    """Action aliases (e.g. uppercase / variant spellings) must resolve
    to canonical names so the disposition counter doesn't silently drop
    the event because of a label mismatch."""
    # The repo's action_aliases.resolve_action accepts uppercase variants
    # and a couple of synonyms; pick one that's known to resolve.
    out = compute_violation_disposition(
        temp_violations=[True],
        quality_violations=[False],
        actions=["COLD_CHAIN"],
    )
    assert out["violation_event_count"] == 1
    assert out["downstream_violation_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Defensive: trace-length mismatch should be flagged loudly
# ---------------------------------------------------------------------------

def test_mismatched_trace_lengths_raise():
    """The function should refuse to aggregate over traces of different
    lengths — silently truncating to the shortest would be a recipe for
    off-by-one errors that drop the last violation event of the
    episode."""
    with pytest.raises(ValueError, match="trace lengths must match"):
        compute_violation_disposition(
            temp_violations=[True, False],
            quality_violations=[False, False, False],
            actions=["cold_chain", "recovery", "cold_chain"],
        )


# ---------------------------------------------------------------------------
# Range-bounded outputs
# ---------------------------------------------------------------------------

def test_all_rates_are_in_unit_interval_for_arbitrary_input():
    """Every rate must be in [0, 1] regardless of input distribution."""
    import random
    random.seed(0)
    n = 200
    tv = [random.random() < 0.3 for _ in range(n)]
    qv = [random.random() < 0.2 for _ in range(n)]
    actions = [
        random.choice(["cold_chain", "local_redistribute", "recovery"])
        for _ in range(n)
    ]
    out = compute_violation_disposition(
        temp_violations=tv, quality_violations=qv, actions=actions,
    )
    for key in ("downstream_violation_rate", "redistribute_violation_rate",
                "contained_violation_rate"):
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of [0,1]"
    if out["violation_event_count"] > 0:
        total = (
            out["downstream_violation_rate"]
            + out["redistribute_violation_rate"]
            + out["contained_violation_rate"]
        )
        assert total == pytest.approx(1.0, abs=1e-9)
