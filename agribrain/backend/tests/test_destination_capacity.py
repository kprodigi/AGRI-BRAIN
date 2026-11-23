"""Destination capacity, acceptance criteria, guards and rule references."""
from __future__ import annotations

import numpy as np
import pytest
from pirag.mcp.tools.destination_capacity import query_destination_capacity
from src.models.action_selection import (
    categorical_action_from_uniform,
    fixed_rule_action,
    greedy_informed_action,
    predicted_route_ari,
    select_action,
)
from src.models.destination_capacity import (
    DestinationCapacity,
    DestinationCapacityConfig,
    accepts,
    route_feasibility,
)
from src.models.policy import Policy
from src.models.resilience import RLETracker

REDISTRIBUTION = "local_redistribute"
RECOVERY = "recovery"


def _enabled(**overrides) -> DestinationCapacityConfig:
    return DestinationCapacityConfig(enabled=True, **overrides)


def test_defaults_match_the_declared_design() -> None:
    config = DestinationCapacityConfig.from_policy(Policy())
    assert config.enabled is False
    assert (config.kappa_lr, config.kappa_rec) == (0.5, 0.25)
    assert (config.bmax_lr, config.bmax_rec) == (8.0, 4.0)
    assert (config.rho_at, config.rho_cut) == (0.10, 0.50)
    assert config.demand_ref == 15.0 and config.inventory_ref == 12_000.0
    assert config.missing_confirmation_closed is True
    assert config.acceptance_enabled is True
    assert config.refused_share_lost is False
    assert config.reservation_share == 1.0
    record = DestinationCapacityConfig.from_policy(Policy(
        refused_share_lost=True, missing_confirmation_closed=False,
        reservation_share=0.8)).as_record()
    assert record["refused_share_lost"] is True
    assert record["missing_confirmation_closed"] is False
    assert record["reservation_share"] == 0.8


@pytest.mark.parametrize("kwargs", [
    {"kappa_lr": -0.1}, {"bmax_rec": -1.0}, {"demand_ref": 0.0},
    {"rho_at": 0.5, "rho_cut": 0.5}, {"rho_cut": 1.2},
    {"reservation_share": 0.0}, {"reservation_share": 1.1},
])
def test_invalid_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        DestinationCapacityConfig(**kwargs)


def test_acceptance_criteria() -> None:
    config = _enabled()
    assert accepts(config, "cold_chain", 0.9)
    assert accepts(config, REDISTRIBUTION, 0.05)
    assert accepts(config, REDISTRIBUTION, 0.50)
    assert not accepts(config, REDISTRIBUTION, 0.51)
    assert not accepts(config, RECOVERY, 0.10)
    assert accepts(config, RECOVERY, 0.11)
    strict = _enabled(strict_redistribution=True)
    assert not accepts(strict, REDISTRIBUTION, 0.05)
    assert accepts(strict, REDISTRIBUTION, 0.30)
    open_door = _enabled(acceptance_enabled=False)
    assert accepts(open_door, REDISTRIBUTION, 0.99)
    assert accepts(open_door, RECOVERY, 0.0)


def test_disabled_model_accepts_everything_and_keeps_no_state() -> None:
    capacity = DestinationCapacity(DestinationCapacityConfig())
    before = capacity.report()
    assert capacity.accepted_share(REDISTRIBUTION, 0.9, 5.0) == 1.0
    capacity.commit(REDISTRIBUTION, 1.0, 5.0)
    capacity.replenish(30.0)
    assert capacity.report() == before


def test_accepted_share_commit_and_replenishment() -> None:
    capacity = DestinationCapacity(_enabled())
    assert capacity.dispatch_size(24_000.0) == 2.0
    assert capacity.accepted_share(REDISTRIBUTION, 0.2, 2.0) == 1.0
    capacity.commit(REDISTRIBUTION, 1.0, 7.5)
    assert capacity.report()[REDISTRIBUTION] == pytest.approx(0.5)
    # half a dispatch of room left for a dispatch of one
    assert capacity.accepted_share(REDISTRIBUTION, 0.2, 1.0) == pytest.approx(0.5)
    # an unacceptable product is refused whatever the room
    assert capacity.accepted_share(REDISTRIBUTION, 0.6, 0.1) == 0.0
    assert capacity.accepted_share(RECOVERY, 0.05, 0.1) == 0.0
    capacity.replenish(15.0)
    assert capacity.report()[REDISTRIBUTION] == pytest.approx(1.0)
    assert capacity.report()[RECOVERY] == pytest.approx(4.0)  # capped at the maximum
    capacity.replenish(15.0 * 100)
    assert capacity.report()[REDISTRIBUTION] == pytest.approx(8.0)
    with pytest.raises(ValueError):
        capacity.commit(RECOVERY, 1.0, 100.0)


def test_capacity_bookkeeping_worked_by_hand() -> None:
    """Three decisions followed with pencil and paper.

    Redistribution starts at 8. A dispatch of 6 is taken whole (2 left), demand
    30 refills 0.5 * 30 / 15 = 1 (3 left). A dispatch of 4 is taken at
    3 / 4 = 0.75 (0 left), demand 0 refills nothing. A dispatch of 2 finds no
    room and none of it is taken; demand 15 refills 0.5. A dispatch of size
    zero is taken in full whatever the room.
    """
    capacity = DestinationCapacity(_enabled())
    assert capacity.accepted_share(REDISTRIBUTION, 0.3, 6.0) == 1.0
    capacity.commit(REDISTRIBUTION, 1.0, 6.0)
    capacity.replenish(30.0)
    assert capacity.report() == {REDISTRIBUTION: 3.0, RECOVERY: 4.0}
    assert capacity.accepted_share(REDISTRIBUTION, 0.3, 4.0) == 0.75
    capacity.commit(REDISTRIBUTION, 0.75, 4.0)
    capacity.replenish(0.0)
    assert capacity.report()[REDISTRIBUTION] == 0.0
    assert capacity.accepted_share(REDISTRIBUTION, 0.3, 2.0) == 0.0
    assert capacity.accepted_share(REDISTRIBUTION, 0.3, 0.0) == 1.0
    capacity.commit(REDISTRIBUTION, 0.0, 2.0)
    capacity.replenish(15.0)
    assert capacity.report()[REDISTRIBUTION] == 0.5
    # recovery: 4 - 1 = 3, refilled by 0.25 each interval and capped at 4
    capacity.commit(RECOVERY, 1.0, 1.0)
    for expected in (3.25, 3.5, 3.75, 4.0, 4.0):
        capacity.replenish(15.0)
        assert capacity.report()[RECOVERY] == expected


CRITERIA = {"rho_at": 0.10, "rho_cut": 0.50}
REPORT = {"dispatch_size": 1.0, "redistribution_capacity": 2.0, "recovery_capacity": 2.0}
RESERVATION = {"guard_kind": "reservation", **CRITERIA, **REPORT}
ELIGIBILITY = {"guard_kind": "eligibility", **CRITERIA, **REPORT}
NO_REPORT = {"redistribution_capacity": None, "recovery_capacity": None}
NO_CRITERIA = {"rho_at": None, "rho_cut": None, "criteria": "unavailable"}


@pytest.mark.parametrize(("base", "rho", "change", "feasible", "reasons"), [
    # eligibility, both guards
    (RESERVATION, 0.05, {}, [True, True, False], [None, None, "acceptance"]),
    (ELIGIBILITY, 0.05, {}, [True, True, False], [None, None, "acceptance"]),
    (RESERVATION, 0.60, {}, [True, False, True], [None, "acceptance", None]),
    (ELIGIBILITY, 0.60, {}, [True, False, True], [None, "acceptance", None]),
    (RESERVATION, 0.05, {"strict_redistribution": True}, [True, False, False],
     [None, "acceptance", "acceptance"]),
    # capacity: the reservation guard needs room for the whole dispatch
    (RESERVATION, 0.30, {"redistribution_capacity": 0.5}, [True, False, True],
     [None, "capacity", None]),
    (RESERVATION, 0.30, {"redistribution_capacity": 1.0}, [True, True, True], [None, None, None]),
    # ... or for the declared share of it
    (RESERVATION, 0.30, {"redistribution_capacity": 0.5, "reservation_share": 0.5},
     [True, True, True], [None, None, None]),
    (RESERVATION, 0.30, {"redistribution_capacity": 0.49, "reservation_share": 0.5},
     [True, False, True], [None, "capacity", None]),
    # capacity: the eligibility guard leaves partial room selectable
    (ELIGIBILITY, 0.30, {"redistribution_capacity": 0.5}, [True, True, True], [None, None, None]),
    (ELIGIBILITY, 0.30, {"redistribution_capacity": 0.0}, [True, False, True],
     [None, "no_capacity", None]),
    # a dispatch of size zero is never removed for capacity
    (ELIGIBILITY, 0.30, {"redistribution_capacity": 0.0, "dispatch_size": 0.0},
     [True, True, True], [None, None, None]),
    (RESERVATION, 0.30, {"redistribution_capacity": 0.0, "dispatch_size": 0.0},
     [True, True, True], [None, None, None]),
    # a requested report that is missing: the rule decides for the reservation
    # guard; the eligibility guard needs no confirmation
    (RESERVATION, 0.30, NO_REPORT, [True, False, False],
     [None, "no_capacity_report", "no_capacity_report"]),
    (RESERVATION, 0.30, {**NO_REPORT, "missing_confirmation_closed": False},
     [True, True, True], [None, None, None]),
    (ELIGIBILITY, 0.30, NO_REPORT, [True, True, True], [None, None, None]),
    # requested criteria that are missing: the rule decides for both guards
    (RESERVATION, 0.60, NO_CRITERIA, [True, False, False],
     [None, "no_acceptance_criteria", "no_acceptance_criteria"]),
    (ELIGIBILITY, 0.60, NO_CRITERIA, [True, False, False],
     [None, "no_acceptance_criteria", "no_acceptance_criteria"]),
    (RESERVATION, 0.60, {**NO_CRITERIA, "missing_confirmation_closed": False},
     [True, True, True], [None, None, None]),
    # information the mode never requests: the check is not applied
    (RESERVATION, 0.60, {"rho_at": None, "rho_cut": None, "criteria": "withheld"},
     [True, True, True], [None, None, None]),
    (RESERVATION, 0.30, {**NO_REPORT, "capacity": "withheld"}, [True, True, True],
     [None, None, None]),
    (RESERVATION, 0.60, {**NO_REPORT, "capacity": "withheld"}, [True, False, True],
     [None, "acceptance", None]),
    # no guard
    (RESERVATION, 0.60, {"guard_kind": "none"}, [True, True, True], [None, None, None]),
])
def test_route_feasibility(base, rho, change, feasible, reasons) -> None:
    assert route_feasibility(rho, {**base, **change}) == (feasible, reasons)


def test_route_feasibility_without_information() -> None:
    assert route_feasibility(0.9, None) == ([True, True, True], [None, None, None])
    with pytest.raises(ValueError):
        route_feasibility(0.3, {**RESERVATION, "criteria": "sometimes"})
    with pytest.raises(ValueError):
        route_feasibility(0.3, {**RESERVATION, "capacity": "available",
                                "redistribution_capacity": None})


RULE_INFORMATION = {**CRITERIA, **REPORT}


def test_rule_references() -> None:
    info = RULE_INFORMATION
    assert fixed_rule_action("always_redistribute", 0.9, None) == 1
    assert fixed_rule_action("always_recovery", 0.0, None) == 2
    assert [fixed_rule_action("risk_threshold", r, info) for r in (0.05, 0.30, 0.60)] == [0, 1, 2]
    assert [fixed_rule_action("capacity_informed", r, info) for r in (0.05, 0.30, 0.60)] == [1, 1, 2]
    assert fixed_rule_action("capacity_informed", 0.30, {**info, "redistribution_capacity": 0.2}) == 2
    # a requested report that is missing
    blocked = {**info, **NO_REPORT}
    assert fixed_rule_action("capacity_informed", 0.30, blocked) == 0
    assert fixed_rule_action(
        "capacity_informed", 0.30, {**blocked, "missing_confirmation_closed": False}) == 1
    # missing criteria: the rules follow the same rule as the guard
    unread = {**info, **NO_CRITERIA}
    assert fixed_rule_action("risk_threshold", 0.30, unread) == 0
    assert fixed_rule_action("capacity_informed", 0.30, unread) == 0
    assert fixed_rule_action(
        "capacity_informed", 0.30, {**unread, "missing_confirmation_closed": False}) == 1
    # a lower reservation share lets a partly filled outlet be used
    assert fixed_rule_action("capacity_informed", 0.30, {
        **info, "redistribution_capacity": 0.6, "reservation_share": 0.5}) == 1
    with pytest.raises(ValueError):
        fixed_rule_action("unknown_rule", 0.2, info)
    with pytest.raises(ValueError):
        fixed_rule_action("greedy_informed", 0.2, info)


# ---------------------------------------------------------------------------
# Predicted step value and the greedy rule, against values worked by hand
# ---------------------------------------------------------------------------
# At 4 degrees C and reference inventory there is no thermal stress and no
# surplus, so the social-performance attenuation is 1 and the emissions are
# distance times 0.12: cold chain 14.4 kg, redistribution 5.4 kg, recovery
# 9.6 kg. With weights 0.30 / 0.20 / 0.25 / 0.25 and a 50 kg normalizer:
#   cold chain      0.30 * (1 - 14.4/50) + 0.20*0.60 + 0.25*0.55 + 0.25*0.55 = 0.6086
#   redistribution  0.30 * (1 -  5.4/50) + 0.20*0.82 + 0.25*0.78 + 0.25*0.78 = 0.8216
#   recovery        0.30 * (1 -  9.6/50) + 0.20*0.70 + 0.25*0.72 + 0.25*0.68 = 0.7324
#   refused after redistribution: cold-chain scores at 5.4 + 14.4 = 19.8 kg  -> 0.5762
#   refused after recovery:       cold-chain scores at 9.6 + 14.4 = 24.0 kg  -> 0.5510
# With raw waste w: redistribution keeps 0.55 w, recovery 0.75 w; the refused
# share loses w * (1 + 45/120) = 1.375 w and w * (1 + 80/120) = 5 w / 3.
STATE = {"rho": 0.30, "inventory": 12_000.0, "temperature": 4.0, "humidity": 90.0}
SOCIAL = {"cold": 0.6086, "redistribution": 0.8216, "recovery": 0.7324,
          "refused_redistribution": 0.5762, "refused_recovery": 0.5510}


def _predicted(policy, action, share):
    return predicted_route_ari(policy, action, share, **STATE)


def _raw_waste(policy):
    # the cold chain saves nothing, so its waste is the raw waste
    return 1.0 - _predicted(policy, 0, 1.0) / (SOCIAL["cold"] * (1.0 - STATE["rho"]))


def test_predicted_step_value_matches_values_worked_by_hand() -> None:
    policy = Policy(destination_capacity_enabled=True)
    w = _raw_waste(policy)
    assert 0.0 < w < 0.15
    sound = 1.0 - STATE["rho"]
    for share in (1.0, 0.75, 0.5, 0.2, 0.0):
        redistribution = ((1.0 - w * (0.55 * share + 1.375 * (1.0 - share)))
                          * (SOCIAL["redistribution"] * share
                             + SOCIAL["refused_redistribution"] * (1.0 - share)) * sound)
        recovery = ((1.0 - w * (0.75 * share + (5.0 / 3.0) * (1.0 - share)))
                    * (SOCIAL["recovery"] * share
                       + SOCIAL["refused_recovery"] * (1.0 - share)) * sound)
        assert _predicted(policy, 1, share) == pytest.approx(redistribution, abs=1e-12)
        assert _predicted(policy, 2, share) == pytest.approx(recovery, abs=1e-12)


def test_refusal_penalty_variants_worked_by_hand() -> None:
    w = _raw_waste(Policy(destination_capacity_enabled=True))
    sound = 1.0 - STATE["rho"]
    # penalty doubled: the diverted leg counts twice for loss and emissions
    doubled = Policy(destination_capacity_enabled=True, refusal_penalty_scale=2.0)
    social = 0.30 * (1.0 - (2 * 5.4 + 14.4) / 50.0) + 0.20 * 0.60 + 0.25 * 0.55 + 0.25 * 0.55
    assert _predicted(doubled, 1, 0.0) == pytest.approx(
        (1.0 - w * (1.0 + 2 * 45.0 / 120.0)) * social * sound, abs=1e-12)
    # refused share lost: all of it is waste, so the step value of a fully
    # refused dispatch is zero and half a refusal halves what is kept
    lost = Policy(destination_capacity_enabled=True, refused_share_lost=True)
    assert _predicted(lost, 1, 0.0) == 0.0
    assert _predicted(lost, 1, 0.5) == pytest.approx(
        (1.0 - (0.5 * 0.55 * w + 0.5))
        * (0.5 * SOCIAL["redistribution"] + 0.5 * SOCIAL["refused_redistribution"]) * sound,
        abs=1e-12)


def test_break_even_share_for_redistribution() -> None:
    """With raw waste 0.10, redistribution equals the cold chain at about 19.3%."""
    def normalized(share, w=0.10):
        return ((1.0 - w * (0.55 * share + 1.375 * (1.0 - share)))
                * (SOCIAL["redistribution"] * share + SOCIAL["refused_redistribution"] * (1.0 - share)))
    cold = (1.0 - 0.10) * SOCIAL["cold"]
    assert cold == pytest.approx(0.54774)
    assert normalized(0.5) == pytest.approx(0.6316, abs=5e-5)
    assert normalized(0.19) < cold < normalized(0.20)
    # the equations of the model give the same ordering at the same shares
    policy = Policy(destination_capacity_enabled=True)
    assert _predicted(policy, 1, 0.10) < _predicted(policy, 0, 1.0) < _predicted(policy, 1, 0.30)


def test_greedy_rule() -> None:
    policy = Policy(destination_capacity_enabled=True)
    observed = {"inventory": STATE["inventory"], "temperature": STATE["temperature"],
                "humidity": STATE["humidity"]}
    info = {**RULE_INFORMATION, "observed": observed}
    # room for everything: redistribution has the highest value
    assert greedy_informed_action(0.30, info, policy) == 1
    assert fixed_rule_action("greedy_informed", 0.30, info, policy) == 1
    # With redistribution half full, recovery in full is worth more:
    # 0.7324 * (1 - 0.75 w) against 0.6989 * (1 - 0.9625 w).
    assert greedy_informed_action(0.30, {**info, "redistribution_capacity": 0.5}, policy) == 2
    # Without recovery, half a dispatch accepted still beats the cold chain
    # (0.6989 against 0.6086 before waste); a tenth does not (0.6007).
    no_recovery = {**info, "recovery_capacity": 0.0}
    assert greedy_informed_action(
        0.30, {**no_recovery, "redistribution_capacity": 0.5}, policy) == 1
    assert greedy_informed_action(
        0.30, {**no_recovery, "redistribution_capacity": 0.1}, policy) == 0
    assert greedy_informed_action(0.30, {**info, "redistribution_capacity": 0.1}, policy) == 2
    # outside the acceptance band a diversion is not considered
    assert greedy_informed_action(0.60, info, policy) == 2
    assert greedy_informed_action(0.05, info, policy) == 1
    # a requested report that is missing
    assert greedy_informed_action(0.30, {**info, **NO_REPORT}, policy) == 0
    assert greedy_informed_action(
        0.30, {**info, **NO_REPORT, "missing_confirmation_closed": False}, policy) == 1
    # missing criteria
    assert greedy_informed_action(0.30, {**info, **NO_CRITERIA}, policy) == 0
    with pytest.raises(ValueError):
        greedy_informed_action(0.30, RULE_INFORMATION, policy)


def test_rle_counts_credit_on_the_accepted_share_only() -> None:
    full, half, none = RLETracker(), RLETracker(), RLETracker()
    for tracker, share in ((full, 1.0), (half, 0.5), (none, 0.0)):
        tracker.update(0.30, REDISTRIBUTION, share)
        tracker.update(0.05, REDISTRIBUTION, share)  # not at risk: not counted
    assert full.rle == pytest.approx(1.0)
    assert half.rle == pytest.approx(0.5)
    assert none.rle == 0.0
    with pytest.raises(ValueError):
        full.update(0.3, REDISTRIBUTION, 1.5)


def _select(route_information, uniform=0.5, mode="no_context", **kwargs):
    out: dict = {}
    action, probs = select_action(
        mode=mode, rho=0.30, inv=12_000.0, y_hat=15.0, temp=6.0, tau=0.0,
        policy=Policy(), rng=np.random.default_rng(1), categorical_uniform=uniform,
        out=out, route_information=route_information, **kwargs,
    )
    return action, probs, out


@pytest.mark.parametrize(("information", "reason"), [
    ({**RESERVATION, "redistribution_capacity": 0.5}, "capacity"),
    ({**ELIGIBILITY, "redistribution_capacity": 0.0}, "no_capacity"),
])
def test_guard_removes_routes_and_renormalizes(information, reason) -> None:
    _, free, _ = _select(None)
    action, probs, out = _select(information)
    assert out["guard_kind"] == information["guard_kind"]
    assert out["guard_removed_routes"] == [1]
    assert out["guard_reasons"] == [None, reason, None]
    assert probs[1] == 0.0 and probs.sum() == pytest.approx(1.0)
    assert np.allclose(out["policy_probs_pre_guard"], free)
    expected = free * np.array([1.0, 0.0, 1.0])
    assert np.allclose(probs, expected / expected.sum())
    assert action != 1
    assert action == categorical_action_from_uniform(probs, 0.5)
    assert out["sampled_action_pre_guard"] == categorical_action_from_uniform(free, 0.5)


def test_eligibility_guard_leaves_partial_room_selectable() -> None:
    _, free, _ = _select(None)
    _, probs, out = _select({**ELIGIBILITY, "redistribution_capacity": 0.5})
    assert out["guard_removed_routes"] == []
    assert np.allclose(probs, free)


def test_guard_leaves_cold_chain_when_nothing_else_is_available() -> None:
    blocked = {**RESERVATION, **NO_REPORT}
    for uniform in (0.01, 0.5, 0.99):
        action, probs, out = _select(blocked, uniform=uniform)
        assert action == 0 and probs.tolist() == [1.0, 0.0, 0.0]
        assert out["guard_removed_routes"] == [1, 2]


def test_no_guard_record_without_a_guard() -> None:
    _, _, out = _select(None)
    assert "guard_kind" not in out
    _, _, out = _select({**RESERVATION, "guard_kind": "none"})
    assert "guard_kind" not in out


def test_probability_gap_override_can_be_switched_off() -> None:
    # A modifier that makes cold chain negligible triggers the submitted
    # override; the revised design switches it off.
    modifier = np.array([-8.0, 8.0, -8.0])
    action, probs, out = _select(None, mode="agribrain", context_modifier=modifier)
    assert out["governance_override"] is True and action == 1
    action, probs, out = _select(
        None, mode="agribrain", context_modifier=modifier,
        probability_gap_override=False,
    )
    assert out["governance_override"] is False
    assert probs.sum() == pytest.approx(1.0) and probs[1] < 1.0


def test_capacity_tool_report() -> None:
    report = query_destination_capacity(0.6, 3.0, 1.2)
    assert report["redistribution_share"] == pytest.approx(0.5)
    assert report["redistribution_shortfall"] == pytest.approx(0.5)
    assert report["recovery_shortfall"] == 0.0
    assert report["redistribution_has_room"] is False
    assert report["recovery_has_room"] is True
    assert query_destination_capacity(0.0, 0.0, 0.0)["redistribution_share"] == 1.0
    with pytest.raises(ValueError):
        query_destination_capacity(-1.0, 1.0, 1.0)
