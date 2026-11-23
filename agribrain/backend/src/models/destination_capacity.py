"""Destination capacity and acceptance criteria for the routing outcome model.

Local redistribution outlets and the recovery facility each hold a finite
absorption capacity and apply an acceptance criterion to the product offered.
A dispatch that a destination cannot or will not take continues by cold chain
after the diverted leg. In the ``refused_share_lost`` variant it returns the
same way but none of it is sold.

All quantities are synthetic design values. They are not measured capacities,
food-safety limits or legal acceptance rules.

Definitions for one routing opportunity
---------------------------------------
dispatch size      q = inventory / inventory_ref
acceptance         redistribution takes marketable product (rho <= rho_cut);
                   recovery takes at-risk product (rho > rho_at);
                   in the strict variant redistribution also requires rho > rho_at
accepted share     alpha = acceptance * min(1, remaining_capacity / q)
capacity update    remaining <- min(maximum, remaining - alpha * q + replenishment)
replenishment      redistribution: kappa_lr * demand / demand_ref
                   recovery:       kappa_rec

The decision side may guard a diversion in two ways (``route_feasibility``).
The eligibility guard removes a route when the product is outside the
acceptance band stated in the acceptance document, or when the destination
reports no room at all. The reservation guard also requires reported room for
a declared share of the dispatch, by default the whole dispatch. A confirmation
that was requested but could not be obtained removes the diversion under
``missing_confirmation_closed`` and leaves it available otherwise, in which
case the destination enforces its own limits. A check whose information an
operating mode never requests is not applied.

With ``enabled=False`` every dispatch is accepted in full and no capacity state
is kept, which is the outcome model without destination constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .action_aliases import resolve_action as _resolve_action

COLD_CHAIN = "cold_chain"
REDISTRIBUTION = "local_redistribute"
RECOVERY = "recovery"
CONSTRAINED_DESTINATIONS = (REDISTRIBUTION, RECOVERY)


@dataclass(frozen=True)
class DestinationCapacityConfig:
    """Declared parameters of the destination capacity model."""

    enabled: bool = False
    kappa_lr: float = 0.5
    kappa_rec: float = 0.25
    bmax_lr: float = 8.0
    bmax_rec: float = 4.0
    demand_ref: float = 15.0
    inventory_ref: float = 12_000.0
    rho_at: float = 0.10
    rho_cut: float = 0.50
    strict_redistribution: bool = False
    refusal_penalty_scale: float = 1.0
    missing_confirmation_closed: bool = True
    acceptance_enabled: bool = True
    refused_share_lost: bool = False
    reservation_share: float = 1.0

    def __post_init__(self) -> None:
        for name in ("kappa_lr", "kappa_rec", "bmax_lr", "bmax_rec",
                     "refusal_penalty_scale"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("demand_ref", "inventory_ref"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= float(self.rho_at) < float(self.rho_cut) <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= rho_at < rho_cut <= 1")
        if not 0.0 < float(self.reservation_share) <= 1.0:
            raise ValueError("reservation_share must be in (0, 1]")

    @classmethod
    def from_policy(cls, policy: Any) -> "DestinationCapacityConfig":
        """Read the configuration from a ``Policy`` object."""

        return cls(
            enabled=bool(getattr(policy, "destination_capacity_enabled", False)),
            kappa_lr=float(getattr(policy, "capacity_kappa_lr", 0.5)),
            kappa_rec=float(getattr(policy, "capacity_kappa_rec", 0.25)),
            bmax_lr=float(getattr(policy, "capacity_bmax_lr", 8.0)),
            bmax_rec=float(getattr(policy, "capacity_bmax_rec", 4.0)),
            demand_ref=float(getattr(policy, "capacity_demand_ref", 15.0)),
            inventory_ref=float(getattr(policy, "capacity_inventory_ref", 12_000.0)),
            rho_at=float(getattr(policy, "acceptance_rho_at", 0.10)),
            rho_cut=float(getattr(policy, "acceptance_rho_cut", 0.50)),
            strict_redistribution=bool(
                getattr(policy, "acceptance_strict_redistribution", False)
            ),
            refusal_penalty_scale=float(
                getattr(policy, "refusal_penalty_scale", 1.0)
            ),
            missing_confirmation_closed=bool(
                getattr(policy, "missing_confirmation_closed", True)
            ),
            acceptance_enabled=bool(
                getattr(policy, "destination_acceptance_enabled", True)
            ),
            refused_share_lost=bool(getattr(policy, "refused_share_lost", False)),
            reservation_share=float(getattr(policy, "reservation_share", 1.0)),
        )

    def as_record(self) -> Dict[str, Any]:
        """JSON-native record of the declared parameters."""

        return {
            "enabled": bool(self.enabled),
            "kappa_lr": float(self.kappa_lr),
            "kappa_rec": float(self.kappa_rec),
            "bmax_lr": float(self.bmax_lr),
            "bmax_rec": float(self.bmax_rec),
            "demand_ref": float(self.demand_ref),
            "inventory_ref": float(self.inventory_ref),
            "rho_at": float(self.rho_at),
            "rho_cut": float(self.rho_cut),
            "strict_redistribution": bool(self.strict_redistribution),
            "refusal_penalty_scale": float(self.refusal_penalty_scale),
            "missing_confirmation_closed": bool(self.missing_confirmation_closed),
            "acceptance_enabled": bool(self.acceptance_enabled),
            "refused_share_lost": bool(self.refused_share_lost),
            "reservation_share": float(self.reservation_share),
        }


def accepts(config: DestinationCapacityConfig, action: str, rho: float) -> bool:
    """Whether the destination of ``action`` accepts product at risk ``rho``."""

    family = _resolve_action(action)
    rho = float(rho)
    if family == COLD_CHAIN:
        return True
    if family not in CONSTRAINED_DESTINATIONS:
        raise ValueError(f"unknown routing action {action!r}")
    if not config.acceptance_enabled:
        return True
    if family == REDISTRIBUTION:
        marketable = rho <= config.rho_cut
        if config.strict_redistribution:
            return bool(marketable and rho > config.rho_at)
        return bool(marketable)
    if family == RECOVERY:
        return bool(rho > config.rho_at)
    raise ValueError(f"unknown routing action {action!r}")


class DestinationCapacity:
    """Remaining capacity of the two constrained destinations in one episode."""

    def __init__(self, config: DestinationCapacityConfig) -> None:
        self.config = config
        self.remaining: Dict[str, float] = {
            REDISTRIBUTION: float(config.bmax_lr),
            RECOVERY: float(config.bmax_rec),
        }

    def dispatch_size(self, inventory: float) -> float:
        """Dispatch size relative to the reference inventory."""

        inventory = float(inventory)
        if not isfinite(inventory) or inventory < 0.0:
            raise ValueError("inventory must be finite and non-negative")
        return inventory / self.config.inventory_ref

    def report(self) -> Dict[str, float]:
        """Capacity each destination holds before the current dispatch."""

        return {
            REDISTRIBUTION: float(self.remaining[REDISTRIBUTION]),
            RECOVERY: float(self.remaining[RECOVERY]),
        }

    def accepted_share(self, action: str, rho: float, dispatch_size: float) -> float:
        """Share of the dispatch that the destination of ``action`` takes."""

        family = _resolve_action(action)
        if family == COLD_CHAIN or not self.config.enabled:
            return 1.0
        if not accepts(self.config, family, rho):
            return 0.0
        dispatch_size = float(dispatch_size)
        if dispatch_size <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, self.remaining[family]) / dispatch_size))

    def commit(self, action: str, accepted_share: float, dispatch_size: float) -> None:
        """Draw the accepted quantity from the destination of ``action``."""

        family = _resolve_action(action)
        if family == COLD_CHAIN or not self.config.enabled:
            return
        taken = float(accepted_share) * float(dispatch_size)
        if taken < 0.0 or taken > self.remaining[family] + 1e-9:
            raise ValueError("accepted quantity exceeds remaining capacity")
        self.remaining[family] = max(0.0, self.remaining[family] - taken)

    def replenish(self, demand: float) -> None:
        """Advance both destinations by one decision interval."""

        if not self.config.enabled:
            return
        demand = float(demand)
        if not isfinite(demand) or demand < 0.0:
            raise ValueError("demand must be finite and non-negative")
        self.remaining[REDISTRIBUTION] = min(
            self.config.bmax_lr,
            self.remaining[REDISTRIBUTION]
            + self.config.kappa_lr * demand / self.config.demand_ref,
        )
        self.remaining[RECOVERY] = min(
            self.config.bmax_rec,
            self.remaining[RECOVERY] + self.config.kappa_rec,
        )


GUARD_KINDS = ("none", "eligibility", "reservation")
INFORMATION_STATES = ("available", "unavailable", "withheld")


def information_states(information: Mapping[str, Any]) -> Tuple[str, str]:
    """States of the acceptance criteria and of the capacity report.

    A state is ``"available"``, ``"unavailable"`` (requested, not obtained) or
    ``"withheld"`` (the operating mode never requests it). Without an explicit
    state, criteria are available when both thresholds are given and withheld
    otherwise; a capacity report is available when both capacities are given
    and unavailable otherwise.
    """

    criteria = information.get("criteria") or (
        "available"
        if information.get("rho_at") is not None
        and information.get("rho_cut") is not None
        else "withheld"
    )
    capacity = information.get("capacity") or (
        "available"
        if information.get("redistribution_capacity") is not None
        and information.get("recovery_capacity") is not None
        else "unavailable"
    )
    for name, state in (("criteria", criteria), ("capacity", capacity)):
        if state not in INFORMATION_STATES:
            raise ValueError(f"unknown {name} state {state!r}")
    return criteria, capacity


def route_feasibility(
    rho: float,
    information: Optional[Mapping[str, Any]],
) -> Tuple[List[bool], List[Optional[str]]]:
    """Routes the decision side may choose, from the information it holds.

    Returns ``(feasible, reason)`` in action order (cold chain,
    redistribution, recovery). Cold chain is always feasible.

    ``information`` carries

    * ``guard_kind``: ``"eligibility"`` or ``"reservation"``; anything else
      applies no check;
    * ``criteria``: state of the acceptance criteria, with ``rho_at``,
      ``rho_cut`` and ``strict_redistribution`` when available;
    * ``capacity``: state of the capacity report, with ``dispatch_size``,
      ``redistribution_capacity`` and ``recovery_capacity`` when available;
    * ``reservation_share``: share of the dispatch for which the reservation
      guard requires reported room (default 1);
    * ``missing_confirmation_closed``: what a requested but unobtainable
      confirmation does (default true).

    The states are those of :func:`information_states`; a check whose
    information is withheld is not applied.

    Eligibility. A route is removed when ``rho``, the decision side's own risk
    estimate, is outside the destination's acceptance band. When the criteria
    are unavailable, both diversions are removed under
    ``missing_confirmation_closed``.

    Capacity. The eligibility guard removes a destination that reports no
    room for a dispatch of positive size, and needs no confirmation otherwise.
    The reservation guard removes a destination that reports less room than
    ``reservation_share`` of the dispatch, and a destination whose report is
    unavailable under ``missing_confirmation_closed``. A dispatch of size zero
    is never removed for capacity.

    Reasons: ``"acceptance"``, ``"no_acceptance_criteria"``, ``"capacity"``,
    ``"no_capacity"`` and ``"no_capacity_report"``.
    """

    feasible: List[bool] = [True, True, True]
    reasons: List[Optional[str]] = [None, None, None]
    if not information:
        return feasible, reasons
    kind = information.get("guard_kind", "none")
    if kind not in ("eligibility", "reservation"):
        return feasible, reasons
    rho = float(rho)
    closed = bool(information.get("missing_confirmation_closed", True))

    criteria, capacity = information_states(information)
    rho_at = information.get("rho_at")
    rho_cut = information.get("rho_cut")
    if criteria == "available":
        if rho_at is None or rho_cut is None:
            raise ValueError("available criteria need both thresholds")
        strict = bool(information.get("strict_redistribution", False))
        if rho > float(rho_cut) or (strict and rho <= float(rho_at)):
            feasible[1], reasons[1] = False, "acceptance"
        if rho <= float(rho_at):
            feasible[2], reasons[2] = False, "acceptance"
    elif criteria == "unavailable" and closed:
        for index in (1, 2):
            feasible[index], reasons[index] = False, "no_acceptance_criteria"

    rooms = (
        information.get("redistribution_capacity"),
        information.get("recovery_capacity"),
    )
    dispatch_size = information.get("dispatch_size")
    share = float(information.get("reservation_share", 1.0))
    for index, room in ((1, rooms[0]), (2, rooms[1])):
        if not feasible[index]:
            continue
        if capacity == "available":
            if room is None or dispatch_size is None:
                raise ValueError(
                    "an available capacity report needs the room and the dispatch size"
                )
            if kind == "reservation":
                if float(room) < share * float(dispatch_size):
                    feasible[index], reasons[index] = False, "capacity"
            elif float(dispatch_size) > 0.0 and float(room) <= 0.0:
                feasible[index], reasons[index] = False, "no_capacity"
        elif capacity == "unavailable" and kind == "reservation" and closed:
            feasible[index], reasons[index] = False, "no_capacity_report"
    return feasible, reasons


__all__ = [
    "COLD_CHAIN", "REDISTRIBUTION", "RECOVERY", "CONSTRAINED_DESTINATIONS",
    "GUARD_KINDS", "INFORMATION_STATES", "DestinationCapacityConfig",
    "DestinationCapacity", "information_states",
    "accepts", "route_feasibility",
]
