"""Locked per-seed trace schema shared by simulation, validation, and figures."""
from __future__ import annotations

import math
from typing import Any


TRACE_LENGTH = 288
TRACE_MODES = ("static", "hybrid_rl", "agribrain")
TRACE_FIELDS = (
    "ari_trace",
    "waste_trace",
    "rho_trace",
    "rho_policy_observed_trace",
    "rho_outcome_environmental_trace",
    "action_trace",
    "prob_trace",
    "carbon_trace",
    "hours",
    "temp_trace",
    "rh_trace",
    "inventory_trace",
    "demand_trace",
    "temp_policy_observed_trace",
    "temp_outcome_environmental_trace",
    "rh_policy_observed_trace",
    "rh_outcome_environmental_trace",
    "inventory_policy_observed_trace",
    "inventory_outcome_environmental_trace",
    "demand_policy_observed_trace",
    "demand_forecast_policy_observed_trace",
    "demand_regime_flag_trace",
    "price_signal_trace",
    "supply_forecast_policy_observed_trace",
    "demand_outcome_environmental_trace",
    "transport_multiplier_outcome_environmental_trace",
    "simulated_dispatch_accounted_trace",
    "slca_component_trace",
    "slca_trace",
    "equity_trace",
    "reward_trace",
)
ACTION_FAMILIES = ("cold_chain", "local_redistribute", "recovery")
SLCA_COMPONENT_KEYS = {
    "C", "L", "R", "P", "composite", "action_family",
    "slca_quality", "composite_attenuated",
}
BOUNDED_FIELDS = {
    "ari_trace", "waste_trace", "rho_trace",
    "rho_policy_observed_trace", "rho_outcome_environmental_trace",
    "slca_trace", "equity_trace",
}
NONNEGATIVE_FIELDS = {
    "carbon_trace", "inventory_trace", "demand_trace",
    "inventory_policy_observed_trace", "inventory_outcome_environmental_trace",
    "demand_policy_observed_trace", "demand_forecast_policy_observed_trace",
    "supply_forecast_policy_observed_trace", "demand_outcome_environmental_trace",
    "transport_multiplier_outcome_environmental_trace",
}


def _numeric_sequence(value: Any, *, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != TRACE_LENGTH:
        raise ValueError(f"{where} is not an exact {TRACE_LENGTH}-step list")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not numeric") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{where} contains a non-finite value")
    return values


def _canonical_action_index(value: Any, *, where: str) -> int:
    """Return one exact discrete action, accepting legacy JSON ``1.0``.

    Simulation commit d3286ae serialized every Python integer through
    ``round(float(value), 4)``, so its immutable preserved action traces use
    0.0/1.0/2.0.  Those values are exactly the documented discrete action set;
    fractional, non-finite, Boolean, string, and out-of-range values remain
    invalid.  The current writer preserves integer types for future runs.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} contains a noncanonical action")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} contains a noncanonical action")
    if value not in (0, 1, 2):
        raise ValueError(f"{where} contains a noncanonical action")
    return int(value)


def validate_trace_cell(cell: Any, *, where: str) -> None:
    """Validate the exact trace fields, shapes, domains, and identities."""

    if not isinstance(cell, dict):
        raise ValueError(f"{where} is not a trace object")
    if set(cell) != set(TRACE_FIELDS):
        raise ValueError(
            f"{where} trace schema mismatch: "
            f"missing={sorted(set(TRACE_FIELDS) - set(cell))}, "
            f"unexpected={sorted(set(cell) - set(TRACE_FIELDS))}"
        )

    numeric: dict[str, list[float]] = {}
    special = {
        "action_trace", "prob_trace", "simulated_dispatch_accounted_trace",
        "slca_component_trace",
    }
    for field in TRACE_FIELDS:
        if field not in special:
            numeric[field] = _numeric_sequence(
                cell[field], where=f"{where}/{field}",
            )

    actions_raw = cell["action_trace"]
    if not isinstance(actions_raw, list) or len(actions_raw) != TRACE_LENGTH:
        raise ValueError(f"{where}/action_trace is not an exact 288-step list")
    actions: list[int] = []
    for value in actions_raw:
        actions.append(_canonical_action_index(
            value, where=f"{where}/action_trace",
        ))

    probabilities = cell["prob_trace"]
    if not isinstance(probabilities, list) or len(probabilities) != TRACE_LENGTH:
        raise ValueError(f"{where}/prob_trace is not an exact 288-step list")
    for index, row in enumerate(probabilities):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{where}/prob_trace[{index}] is not a 3-action vector")
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}/prob_trace[{index}] is not numeric") from exc
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{where}/prob_trace[{index}] leaves [0,1]")
        if not math.isclose(sum(values), 1.0, abs_tol=2e-4):
            raise ValueError(f"{where}/prob_trace[{index}] does not sum to one")

    accounted = cell["simulated_dispatch_accounted_trace"]
    if (
        not isinstance(accounted, list)
        or len(accounted) != TRACE_LENGTH
        or any(value is not True for value in accounted)
    ):
        raise ValueError(f"{where} contains an unaccounted dispatch opportunity")

    components = cell["slca_component_trace"]
    if not isinstance(components, list) or len(components) != TRACE_LENGTH:
        raise ValueError(f"{where}/slca_component_trace is not complete")
    for index, record in enumerate(components):
        if not isinstance(record, dict) or set(record) != SLCA_COMPONENT_KEYS:
            raise ValueError(f"{where}/slca_component_trace[{index}] schema mismatch")
        if record.get("action_family") != ACTION_FAMILIES[actions[index]]:
            raise ValueError(
                f"{where}/slca_component_trace[{index}] action mismatch"
            )
        for key in SLCA_COMPONENT_KEYS - {"action_family"}:
            try:
                value = float(record[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{where}/slca_component_trace[{index}]/{key} is not numeric"
                ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{where}/slca_component_trace[{index}]/{key} leaves [0,1]"
                )
        if not math.isclose(
            float(record["composite_attenuated"]),
            numeric["slca_trace"][index],
            abs_tol=2e-4,
        ):
            raise ValueError(
                f"{where}/slca_component_trace[{index}] disagrees with slca_trace"
            )

    hours = numeric["hours"]
    if not math.isclose(hours[0], 0.0, abs_tol=1e-9) or any(
        not math.isclose(hours[index] - hours[index - 1], 0.25, abs_tol=1e-9)
        for index in range(1, TRACE_LENGTH)
    ):
        raise ValueError(f"{where}/hours does not use the exact 15-minute grid")
    for field in BOUNDED_FIELDS:
        if any(value < 0.0 or value > 1.0 for value in numeric[field]):
            raise ValueError(f"{where}/{field} leaves [0,1]")
    for field in NONNEGATIVE_FIELDS:
        if any(value < 0.0 for value in numeric[field]):
            raise ValueError(f"{where}/{field} contains a negative value")
    for field in ("rho_policy_observed_trace", "rho_outcome_environmental_trace"):
        values = numeric[field]
        if any(values[index] + 1e-12 < values[index - 1] for index in range(1, TRACE_LENGTH)):
            raise ValueError(f"{where}/{field} is not monotone")
    for alias, explicit in (
        ("rho_trace", "rho_policy_observed_trace"),
        ("temp_trace", "temp_policy_observed_trace"),
        ("rh_trace", "rh_policy_observed_trace"),
        ("inventory_trace", "inventory_policy_observed_trace"),
        ("demand_trace", "demand_forecast_policy_observed_trace"),
    ):
        if cell[alias] != cell[explicit]:
            raise ValueError(f"{where} legacy alias {alias} is ambiguous")
    for index, (ari, waste, social, rho) in enumerate(zip(
        numeric["ari_trace"], numeric["waste_trace"], numeric["slca_trace"],
        numeric["rho_outcome_environmental_trace"], strict=True,
    )):
        expected = (1.0 - waste) * social * (1.0 - rho)
        if not math.isclose(ari, expected, rel_tol=2e-3, abs_tol=2e-4):
            raise ValueError(f"{where}/ari_trace[{index}] violates the ARI equation")
