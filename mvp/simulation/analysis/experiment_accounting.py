"""Pure helpers for unambiguous benchmark episode accounting.

The publication distinguishes a retained endpoint cell from an executed
72-hour episode.  Learned arms may execute several episodes while retaining
only the final evaluation record, so multiplying scenarios, modes, and seeds
does not in general give the number of executed episodes.

This module deliberately has no simulator imports.  Callers supply the
declared episode budget for each configured mode, which keeps the arithmetic
testable without importing the backend stack and avoids creating another
source of truth for mode capabilities.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


PRIMARY_PUBLICATION_MODES: tuple[str, ...] = (
    "static",
    "hybrid_rl",
    "no_pinn",
    "no_slca",
    "no_context",
    "mcp_only",
    "pirag_only",
    "agribrain",
)


def build_episode_accounting(
    *,
    scenarios: Iterable[str],
    configured_modes: Iterable[str],
    episode_budget_by_mode: Mapping[str, int],
    n_seeds: int,
    steps_per_episode: int = 288,
    primary_modes: Iterable[str] = PRIMARY_PUBLICATION_MODES,
) -> dict[str, Any]:
    """Return retained-cell and executed-episode counts for one run design.

    A retained cell is one final scenario-mode-seed endpoint.  Executed
    episodes include every adaptation/training episode used to produce that
    endpoint.  Fixed arms may therefore contribute one retained cell and one
    execution while learned arms contribute one retained cell and several
    executions.
    """

    scenario_list = tuple(scenarios)
    configured_list = tuple(configured_modes)
    primary_list = tuple(primary_modes)
    if len(set(scenario_list)) != len(scenario_list) or not scenario_list:
        raise ValueError("scenarios must be a non-empty unique sequence")
    if len(set(configured_list)) != len(configured_list) or not configured_list:
        raise ValueError("configured_modes must be a non-empty unique sequence")
    if len(set(primary_list)) != len(primary_list) or not primary_list:
        raise ValueError("primary_modes must be a non-empty unique sequence")
    if not isinstance(n_seeds, int) or isinstance(n_seeds, bool) or n_seeds <= 0:
        raise ValueError("n_seeds must be a positive integer")
    if (
        not isinstance(steps_per_episode, int)
        or isinstance(steps_per_episode, bool)
        or steps_per_episode <= 0
    ):
        raise ValueError("steps_per_episode must be a positive integer")

    missing_budgets = [m for m in configured_list if m not in episode_budget_by_mode]
    if missing_budgets:
        raise ValueError(f"missing episode budgets for modes: {missing_budgets}")
    missing_primary = [m for m in primary_list if m not in configured_list]
    if missing_primary:
        raise ValueError(f"primary modes are not configured: {missing_primary}")

    budgets: dict[str, int] = {}
    for mode in configured_list:
        budget = episode_budget_by_mode[mode]
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"episode budget for {mode!r} must be a positive integer")
        budgets[mode] = int(budget)

    n_scenarios = len(scenario_list)
    retained_all = n_seeds * n_scenarios * len(configured_list)
    retained_primary = n_seeds * n_scenarios * len(primary_list)
    executed_all = n_seeds * n_scenarios * sum(budgets.values())
    executed_primary = n_seeds * n_scenarios * sum(
        budgets[mode] for mode in primary_list
    )
    return {
        "unit_definition": {
            "retained_endpoint_cell": (
                "one final scenario-mode-seed endpoint record"
            ),
            "executed_episode": "one complete simulated 72-hour episode",
            "inferential_unit": "seed",
        },
        "n_seeds": n_seeds,
        "n_scenarios": n_scenarios,
        "steps_per_episode": steps_per_episode,
        "configured_modes": list(configured_list),
        "primary_modes": list(primary_list),
        "episode_budget_by_mode": budgets,
        "retained_endpoint_cells_all_configured_modes": retained_all,
        "executed_episodes_all_configured_modes": executed_all,
        "simulated_decision_steps_all_configured_modes": (
            executed_all * steps_per_episode
        ),
        "retained_endpoint_cells_primary": retained_primary,
        "executed_episodes_primary": executed_primary,
        "simulated_decision_steps_primary": executed_primary * steps_per_episode,
        "terminology_warning": (
            "Retained endpoint cells are not executed episodes when any mode "
            "has an episode budget greater than one."
        ),
    }


def validate_episode_accounting(accounting: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when serialized accounting is self-contradictory."""

    required = {
        "n_seeds",
        "n_scenarios",
        "steps_per_episode",
        "configured_modes",
        "primary_modes",
        "episode_budget_by_mode",
        "retained_endpoint_cells_all_configured_modes",
        "executed_episodes_all_configured_modes",
        "simulated_decision_steps_all_configured_modes",
        "retained_endpoint_cells_primary",
        "executed_episodes_primary",
        "simulated_decision_steps_primary",
    }
    missing = sorted(required.difference(accounting))
    if missing:
        raise ValueError(f"episode accounting is missing fields: {missing}")
    expected = build_episode_accounting(
        scenarios=[f"scenario_{i}" for i in range(int(accounting["n_scenarios"]))],
        configured_modes=list(accounting["configured_modes"]),
        episode_budget_by_mode=dict(accounting["episode_budget_by_mode"]),
        n_seeds=int(accounting["n_seeds"]),
        steps_per_episode=int(accounting["steps_per_episode"]),
        primary_modes=list(accounting["primary_modes"]),
    )
    for field in required - {
        "configured_modes",
        "primary_modes",
        "episode_budget_by_mode",
    }:
        if accounting[field] != expected[field]:
            raise ValueError(
                f"episode accounting field {field!r} is inconsistent: "
                f"reported={accounting[field]!r}, expected={expected[field]!r}"
            )


def build_h3_episode_accounting(
    *,
    n_seeds: int,
    n_scenarios: int,
    n_stressors: int,
    episodes_per_condition: int,
    nominal_reference_reused: bool,
    steps_per_episode: int = 288,
) -> dict[str, Any]:
    """Return explicit AGRI-BRAIN-only confirmatory H3 run accounting."""

    for name, value in (
        ("n_seeds", n_seeds),
        ("n_scenarios", n_scenarios),
        ("n_stressors", n_stressors),
        ("episodes_per_condition", episodes_per_condition),
        ("steps_per_episode", steps_per_episode),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    stressed_cells = n_seeds * n_scenarios * n_stressors
    stressed_executions = stressed_cells * episodes_per_condition
    nominal_cells = n_seeds * n_scenarios
    nominal_executions = nominal_cells * episodes_per_condition
    incremental_nominal = 0 if nominal_reference_reused else nominal_executions
    incremental_executions = stressed_executions + incremental_nominal
    return {
        "confirmatory_method": "agribrain",
        "n_seeds": n_seeds,
        "n_scenarios": n_scenarios,
        "n_stressors": n_stressors,
        "episodes_per_condition": episodes_per_condition,
        "steps_per_episode": steps_per_episode,
        "formal_contrast_cells": n_scenarios * n_stressors,
        "retained_nominal_endpoint_cells": nominal_cells,
        "retained_stressed_endpoint_cells": stressed_cells,
        "nominal_reference_reused": bool(nominal_reference_reused),
        "executed_stressed_episodes": stressed_executions,
        "incremental_executed_nominal_episodes": incremental_nominal,
        "incremental_executed_episodes": incremental_executions,
        "incremental_simulated_decision_steps": (
            incremental_executions * steps_per_episode
        ),
    }
