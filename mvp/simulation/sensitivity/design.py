"""Reproducible 100-point seed-locked Latin-hypercube design and manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .parameters import PARAMETERS, derived_values, registry_as_dict


DESIGN_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DESIGN_SEED = 20260828
PRIMARY_MODES: tuple[str, ...] = (
    "static",
    "hybrid_rl",
    "no_pinn",
    "no_slca",
    "no_context",
    "mcp_only",
    "pirag_only",
    "agribrain",
)
STRESSORS: tuple[str, ...] = (
    "sensor_noise",
    "missing_data",
    "telemetry_delay",
    "mcp_fault_injection",
    "compounded",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_for(design_seed: int, label: str) -> int:
    material = f"agribrain-lhs-v1|{int(design_seed)}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def load_locked_protocol(path: Path | str) -> dict[str, Any]:
    protocol_path = Path(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_rerun":
        raise ValueError("structural design requires the locked pre-run protocol")
    if tuple(protocol.get("primary_modes", ())) != PRIMARY_MODES:
        raise ValueError("primary mode order does not match the locked protocol")
    h3 = protocol.get("hypotheses", {}).get("h3", {})
    if tuple(h3.get("stressors", ())) != STRESSORS:
        raise ValueError("H3 stressor order does not match the locked protocol")
    declared = protocol.get("counts", {}).get("structural_sensitivity", {})
    expected = {
        "latin_hypercube_points": 100,
        "retained_cells": 6500,
        "executed_episodes": 24500,
        "simulated_steps": 7056000,
    }
    for key, value in expected.items():
        if declared.get(key) != value:
            raise ValueError(
                f"locked protocol structural count {key!r} is "
                f"{declared.get(key)!r}, expected {value}"
            )
    return protocol


def _latin_column(n_points: int, design_seed: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(_seed_for(design_seed, f"factor|{key}"))
    strata = rng.permutation(n_points)
    within = rng.random(n_points)
    return (strata.astype(float) + within) / float(n_points)


def _balanced_seed_assignment(
    seeds: tuple[int, ...], n_points: int, design_seed: int,
) -> list[int]:
    if not seeds:
        raise ValueError("the locked seed panel is empty")
    if n_points % len(seeds) != 0:
        raise ValueError(
            "seed-locked LHS requires n_points to be an exact multiple of "
            "the declared seed-panel size"
        )
    slots = np.repeat(np.asarray(seeds, dtype=np.int64), n_points // len(seeds))
    rng = np.random.default_rng(_seed_for(design_seed, "balanced-seed-assignment"))
    rng.shuffle(slots)
    return [int(value) for value in slots]


def build_design(
    protocol: Mapping[str, Any], *, design_seed: int = DEFAULT_DESIGN_SEED,
) -> dict[str, Any]:
    """Build the exact locked 100-point structural LHS in memory.

    Every one of the 20 confirmatory seeds is assigned to exactly five design
    points.  A given assigned seed keeps the simulator's normal
    ``(seed, scenario, episode_index, stream)`` keys; the design point is not
    injected into an exogenous random-stream key.  Consequently changes across
    factor settings are not confounded by gratuitous random-stream changes.
    """

    n_points = int(
        protocol["counts"]["structural_sensitivity"]["latin_hypercube_points"]
    )
    if n_points != 100:
        raise ValueError("the locked structural design must contain exactly 100 points")
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    assigned_seeds = _balanced_seed_assignment(seeds, n_points, design_seed)
    columns = {
        parameter.key: _latin_column(n_points, design_seed, parameter.key)
        for parameter in PARAMETERS
    }
    points: list[dict[str, Any]] = []
    for index in range(n_points):
        unit_coordinates = {
            parameter.key: round(float(columns[parameter.key][index]), 15)
            for parameter in PARAMETERS
        }
        values = {
            parameter.key: parameter.transform(columns[parameter.key][index])
            for parameter in PARAMETERS
        }
        # Round continuous coordinates only for cross-platform canonical JSON;
        # integer factors remain integers.
        values = {
            key: (value if isinstance(value, int) else round(float(value), 12))
            for key, value in values.items()
        }
        point_parameters_hash = canonical_sha256(values)
        points.append({
            "point_index": index,
            "point_id": f"lhs_{index:03d}",
            "seed": assigned_seeds[index],
            "lhs_unit_coordinates": unit_coordinates,
            "parameters": values,
            "derived_parameters": derived_values(values),
            "parameters_sha256": point_parameters_hash,
        })

    payload: dict[str, Any] = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "range_semantics": (
            "space-filling deterministic bounds; not probability "
            "distributions, priors, or confidence intervals"
        ),
        "design_method": "independently permuted Latin hypercube with within-stratum jitter",
        "design_seed": int(design_seed),
        "n_points": n_points,
        "seed_locking": {
            "assignment": "balanced deterministic assignment",
            "each_declared_seed_used": n_points // len(seeds),
            "simulator_stream_key": [
                "seed", "scenario", "episode_index", "stream",
            ],
            "design_point_excluded_from_stream_key": True,
        },
        "parameter_registry": registry_as_dict(),
        "points": points,
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


def build_structural_accounting(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return independently checkable retained/executed/step accounting."""

    n_points = int(
        protocol["counts"]["structural_sensitivity"]["latin_hypercube_points"]
    )
    scenarios = tuple(str(value) for value in protocol["scenarios"])
    learned_episodes = (
        int(protocol["episode_protocol"]["learned_adaptation_episodes"])
        + int(protocol["episode_protocol"]["learned_frozen_evaluation_episodes"])
    )
    static_episodes = int(protocol["episode_protocol"]["static_evaluation_episodes"])
    steps = int(protocol["episode_protocol"]["steps_per_episode"])
    budgets = {
        mode: (static_episodes if mode == "static" else learned_episodes)
        for mode in PRIMARY_MODES
    }
    primary_retained_per_point = len(scenarios) * len(PRIMARY_MODES)
    primary_executed_per_point = len(scenarios) * sum(budgets.values())
    stress_retained_per_point = len(scenarios) * len(STRESSORS)
    stress_executed_per_point = (
        len(scenarios) * len(STRESSORS) * learned_episodes
    )
    accounting = {
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "n_design_points": n_points,
        "n_scenarios": len(scenarios),
        "primary_modes": list(PRIMARY_MODES),
        "stress_mode": "agribrain",
        "stressors": list(STRESSORS),
        "episode_budget_by_primary_mode": budgets,
        "episodes_per_stressed_agribrain_cell": learned_episodes,
        "steps_per_episode": steps,
        "per_design_point": {
            "primary_retained_cells": primary_retained_per_point,
            "primary_executed_episodes": primary_executed_per_point,
            "h3_stressed_retained_cells": stress_retained_per_point,
            "h3_stressed_executed_episodes": stress_executed_per_point,
            "total_retained_cells": (
                primary_retained_per_point + stress_retained_per_point
            ),
            "total_executed_episodes": (
                primary_executed_per_point + stress_executed_per_point
            ),
        },
        "total": {
            "retained_cells": n_points * (
                primary_retained_per_point + stress_retained_per_point
            ),
            "executed_episodes": n_points * (
                primary_executed_per_point + stress_executed_per_point
            ),
            "simulated_steps": n_points * (
                primary_executed_per_point + stress_executed_per_point
            ) * steps,
        },
        "nominal_agribrain_reference": (
            "reuse the primary-panel agribrain endpoint for the same point, "
            "scenario, and seed; do not execute a duplicate nominal H3 arm"
        ),
        "terminology_warning": (
            "retained cells are not executed episodes; this design contains "
            "6,500 retained cells and 24,500 complete episode executions"
        ),
    }
    declared = protocol["counts"]["structural_sensitivity"]
    expected = accounting["total"]
    for key in ("retained_cells", "executed_episodes", "simulated_steps"):
        if int(declared[key]) != int(expected[key]):
            raise ValueError(
                f"computed structural {key}={expected[key]} conflicts with "
                f"locked protocol value {declared[key]}"
            )
    return accounting


def build_task_manifest(
    design: Mapping[str, Any], protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build 3,000 independently runnable scenario-panel task records."""

    if int(design.get("n_points", 0)) != 100:
        raise ValueError("task manifest requires the complete 100-point design")
    scenarios = tuple(str(value) for value in protocol["scenarios"])
    accounting = build_structural_accounting(protocol)
    primary_episodes = sum(
        accounting["episode_budget_by_primary_mode"].values()
    )
    stress_episodes = int(accounting["episodes_per_stressed_agribrain_cell"])
    steps = int(accounting["steps_per_episode"])
    tasks: list[dict[str, Any]] = []
    task_index = 0
    for point in design["points"]:
        point_id = str(point["point_id"])
        common = {
            "design_sha256": str(design["design_sha256"]),
            "point_index": int(point["point_index"]),
            "point_id": point_id,
            "seed": int(point["seed"]),
            "parameters_sha256": str(point["parameters_sha256"]),
        }
        for scenario in scenarios:
            task = {
                **common,
                "task_index": task_index,
                "task_id": f"{point_id}__{scenario}__primary",
                "panel": "primary",
                "scenario": scenario,
                "modes": list(PRIMARY_MODES),
                "retained_cells": len(PRIMARY_MODES),
                "executed_episodes": primary_episodes,
                "simulated_steps": primary_episodes * steps,
                "output_relpath": f"tasks/{point_id}/{scenario}__primary.json",
            }
            task["task_sha256"] = canonical_sha256(task)
            tasks.append(task)
            task_index += 1
        for scenario in scenarios:
            for stressor in STRESSORS:
                task = {
                    **common,
                    "task_index": task_index,
                    "task_id": f"{point_id}__{scenario}__h3__{stressor}",
                    "panel": "h3_stressed",
                    "scenario": scenario,
                    "stressor": stressor,
                    "modes": ["agribrain"],
                    "nominal_reference_task_id": (
                        f"{point_id}__{scenario}__primary"
                    ),
                    "retained_cells": 1,
                    "executed_episodes": stress_episodes,
                    "simulated_steps": stress_episodes * steps,
                    "output_relpath": (
                        f"tasks/{point_id}/{scenario}__h3__{stressor}.json"
                    ),
                }
                task["task_sha256"] = canonical_sha256(task)
                tasks.append(task)
                task_index += 1
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "design_sha256": str(design["design_sha256"]),
        "n_tasks": len(tasks),
        "task_granularity": (
            "one primary seven-mode scenario panel or one stressed "
            "AGRI-BRAIN scenario-stressor cell"
        ),
        "accounting": accounting,
        "tasks": tasks,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_task_manifest(manifest, protocol)
    return manifest


def validate_design(design: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    if design.get("analysis_label") != "structural sensitivity":
        raise ValueError("design is not labelled structural sensitivity")
    if design.get("probability_interpretation") is not False:
        raise ValueError("structural design must explicitly reject probability interpretation")
    unsigned = dict(design)
    digest = unsigned.pop("design_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError("design SHA-256 does not match canonical content")
    points = list(design.get("points", ()))
    if len(points) != 100:
        raise ValueError(f"expected 100 LHS points, found {len(points)}")
    seeds = [int(point["seed"]) for point in points]
    declared_seeds = tuple(int(seed) for seed in protocol["seeds"])
    expected_repeats = len(points) // len(declared_seeds)
    counts = {seed: seeds.count(seed) for seed in declared_seeds}
    if any(count != expected_repeats for count in counts.values()):
        raise ValueError(f"seed assignment is not balanced: {counts}")
    n = len(points)
    for parameter in PARAMETERS:
        strata = sorted(
            int(float(point["lhs_unit_coordinates"][parameter.key]) * n)
            for point in points
        )
        if strata != list(range(n)):
            raise ValueError(f"{parameter.key}: LHS strata are not a permutation")
        for point in points:
            value = point["parameters"][parameter.key]
            if not parameter.lower <= float(value) <= parameter.upper:
                raise ValueError(f"{parameter.key}: transformed value outside bounds")
            if point["parameters_sha256"] != canonical_sha256(point["parameters"]):
                raise ValueError(f"{point['point_id']}: parameter hash mismatch")


def validate_task_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any],
) -> None:
    if manifest.get("analysis_label") != "structural sensitivity":
        raise ValueError("task manifest is not labelled structural sensitivity")
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError("manifest SHA-256 does not match canonical content")
    tasks = list(manifest.get("tasks", ()))
    if len(tasks) != 3000:
        raise ValueError(f"expected 3,000 structural tasks, found {len(tasks)}")
    ids = [str(task["task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task ids are not unique")
    if [int(task["task_index"]) for task in tasks] != list(range(len(tasks))):
        raise ValueError("task indices must be contiguous and ordered")
    for task in tasks:
        unsigned_task = dict(task)
        task_digest = unsigned_task.pop("task_sha256", None)
        if task_digest != canonical_sha256(unsigned_task):
            raise ValueError(f"{task['task_id']}: task SHA-256 mismatch")
    actual = {
        "retained_cells": sum(int(task["retained_cells"]) for task in tasks),
        "executed_episodes": sum(int(task["executed_episodes"]) for task in tasks),
        "simulated_steps": sum(int(task["simulated_steps"]) for task in tasks),
    }
    expected = manifest["accounting"]["total"]
    if actual != expected:
        raise ValueError(f"task totals {actual} conflict with accounting {expected}")
    declared = protocol["counts"]["structural_sensitivity"]
    if any(int(actual[key]) != int(declared[key]) for key in actual):
        raise ValueError("task totals conflict with the locked protocol")
