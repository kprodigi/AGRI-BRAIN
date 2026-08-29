"""Descriptive rank/sign/margin stability and PRCC/Spearman analysis.

Each LHS row is paired with one locked seed, so the 100 rows are structural
design points, not 100 independent draws from a parameter population.  This
module therefore reports descriptive stability over the declared factor box.
It does not repeat the confirmatory H1/H2 tests or the 20-seed H3 TOST at each
point, and it never attaches probability coverage to the parameter bounds.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import stats

from .design import PRIMARY_MODES, STRESSORS, canonical_sha256
from .parameters import PARAMETERS


H2_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("mcp_only_minus_no_context", "mcp_only", "no_context"),
    ("pirag_only_minus_no_context", "pirag_only", "no_context"),
    ("agribrain_minus_mcp_only", "agribrain", "mcp_only"),
    ("agribrain_minus_pirag_only", "agribrain", "pirag_only"),
)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires a non-empty finite vector")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _fraction(predicate: Iterable[bool]) -> float:
    values = [bool(value) for value in predicate]
    if not values:
        raise ValueError("fraction requires at least one value")
    return float(sum(values) / len(values))


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _residualize(values: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, int]:
    if controls.ndim != 2:
        raise ValueError("controls must be a matrix")
    design = np.column_stack([np.ones(len(values)), controls])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients, int(np.linalg.matrix_rank(design))


def _prcc(
    parameter_ranks: np.ndarray,
    response: np.ndarray,
    parameter_index: int,
    nuisance_controls: np.ndarray,
) -> tuple[float, float]:
    n, k = parameter_ranks.shape
    y_rank = stats.rankdata(response, method="average").astype(float)
    x_rank = parameter_ranks[:, parameter_index]
    controls = np.column_stack([
        np.delete(parameter_ranks, parameter_index, axis=1),
        nuisance_controls,
    ])
    x_resid, control_rank = _residualize(x_rank, controls)
    y_resid, _ = _residualize(y_rank, controls)
    if np.std(x_resid) <= 1e-14 or np.std(y_resid) <= 1e-14:
        return float("nan"), float("nan")
    coefficient = float(np.corrcoef(x_resid, y_resid)[0, 1])
    coefficient = float(np.clip(coefficient, -1.0, 1.0))
    # Partial-correlation df = n - q - 2 where q is the number of effective
    # control columns. ``control_rank`` includes the intercept, hence n-rank-1.
    df = n - control_rank - 1
    if df <= 0 or abs(coefficient) >= 1.0:
        p_value = 0.0 if abs(coefficient) >= 1.0 else float("nan")
    else:
        statistic = coefficient * math.sqrt(df / max(1e-15, 1.0 - coefficient**2))
        p_value = float(2.0 * stats.t.sf(abs(statistic), df))
    return coefficient, p_value


def _associations(
    parameter_matrix: np.ndarray,
    response_names: list[str],
    response_matrix: np.ndarray,
    seed_vector: np.ndarray,
) -> dict[str, Any]:
    if parameter_matrix.shape[0] != response_matrix.shape[0]:
        raise ValueError("parameter and response matrices have different row counts")
    if seed_vector.shape != (parameter_matrix.shape[0],):
        raise ValueError("seed vector does not match the design rows")
    seed_levels = sorted({int(seed) for seed in seed_vector})
    # Drop the first seed indicator because the residualizer adds an intercept.
    seed_dummies = np.column_stack([
        (seed_vector == seed).astype(float) for seed in seed_levels[1:]
    ]) if len(seed_levels) > 1 else np.empty((len(seed_vector), 0), dtype=float)
    n_control_columns = parameter_matrix.shape[1] - 1 + seed_dummies.shape[1]
    if parameter_matrix.shape[0] <= n_control_columns + 2:
        raise ValueError(
            "PRCC requires more design points than effective factor and seed controls plus two"
        )
    parameter_ranks = np.column_stack([
        stats.rankdata(parameter_matrix[:, index], method="average")
        for index in range(parameter_matrix.shape[1])
    ]).astype(float)
    records: list[dict[str, Any]] = []
    def finite_or_none(value: Any) -> float | None:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    for response_index, response_name in enumerate(response_names):
        response = response_matrix[:, response_index]
        for parameter_index, parameter in enumerate(PARAMETERS):
            factor = parameter_matrix[:, parameter_index]
            if np.ptp(factor) <= 0.0 or np.ptp(response) <= 0.0:
                spearman_statistic = float("nan")
                spearman_p_value = float("nan")
            else:
                spearman = stats.spearmanr(factor, response)
                spearman_statistic = float(spearman.statistic)
                spearman_p_value = float(spearman.pvalue)
            prcc_coefficient, prcc_p = _prcc(
                parameter_ranks, response, parameter_index, seed_dummies,
            )
            records.append({
                "response": response_name,
                "parameter": parameter.key,
                "n_design_points": int(len(response)),
                "spearman_rho": finite_or_none(spearman_statistic),
                "spearman_p_value": finite_or_none(spearman_p_value),
                "prcc": finite_or_none(prcc_coefficient),
                "prcc_p_value": finite_or_none(prcc_p),
            })
    for method in ("spearman", "prcc"):
        p_key = f"{method}_p_value"
        finite_indices = [
            index for index, record in enumerate(records)
            if record[p_key] is not None
        ]
        adjusted = _holm_adjust([
            float(records[index][p_key]) for index in finite_indices
        ])
        for index, value in zip(finite_indices, adjusted):
            records[index][f"{method}_p_value_holm_all_associations"] = value
        for index, record in enumerate(records):
            record.setdefault(f"{method}_p_value_holm_all_associations", None)
    return {
        "methods": {
            "spearman": "pairwise rank correlation",
            "prcc": (
                "partial correlation of ranked factor and ranked response "
                "after least-squares residualization on all other ranked "
                "factors and fixed indicators for the balanced locked-seed blocks"
            ),
            "seed_blocking": {
                "n_seed_levels": len(seed_levels),
                "reference_seed": seed_levels[0],
                "n_seed_indicator_controls": int(seed_dummies.shape[1]),
                "spearman_is_unadjusted_bivariate": True,
                "prcc_controls_seed_indicators": True,
            },
            "multiplicity": (
                "Holm adjustment separately across all parameter-response "
                "associations for each method"
            ),
        },
        "records": records,
    }


def _validate_payload(
    payload: Mapping[str, Any], task: Mapping[str, Any], source_commit: str,
) -> None:
    unsigned = dict(payload)
    digest = unsigned.pop("result_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError(f"{task['task_id']}: result SHA-256 mismatch")
    expected = {
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "source_commit": source_commit,
        "task_sha256": task["task_sha256"],
        "task_id": task["task_id"],
        "task_index": task["task_index"],
        "point_id": task["point_id"],
        "seed": task["seed"],
        "scenario": task["scenario"],
        "panel": task["panel"],
        "parameters_sha256": task["parameters_sha256"],
        "retained_cells": task["retained_cells"],
        "executed_episodes": task["executed_episodes"],
        "simulated_steps": task["simulated_steps"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{task['task_id']}: result field {key!r}={payload.get(key)!r}, "
                f"expected {value!r}"
            )


def analyze_payloads(
    design: Mapping[str, Any],
    manifest: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Analyse a complete, already hash-validated structural result panel."""

    tasks = list(manifest["tasks"])
    expected_ids = {str(task["task_id"]) for task in tasks}
    if set(payloads) != expected_ids:
        raise ValueError(
            "result task set is incomplete or unexpected: "
            f"missing={len(expected_ids-set(payloads))}, "
            f"unexpected={len(set(payloads)-expected_ids)}"
        )
    primary: dict[tuple[int, str], Mapping[str, Any]] = {}
    stressed: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for task in tasks:
        payload = payloads[str(task["task_id"])]
        _validate_payload(payload, task, source_commit)
        point = int(task["point_index"])
        scenario = str(task["scenario"])
        if task["panel"] == "primary":
            result_modes = payload.get("results", {})
            if set(result_modes) != set(PRIMARY_MODES):
                raise ValueError(f"{task['task_id']}: incomplete primary modes")
            primary[(point, scenario)] = result_modes
        else:
            result_modes = payload.get("results", {})
            if set(result_modes) != {"agribrain"}:
                raise ValueError(f"{task['task_id']}: invalid stressed modes")
            stressed[(point, scenario, str(task["stressor"]))] = result_modes["agribrain"]

    scenarios = tuple(
        dict.fromkeys(str(task["scenario"]) for task in tasks if task["panel"] == "primary")
    )
    n_points = int(design["n_points"])
    if len(primary) != n_points * len(scenarios):
        raise ValueError("primary result panel is not rectangular")
    if len(stressed) != n_points * len(scenarios) * len(STRESSORS):
        raise ValueError("H3 stressed result panel is not rectangular")

    # Fail closed on common-random-number pairing and stress exposure before
    # computing any stability statistic.
    exposure_report: dict[str, dict[str, int]] = {
        stressor: {"cells": 0, "cells_with_nonzero_exposure": 0}
        for stressor in STRESSORS
    }
    for point_index in range(n_points):
        for scenario in scenarios:
            nominal = primary[(point_index, scenario)]["agribrain"]
            for stressor in STRESSORS:
                stress = stressed[(point_index, scenario, stressor)]
                if (
                    stress.get("latent_environment_sha256")
                    != nominal.get("latent_environment_sha256")
                ):
                    raise ValueError(
                        f"latent truth mismatch at point={point_index}, "
                        f"scenario={scenario}, stressor={stressor}"
                    )
                exposure_report[stressor]["cells"] += 1
                if stressor in {"mcp_fault_injection", "compounded"}:
                    exposed = int(stress.get("fault_injection_trigger_steps", 0)) > 0
                else:
                    exposed = (
                        stress.get("observed_policy_input_sha256")
                        != nominal.get("observed_policy_input_sha256")
                    )
                exposure_report[stressor]["cells_with_nonzero_exposure"] += int(exposed)
                if not exposed:
                    raise ValueError(
                        f"zero stress exposure at point={point_index}, "
                        f"scenario={scenario}, stressor={stressor}"
                    )

    # Rank stability by scenario and for the across-scenario mean ARI.
    rank_panels: dict[str, Any] = {}
    rank_vectors_by_panel: dict[str, list[np.ndarray]] = {
        scenario: [] for scenario in scenarios
    }
    rank_vectors_by_panel["pooled_scenario_mean"] = []
    orderings_by_panel: dict[str, list[tuple[str, ...]]] = {
        panel: [] for panel in rank_vectors_by_panel
    }
    for point_index in range(n_points):
        pooled = {
            mode: float(np.mean([
                primary[(point_index, scenario)][mode]["ari"]
                for scenario in scenarios
            ]))
            for mode in PRIMARY_MODES
        }
        for panel in (*scenarios, "pooled_scenario_mean"):
            values = (
                pooled if panel == "pooled_scenario_mean" else {
                    mode: float(primary[(point_index, panel)][mode]["ari"])
                    for mode in PRIMARY_MODES
                }
            )
            vector = np.asarray([values[mode] for mode in PRIMARY_MODES], dtype=float)
            ranks = stats.rankdata(-vector, method="average")
            rank_vectors_by_panel[panel].append(ranks)
            ordering = tuple(sorted(
                PRIMARY_MODES,
                key=lambda mode: (-values[mode], PRIMARY_MODES.index(mode)),
            ))
            orderings_by_panel[panel].append(ordering)
    for panel in rank_vectors_by_panel:
        ranks = np.vstack(rank_vectors_by_panel[panel])
        ordering_counts = Counter(orderings_by_panel[panel])
        modal, modal_count = ordering_counts.most_common(1)[0]
        rank_panels[panel] = {
            "modal_complete_ordering": list(modal),
            "modal_complete_ordering_fraction": float(modal_count / n_points),
            "n_unique_complete_orderings": len(ordering_counts),
            "all_points_identical_complete_ordering": len(ordering_counts) == 1,
            "by_mode": {
                mode: {
                    "mean_rank": float(np.mean(ranks[:, index])),
                    "rank_std": float(np.std(ranks[:, index], ddof=1)),
                    "best_rank": float(np.min(ranks[:, index])),
                    "worst_rank": float(np.max(ranks[:, index])),
                    "first_place_fraction": float(np.mean(ranks[:, index] == 1.0)),
                }
                for index, mode in enumerate(PRIMARY_MODES)
            },
        }

    h1: dict[str, Any] = {}
    h2: dict[str, Any] = {}
    h3: dict[str, Any] = {}
    point_responses: list[dict[str, float]] = []
    point_h3_all_inside: list[bool] = []
    for scenario in scenarios:
        differences = np.asarray([
            float(primary[(point, scenario)]["agribrain"]["ari"])
            - float(primary[(point, scenario)]["no_context"]["ari"])
            for point in range(n_points)
        ])
        h1[scenario] = {
            "contrast": "agribrain - no_context",
            "descriptive_over_structural_points": _summary(differences),
            "positive_sign_fraction": float(np.mean(differences > 0.0)),
            "point_difference_above_0p005_fraction": float(np.mean(differences > 0.005)),
        }
        h2[scenario] = {}
        for name, left, right in H2_CONTRASTS:
            contrast = np.asarray([
                float(primary[(point, scenario)][left]["ari"])
                - float(primary[(point, scenario)][right]["ari"])
                for point in range(n_points)
            ])
            h2[scenario][name] = {
                "descriptive_over_structural_points": _summary(contrast),
                "positive_sign_fraction": float(np.mean(contrast > 0.0)),
            }
        synergy = np.asarray([
            float(primary[(point, scenario)]["agribrain"]["ari"])
            - float(primary[(point, scenario)]["mcp_only"]["ari"])
            - float(primary[(point, scenario)]["pirag_only"]["ari"])
            + float(primary[(point, scenario)]["no_context"]["ari"])
            for point in range(n_points)
        ])
        h2[scenario]["synergy_full_minus_mcp_minus_retrieval_plus_no_context"] = {
            "descriptive_over_structural_points": _summary(synergy),
            "positive_sign_fraction": float(np.mean(synergy > 0.0)),
            "separate_from_h2_universal_directional_family": True,
        }

    for scenario in scenarios:
        h3[scenario] = {}
        for stressor in STRESSORS:
            differences = np.asarray([
                float(stressed[(point, scenario, stressor)]["ari"])
                - float(primary[(point, scenario)]["agribrain"]["ari"])
                for point in range(n_points)
            ])
            h3[scenario][stressor] = {
                "contrast": "stressed agribrain - primary nominal agribrain",
                "descriptive_over_structural_points": _summary(differences),
                "inside_strict_0p01_margin_fraction": float(
                    np.mean(np.abs(differences) < 0.01)
                ),
                "max_absolute_delta": float(np.max(np.abs(differences))),
                "all_cells_have_nonzero_exposure": (
                    exposure_report[stressor]["cells_with_nonzero_exposure"]
                    == exposure_report[stressor]["cells"]
                ),
            }

    response_names = [
        "agribrain_mean_ari",
        "agribrain_mean_waste",
        "agribrain_mean_slca",
        "agribrain_mean_carbon",
        "h1_mean_contrast",
        "h1_min_scenario_contrast",
        *[f"h2_mean_{name}" for name, _, _ in H2_CONTRASTS],
        "h2_mean_synergy",
        "h3_max_absolute_delta",
        "h3_mean_absolute_delta",
        "h3_fraction_inside_strict_0p01_margin",
    ]
    for point in range(n_points):
        agri_cells = [primary[(point, scenario)]["agribrain"] for scenario in scenarios]
        h1_values = [
            float(primary[(point, scenario)]["agribrain"]["ari"])
            - float(primary[(point, scenario)]["no_context"]["ari"])
            for scenario in scenarios
        ]
        responses: dict[str, float] = {
            "agribrain_mean_ari": float(np.mean([cell["ari"] for cell in agri_cells])),
            "agribrain_mean_waste": float(np.mean([cell["waste"] for cell in agri_cells])),
            "agribrain_mean_slca": float(np.mean([cell["slca"] for cell in agri_cells])),
            "agribrain_mean_carbon": float(np.mean([cell["carbon"] for cell in agri_cells])),
            "h1_mean_contrast": float(np.mean(h1_values)),
            "h1_min_scenario_contrast": float(np.min(h1_values)),
        }
        for name, left, right in H2_CONTRASTS:
            responses[f"h2_mean_{name}"] = float(np.mean([
                float(primary[(point, scenario)][left]["ari"])
                - float(primary[(point, scenario)][right]["ari"])
                for scenario in scenarios
            ]))
        responses["h2_mean_synergy"] = float(np.mean([
            float(primary[(point, scenario)]["agribrain"]["ari"])
            - float(primary[(point, scenario)]["mcp_only"]["ari"])
            - float(primary[(point, scenario)]["pirag_only"]["ari"])
            + float(primary[(point, scenario)]["no_context"]["ari"])
            for scenario in scenarios
        ]))
        h3_abs = np.asarray([
            abs(
                float(stressed[(point, scenario, stressor)]["ari"])
                - float(primary[(point, scenario)]["agribrain"]["ari"])
            )
            for scenario in scenarios for stressor in STRESSORS
        ])
        responses["h3_max_absolute_delta"] = float(np.max(h3_abs))
        responses["h3_mean_absolute_delta"] = float(np.mean(h3_abs))
        responses["h3_fraction_inside_strict_0p01_margin"] = float(
            np.mean(h3_abs < 0.01)
        )
        point_responses.append(responses)
        point_h3_all_inside.append(bool(np.all(h3_abs < 0.01)))

    parameter_matrix = np.asarray([
        [float(point["parameters"][parameter.key]) for parameter in PARAMETERS]
        for point in design["points"]
    ], dtype=float)
    response_matrix = np.asarray([
        [record[name] for name in response_names] for record in point_responses
    ], dtype=float)
    associations = _associations(
        parameter_matrix,
        response_names,
        response_matrix,
        np.asarray([point["seed"] for point in design["points"]], dtype=int),
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "interpretation_boundary": (
            "Fractions and quantiles describe stability over the prespecified "
            "100-point factor box. They are not probabilities over a parameter "
            "population and do not replace the seed-level confirmatory H1/H2 "
            "tests or paired H3 TOST."
        ),
        "source_commit": source_commit,
        "design_sha256": design["design_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "n_design_points": n_points,
        "n_parameters": len(PARAMETERS),
        "rank_stability_method": {
            "numeric_ranks": "descending ARI with average ranks for ties",
            "complete_ordering_tie_break": (
                "locked PRIMARY_MODES order, used only to serialize a complete ordering"
            ),
        },
        "rank_stability": rank_panels,
        "h1_sign_stability": h1,
        "h2_sign_stability": h2,
        "h3_margin_stability": {
            "strict_margin": 0.01,
            "cells": h3,
            "fraction_design_points_all_25_cells_inside_margin": _fraction(
                point_h3_all_inside
            ),
            "stress_exposure": exposure_report,
        },
        "responses_by_point": [
            {
                "point_id": design["points"][index]["point_id"],
                "seed": design["points"][index]["seed"],
                **responses,
            }
            for index, responses in enumerate(point_responses)
        ],
        "rank_associations": associations,
    }
    report["analysis_sha256"] = canonical_sha256(report)
    return report


def analyze_run(run_plan_path: Path | str) -> dict[str, Any]:
    """Load and analyse a complete task tree anchored by ``run_plan.json``."""

    # Lazy import avoids a circular dependency when the CLI's analyze command
    # imports this module.
    from .run_structural_sensitivity import (
        _load_plan_bundle,
        validate_completed_results,
    )

    plan_path = Path(run_plan_path).resolve()
    status = validate_completed_results(plan_path)
    if status["status"] != "complete":
        raise RuntimeError(
            "BLOCK: structural sensitivity analysis requires all 3,000 "
            f"hash-valid tasks; missing {status['n_missing_tasks']}"
        )
    plan, _protocol, design, manifest = _load_plan_bundle(plan_path)
    payloads = {
        str(task["task_id"]): json.loads(
            (plan_path.parent / task["output_relpath"]).read_text(encoding="utf-8")
        )
        for task in manifest["tasks"]
    }
    return analyze_payloads(
        design,
        manifest,
        payloads,
        source_commit=str(plan["source_commit"]),
    )
