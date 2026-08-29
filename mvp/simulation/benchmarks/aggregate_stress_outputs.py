#!/usr/bin/env python3
"""Combine scenario-parallel stress-suite outputs into canonical H3 files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

try:
    from ..analysis.experiment_accounting import build_h3_episode_accounting
except ImportError:
    import sys as _accounting_sys

    _ACCOUNTING_REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_ACCOUNTING_REPO_ROOT) not in _accounting_sys.path:
        _accounting_sys.path.insert(0, str(_ACCOUNTING_REPO_ROOT))
    from mvp.simulation.analysis.experiment_accounting import (  # noqa: E402
        build_h3_episode_accounting,
    )

try:
    from ..generate_results import SCENARIOS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from generate_results import SCENARIOS

STRESSORS = (
    "sensor_noise", "missing_data", "telemetry_delay",
    "mcp_fault_injection", "compounded",
)
CANONICAL_SEEDS = (
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
)


def _as_bool(value, *, where: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise RuntimeError(f"{where} is not Boolean")


def _assert_same_number(left, right, *, where: str) -> None:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{where} is not numeric") from exc
    if not __import__("math").isfinite(a) or not __import__("math").isfinite(b):
        raise RuntimeError(f"{where} is non-finite")
    if not __import__("math").isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError(f"{where} differs between CSV and H3 JSON")


def _validate_exact_h3_frame(
    frame: pd.DataFrame, *, scenario: str, where: Path,
) -> None:
    """Reject stale comparator rows or an incomplete AGRI-BRAIN stress grid."""

    required = {"Scenario", "Stressor", "Method", "n_seeds"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{where} lacks required H3 columns")
    keys = list(zip(frame["Scenario"], frame["Stressor"], frame["Method"]))
    expected = {
        (scenario, stressor, "agribrain") for stressor in STRESSORS
    }
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError(
            f"{where} is not the exact AGRI-BRAIN-only five-stressor panel"
        )
    if not (frame["n_seeds"] == 20).all():
        raise RuntimeError(f"{where} does not use exactly 20 seeds per H3 cell")


def _ledger_set_sha256(
    seed_panel: dict, *, mode: str = "agribrain",
) -> str:
    records = []
    for seed in CANONICAL_SEEDS:
        mode_panel = seed_panel.get(str(seed), seed_panel.get(seed))
        if not isinstance(mode_panel, dict) or mode not in mode_panel:
            raise RuntimeError(f"missing ledger-bound H3 seed cell: {seed}/{mode}")
        cell = mode_panel[mode]
        record = {
            "seed": seed,
            "path": cell.get("decision_ledger_path"),
            "sha256": cell.get("decision_ledger_sha256"),
            "merkle_root": cell.get("decision_ledger_merkle_root"),
            "n_records": cell.get("decision_ledger_n_records"),
        }
        if (
            not isinstance(record["path"], str)
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            or not isinstance(record["merkle_root"], str)
            or len(record["merkle_root"]) != 64
            or record["n_records"] != 288
        ):
            raise RuntimeError(f"invalid H3 ledger binding: seed={seed}/{mode}")
        records.append(record)
    return hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    canonical_results = Path(__file__).resolve().parent.parent / "results"
    canonical_results = canonical_results.resolve()
    run_tag = os.environ.get("RUN_TAG", "").strip()
    if output_dir == canonical_results:
        expected_input = (
            canonical_results / "stress_runs" / run_tag
        ).resolve() if run_tag else None
        if (
            not args.publication
            or os.environ.get("STRICT_VALIDATION") != "1"
            or os.environ.get("AGRIBRAIN_PUBLICATION_AGGREGATION") != "1"
            or expected_input is None
            or input_root != expected_input
        ):
            raise RuntimeError(
                "canonical H3 aggregation is restricted to the locked HPC "
                "publisher and exact run-tagged stress task directory"
            )
    elif args.publication:
        raise RuntimeError("--publication requires the canonical results directory")

    expected = set(SCENARIOS)
    found = {p.name for p in input_root.iterdir() if p.is_dir()}
    if found != expected:
        raise RuntimeError(
            f"Incomplete stress scenario panel: missing={sorted(expected-found)}, "
            f"unexpected={sorted(found-expected)}"
        )

    combined_results = {}
    degradation_frames = []
    pass_frames = []
    h3_cells = []
    thresholds = None
    design_meta = None
    for scenario in SCENARIOS:
        root = input_root / scenario
        summary_path = root / "stress_summary.json"
        degradation_path = root / "stress_degradation.csv"
        pass_path = root / "stress_passfail.csv"
        h3_path = root / "stress_h3_test.json"
        for path in (summary_path, degradation_path, pass_path, h3_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing scenario stress artifact: {path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result_block = summary.get("results", {})
        if set(result_block) != {scenario}:
            raise RuntimeError(
                f"{summary_path} must contain exactly scenario {scenario!r}"
            )
        combined_results[scenario] = result_block[scenario]
        thresholds_here = summary.get("meta", {}).get("thresholds")
        if thresholds is None:
            thresholds = thresholds_here
        elif thresholds_here != thresholds:
            raise RuntimeError("Stress thresholds differ across scenario tasks")
        design_here = {
            "source_commit": summary.get("meta", {}).get("source_commit"),
            "run_tag": summary.get("meta", {}).get("run_tag"),
            "max_rows": summary.get("meta", {}).get("max_rows"),
            "adaptation_episodes_per_stressed_condition": summary.get(
                "meta", {}
            ).get("adaptation_episodes_per_stressed_condition"),
            "frozen_evaluation_episodes_per_stressed_condition": summary.get(
                "meta", {}
            ).get("frozen_evaluation_episodes_per_stressed_condition"),
            "nominal_reference": summary.get("meta", {}).get(
                "nominal_reference"
            ),
            "adaptation_posture": summary.get("meta", {}).get("adaptation_posture"),
            "decision_history_posture": summary.get("meta", {}).get(
                "decision_history_posture"
            ),
            "mcp_reliability_posture": summary.get("meta", {}).get(
                "mcp_reliability_posture"
            ),
            "mcp_fault_dose": summary.get("meta", {}).get("mcp_fault_dose"),
            "retained_ledger_design": summary.get("meta", {}).get(
                "retained_ledger_design"
            ),
        }
        if (
            design_here["max_rows"] is not None
            or design_here["adaptation_episodes_per_stressed_condition"] != 3
            or design_here["frozen_evaluation_episodes_per_stressed_condition"] != 1
            or design_here["nominal_reference"]
            != "reused_primary_benchmark_episode_3"
            or design_here["mcp_reliability_posture"] != "false"
        ):
            raise RuntimeError(
                f"{summary_path} does not use the locked H3 execution design"
            )
        expected_ledger_design = {
            "stressed_ledgers_per_scenario_task": len(STRESSORS) * 20,
            "stressed_decisions_per_scenario_task": len(STRESSORS) * 20 * 288,
            "reused_primary_nominal_ledgers_per_scenario_task": 20,
            "newly_executed_nominal_episodes": 0,
            "canonical_stressed_ledger_root": (
                f"decision_ledger_h3/{design_here['run_tag']}"
            ),
            "canonical_nominal_ledger_root": (
                f"decision_ledger_per_seed/{design_here['run_tag']}"
            ),
        }
        if design_here["retained_ledger_design"] != expected_ledger_design:
            raise RuntimeError(f"{summary_path} has incorrect retained-ledger design")
        if design_meta is None:
            design_meta = design_here
        elif design_here != design_meta:
            raise RuntimeError("Stress-design metadata differ across scenario tasks")

        degradation_df = pd.read_csv(degradation_path)
        _validate_exact_h3_frame(
            degradation_df, scenario=scenario, where=degradation_path,
        )
        degradation_frames.append(degradation_df)
        pass_df = pd.read_csv(pass_path)
        _validate_exact_h3_frame(pass_df, scenario=scenario, where=pass_path)
        h3_pass = pass_df
        pass_keys = list(zip(h3_pass["Scenario"], h3_pass["Stressor"]))
        expected_keys = {(scenario, stressor) for stressor in STRESSORS}
        if len(pass_keys) != len(expected_keys) or set(pass_keys) != expected_keys:
            raise RuntimeError(f"{pass_path} does not contain the exact H3 cell panel")
        pass_frames.append(pass_df)

        h3 = json.loads(h3_path.read_text(encoding="utf-8"))
        if h3.get("source_commit") != design_here["source_commit"]:
            raise RuntimeError(f"Source commit mismatch in {h3_path}")
        if h3.get("run_tag") != design_here["run_tag"]:
            raise RuntimeError(f"Run tag mismatch in {h3_path}")
        if (
            h3.get("retained_stressed_decision_ledger_count") != 100
            or h3.get("reused_nominal_decision_ledger_references") != 20
            or h3.get("newly_executed_nominal_episodes") != 0
        ):
            raise RuntimeError(f"Retained-ledger accounting mismatch in {h3_path}")
        expected_accounting = build_h3_episode_accounting(
            n_seeds=20, n_scenarios=1, n_stressors=len(STRESSORS),
            episodes_per_condition=4, nominal_reference_reused=True,
        )
        if h3.get("episode_accounting") != expected_accounting:
            raise RuntimeError(f"Incorrect scenario-task episode accounting in {h3_path}")
        cells = h3.get("cells", [])
        if len(cells) != 5:
            raise RuntimeError(f"Expected five AGRI-BRAIN H3 cells in {h3_path}")
        h3_keys = [(cell.get("Scenario"), cell.get("Stressor")) for cell in cells]
        if len(h3_keys) != len(expected_keys) or set(h3_keys) != expected_keys:
            raise RuntimeError(f"{h3_path} does not contain the exact H3 cell panel")
        h3_by_key = {key: cell for key, cell in zip(h3_keys, cells)}
        pass_by_key = {
            (row["Scenario"], row["Stressor"]): row
            for _, row in h3_pass.iterrows()
        }
        for key in sorted(expected_keys):
            json_cell = h3_by_key[key]
            csv_cell = pass_by_key[key]
            if json_cell.get("Method") != "agribrain":
                raise RuntimeError(f"{h3_path} {key} is not an AGRI-BRAIN cell")
            if not _as_bool(
                json_cell.get("Confirmatory_H3"),
                where=f"{h3_path} {key}/Confirmatory_H3",
            ):
                raise RuntimeError(f"{h3_path} {key} is not marked confirmatory")
            if int(json_cell.get("n_seeds", -1)) != 20:
                raise RuntimeError(f"{h3_path} {key} does not contain 20 seeds")
            for field in (
                "ari_delta", "ari_tost_p_tost",
                "ari_tost_ci90_low", "ari_tost_ci90_high",
                "ari_tost_one_sided_95_lower_bound",
                "ari_tost_one_sided_95_upper_bound",
                "ari_tost_max_abs_one_sided_95_bound",
                "ari_tost_margin_clearance",
                "fault_injection_scheduled_opportunity_steps_mean",
                "fault_injection_scheduled_opportunity_steps_min",
                "fault_injection_scheduled_opportunity_steps_max",
                "fault_injection_trigger_steps_mean",
                "fault_injection_trigger_steps_min",
                "fault_injection_trigger_steps_max",
                "fault_injected_tool_result_count_mean",
                "fault_injected_tool_result_count_min",
                "fault_injected_tool_result_count_max",
                "retained_stressed_decision_ledger_count",
                "retained_stressed_decision_count",
                "reused_nominal_decision_ledger_count",
                "reused_nominal_decision_count",
            ):
                _assert_same_number(
                    csv_cell.get(field), json_cell.get(field),
                    where=f"{scenario}/{key[1]}/{field}",
                )
            expected_stressed_set = _ledger_set_sha256(
                result_block[scenario][key[1]],
            )
            expected_nominal_set = _ledger_set_sha256(
                result_block[scenario]["baseline_by_seed"],
            )
            for field, expected_hash in (
                ("retained_stressed_decision_ledger_set_sha256", expected_stressed_set),
                ("reused_nominal_decision_ledger_set_sha256", expected_nominal_set),
            ):
                if (
                    json_cell.get(field) != expected_hash
                    or str(csv_cell.get(field)) != expected_hash
                ):
                    raise RuntimeError(
                        f"{scenario}/{key[1]}/{field} does not bind its seed ledgers"
                    )
            for field in (
                "Pass", "Pass_Equivalence", "Confirmatory_H3",
                "H3_Pass", "treatment_exposure_verified",
                "ari_tost_one_sided_95_bound_below_margin",
            ):
                if _as_bool(csv_cell.get(field), where=f"CSV {key}/{field}") != _as_bool(
                    json_cell.get(field), where=f"JSON {key}/{field}",
                ):
                    raise RuntimeError(
                        f"{scenario}/{key[1]}/{field} differs between CSV and H3 JSON"
                    )
        h3_cells.extend(cells)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_out = {
        "meta": {
            "scenarios": list(SCENARIOS),
            "thresholds": thresholds,
            "parallel_aggregation": True,
            "stressors": [
                *STRESSORS,
            ],
            **(design_meta or {}),
        },
        "results": combined_results,
    }
    (output_dir / "stress_summary.json").write_text(
        json.dumps(summary_out, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.concat(degradation_frames, ignore_index=True).to_csv(
        output_dir / "stress_degradation.csv", index=False
    )
    pd.concat(pass_frames, ignore_index=True).to_csv(
        output_dir / "stress_passfail.csv", index=False
    )
    h3_out = {
        "hypothesis": (
            "H3: for every declared scenario-stressor cell, the mean paired "
            "seed-level AGRI-BRAIN ARI change is equivalent to zero within "
            "the ±0.01 margin."
        ),
        "test": "paired one-sample TOST on seed-level ARI differences",
        "alpha": 0.05,
        "equivalence_margin": float(thresholds["ari_abs_delta_max"]),
        "confirmatory_method": "agribrain",
        "expected_scenarios": list(SCENARIOS),
        "expected_stressors": list(STRESSORS),
        "expected_n_cells": len(SCENARIOS) * len(STRESSORS),
        "global_decision_rule": (
            "intersection-union: supported only when every prespecified "
            "AGRI-BRAIN scenario-stressor cell passes TOST and has verified "
            "nonzero treatment exposure"
        ),
        "one_sided_bound_rule": (
            "max(-one_sided_95_lower_bound, one_sided_95_upper_bound) < 0.01"
        ),
        **(design_meta or {}),
        "episode_accounting": build_h3_episode_accounting(
            n_seeds=20,
            n_scenarios=len(SCENARIOS),
            n_stressors=len(STRESSORS),
            episodes_per_condition=(
                int((design_meta or {}).get(
                    "adaptation_episodes_per_stressed_condition"
                ) or 3)
                + int((design_meta or {}).get(
                    "frozen_evaluation_episodes_per_stressed_condition"
                ) or 1)
            ),
            nominal_reference_reused=True,
        ),
        "n_cells": len(h3_cells),
        "n_cells_equivalent": sum(
            _as_bool(
                cell.get("Pass_Equivalence"),
                where=(
                    f"combined H3 {cell.get('Scenario')}/"
                    f"{cell.get('Stressor')}/Pass_Equivalence"
                ),
            )
            for cell in h3_cells
        ),
        "n_cells_with_verified_exposure": sum(
            _as_bool(
                cell.get("treatment_exposure_verified"),
                where=(
                    f"combined H3 {cell.get('Scenario')}/"
                    f"{cell.get('Stressor')}/treatment_exposure_verified"
                ),
            )
            for cell in h3_cells
        ),
        "retained_stressed_decision_ledger_count": sum(
            int(cell.get("retained_stressed_decision_ledger_count", 0))
            for cell in h3_cells
        ),
        "reused_nominal_decision_ledger_references": (
            len(SCENARIOS) * len(CANONICAL_SEEDS)
        ),
        "newly_executed_nominal_episodes": 0,
        "supported_all_cells": bool(h3_cells) and all(
            _as_bool(
                cell.get("Pass_Equivalence"),
                where=(
                    f"combined H3 {cell.get('Scenario')}/"
                    f"{cell.get('Stressor')}/Pass_Equivalence"
                ),
            )
            and _as_bool(
                cell.get("treatment_exposure_verified"),
                where=(
                    f"combined H3 {cell.get('Scenario')}/"
                    f"{cell.get('Stressor')}/treatment_exposure_verified"
                ),
            )
            for cell in h3_cells
        ),
        "cells": h3_cells,
    }
    combined_keys = [(cell.get("Scenario"), cell.get("Stressor")) for cell in h3_cells]
    expected_combined = {
        (scenario, stressor) for scenario in SCENARIOS for stressor in STRESSORS
    }
    if len(combined_keys) != len(expected_combined) or set(combined_keys) != expected_combined:
        raise RuntimeError("Combined H3 output is not the exact 5 x 5 Cartesian panel")
    (output_dir / "stress_h3_test.json").write_text(
        json.dumps(h3_out, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Combined {len(SCENARIOS)} scenarios and {len(h3_cells)} H3 cells; "
        f"all-equivalent={h3_out['supported_all_cells']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
