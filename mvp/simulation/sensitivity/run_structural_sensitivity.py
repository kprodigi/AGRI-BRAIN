#!/usr/bin/env python3
"""Generate, validate, execute, or analyse the locked structural LHS study.

Examples (run from the repository root)::

    python -m mvp.simulation.sensitivity.run_structural_sensitivity generate \
        --output-dir /scratch/agribrain_lhs_run
    python -m mvp.simulation.sensitivity.run_structural_sensitivity run-task \
        --run-plan /scratch/agribrain_lhs_run/run_plan.json --task-index 0
    python -m mvp.simulation.sensitivity.run_structural_sensitivity analyze \
        --run-plan /scratch/agribrain_lhs_run/run_plan.json

``generate`` does not execute an episode.  The full manifest contains 3,000
scenario-panel jobs which jointly execute 24,500 complete episodes.  A source
commit with tracked-file changes is rejected so a numerical result can never
be attributed to an uncommitted implementation accidentally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from hpc.slurm_execution_provenance import (
    build_array_execution_provenance,
    validate_structural_array_provenance,
)
from hpc.validate_complete_episode_evidence import validate_complete_evidence

from .design import (
    PRIMARY_MODES,
    STRESSORS,
    build_design,
    build_structural_accounting,
    build_task_manifest,
    canonical_sha256,
    file_sha256,
    load_locked_protocol,
    validate_design,
    validate_task_manifest,
)
from .overrides import (
    applied_structural_parameters,
    validate_dynamic_influence,
)
from .parameters import (
    PARAMETERS,
    registry_as_dict,
    validate_parameter_registry,
)

RUN_PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = REPO_ROOT / "mvp" / "simulation" / "experiment_protocol.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    source_status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "source_commit": commit,
        "tracked_tree_clean": not bool(tracked_status),
        "tracked_status": tracked_status.splitlines(),
        "source_tree_clean": not bool(source_status),
        "source_status": source_status.splitlines(),
    }


def _write_design_csv(path: Path, design: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ["point_index", "point_id", "seed"] + [
        parameter.key for parameter in PARAMETERS
    ] + ["slca_weight_price_transparency", "parameters_sha256"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in design["points"]:
            writer.writerow({
                "point_index": point["point_index"],
                "point_id": point["point_id"],
                "seed": point["seed"],
                **point["parameters"],
                "slca_weight_price_transparency": point[
                    "derived_parameters"
                ]["slca_weight_price_transparency"],
                "parameters_sha256": point["parameters_sha256"],
            })
    temporary.replace(path)


def _write_manifest_jsonl(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for task in manifest["tasks"]:
            handle.write(json.dumps(
                task, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ))
            handle.write("\n")
    temporary.replace(path)


def generate_run_plan(
    output_dir: Path,
    protocol_path: Path,
    *,
    run_tag: str | None = None,
    allow_dirty: bool = False,
    skip_dynamic_audit: bool = False,
) -> Path:
    protocol = load_locked_protocol(protocol_path)
    static_audit = validate_parameter_registry(REPO_ROOT)
    dynamic_audit = (
        {"status": "skipped_by_explicit_development_flag"}
        if skip_dynamic_audit else validate_dynamic_influence(REPO_ROOT)
    )
    git = _git_state(REPO_ROOT)
    if not git["source_tree_clean"] and not allow_dirty:
        raise RuntimeError(
            "BLOCK: source changes or untracked files are present. Commit the final code "
            "before generating a publication sensitivity run plan."
        )

    design = build_design(protocol)
    validate_design(design, protocol)
    manifest = build_task_manifest(design, protocol)
    accounting = build_structural_accounting(protocol)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "parameter_registry.json"
    design_path = output_dir / "lhs_design.json"
    design_csv_path = output_dir / "lhs_design.csv"
    manifest_path = output_dir / "task_manifest.json"
    manifest_jsonl_path = output_dir / "task_manifest.jsonl"
    accounting_path = output_dir / "episode_accounting.json"
    protocol_snapshot_path = output_dir / "experiment_protocol.json"
    _atomic_json(registry_path, registry_as_dict())
    _atomic_json(design_path, design)
    _write_design_csv(design_csv_path, design)
    _atomic_json(manifest_path, manifest)
    _write_manifest_jsonl(manifest_jsonl_path, manifest)
    _atomic_json(accounting_path, accounting)
    # Preserve the exact locked protocol beside the plan.  A relative bundle
    # reference keeps the evidence independently verifiable after transfer;
    # the repository path remains provenance metadata rather than a runtime
    # dependency of the copied bundle.
    protocol_snapshot_path.write_bytes(protocol_path.resolve().read_bytes())

    plan: dict[str, Any] = {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "execution_scope": "structural_sensitivity_only",
        "run_tag": run_tag,
        "source_commit": git["source_commit"],
        "source_tracked_tree_clean_at_generation": git["tracked_tree_clean"],
        "source_tree_clean_at_generation": git["source_tree_clean"],
        "development_only_dirty_plan": bool(not git["source_tree_clean"]),
        "protocol": {
            "path": protocol_snapshot_path.name,
            "source_path": str(protocol_path.resolve()),
            "sha256": file_sha256(protocol_path),
        },
        "audits": {
            "static_parameter_links": static_audit,
            "dynamic_influence": dynamic_audit,
        },
        "artifacts": {
            "parameter_registry": registry_path.name,
            "lhs_design": design_path.name,
            "lhs_design_csv": design_csv_path.name,
            "task_manifest": manifest_path.name,
            "task_manifest_jsonl": manifest_jsonl_path.name,
            "episode_accounting": accounting_path.name,
            "locked_protocol": protocol_snapshot_path.name,
        },
        "artifact_sha256": {
            path.name: file_sha256(path)
            for path in (
                registry_path,
                design_path,
                design_csv_path,
                manifest_path,
                manifest_jsonl_path,
                accounting_path,
                protocol_snapshot_path,
            )
        },
        "execution_status": "planned_not_executed",
        "full_run_warning": (
            "This plan contains 24,500 complete 72-hour episode executions; "
            "generating the plan does not execute them."
        ),
    }
    plan["run_plan_sha256"] = canonical_sha256(plan)
    plan_path = output_dir / "run_plan.json"
    _atomic_json(plan_path, plan)
    return plan_path


def _load_plan_bundle(plan_path: Path) -> tuple[dict, dict, dict, dict]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    digest = unsigned.pop("run_plan_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError("run plan SHA-256 does not match canonical content")
    root = plan_path.parent
    artifacts = plan["artifacts"]
    for filename, expected in plan["artifact_sha256"].items():
        path = root / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"run-plan artifact is missing or altered: {path}")
    protocol_reference = Path(plan["protocol"]["path"])
    protocol_path = (
        protocol_reference
        if protocol_reference.is_absolute()
        else root / protocol_reference
    )
    if not protocol_path.is_file() or file_sha256(protocol_path) != plan["protocol"]["sha256"]:
        raise ValueError("locked protocol is missing or changed since plan generation")
    protocol = load_locked_protocol(protocol_path)
    design = json.loads((root / artifacts["lhs_design"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / artifacts["task_manifest"]).read_text(encoding="utf-8"))
    validate_design(design, protocol)
    validate_task_manifest(manifest, protocol)
    if manifest["design_sha256"] != design["design_sha256"]:
        raise ValueError("task manifest and LHS design hashes do not match")
    return plan, protocol, design, manifest


def _assert_execution_source(plan: Mapping[str, Any]) -> None:
    git = _git_state(REPO_ROOT)
    if git["source_commit"] != plan["source_commit"]:
        raise RuntimeError(
            "BLOCK: execution commit differs from the commit recorded by the run plan"
        )
    if not git["source_tree_clean"]:
        raise RuntimeError(
            "BLOCK: source changes or untracked files are present during execution"
        )
    if plan.get("development_only_dirty_plan"):
        raise RuntimeError(
            "BLOCK: a development-only dirty run plan cannot produce publication results"
        )
    run_tag = plan.get("run_tag")
    if run_tag and os.environ.get("RUN_TAG") != run_tag:
        raise RuntimeError(
            "BLOCK: execution RUN_TAG differs from the run plan identity"
        )
    sensitivity_commit = os.environ.get(
        "AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT", ""
    ).strip()
    if sensitivity_commit and sensitivity_commit != plan["source_commit"]:
        raise RuntimeError(
            "BLOCK: structural-sensitivity source identity differs from the run plan"
        )
    dynamic = plan.get("audits", {}).get("dynamic_influence", {})
    if dynamic.get("status") != "pass":
        raise RuntimeError(
            "BLOCK: publication execution requires a passed dynamic parameter-influence audit"
        )


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in task result")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"task result contains unsupported value {type(value).__name__}")


_ENDPOINT_FIELDS: tuple[str, ...] = (
    "ari",
    "waste",
    "rle",
    "slca",
    "carbon",
    "equity",
    "mean_decision_latency_ms",
    "constraint_violation_rate",
    "operating_envelope_violation_rate",
    "downstream_violation_rate",
    "redistribute_violation_rate",
    "contained_violation_rate",
    "violation_event_count",
    "benchmark_seed",
    "episode_index",
    "episode_phase",
    "learning_enabled",
    "environment_stream_id",
    "policy_stream_id",
    "stochastic_stream_id",
    "latent_environment_sha256",
    "observed_policy_input_sha256",
    "demand_observation_sha256",
    "spoilage_estimator",
    "latent_spoilage_model",
    "effective_k_ref",
    "effective_Ea_R",
    "fault_injection_scheduled_opportunity_steps",
    "fault_injection_trigger_steps",
    "fault_injected_tool_result_count",
    "trace_schema_version",
    "message_count",
    "learner_summary",
    "theta_learner_summary",
    "reward_shaping_learner_summary",
    "learner_freeze_summary",
)


def _extract_endpoint(
    episode: Mapping[str, Any], expected_seed: int, *, expected_phase: str,
) -> dict[str, Any]:
    missing = [
        field for field in ("ari", "waste", "rle", "slca", "carbon", "equity")
        if field not in episode
    ]
    if missing:
        raise ValueError(f"retained endpoint is missing metrics: {missing}")
    provenance_fields = ("spoilage_estimator", "latent_spoilage_model")
    missing_provenance = [
        field for field in provenance_fields
        if not isinstance(episode.get(field), Mapping)
    ]
    if missing_provenance:
        raise ValueError(
            "retained endpoint is missing spoilage provenance: "
            f"{missing_provenance}"
        )
    if int(episode.get("benchmark_seed", -1)) != int(expected_seed):
        raise ValueError("retained endpoint benchmark seed does not match task")
    if int(episode.get("episode_index", -1)) != 3:
        raise ValueError("retained endpoint is not episode index 3")
    if expected_phase not in {"fixed_evaluation", "frozen_evaluation"}:
        raise ValueError(f"unsupported retained endpoint phase {expected_phase!r}")
    if episode.get("episode_phase") != expected_phase:
        raise ValueError(
            f"retained endpoint is not labelled {expected_phase}"
        )
    if episode.get("learning_enabled") is not False:
        raise ValueError("learning remained enabled during the retained endpoint")
    freeze = episode.get("learner_freeze_summary", {}) or {}
    if freeze and freeze.get("learners_frozen") is not True:
        raise ValueError("retained endpoint reports an incomplete learner freeze")
    extracted = {
        field: episode[field] for field in _ENDPOINT_FIELDS if field in episode
    }
    return _strict_json_value(extracted)


def _bind_retained_ledger(
    episode: Mapping[str, Any],
    endpoint: dict[str, Any],
    *,
    task_root: Path,
    run_root: Path,
    ledger_path: Path | None = None,
) -> None:
    """Attach one portable literal-byte/Merkle binding to an endpoint."""

    if ledger_path is None:
        raw_path = episode.get("decision_ledger_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("retained endpoint lacks its decision-ledger path")
        ledger_path = Path(raw_path)
    resolved_run_root = run_root.resolve()
    resolved_task_root = task_root.resolve()
    if not resolved_task_root.is_relative_to(resolved_run_root):
        raise ValueError("structural task root escapes the structural run root")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError(f"retained decision ledger is missing: {ledger_path}")
    resolved = ledger_path.resolve()
    if not resolved.is_relative_to(resolved_task_root):
        raise ValueError("retained decision ledger escapes its structural task root")
    relative = resolved.relative_to(resolved_run_root).as_posix()
    with resolved.open("r", encoding="utf-8") as handle:
        try:
            header = json.loads(handle.readline())
        except json.JSONDecodeError as exc:
            raise ValueError("retained decision ledger has an invalid header") from exc
    if not isinstance(header, dict) or header.get("_header") is not True:
        raise ValueError("retained decision ledger lacks its canonical header")
    root = header.get("merkle_root")
    n_records = header.get("n_records")
    if (
        not isinstance(root, str)
        or not re.fullmatch(r"[0-9a-f]{64}", root)
        or n_records != 288
    ):
        raise ValueError("retained decision ledger has an invalid Merkle binding")
    episode_root = episode.get(
        "decision_ledger_root", episode.get("decision_ledger_merkle_root")
    )
    episode_count = episode.get(
        "decision_ledger_n", episode.get("decision_ledger_n_records")
    )
    if episode_root != root or (
        episode_count != n_records
    ):
        raise ValueError("endpoint and retained-ledger header binding disagree")
    endpoint.update({
        "decision_ledger_path": relative,
        "decision_ledger_sha256": file_sha256(resolved),
        "decision_ledger_merkle_root": root,
        "decision_ledger_n_records": n_records,
    })


def _expected_ledger_relative_path(
    task: Mapping[str, Any], mode: str,
) -> str:
    """Return the one run-relative retained-ledger path allowed for a cell."""

    output = PurePosixPath(str(task["output_relpath"]))
    if output.is_absolute() or output.suffix != ".json" or any(
        part in {"", ".", ".."} for part in output.parts
    ):
        raise ValueError("structural task has an unsafe output path")
    artifact_root = output.parent / f"{output.stem}__artifacts"
    scenario = str(task["scenario"])
    if task["panel"] == "primary":
        expected = (
            artifact_root / "runtime_artifacts" / "decision_ledger"
            / f"{mode}__{scenario}.jsonl"
        )
    elif task["panel"] == "h3_stressed":
        expected = (
            artifact_root / "decision_ledgers" / scenario
            / f"structural__{task['point_id']}__{task['stressor']}"
            / f"seed_{int(task['seed'])}" / f"{mode}__{scenario}.jsonl"
        )
    else:
        raise ValueError(f"unsupported structural task panel {task['panel']!r}")
    return expected.as_posix()


def _run_primary_task(
    task: Mapping[str, Any], point: Mapping[str, Any], task_root: Path,
    *,
    run_root: Path,
) -> dict[str, Any]:
    scenario = str(task["scenario"])
    with applied_structural_parameters(point["parameters"], REPO_ROOT) as applied:
        gr = applied["generate_results_module"]
        saved = {
            "scenarios": gr.SCENARIOS,
            "modes": gr.MODES,
            "primary_modes": gr.PRIMARY_MODES,
            "results_dir": gr.RESULTS_DIR,
        }
        try:
            gr.SCENARIOS = [scenario]
            gr.MODES = list(PRIMARY_MODES)
            gr.PRIMARY_MODES = list(PRIMARY_MODES)
            gr.RESULTS_DIR = task_root / "runtime_artifacts"
            data = gr.run_all(seed=int(task["seed"]))
        finally:
            gr.SCENARIOS = saved["scenarios"]
            gr.MODES = saved["modes"]
            gr.PRIMARY_MODES = saved["primary_modes"]
            gr.RESULTS_DIR = saved["results_dir"]
    results = data["results"][scenario]
    if tuple(results) != PRIMARY_MODES:
        raise ValueError(
            f"primary task returned mode order {tuple(results)}, expected {PRIMARY_MODES}"
        )
    endpoints: dict[str, dict[str, Any]] = {}
    for mode in PRIMARY_MODES:
        endpoint = _extract_endpoint(
            results[mode],
            int(task["seed"]),
            expected_phase=(
                "fixed_evaluation" if mode == "static" else "frozen_evaluation"
            ),
        )
        _bind_retained_ledger(
            results[mode], endpoint, task_root=task_root, run_root=run_root,
        )
        endpoints[mode] = endpoint
    latent = {endpoint["latent_environment_sha256"] for endpoint in endpoints.values()}
    if len(latent) != 1:
        raise ValueError("primary paired modes do not share retained latent truth")
    return {"results": endpoints}


def _stress_frame(
    base: pd.DataFrame,
    *,
    scenario: str,
    stressor: str,
    seed: int,
    episode_index: int,
    stress_module: Any,
) -> pd.DataFrame:
    material = (
        f"stress|{scenario}|{stressor}|{seed}|{episode_index}"
    ).encode("utf-8")
    perturb_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return stress_module._perturb_df(
        base, stressor, np.random.default_rng(perturb_seed),
    )


def _run_h3_stressed_task(
    task: Mapping[str, Any], point: Mapping[str, Any], task_root: Path,
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Execute only the stressed arm; nominal AGRI is reused from primary."""

    scenario = str(task["scenario"])
    stressor = str(task["stressor"])
    seed = int(task["seed"])
    if stressor not in STRESSORS:
        raise ValueError(f"unexpected stressor {stressor!r}")
    with applied_structural_parameters(point["parameters"], REPO_ROOT) as applied:
        gr = applied["generate_results_module"]
        from mvp.simulation import stochastic
        from mvp.simulation.benchmarks import run_stress_suite as stress
        base_data = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"])
        stressed_frames: dict[int, pd.DataFrame] = {}
        policy_for_scenario = applied["policy_factory"]()
        for episode_index in range(4):
            scenario_seed = gr._stream_seed(
                seed, scenario, episode_index, "scenario",
            )
            environment_seed = gr._stream_seed(
                seed, scenario, episode_index, "environment",
            )
            scenario_frame = gr.apply_scenario(
                base_data,
                scenario,
                policy_for_scenario,
                np.random.default_rng(scenario_seed),
                stoch=stochastic.make_stochastic_layer(
                    np.random.default_rng(environment_seed),
                    stream_seed=environment_seed,
                ),
            )
            stressed_frames[episode_index] = _stress_frame(
                scenario_frame,
                scenario=scenario,
                stressor=stressor,
                seed=seed,
                episode_index=episode_index,
                stress_module=stress,
            )

        # Reuse the production H3 arm runner so episode freezing, feature-flag
        # posture, Source-7 policy prior, and exposure evidence cannot drift
        # between the core H3 and structural-sensitivity implementations.
        saved_policy = stress.Policy
        saved_ledger_root = stress.STRESS_LEDGER_ROOT
        try:
            stress.Policy = applied["policy_factory"]
            stress.STRESS_LEDGER_ROOT = task_root / "decision_ledgers"
            episode = stress._run_pair(
                stressed_frames,
                scenario,
                seed=seed,
                with_faults=stressor in {"mcp_fault_injection", "compounded"},
                modes=("agribrain",),
                ledger_condition=f"structural__{point['point_id']}__{stressor}",
            )["agribrain"]
        finally:
            stress.Policy = saved_policy
            stress.STRESS_LEDGER_ROOT = saved_ledger_root
    endpoint = _extract_endpoint(
        episode, seed, expected_phase="frozen_evaluation",
    )
    retained_ledger = (
        task_root / "decision_ledgers" / scenario
        / f"structural__{point['point_id']}__{stressor}"
        / f"seed_{seed}" / f"agribrain__{scenario}.jsonl"
    )
    _bind_retained_ledger(
        episode,
        endpoint,
        task_root=task_root,
        run_root=run_root,
        ledger_path=retained_ledger,
    )
    treatment = dict(episode.get("observation_treatment") or {})
    if treatment.get("stressor") != stressor:
        raise ValueError("stress result does not carry the requested treatment label")
    if stressor in {"mcp_fault_injection", "compounded"}:
        if int(endpoint.get("fault_injection_trigger_steps", 0)) <= 0:
            raise ValueError("MCP fault stressor had zero observed trigger exposure")
    return {
        "results": {"agribrain": endpoint},
        "observation_treatment": _strict_json_value(treatment),
    }


def _structural_episode_evidence_expectations(
    task: Mapping[str, Any], task_root: Path,
) -> tuple[Path, dict[str, int]]:
    """Return the task-local evidence root and its exact locked inventory."""

    if task["panel"] == "primary":
        return task_root / "runtime_artifacts" / "decision_ledger", {
            "expected_groups": 8,
            "expected_episodes": 29,
            "expected_adaptation_ledgers": 21,
            "expected_final_ledgers": 8,
        }
    if task["panel"] == "h3_stressed":
        return task_root / "decision_ledgers", {
            "expected_groups": 1,
            "expected_episodes": 4,
            "expected_adaptation_ledgers": 3,
            "expected_final_ledgers": 1,
        }
    raise ValueError(f"unsupported structural task panel {task['panel']!r}")


def _validate_structural_h3_ledger(
    ledger_path: Path,
    *,
    task: Mapping[str, Any],
    payload: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    metadata: Mapping[str, Any],
    stochastic_layer: Any,
) -> None:
    """Bind a structural H3 task to its exact dose and observed trajectory."""

    from mvp.simulation.benchmarks import run_stress_suite as stress

    stressor = str(task.get("stressor") or "")
    scenario = str(task["scenario"])
    seed = int(task["seed"])
    if stressor not in STRESSORS:
        raise ValueError(f"structural H3 task has unknown stressor: {stressor!r}")
    dummy = pd.DataFrame({
        "inventory_units": np.zeros(288, dtype=float),
        "demand_units": np.zeros(288, dtype=float),
    })
    dose_frame = _stress_frame(
        dummy,
        scenario=scenario,
        stressor=stressor,
        seed=seed,
        episode_index=3,
        stress_module=stress,
    )
    expected_treatment = _strict_json_value(
        dict(dose_frame.attrs["observation_treatment"]),
    )
    if payload.get("observation_treatment") != expected_treatment or (
        metadata.get("observation_treatment") != expected_treatment
    ):
        raise ValueError(
            f"structural H3 treatment does not match its task: {ledger_path}"
        )

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines[1:]]
    if len(records) != 288:
        raise ValueError(f"structural H3 ledger has wrong length: {ledger_path}")
    expected_temp_noise = dose_frame["h3_temp_noise_c"].to_numpy(dtype=float)
    expected_rh_noise = dose_frame["h3_rh_noise_pct"].to_numpy(dtype=float)
    expected_missing = dose_frame[
        "h3_missing_observation"
    ].to_numpy(dtype=bool)
    expected_sources = dose_frame[
        "h3_telemetry_source_step_index"
    ].to_numpy(dtype=int)

    canonical_temp: list[float] = []
    canonical_rh: list[float] = []
    predelay_temp: list[float] = []
    predelay_rh: list[float] = []
    scheduled_count = 0
    triggered_count = 0
    replaced_count = 0
    for index, record in enumerate(records):
        required = {
            "hour", "temp_outcome_environmental",
            "rh_outcome_environmental", "temp_policy_observed",
            "rh_policy_observed", "h3_stressor",
            "h3_data_observation_treatment", "h3_temp_noise_c",
            "h3_rh_noise_pct", "h3_missing_observation",
            "h3_telemetry_source_step_index",
            "h3_fault_injection_scheduled_opportunity",
            "h3_fault_injection_triggered",
            "h3_fault_injected_tool_result_count",
            "primary_mcp_tools_invoked_step",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"structural H3 ledger row lacks fields {sorted(missing)}: "
                f"{ledger_path}:{index + 2}"
            )
        if record["h3_stressor"] != stressor or record[
            "h3_data_observation_treatment"
        ] is not (stressor != "mcp_fault_injection"):
            raise ValueError(
                f"structural H3 row treatment label mismatch: "
                f"{ledger_path}:{index + 2}"
            )
        primitive_checks = (
            (float(record["h3_temp_noise_c"]), expected_temp_noise[index]),
            (float(record["h3_rh_noise_pct"]), expected_rh_noise[index]),
        )
        if any(left != float(right) for left, right in primitive_checks) or (
            bool(record["h3_missing_observation"])
            is not bool(expected_missing[index])
        ) or int(record["h3_telemetry_source_step_index"]) != int(
            expected_sources[index]
        ):
            raise ValueError(
                f"structural H3 primitive dose mismatch: "
                f"{ledger_path}:{index + 2}"
            )

        temp = stochastic_layer.perturb_temperature(
            float(record["temp_outcome_environmental"]), counter=index,
        )
        rh = stochastic_layer.perturb_humidity(
            float(record["rh_outcome_environmental"]), counter=index,
        )
        if index > 0 and stochastic_layer.should_delay(counter=index):
            temp = canonical_temp[-1]
            rh = canonical_rh[-1]
        canonical_temp.append(float(temp))
        canonical_rh.append(float(rh))
        temp += float(expected_temp_noise[index])
        rh = float(np.clip(rh + float(expected_rh_noise[index]), 15.0, 100.0))
        if bool(expected_missing[index]):
            if index == 0:
                raise ValueError("structural H3 dose masks the first observation")
            temp = predelay_temp[-1]
            rh = predelay_rh[-1]
        predelay_temp.append(float(temp))
        predelay_rh.append(float(rh))
        source_index = int(expected_sources[index])
        expected_temp = predelay_temp[source_index]
        expected_rh = predelay_rh[source_index]
        if not math.isclose(
            float(record["temp_policy_observed"]), expected_temp,
            rel_tol=0.0, abs_tol=1e-12,
        ) or not math.isclose(
            float(record["rh_policy_observed"]), expected_rh,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(
                f"structural H3 observed state does not reconstruct: "
                f"{ledger_path}:{index + 2}"
            )

        hour = float(record["hour"])
        scheduled = bool(
            stressor in {"mcp_fault_injection", "compounded"}
            and int(hour) % 11 == 0
        )
        channel_available = not (
            scenario == "cyber_outage" and hour >= 24.0
        )
        triggered = bool(scheduled and channel_available)
        invoked = record["primary_mcp_tools_invoked_step"]
        if not isinstance(invoked, list):
            raise ValueError(
                f"structural H3 primary tool list is invalid: "
                f"{ledger_path}:{index + 2}"
            )
        replaced = len(invoked) if triggered else 0
        if record["h3_fault_injection_scheduled_opportunity"] is not scheduled or (
            record["h3_fault_injection_triggered"] is not triggered
        ) or record["h3_fault_injected_tool_result_count"] != replaced:
            raise ValueError(
                f"structural H3 fault exposure does not reconstruct: "
                f"{ledger_path}:{index + 2}"
            )
        scheduled_count += int(scheduled)
        triggered_count += int(triggered)
        replaced_count += int(replaced)

    expected_counts = {
        "fault_injection_scheduled_opportunity_steps": scheduled_count,
        "fault_injection_trigger_steps": triggered_count,
        "fault_injected_tool_result_count": replaced_count,
    }
    for field, expected in expected_counts.items():
        if endpoint.get(field) != expected:
            raise ValueError(
                f"structural H3 endpoint {field} differs from its ledger: "
                f"{ledger_path}"
            )


def _validate_existing_result(
    path: Path,
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    point: Mapping[str, Any],
    submission_receipt: Mapping[str, Any] | None = None,
    task_root_override: Path | None = None,
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    digest = unsigned.pop("result_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise ValueError(f"existing task result hash mismatch: {path}")
    if payload.get("task_sha256") != task["task_sha256"]:
        raise ValueError(f"existing result belongs to a different task: {path}")
    if payload.get("source_commit") != plan["source_commit"]:
        raise ValueError(f"existing result belongs to a different source commit: {path}")
    expected_identity = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "protocol_sha256": plan["protocol"]["sha256"],
        "design_sha256": task["design_sha256"],
        "task_id": task["task_id"],
        "task_index": task["task_index"],
        "point_id": task["point_id"],
        "point_index": task["point_index"],
        "seed": task["seed"],
        "scenario": task["scenario"],
        "panel": task["panel"],
        "stressor": task.get("stressor"),
        "nominal_reference_task_id": task.get("nominal_reference_task_id"),
        "parameters_sha256": task["parameters_sha256"],
        "retained_cells": task["retained_cells"],
        "executed_episodes": task["executed_episodes"],
        "simulated_steps": task["simulated_steps"],
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"existing result identity {field!r} differs from task: {path}"
            )
    if submission_receipt is not None:
        validate_structural_array_provenance(
            payload.get("execution_provenance"),
            logical_task_index=int(task["task_index"]),
            submission_receipt=submission_receipt,
        )
    task_root = (
        task_root_override
        if task_root_override is not None
        else path.parent / (path.stem + "__artifacts")
    )
    evidence_root, evidence_expected = _structural_episode_evidence_expectations(
        task, task_root,
    )
    expected_evidence_manifest = task_root / "complete_episode_evidence_manifest.json"
    evidence_binding = payload.get("complete_episode_evidence")
    if not isinstance(evidence_binding, dict) or (
        evidence_binding.get("status") != "COMPLETE"
    ):
        raise ValueError(f"existing result lacks complete episode evidence: {path}")
    expected_evidence_relative = expected_evidence_manifest.resolve().relative_to(
        run_root.resolve(),
    ).as_posix()
    if (
        evidence_binding.get("manifest_path") != expected_evidence_relative
        or expected_evidence_manifest.is_symlink()
        or not expected_evidence_manifest.is_file()
        or evidence_binding.get("manifest_file_sha256")
        != file_sha256(expected_evidence_manifest)
    ):
        raise ValueError(f"existing result evidence-manifest binding mismatch: {path}")
    stored_evidence_manifest = json.loads(
        expected_evidence_manifest.read_text(encoding="utf-8")
    )
    recomputed_evidence_manifest = validate_complete_evidence(
        evidence_root,
        **evidence_expected,
        manifest_path=None,
    )
    if (
        stored_evidence_manifest != recomputed_evidence_manifest
        or evidence_binding.get("manifest_sha256")
        != recomputed_evidence_manifest["manifest_sha256"]
        or evidence_binding.get("counts")
        != recomputed_evidence_manifest["counts"]
    ):
        raise ValueError(f"existing result episode evidence is not reproducible: {path}")
    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != set(task["modes"]):
        raise ValueError(f"existing result has the wrong retained endpoint panel: {path}")
    from hpc.validate_decision_ledgers import (
        PUBLICATION_DATA_CSV,
        validate_learner_snapshot_binding,
        validate_ledger,
    )
    from hpc.validate_raw_publication_inputs import _validate_learner_provenance

    # Recreate every structural policy surface while the exact LHS overrides
    # are active.  The outcome equations alone are insufficient: gamma tilts,
    # the Source-7 policy prior, the context prior, and the per-episode policy
    # temperature also feed retained actions and must be independently bound.
    with applied_structural_parameters(
        dict(point["parameters"]), REPO_ROOT,
    ) as applied:
        from pirag import context_to_logits
        from src.models import action_selection

        from mvp.simulation import stochastic

        gr = applied["generate_results_module"]
        environment_seed = gr._stream_seed(
            int(task["seed"]), str(task["scenario"]), 3, "environment",
        )
        structural_layer = stochastic.make_stochastic_layer(
            np.random.default_rng(environment_seed),
            stream_seed=environment_seed,
        )
        expected_policy = applied["policy_factory"]()
        expected_policy_theta = gr.policy_theta_for_seed(
            np.asarray(action_selection.DECLARED_THETA, dtype=float),
            int(task["seed"]),
        )
        expected_context_prior = np.asarray(
            context_to_logits.THETA_CONTEXT, dtype=float,
        ).copy()
        expected_policy_temperature = structural_layer.policy_temperature(
            base=1.0, counter=0,
        )
        expected_outcome_contract = gr.build_outcome_equation_contract(
            expected_policy,
            effective_k_ref=structural_layer.perturb_k_ref(
                expected_policy.k_ref, counter=0,
            ),
            effective_ea_r=structural_layer.perturb_ea_r(
                expected_policy.Ea_R, counter=0,
            ),
            stochastic_layer=structural_layer,
        )
        committed_base_data = pd.read_csv(
            PUBLICATION_DATA_CSV, parse_dates=["timestamp"],
        )
        scenario_seed = gr._stream_seed(
            int(task["seed"]), str(task["scenario"]), 3, "scenario",
        )
        expected_scenario_frame = gr.apply_scenario(
            committed_base_data,
            str(task["scenario"]),
            expected_policy,
            np.random.default_rng(scenario_seed),
            stoch=structural_layer,
        )

    ledger_paths: set[str] = set()
    for mode, endpoint in results.items():
        if not isinstance(endpoint, dict):
            raise ValueError(f"existing result has an invalid endpoint: {path}/{mode}")
        raw_relative = endpoint.get("decision_ledger_path")
        if not isinstance(raw_relative, str) or "\\" in raw_relative:
            raise ValueError(f"existing result has an invalid ledger path: {path}/{mode}")
        relative = PurePosixPath(raw_relative)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(f"existing result has an unsafe ledger path: {path}/{mode}")
        normalized = relative.as_posix()
        expected_relative = _expected_ledger_relative_path(task, str(mode))
        if normalized != expected_relative:
            raise ValueError(
                "existing result ledger path does not match its task/mode: "
                f"{path}/{mode}"
            )
        if normalized in ledger_paths:
            raise ValueError(f"existing result reuses one ledger for multiple endpoints: {path}")
        ledger_paths.add(normalized)
        ledger_path = run_root.joinpath(*relative.parts)
        if ledger_path.is_symlink() or not ledger_path.is_file() or (
            not ledger_path.resolve().is_relative_to(run_root.resolve())
        ):
            raise ValueError(f"existing result ledger is missing or unsafe: {ledger_path}")
        if endpoint.get("decision_ledger_sha256") != file_sha256(ledger_path):
            raise ValueError(f"existing result ledger SHA-256 mismatch: {ledger_path}")
        if endpoint.get("decision_ledger_n_records") != 288:
            raise ValueError(f"existing result ledger count mismatch: {ledger_path}")
        try:
            with ledger_path.open("r", encoding="utf-8") as handle:
                header = json.loads(handle.readline())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"existing result ledger header is invalid: {ledger_path}"
            ) from exc
        if (
            not isinstance(header, dict)
            or header.get("_header") is not True
            or endpoint.get("decision_ledger_merkle_root")
            != header.get("merkle_root")
            or endpoint.get("decision_ledger_n_records")
            != header.get("n_records")
        ):
            raise ValueError(
                f"existing result ledger Merkle/count binding mismatch: {ledger_path}"
            )
        metadata = header.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"existing result ledger metadata is missing: {ledger_path}")
        expected_phase = (
            "fixed_evaluation" if mode == "static" else "frozen_evaluation"
        )
        endpoint_metadata = {
            "benchmark_seed": int(task["seed"]),
            "episode_index": 3,
            "episode_phase": expected_phase,
            "learning_enabled": False,
        }
        for field, expected in endpoint_metadata.items():
            if endpoint.get(field) != expected or metadata.get(field) != expected:
                raise ValueError(
                    f"existing result endpoint/header {field} mismatch: {ledger_path}"
                )
        for field in (
            "latent_environment_sha256",
            "observed_policy_input_sha256",
            "demand_observation_sha256",
            "trace_schema_version",
            "spoilage_estimator",
            "latent_spoilage_model",
        ):
            if endpoint.get(field) != metadata.get(field):
                raise ValueError(
                    f"existing result endpoint/header {field} mismatch: {ledger_path}"
                )
        for field in ("effective_k_ref", "effective_Ea_R"):
            if field in endpoint and not math.isclose(
                float(endpoint[field]),
                float(metadata.get(field, math.nan)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"existing result endpoint/header {field} mismatch: {ledger_path}"
                )
        summary = validate_ledger(
            ledger_path,
            mode=str(mode),
            scenario=str(task["scenario"]),
            benchmark_seed=int(task["seed"]),
            expected_outcome_equation_contract=expected_outcome_contract,
            expected_policy=expected_policy,
            expected_policy_theta=expected_policy_theta,
            expected_context_prior=expected_context_prior,
            expected_policy_temperature=expected_policy_temperature,
            expected_scenario_frame=expected_scenario_frame,
            expected_stochastic_layer=structural_layer,
        )
        validate_learner_snapshot_binding(
            endpoint,
            summary["learner_snapshots"],
            mode=str(mode),
            where=f"{path}/{mode}",
        )
        _validate_learner_provenance(
            endpoint,
            mode=str(mode),
            where=f"{path}/{mode}",
        )
        if task.get("panel") == "h3_stressed":
            _validate_structural_h3_ledger(
                ledger_path,
                task=task,
                payload=payload,
                endpoint=endpoint,
                metadata=metadata,
                stochastic_layer=structural_layer,
            )
        if summary["latent_environment_sha256"] != endpoint.get(
            "latent_environment_sha256"
        ):
            raise ValueError(f"existing result ledger latent identity mismatch: {ledger_path}")
        headlines = summary["headline_metrics"]
        required_headlines = (
            "ari", "waste", "slca", "carbon", "equity", "rle",
            "constraint_violation_rate",
        )
        optional_headlines = (
            "operating_envelope_violation_rate", "downstream_violation_rate",
            "redistribute_violation_rate", "contained_violation_rate",
        )
        for field in required_headlines:
            if not math.isclose(
                float(endpoint.get(field, math.nan)),
                float(headlines[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"existing result endpoint {field} differs from ledger: "
                    f"{ledger_path}"
                )
        for field in optional_headlines:
            if field in endpoint and not math.isclose(
                float(endpoint[field]),
                float(headlines[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"existing result endpoint {field} differs from ledger: "
                    f"{ledger_path}"
                )
        if "violation_event_count" in endpoint and endpoint[
            "violation_event_count"
        ] != headlines["violation_event_count"]:
            raise ValueError(
                f"existing result violation count differs from ledger: {ledger_path}"
            )
    return True


def _canonicalize_structural_ledger_paths_for_install(
    *,
    task: Mapping[str, Any],
    panel_payload: Mapping[str, Any],
    attempt_root: Path,
    final_task_root: Path,
    run_root: Path,
) -> None:
    """Replace attempt-local endpoint paths with their final canonical paths.

    The task executes in a unique sibling directory.  Only after every episode
    archive and ledger has validated is that directory atomically renamed to
    ``*__artifacts``.  This helper proves that each endpoint currently points
    to the matching file inside the attempt before recording its post-rename
    path in the durable task result.
    """

    results = panel_payload.get("results")
    if not isinstance(results, dict) or set(results) != set(task["modes"]):
        raise ValueError("structural attempt returned the wrong mode panel")
    final_root_relative = PurePosixPath(
        final_task_root.resolve().relative_to(run_root.resolve()).as_posix()
    )
    for mode, endpoint in results.items():
        if not isinstance(endpoint, dict):
            raise ValueError(f"structural endpoint is not an object: {mode}")
        expected_relative = PurePosixPath(
            _expected_ledger_relative_path(task, str(mode))
        )
        try:
            suffix = expected_relative.relative_to(final_root_relative)
        except ValueError as exc:
            raise ValueError(
                f"canonical ledger path is outside its task root: {mode}"
            ) from exc
        expected_attempt_path = attempt_root.joinpath(*suffix.parts).resolve()
        raw_current = endpoint.get("decision_ledger_path")
        if not isinstance(raw_current, str):
            raise ValueError(f"structural endpoint lacks a ledger path: {mode}")
        current_relative = PurePosixPath(raw_current)
        if current_relative.is_absolute() or any(
            part in {"", ".", ".."} for part in current_relative.parts
        ):
            raise ValueError(f"structural attempt ledger path is unsafe: {mode}")
        current_path = run_root.joinpath(*current_relative.parts).resolve()
        if current_path != expected_attempt_path:
            raise ValueError(
                f"structural attempt ledger path does not match its task: {mode}"
            )
        if current_path.is_symlink() or not current_path.is_file():
            raise ValueError(f"structural attempt ledger is missing: {mode}")
        if endpoint.get("decision_ledger_sha256") != file_sha256(current_path):
            raise ValueError(f"structural attempt ledger hash changed: {mode}")
        endpoint["decision_ledger_path"] = expected_relative.as_posix()


def _structural_attempt_failure_payload(
    *,
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    attempt_root: Path,
    exc: Exception,
) -> dict[str, Any]:
    """Return a compact failure record without deleting partial evidence."""

    partial_files = sorted(
        path.relative_to(attempt_root).as_posix()
        for path in attempt_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    return {
        "schema_version": 1,
        "status": "FAILED_ATTEMPT_RETAINED",
        "task_id": task["task_id"],
        "task_index": task["task_index"],
        "task_sha256": task["task_sha256"],
        "source_commit": plan["source_commit"],
        "protocol_sha256": plan["protocol"]["sha256"],
        "slurm": {
            name: os.environ.get(name, "")
            for name in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_RESTART_COUNT",
            )
        },
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "partial_file_count": len(partial_files),
        "partial_files": partial_files,
    }


def run_one_task(
    plan_path: Path, *, task_index: int, resume: bool = False,
) -> Path:
    plan, _protocol, design, manifest = _load_plan_bundle(plan_path)
    _assert_execution_source(plan)
    if not 0 <= task_index < len(manifest["tasks"]):
        raise IndexError(f"task index {task_index} outside manifest range")
    task = manifest["tasks"][task_index]
    if int(task["task_index"]) != task_index:
        raise ValueError("manifest task index mismatch")
    point = design["points"][int(task["point_index"])]
    if point["parameters_sha256"] != task["parameters_sha256"]:
        raise ValueError("task parameter hash does not match LHS point")

    execution_provenance = None
    strict_publication = os.environ.get("STRICT_VALIDATION", "0") == "1"
    if strict_publication:
        execution_provenance = build_array_execution_provenance(
            stage="structural_task_array",
            logical_task_index=task_index,
        )
        offset_raw = os.environ.get("SENSITIVITY_TASK_OFFSET", "").strip()
        if not offset_raw.isdigit() or (
            int(offset_raw) + execution_provenance["slurm_array_task_id"]
            != task_index
        ):
            raise RuntimeError(
                "structural Slurm offset/local array index does not map to task_index"
            )

    run_root = plan_path.resolve().parent
    output_path = run_root / task["output_relpath"]
    task_root = output_path.parent / (output_path.stem + "__artifacts")
    completion_record = task_root / "_completion" / "task_result.json"
    submission_receipt = None
    if strict_publication:
        submission_path = run_root / "slurm_submission.json"
        if not submission_path.is_file():
            raise RuntimeError(
                "strict structural resume requires slurm_submission.json"
            )
        submission_receipt = json.loads(
            submission_path.read_text(encoding="utf-8")
        )
    if output_path.exists():
        if resume and _validate_existing_result(
            output_path,
            task,
            plan,
            run_root=run_root,
            point=point,
            submission_receipt=submission_receipt,
        ):
            return output_path
        raise FileExistsError(
            f"refusing to overwrite task result {output_path}; use --resume "
            "only to accept an exact hash-validated result"
        )

    # A completed attempt is installed atomically with its self-contained task
    # result.  If a worker died after the directory rename but before copying
    # the result to its canonical location, --resume can finish that install
    # without executing a single episode again.
    if task_root.exists():
        if resume and completion_record.is_file() and _validate_existing_result(
            completion_record,
            task,
            plan,
            run_root=run_root,
            point=point,
            submission_receipt=submission_receipt,
            task_root_override=task_root,
        ):
            recovered = json.loads(completion_record.read_text(encoding="utf-8"))
            _atomic_json(output_path, recovered)
            _validate_existing_result(
                output_path,
                task,
                plan,
                run_root=run_root,
                point=point,
                submission_receipt=submission_receipt,
            )
            return output_path
        raise FileExistsError(
            f"refusing to overwrite or discard structural task artifacts "
            f"{task_root}; no exact recoverable completion record was accepted"
        )

    attempts_root = output_path.parent / (output_path.stem + "__attempts")
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing_attempts = [
        path for path in sorted(attempts_root.glob("attempt_*"))
        if path.is_dir() and not path.is_symlink()
    ]
    if len(existing_attempts) > 1:
        raise RuntimeError(
            f"multiple preserved structural attempts require manual audit: "
            f"{existing_attempts}"
        )
    attempt_root = (
        existing_attempts[0]
        if existing_attempts
        else Path(tempfile.mkdtemp(prefix="attempt_", dir=attempts_root))
    )
    try:
        panel_payload = (
            _run_primary_task(task, point, attempt_root, run_root=run_root)
            if task["panel"] == "primary"
            else _run_h3_stressed_task(
                task, point, attempt_root, run_root=run_root,
            )
        )
        evidence_root, evidence_expected = _structural_episode_evidence_expectations(
            task, attempt_root,
        )
        evidence_manifest_path = (
            attempt_root / "complete_episode_evidence_manifest.json"
        )
        evidence_manifest = validate_complete_evidence(
            evidence_root,
            **evidence_expected,
            manifest_path=evidence_manifest_path,
        )
        _canonicalize_structural_ledger_paths_for_install(
            task=task,
            panel_payload=panel_payload,
            attempt_root=attempt_root,
            final_task_root=task_root,
            run_root=run_root,
        )
        final_manifest_path = task_root / evidence_manifest_path.name
        evidence_manifest_relative = final_manifest_path.resolve().relative_to(
            run_root.resolve(),
        ).as_posix()
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "analysis_label": "structural sensitivity",
            "probability_interpretation": False,
            "source_commit": plan["source_commit"],
            "protocol_sha256": plan["protocol"]["sha256"],
            "design_sha256": design["design_sha256"],
            "task_sha256": task["task_sha256"],
            "task_id": task["task_id"],
            "task_index": task_index,
            "point_id": point["point_id"],
            "point_index": point["point_index"],
            "seed": task["seed"],
            "scenario": task["scenario"],
            "panel": task["panel"],
            "stressor": task.get("stressor"),
            "nominal_reference_task_id": task.get("nominal_reference_task_id"),
            "parameters_sha256": point["parameters_sha256"],
            "retained_cells": task["retained_cells"],
            "executed_episodes": task["executed_episodes"],
            "simulated_steps": task["simulated_steps"],
            "execution_provenance": execution_provenance,
            "complete_episode_evidence": {
                "status": "COMPLETE",
                "manifest_path": evidence_manifest_relative,
                "manifest_file_sha256": file_sha256(evidence_manifest_path),
                "manifest_sha256": evidence_manifest["manifest_sha256"],
                "counts": evidence_manifest["counts"],
            },
            **panel_payload,
        }
        result["result_sha256"] = canonical_sha256(result)
        attempt_completion = attempt_root / "_completion" / "task_result.json"
        _atomic_json(attempt_completion, result)
        attempt_root.replace(task_root)
        _atomic_json(output_path, result)
        _validate_existing_result(
            output_path,
            task,
            plan,
            run_root=run_root,
            point=point,
            submission_receipt=submission_receipt,
        )
        return output_path
    except Exception as exc:
        # Keep all partial bytes for diagnosis.  Never delete or reuse an
        # attempt directory; a retry always starts in a new unique location.
        if attempt_root.exists():
            try:
                failure = _structural_attempt_failure_payload(
                    task=task,
                    plan=plan,
                    attempt_root=attempt_root,
                    exc=exc,
                )
                failure["failure_sha256"] = canonical_sha256(failure)
                failure_root = attempt_root / "_attempt_failures"
                failure_index = len(list(failure_root.glob("failure_*.json")))
                _atomic_json(
                    failure_root / f"failure_{failure_index:04d}.json",
                    failure,
                )
            except Exception:
                # Preserve the scientific exception even if a full disk or
                # permission error prevents writing the diagnostic marker.
                pass
        raise


def _structural_pairing_signature(ledger_path: Path) -> dict[str, str]:
    """Hash the latent world and every H3-unaffected policy input."""

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 289:
        raise ValueError(f"structural pairing ledger has wrong length: {ledger_path}")
    rows = [json.loads(line) for line in lines[1:]]
    latent_fields = (
        "step_index", "hour", "temp_outcome_environmental",
        "rh_outcome_environmental", "rho_outcome_environmental",
        "inventory_outcome_environmental", "demand_outcome_environmental",
        "transport_multiplier_outcome_environmental",
    )
    unaffected_fields = (
        "step_index", "hour", "inventory_policy_observed",
        "demand_policy_observed", "demand_forecast_policy_observed",
        "demand_forecast_std_policy_observed",
        "supply_forecast_policy_observed",
        "supply_forecast_std_policy_observed", "bollinger_regime_flag",
        "price_signal",
    )
    for field in (*latent_fields, *unaffected_fields):
        if any(field not in row for row in rows):
            raise ValueError(
                f"structural pairing field {field!r} is missing: {ledger_path}"
            )
    return {
        "latent_sha256": canonical_sha256([
            [row[field] for field in latent_fields] for row in rows
        ]),
        "h3_unaffected_policy_inputs_sha256": canonical_sha256([
            [row[field] for field in unaffected_fields] for row in rows
        ]),
    }


def validate_completed_results_with_ledgers(
    plan_path: Path,
    *,
    submission_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate all task bytes/ledgers and return status plus exact inventory."""

    plan, _protocol, design, manifest = _load_plan_bundle(plan_path)
    root = plan_path.resolve().parent
    missing: list[str] = []
    valid = 0
    ledger_paths: list[str] = []
    primary_agribrain_signatures: dict[str, dict[str, str]] = {}
    h3_pairing_requests: list[tuple[str, str, dict[str, str]]] = []
    for task in manifest["tasks"]:
        path = root / task["output_relpath"]
        if not path.is_file():
            missing.append(task["task_id"])
            continue
        point = design["points"][int(task["point_index"])]
        _validate_existing_result(
            path, task, plan, run_root=root, point=point,
            submission_receipt=submission_receipt,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        ledger_paths.extend(
            str(endpoint["decision_ledger_path"])
            for endpoint in payload["results"].values()
        )
        if task["panel"] == "primary":
            agribrain_endpoint = payload["results"].get("agribrain")
            if not isinstance(agribrain_endpoint, dict):
                raise ValueError(
                    f"primary structural task lacks AGRI-BRAIN: {task['task_id']}"
                )
            agribrain_path = root.joinpath(*PurePosixPath(
                str(agribrain_endpoint["decision_ledger_path"])
            ).parts)
            primary_agribrain_signatures[str(task["task_id"])] = (
                _structural_pairing_signature(agribrain_path)
            )
        elif task["panel"] == "h3_stressed":
            reference = task.get("nominal_reference_task_id")
            agribrain_endpoint = payload["results"].get("agribrain")
            if not isinstance(reference, str) or not isinstance(
                agribrain_endpoint, dict
            ):
                raise ValueError(
                    f"H3 structural task lacks nominal reference: {task['task_id']}"
                )
            stressed_path = root.joinpath(*PurePosixPath(
                str(agribrain_endpoint["decision_ledger_path"])
            ).parts)
            h3_pairing_requests.append((
                str(task["task_id"]),
                reference,
                _structural_pairing_signature(stressed_path),
            ))
        valid += 1
    status = {
        "status": "complete" if not missing else "incomplete",
        "n_expected_tasks": len(manifest["tasks"]),
        "n_valid_tasks": valid,
        "n_missing_tasks": len(missing),
        "missing_task_ids": missing,
    }
    if not missing and (
        len(ledger_paths) != 6_500 or len(set(ledger_paths)) != 6_500
    ):
        raise ValueError(
            "structural retained-ledger inventory must contain 6,500 unique paths"
        )
    if not missing:
        for task_id, reference, stressed_signature in h3_pairing_requests:
            nominal_signature = primary_agribrain_signatures.get(reference)
            if nominal_signature is None:
                raise ValueError(
                    f"structural H3 task {task_id} references missing nominal "
                    f"task {reference}"
                )
            if stressed_signature != nominal_signature:
                raise ValueError(
                    f"structural H3 task {task_id} changed latent truth or an "
                    "H3-unaffected policy input"
                )
    return status, sorted(ledger_paths)


def validate_completed_results(
    plan_path: Path,
    *,
    submission_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status, _ledger_paths = validate_completed_results_with_ledgers(
        plan_path, submission_receipt=submission_receipt,
    )
    return status


def retained_ledger_relative_paths(plan_path: Path) -> list[str]:
    """Return the exact validated 6,500-ledger inventory for finalization."""

    status, paths = validate_completed_results_with_ledgers(plan_path)
    if status["status"] != "complete":
        raise ValueError("cannot inventory retained ledgers for an incomplete run")
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write the design and task manifest only")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    generate.add_argument(
        "--run-tag",
        help="bind this plan to one run-scoped Slurm identity",
    )
    generate.add_argument(
        "--allow-dirty", action="store_true",
        help="development plan only; such a plan is blocked from execution",
    )
    generate.add_argument(
        "--skip-dynamic-audit", action="store_true",
        help="development plan only; such a plan is blocked from execution",
    )

    validate = sub.add_parser("validate", help="validate registry or a run plan")
    validate.add_argument("--run-plan", type=Path)
    validate.add_argument("--dynamic-influence", action="store_true")

    run_task = sub.add_parser("run-task", help="execute one manifest task")
    run_task.add_argument("--run-plan", type=Path, required=True)
    run_task.add_argument("--task-index", type=int, required=True)
    run_task.add_argument("--resume", action="store_true")

    status = sub.add_parser("status", help="hash-check all completed task outputs")
    status.add_argument("--run-plan", type=Path, required=True)
    status.add_argument("--submission-receipt", type=Path)

    analyze = sub.add_parser("analyze", help="analyse a complete result panel")
    analyze.add_argument("--run-plan", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        plan = generate_run_plan(
            args.output_dir,
            args.protocol,
            run_tag=args.run_tag,
            allow_dirty=args.allow_dirty,
            skip_dynamic_audit=args.skip_dynamic_audit,
        )
        print(plan)
        return 0
    if args.command == "validate":
        report: dict[str, Any] = {
            "static_parameter_registry": validate_parameter_registry(REPO_ROOT),
        }
        if args.dynamic_influence:
            report["dynamic_influence"] = validate_dynamic_influence(REPO_ROOT)
        if args.run_plan:
            _load_plan_bundle(args.run_plan)
            report["run_plan"] = {"status": "pass"}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if args.command == "run-task":
        print(run_one_task(
            args.run_plan, task_index=args.task_index, resume=args.resume,
        ))
        return 0
    if args.command == "status":
        submission_receipt = (
            json.loads(args.submission_receipt.read_text(encoding="utf-8"))
            if args.submission_receipt is not None else None
        )
        report = validate_completed_results(
            args.run_plan, submission_receipt=submission_receipt,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["status"] == "complete" else 2
    if args.command == "analyze":
        from .analyze_structural_sensitivity import analyze_run
        output = args.output or (
            args.run_plan.resolve().parent / "structural_sensitivity_analysis.json"
        )
        report = analyze_run(args.run_plan)
        _atomic_json(output, report)
        print(output)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
