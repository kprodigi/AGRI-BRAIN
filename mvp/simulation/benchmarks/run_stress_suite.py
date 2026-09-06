#!/usr/bin/env python3
"""Stress-test suite for H3 robustness reporting.

For each scenario and stressor, the suite pairs a freshly executed stressed
AGRI-BRAIN arm with the hash-identified primary nominal endpoint by seed. The
nominal arm is reused, not rerun. The formal H3 test is a two-one-sided
equivalence test (TOST) on
the seed-level Adaptive Resilience Index differences with a ±0.01 margin.
Observed mean drift and the fraction of seeds inside the margin are retained as
descriptive diagnostics; they are not substitutes for the equivalence test.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

try:
    from ..analysis.experiment_accounting import build_h3_episode_accounting
    from ..analysis.protocol_statistics import equivalence_tost
except ImportError:
    import sys as _accounting_sys

    _ACCOUNTING_REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_ACCOUNTING_REPO_ROOT) not in _accounting_sys.path:
        _accounting_sys.path.insert(0, str(_ACCOUNTING_REPO_ROOT))
    from mvp.simulation.analysis.experiment_accounting import (  # noqa: E402
        build_h3_episode_accounting,
    )
    from mvp.simulation.analysis.protocol_statistics import (  # noqa: E402
        equivalence_tost,
    )

from hpc.slurm_execution_provenance import (  # noqa: E402
    CORE_SCENARIOS,
    build_array_execution_provenance,
)

try:
    from ..generate_results import (
        DATA_CSV, SCENARIOS, Policy, apply_scenario, run_episode,
        TRACE_SCHEMA_VERSION, _stream_id, _stream_seed,
        decision_ledger_scope, policy_theta_for_seed,
    )
    from ..stochastic import make_stochastic_layer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from generate_results import (
        DATA_CSV, SCENARIOS, Policy, apply_scenario, run_episode,
        TRACE_SCHEMA_VERSION, _stream_id, _stream_seed,
        decision_ledger_scope, policy_theta_for_seed,
    )
    from stochastic import make_stochastic_layer


_SIMULATION_DIR = Path(__file__).resolve().parent.parent
_DEVELOPMENT_STRESS_ROOT = (
    _SIMULATION_DIR / "development_results" / "unpublished_h3"
)
RESULTS_DIR = Path(os.environ.get("STRESS_OUTPUT_DIR", str(_DEVELOPMENT_STRESS_ROOT)))
STRESS_LEDGER_ROOT = Path(
    os.environ.get(
        "STRESS_LEDGER_ROOT",
        str(_DEVELOPMENT_STRESS_ROOT / "decision_ledgers"),
    )
)
PRIMARY_SEEDS_DIR = Path(
    os.environ.get(
        "STRESS_PRIMARY_SEEDS_DIR",
        str(Path(__file__).resolve().parent.parent / "results" / "benchmark_seeds"),
    )
)

# Formal H3 equivalence margin. Robustness is supported only when the absolute
# mean change in Adaptive Resilience Index is at most one percentage point.
# Other deltas remain descriptive diagnostics and are not silently folded into
# the H3 pass criterion.
STRESS_THRESHOLDS = {
    "ari_abs_delta_max": 0.01,
    "waste_delta_max": 0.04,
    "slca_delta_min": -0.10,
    "rle_delta_min": -0.12,
    "carbon_delta_max": 250.0,
    "equity_delta_min": -0.06,
    "constraint_violation_delta_max": 0.15,
    "latency_ms_delta_max": 100.0,
}
CANONICAL_SEEDS = (
    42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
    606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
)


def _json_safe(value):
    """Convert numpy scalars and non-finite descriptive cells to strict JSON.

    Pandas adds NaN columns when ordinary stress rows and descriptive
    cross-mode rows are combined. Those absent fields are represented as JSON
    null; calculated inferential fields remain finite and are validated before
    reaching this helper.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _confirmatory_seed_panel(n_seeds: int) -> list[int]:
    if n_seeds != len(CANONICAL_SEEDS):
        raise ValueError(
            f"Confirmatory H3 requires STRESS_N_SEEDS={len(CANONICAL_SEEDS)}; "
            f"received {n_seeds}"
        )
    return list(CANONICAL_SEEDS)


def _primary_seed_path(seed: int) -> Path:
    """Resolve the canonical flat or per-task primary seed envelope."""
    candidates = (
        PRIMARY_SEEDS_DIR / f"seed_{seed}.json",
        PRIMARY_SEEDS_DIR / f"seed_{seed}" / f"seed_{seed}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing primary benchmark envelope for seed {seed} under "
        f"{PRIMARY_SEEDS_DIR}. H3 reuses, rather than reruns, its nominal arm."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ledger_file_binding(path: Path, *, canonical_path: str) -> Dict[str, Any]:
    """Bind one retained JSONL ledger by literal bytes and Merkle root."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing retained decision ledger: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid retained decision ledger header: {path}") from exc
    root = header.get("merkle_root") if isinstance(header, dict) else None
    n_records = header.get("n_records") if isinstance(header, dict) else None
    if (
        header.get("_header") is not True
        or not isinstance(root, str)
        or len(root) != 64
        or any(char not in "0123456789abcdef" for char in root)
        or n_records != 288
    ):
        raise RuntimeError(f"Invalid retained decision ledger binding: {path}")
    return {
        "decision_ledger_path": canonical_path,
        "decision_ledger_sha256": _sha256_file(path),
        "decision_ledger_merkle_root": root,
        "decision_ledger_n_records": int(n_records),
    }


def _primary_ledger_path(seed: int, scenario: str) -> Path:
    candidates = (
        PRIMARY_SEEDS_DIR / f"decision_ledger_{seed}"
        / f"agribrain__{scenario}.jsonl",
        _primary_seed_path(seed).parent / f"decision_ledger_{seed}"
        / f"agribrain__{scenario}.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Missing retained primary AGRI-BRAIN ledger for H3 nominal reuse: "
        f"seed={seed}, scenario={scenario}, root={PRIMARY_SEEDS_DIR}"
    )


def _load_primary_nominal(
    scenario: str, seed: int,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Load and validate the retained primary AGRI-BRAIN nominal cell."""
    path = _primary_seed_path(seed)
    envelope_bytes = path.read_bytes()
    payload = json.loads(envelope_bytes.decode("utf-8"))
    meta = payload.get("_meta")
    if not isinstance(meta, dict) or payload.get("_trace_failures"):
        raise RuntimeError(f"{path} is not a complete primary benchmark envelope")
    if payload.get("seed") != seed or payload.get("trace_schema_version") != (
        TRACE_SCHEMA_VERSION
    ):
        raise RuntimeError(f"{path} seed/schema does not match the H3 run")
    try:
        cell = payload["scenarios"][scenario]["agribrain"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{path} lacks primary AGRI-BRAIN cell for {scenario}"
        ) from exc
    if (
        cell.get("episode_index") != 3
        or cell.get("learning_enabled") is not False
        or cell.get("episode_phase") != "frozen_evaluation"
    ):
        raise RuntimeError(
            f"{path}:{scenario}/agribrain is not frozen retained episode 3"
        )
    freeze = cell.get("learner_freeze_summary") or {}
    role_freeze = freeze.get("policy_delta_frozen_by_role") or {}
    if (
        freeze.get("learners_frozen") is not True
        or freeze.get("learner_phase") != "frozen_evaluation"
        or freeze.get("freeze_reason") != "retained_episode_3"
        or freeze.get("context_matrix_frozen") is not True
        or freeze.get("reward_shaping_frozen") is not True
        or not role_freeze
        or not all(value is True for value in role_freeze.values())
    ):
        raise RuntimeError(
            f"{path}:{scenario}/agribrain lacks learner-freeze evidence"
        )
    if cell.get("demand_forecast_method") != "holt_linear" or (
        cell.get("supply_forecast_method") != "persistence"
    ):
        raise RuntimeError(f"{path}:{scenario}/agribrain forecast lock mismatch")
    required = (
        "ari", "waste", "slca", "rle", "carbon", "equity",
        "constraint_violation_rate", "mean_decision_latency_ms",
        "latent_environment_sha256", "observed_policy_input_sha256",
        "demand_observation_sha256", "trace_schema_version", "benchmark_seed",
        "episode_index", "environment_stream_id", "policy_stream_id",
        "stochastic_stream_id", "context_prior_sha256",
        "policy_theta_initial_sha256",
        "spoilage_estimator",
        "latent_spoilage_model",
        "demand_forecast_method", "supply_forecast_method",
    )
    missing = [name for name in required if name not in cell]
    if missing:
        raise RuntimeError(f"{path}:{scenario}/agribrain lacks {missing}")
    expected_environment_id = _stream_id(seed, scenario, 3, "environment")
    expected_policy_id = _stream_id(seed, scenario, 3, "policy")
    if (
        cell.get("benchmark_seed") != seed
        or cell.get("environment_stream_id") != expected_environment_id
        or cell.get("stochastic_stream_id") != expected_environment_id
        or cell.get("policy_stream_id") != expected_policy_id
    ):
        raise RuntimeError(
            f"{path}:{scenario}/agribrain has mismatched retained stream identity"
        )
    for field in (
        "latent_environment_sha256", "observed_policy_input_sha256",
        "demand_observation_sha256", "context_prior_sha256",
        "policy_theta_initial_sha256",
    ):
        value = cell[field]
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise RuntimeError(
                f"{path}:{scenario}/agribrain has invalid {field}"
            )
    primary_cell_sha256 = hashlib.sha256(json.dumps(
        cell, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    nominal: Dict[str, object] = {
        "ari": float(cell["ari"]),
        "waste": float(cell["waste"]),
        "slca": float(cell["slca"]),
        "rle": float(cell["rle"]),
        "carbon": float(cell["carbon"]),
        "equity": float(cell["equity"]),
        "message_count": int(cell["message_count"]),
        "constraint_violation_rate": float(cell["constraint_violation_rate"]),
        "decision_latency_ms": float(cell["mean_decision_latency_ms"]),
        "downstream_violation_rate": float(
            cell.get("downstream_violation_rate", 0.0)
        ),
        "contained_violation_rate": float(
            cell.get("contained_violation_rate", 0.0)
        ),
        "trace_schema_version": int(cell["trace_schema_version"]),
        "benchmark_seed": int(cell["benchmark_seed"]),
        "episode_index": int(cell["episode_index"]),
        "environment_stream_id": str(cell["environment_stream_id"]),
        "policy_stream_id": str(cell["policy_stream_id"]),
        "stochastic_stream_id": str(cell["stochastic_stream_id"]),
        "context_prior_sha256": str(cell["context_prior_sha256"]),
        "policy_theta_initial_sha256": str(cell["policy_theta_initial_sha256"]),
        "spoilage_estimator": dict(cell["spoilage_estimator"]),
        "latent_spoilage_model": dict(cell["latent_spoilage_model"]),
        "latent_environment_sha256": str(cell["latent_environment_sha256"]),
        "observed_policy_input_sha256": str(cell["observed_policy_input_sha256"]),
        "demand_observation_sha256": str(cell["demand_observation_sha256"]),
        "demand_forecast_method": str(cell["demand_forecast_method"]),
        "supply_forecast_method": str(cell["supply_forecast_method"]),
        "learning_enabled": False,
        "episode_phase": "frozen_evaluation",
        "learner_freeze_summary": freeze,
        "primary_seed_envelope_sha256": hashlib.sha256(
            envelope_bytes
        ).hexdigest(),
        "primary_nominal_cell_sha256": primary_cell_sha256,
        "learner_summary": cell.get("learner_summary"),
        "theta_learner_summary": cell.get("theta_learner_summary"),
        "reward_shaping_learner_summary": cell.get(
            "reward_shaping_learner_summary"
        ),
        "observation_treatment": {
            "stressor": "nominal",
            "n_steps": 288,
            "data_observation_treatment": False,
            "delay_steps": 0,
            "missing_count": 0,
            "source": "reused_primary_benchmark",
        },
    }
    for field in (
        "protocol_interaction_count",
        "protocol_jsonrpc_error_count",
        "protocol_tool_iserror_count",
        "protocol_real_tool_iserror_count",
        "protocol_error_count",
        "protocol_dropped_interaction_count",
        "dispatcher_tool_failure_count",
        "context_execution_error_count",
        "fault_injection_scheduled_opportunity_steps",
        "fault_injection_trigger_steps",
        "fault_injected_tool_result_count",
        "dispatch_opportunity_count",
    ):
        nominal[field] = int(cell.get(field, 0))
    nominal["dispatch_cadence_hours"] = float(
        cell.get("dispatch_cadence_hours", 0.25)
    )
    identity = {
        "source_commit": str(meta.get("source_commit", "")),
        "run_tag": str(meta.get("run_tag", "")),
    }
    if not identity["source_commit"] or not identity["run_tag"]:
        raise RuntimeError(f"{path} lacks source_commit/run_tag identity")
    nominal.update(_ledger_file_binding(
        _primary_ledger_path(seed, scenario),
        canonical_path=(
            f"decision_ledger_per_seed/{identity['run_tag']}/seed_{seed}/"
            f"agribrain__{scenario}.jsonl"
        ),
    ))
    return nominal, identity


def _perturb_df(df: pd.DataFrame, stressor: str, rng: np.random.Generator) -> pd.DataFrame:
    """Inject a controlled fault into a sensor trace.

    Stress doses are declared operational assumptions, not selected to obtain
    a desired ARI change: sensor noise 2 degC/5 percentage points RH, 10%
    missing telemetry, and a four-step (one-hour) delay at 15-minute cadence.
    """
    out = df.copy()
    # Latent environmental columns remain untouched. This function samples
    # only the primitive H3 dose. ``run_episode`` applies that dose after the
    # canonical stochastic observation stream (including intrinsic telemetry
    # carry-forward) has been constructed, so nominal and stressed streams
    # differ in one declared, replayable layer.
    out["inventory_policy_observed"] = out.get(
        "inventory_policy_observed", out["inventory_units"],
    ).astype(float)
    out["demand_policy_observed"] = out.get(
        "demand_policy_observed", out["demand_units"],
    ).astype(float)
    treatment: dict[str, object] = {
        "stressor": stressor,
        "n_steps": int(len(out)),
        "data_observation_treatment": stressor != "mcp_fault_injection",
        "delay_steps": 0,
        "missing_count": 0,
    }
    temp_noise = np.zeros(len(out), dtype=float)
    rh_noise = np.zeros(len(out), dtype=float)
    miss = np.zeros(len(out), dtype=bool)
    source_step = np.arange(len(out), dtype=int)
    def _array_hash(values) -> str:
        canonical = json.dumps(
            [float(value) for value in np.asarray(values, dtype=float)],
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    if stressor == "sensor_noise":
        temp_noise = rng.normal(0.0, 2.0, size=len(out))
        rh_noise = rng.normal(0.0, 5.0, size=len(out))
        treatment["temp_noise_sha256"] = _array_hash(temp_noise)
        treatment["rh_noise_sha256"] = _array_hash(rh_noise)
    elif stressor == "missing_data":
        miss = rng.random(len(out)) < 0.10
        if len(miss):
            miss[0] = False
        treatment["missing_count"] = int(np.count_nonzero(miss))
        treatment["missing_mask_sha256"] = hashlib.sha256(
            np.asarray(miss, dtype=np.uint8).tobytes()
        ).hexdigest()
    elif stressor == "telemetry_delay":
        delay_steps = 4
        source_step = np.maximum(np.arange(len(out), dtype=int) - delay_steps, 0)
        treatment["delay_steps"] = delay_steps
    elif stressor == "compounded":
        # Sensor noise, missingness, delay, and MCP fault injection are
        # combined; the latter is enabled by the caller.
        temp_noise = rng.normal(0.0, 2.0, size=len(out))
        rh_noise = rng.normal(0.0, 5.0, size=len(out))
        miss = rng.random(len(out)) < 0.10
        if len(miss):
            miss[0] = False
        source_step = np.maximum(np.arange(len(out), dtype=int) - 4, 0)
        treatment.update({
            "delay_steps": 4,
            "missing_count": int(np.count_nonzero(miss)),
            "missing_mask_sha256": hashlib.sha256(
                np.asarray(miss, dtype=np.uint8).tobytes()
            ).hexdigest(),
            "temp_noise_sha256": _array_hash(temp_noise),
            "rh_noise_sha256": _array_hash(rh_noise),
        })
    # Preserve the primitive dose at decision resolution. These columns flow
    # into the Merkle-covered retained ledger, allowing publication validation
    # to reconstruct the summary treatment hashes/counts instead of trusting
    # detached metadata.
    out["h3_temp_noise_c"] = np.asarray(temp_noise, dtype=float)
    out["h3_rh_noise_pct"] = np.asarray(rh_noise, dtype=float)
    out["h3_missing_observation"] = np.asarray(miss, dtype=bool)
    out["h3_telemetry_source_step_index"] = np.asarray(source_step, dtype=int)
    out.attrs["policy_observation_stressor"] = stressor
    treatment["treatment_sha256"] = hashlib.sha256(json.dumps(
        treatment, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    out.attrs["observation_treatment"] = treatment
    return out


def _run_pair_impl(
    episode_frames: Dict[int, pd.DataFrame] | pd.DataFrame,
    scenario: str,
    seed: int,
    with_faults: bool,
    modes: Iterable[str],
    ledger_condition: str,
) -> Dict[str, Dict[str, Any]]:
    """Run nominal or stressed adaptive policies under an equal budget.

    Each stressed arm starts from the same declared priors, adapts on episodes
    0--2, and is evaluated without updates on episode 3. Learner state persists
    only inside one (stressor, scenario, seed, mode) block. Thus H3 evaluates
    the adaptive architecture under a faulted input stream; it is not a
    frozen-policy-only perturbation test.
    """
    policy = Policy()
    # Match the primary publication posture exactly.  The only treatment
    # toggle is deliberate result replacement for the two MCP-fault stressors.
    policy.enable_failure_injection = bool(with_faults)
    policy.enable_mcp_reliability = (
        os.environ.get("MCP_RELIABILITY", "false").lower() == "true"
    )
    if policy.enable_mcp_reliability:
        raise ValueError(
            "Confirmatory H3 requires canonical MCP_RELIABILITY=false; "
            "fault stress is injected only through enable_failure_injection"
        )
    policy.enable_mcp_qos_routing = (
        os.environ.get("MCP_QOS_ROUTING", "false").lower() == "true"
    )
    policy.enable_pirag_counterfactual_eval = (
        os.environ.get("PIRAG_COUNTERFACTUAL", "false").lower() == "true"
    )
    policy.enable_physics_consistency_gate = (
        os.environ.get("PHYSICS_CONSISTENCY_GATE", "false").lower() == "true"
    )
    policy.enable_heterogeneous_profiles = (
        os.environ.get("HETEROGENEOUS_PROFILES", "false").lower() == "true"
    )
    policy.enable_research_metrics = (
        os.environ.get("RESEARCH_METRICS", "false").lower() == "true"
    )
    results: Dict[str, Dict[str, Any]] = {}
    n_episodes = int(os.environ.get("STRESS_LEARNING_EPISODES", "4"))
    if n_episodes != 4:
        raise ValueError(
            "Confirmatory H3 requires exactly three adaptation episodes and "
            "one frozen evaluation episode"
        )
    if isinstance(episode_frames, pd.DataFrame):
        # Compatibility for focused callers.  The confirmatory main path
        # always supplies independently generated, episode-indexed frames.
        frames = {idx: episode_frames for idx in range(n_episodes)}
    else:
        frames = dict(episode_frames)
    if set(frames) != set(range(n_episodes)):
        raise ValueError("stress arm requires episode frames indexed 0, 1, 2, 3")
    for mode in modes:
        print(f"  running mode={mode} scenario={scenario} faults={with_faults}")
        learner_state_cache: dict[str, dict] = {}
        ep = None
        arm_ledger_dir = (
            STRESS_LEDGER_ROOT / scenario / ledger_condition
            / f"seed_{seed}"
        )
        with decision_ledger_scope(arm_ledger_dir, reset=True):
            for episode_index in range(n_episodes):
                df = frames[episode_index]
                environment_seed = _stream_seed(
                    seed, scenario, episode_index, "environment",
                )
                policy_seed = _stream_seed(
                    seed, scenario, episode_index, "policy",
                )
                environment_id = _stream_id(
                    seed, scenario, episode_index, "environment",
                )
                policy_id = _stream_id(
                    seed, scenario, episode_index, "policy",
                )
                mode_rng = np.random.default_rng(policy_seed)
                stoch = make_stochastic_layer(
                    np.random.default_rng(environment_seed),
                    stream_seed=environment_seed,
                )
                ep = run_episode(
                    df, mode, policy, mode_rng, scenario, stoch=stoch,
                    seed=seed, benchmark_seed=seed,
                    episode_index=episode_index,
                    environment_stream_id=environment_id,
                    policy_stream_id=policy_id,
                    stochastic_stream_id=environment_id,
                    learner_state_cache=learner_state_cache,
                    learning_enabled=episode_index < 3,
                )
        assert ep is not None
        if ep.get("learning_enabled") is not False or (
            ep.get("episode_phase") != "frozen_evaluation"
        ):
            raise RuntimeError(
                f"H3 retained arm was not frozen: {scenario}/{mode}/seed={seed}"
            )
        freeze_summary = ep.get("learner_freeze_summary") or {}
        role_freeze = freeze_summary.get("policy_delta_frozen_by_role") or {}
        if (
            freeze_summary.get("learners_frozen") is not True
            or freeze_summary.get("learner_phase") != "frozen_evaluation"
            or freeze_summary.get("freeze_reason") != "retained_episode_3"
            or freeze_summary.get("context_matrix_frozen") is not True
            or freeze_summary.get("reward_shaping_frozen") is not True
            or not role_freeze
            or not all(value is True for value in role_freeze.values())
        ):
            raise RuntimeError(
                f"H3 retained arm lacks learner-freeze evidence: "
                f"{scenario}/{mode}/seed={seed}"
            )
        if not np.isfinite(ep["ari"]) or not np.isfinite(ep["waste"]) or not np.isfinite(ep["slca"]):
            raise ValueError(f"Non-finite episode metrics for mode={mode}, scenario={scenario}")
        run_tag = os.environ.get("RUN_TAG", "").strip()
        if not run_tag:
            raise RuntimeError("H3 ledger binding requires non-empty RUN_TAG")
        ledger_binding = _ledger_file_binding(
            Path(str(ep.get("decision_ledger_path", ""))),
            canonical_path=(
                f"decision_ledger_h3/{run_tag}/"
                f"{scenario}/{ledger_condition}/seed_{seed}/"
                f"{mode}__{scenario}.jsonl"
            ),
        )
        results[mode] = {
            "ari": float(ep["ari"]),
            "waste": float(ep["waste"]),
            "slca": float(ep["slca"]),
            # Single canonical RLE: EU-hierarchy + severity-weighted.
            "rle": float(ep["rle"]),
            "carbon": float(ep["carbon"]),
            "equity": float(ep["equity"]),
            "message_count": int(ep["message_count"]),
            "constraint_violation_rate": float(ep.get("constraint_violation_rate", 0.0)),
            "decision_latency_ms": float(ep.get("mean_decision_latency_ms", 0.0)),
            # Outcome-side disposition: under stressors, did the policy
            # still choose redistribution/recovery on the common operating-
            # envelope event set, or did the noise drive it back toward an
            # undifferentiated cold_chain default? Carrying these through stress_summary
            # lets downstream readers spot stressor-induced regressions
            # in policy decisiveness without re-running the simulator.
            "downstream_violation_rate": float(ep.get("downstream_violation_rate", 0.0)),
            "contained_violation_rate": float(ep.get("contained_violation_rate", 0.0)),
            # Protocol execution evidence for the retained episode. Deliberate
            # H3 post-call result drops are not protocol failures and therefore
            # leave these error counters at zero.
            "protocol_interaction_count": int(ep.get("protocol_interaction_count", 0)),
            "protocol_jsonrpc_error_count": int(ep.get("protocol_jsonrpc_error_count", 0)),
            "protocol_tool_iserror_count": int(ep.get("protocol_tool_iserror_count", 0)),
            "protocol_real_tool_iserror_count": int(ep.get("protocol_real_tool_iserror_count", 0)),
            "protocol_error_count": int(ep.get("protocol_error_count", 0)),
            "protocol_dropped_interaction_count": int(
                ep.get("protocol_dropped_interaction_count", 0)
            ),
            "dispatcher_tool_failure_count": int(
                ep.get("dispatcher_tool_failure_count", 0)
            ),
            "context_execution_error_count": int(
                ep.get("context_execution_error_count", 0)
            ),
            # Deliberate H3 treatment exposure.  The schedule count is not an
            # observed drop count: the MCP channel can already be unavailable
            # (notably cyber_outage after hour 24).  Retain both the number of
            # steps on which injection actually triggered and the number of
            # individual returned tool results replaced on those steps.
            "fault_injection_scheduled_opportunity_steps": int(
                ep.get("fault_injection_scheduled_opportunity_steps", 0)
            ),
            "fault_injection_trigger_steps": int(
                ep.get("fault_injection_trigger_steps", 0)
            ),
            "fault_injected_tool_result_count": int(
                ep.get("fault_injected_tool_result_count", 0)
            ),
            "trace_schema_version": int(ep["trace_schema_version"]),
            "benchmark_seed": int(ep["benchmark_seed"]),
            "episode_index": int(ep["episode_index"]),
            "environment_stream_id": str(ep["environment_stream_id"]),
            "policy_stream_id": str(ep["policy_stream_id"]),
            "stochastic_stream_id": str(ep["stochastic_stream_id"]),
            "context_prior_sha256": str(ep["context_prior_sha256"]),
            "policy_theta_initial_sha256": str(
                ep["policy_theta_initial_sha256"]
            ),
            "spoilage_estimator": dict(ep["spoilage_estimator"]),
            "latent_spoilage_model": dict(ep["latent_spoilage_model"]),
            "latent_environment_sha256": str(ep["latent_environment_sha256"]),
            "observed_policy_input_sha256": str(
                ep["observed_policy_input_sha256"]
            ),
            "demand_observation_sha256": str(ep["demand_observation_sha256"]),
            "demand_forecast_method": str(ep["demand_forecast_method"]),
            "supply_forecast_method": str(ep["supply_forecast_method"]),
            "learning_enabled": bool(ep["learning_enabled"]),
            "episode_phase": str(ep["episode_phase"]),
            "dispatch_opportunity_count": int(ep["dispatch_opportunity_count"]),
            "dispatch_cadence_hours": float(ep["dispatch_cadence_hours"]),
            "learner_summary": ep.get("learner_summary"),
            "theta_learner_summary": ep.get("theta_learner_summary"),
            "reward_shaping_learner_summary": ep.get(
                "reward_shaping_learner_summary"
            ),
            "learner_freeze_summary": ep.get("learner_freeze_summary"),
            "observation_treatment": dict(frames[3].attrs.get(
                "observation_treatment",
                {
                    "stressor": "nominal",
                    "n_steps": int(len(df)),
                    "data_observation_treatment": False,
                    "delay_steps": 0,
                    "missing_count": 0,
                },
            )),
            **ledger_binding,
        }
    return results


def _run_pair(
    episode_frames: Dict[int, pd.DataFrame] | pd.DataFrame,
    scenario: str,
    seed: int,
    with_faults: bool,
    modes: Iterable[str],
    ledger_condition: str,
) -> Dict[str, Dict[str, Any]]:
    """Run a stress arm under the seed-level Source-7 policy prior."""
    import src.models.action_selection as action_selection

    original_theta = action_selection.THETA.copy()
    action_selection.THETA = policy_theta_for_seed(original_theta, seed)
    try:
        return _run_pair_impl(
            episode_frames, scenario, seed, with_faults, modes, ledger_condition,
        )
    finally:
        action_selection.THETA = original_theta


def _degrade(nom: Dict[str, float], stressed: Dict[str, float]) -> Dict[str, float]:
    return {
        "ari_delta": float(stressed["ari"] - nom["ari"]),
        "waste_delta": float(stressed["waste"] - nom["waste"]),
        "slca_delta": float(stressed["slca"] - nom["slca"]),
        "rle_delta": float(stressed["rle"] - nom["rle"]),
        "carbon_delta": float(stressed["carbon"] - nom["carbon"]),
        "equity_delta": float(stressed["equity"] - nom["equity"]),
        "constraint_violation_delta": float(stressed["constraint_violation_rate"] - nom["constraint_violation_rate"]),
        "latency_ms_delta": float(stressed["decision_latency_ms"] - nom["decision_latency_ms"]),
    }


def _ledger_set_binding(
    cells_by_seed: Dict[int, Dict[str, Any]], seed_list: Iterable[int],
) -> Dict[str, Any]:
    """Hash the exact ordered seed-to-ledger inventory for one H3 arm."""
    records = []
    for seed in seed_list:
        cell = cells_by_seed[int(seed)]
        record = {
            "seed": int(seed),
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
            raise RuntimeError(f"invalid retained ledger binding for seed {seed}")
        records.append(record)
    return {
        "count": len(records),
        "decision_count": sum(int(record["n_records"]) for record in records),
        "sha256": hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
    }


def _within_margin(ari_delta: float) -> bool:
    return abs(float(ari_delta)) <= STRESS_THRESHOLDS["ari_abs_delta_max"]


def _treatment_exposure_verified(
    stressor: str, cells: Iterable[Dict[str, Any]],
) -> bool:
    """Require observed, nonzero treatment exposure in every seed cell."""
    records = list(cells)
    if not records:
        return False
    treatments = [record.get("observation_treatment") or {} for record in records]
    if any(treatment.get("stressor") != stressor for treatment in treatments):
        return False
    if any(
        treatment.get("n_steps") != 288
        or treatment.get("data_observation_treatment") is not (
            stressor != "mcp_fault_injection"
        )
        for treatment in treatments
    ):
        return False
    for treatment in treatments:
        claimed_hash = treatment.get("treatment_sha256")
        if (
            not isinstance(claimed_hash, str)
            or len(claimed_hash) != 64
            or any(char not in "0123456789abcdef" for char in claimed_hash)
        ):
            return False
        unhashed = dict(treatment)
        unhashed.pop("treatment_sha256", None)
        expected_hash = hashlib.sha256(json.dumps(
            unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if claimed_hash != expected_hash:
            return False
    if stressor in {"sensor_noise", "compounded"} and any(
        not isinstance(treatment.get(field), str)
        or len(treatment[field]) != 64
        or any(char not in "0123456789abcdef" for char in treatment[field])
        for treatment in treatments
        for field in ("temp_noise_sha256", "rh_noise_sha256")
    ):
        return False
    if stressor in {"missing_data", "compounded"} and any(
        int(treatment.get("missing_count", 0)) <= 0
        for treatment in treatments
    ):
        return False
    if stressor in {"telemetry_delay", "compounded"} and any(
        int(treatment.get("delay_steps", 0)) != 4
        for treatment in treatments
    ):
        return False
    if stressor in {"mcp_fault_injection", "compounded"} and any(
        int(record.get("fault_injection_trigger_steps", 0)) <= 0
        or int(record.get("fault_injected_tool_result_count", 0)) <= 0
        for record in records
    ):
        return False
    return True


def _equivalence_tost(values: Iterable[float], margin: float) -> Dict[str, float | bool | int]:
    """One-sample TOST for mean seed-level change within ``±margin``.

    Equivalence at alpha=0.05 is equivalent to the 90% two-sided confidence
    interval lying wholly inside the margin. The two 90% interval endpoints
    are also the corresponding one-sided 95% lower and upper bounds. Their
    maximum absolute excursion is emitted explicitly so a near-margin point
    estimate cannot be mistaken for demonstrated equivalence. A two-sided 95%
    interval is retained for descriptive uncertainty.
    """
    return equivalence_tost(values, margin)


def main() -> None:
    required_paths = (
        "STRESS_OUTPUT_DIR", "STRESS_LEDGER_ROOT", "STRESS_PRIMARY_SEEDS_DIR",
    )
    missing_paths = [name for name in required_paths if not os.environ.get(name, "").strip()]
    if missing_paths:
        raise RuntimeError(
            "Confirmatory H3 requires explicit run-scoped paths: "
            + ", ".join(missing_paths)
        )
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found: {DATA_CSV}")
    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    stressors = ("sensor_noise", "missing_data", "telemetry_delay",
                 "mcp_fault_injection", "compounded")
    scenarios_env = os.environ.get("STRESS_SCENARIOS", "").strip()
    if scenarios_env:
        scenarios = [s.strip() for s in scenarios_env.split(",") if s.strip()]
    else:
        scenarios = list(SCENARIOS)
    if (
        not scenarios
        or len(set(scenarios)) != len(scenarios)
        or any(scenario not in SCENARIOS for scenario in scenarios)
    ):
        raise ValueError(
            "STRESS_SCENARIOS must be a non-empty unique subset of the "
            f"declared scenario panel {list(SCENARIOS)!r}"
        )
    if len(scenarios) != 1:
        raise ValueError(
            "Confirmatory H3 tasks must execute exactly one scenario; submit "
            "the locked five-task scenario array through hpc/hpc_run.sh"
        )
    execution_provenance = None
    if os.environ.get("STRICT_VALIDATION", "0") == "1":
        scenario = scenarios[0]
        if scenario not in CORE_SCENARIOS:
            raise RuntimeError(
                f"strict H3 scenario {scenario!r} is outside the locked panel"
            )
        logical_task_index = CORE_SCENARIOS.index(scenario)
        execution_provenance = build_array_execution_provenance(
            stage="core_stress_array",
            logical_task_index=logical_task_index,
        )
        if execution_provenance["slurm_array_task_id"] != logical_task_index:
            raise RuntimeError(
                "SLURM_ARRAY_TASK_ID does not map to the requested H3 scenario"
            )
    run_tag = os.environ.get("RUN_TAG", "").strip()
    if not run_tag:
        raise RuntimeError("Confirmatory H3 requires RUN_TAG")
    canonical_results = (_SIMULATION_DIR / "results").resolve()
    expected_output = (
        canonical_results / "stress_runs" / run_tag / scenarios[0]
    ).resolve()
    expected_ledgers = (
        canonical_results / "decision_ledger_h3" / run_tag
    ).resolve()
    expected_primary = (
        canonical_results / "benchmark_seeds" / run_tag
    ).resolve()
    resolved_paths = {
        "STRESS_OUTPUT_DIR": RESULTS_DIR.resolve(),
        "STRESS_LEDGER_ROOT": STRESS_LEDGER_ROOT.resolve(),
        "STRESS_PRIMARY_SEEDS_DIR": PRIMARY_SEEDS_DIR.resolve(),
    }
    expected_paths = {
        "STRESS_OUTPUT_DIR": expected_output,
        "STRESS_LEDGER_ROOT": expected_ledgers,
        "STRESS_PRIMARY_SEEDS_DIR": expected_primary,
    }
    for name, actual in resolved_paths.items():
        if actual != expected_paths[name]:
            raise RuntimeError(
                f"{name} is not the exact run-scoped H3 path: "
                f"{actual} != {expected_paths[name]}"
            )
    if RESULTS_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing H3 task output: {RESULTS_DIR}"
        )
    RESULTS_DIR.parent.mkdir(parents=True, exist_ok=True)
    max_rows_env = os.environ.get("STRESS_MAX_ROWS", "").strip()
    max_rows = int(max_rows_env) if max_rows_env else 0
    # H3 is an AGRI-BRAIN-only prespecified equivalence panel.  Adding
    # comparator modes here would change both the estimand and episode count.
    stress_modes = {stressor: ("agribrain",) for stressor in stressors}
    if max_rows > 0:
        raise ValueError(
            "STRESS_MAX_ROWS is diagnostic-only and cannot be combined with "
            "the primary-nominal reuse required by confirmatory H3"
        )
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    rows = []

    # This confirmatory entry point is locked to the exact 20-seed primary
    # panel. Smaller smoke runs must use a separately labelled diagnostic
    # driver and cannot emit publication H3 artifacts.
    n_seeds = int(os.environ.get("STRESS_N_SEEDS", "20"))
    seed_list = _confirmatory_seed_panel(n_seeds)
    primary_identities: set[tuple[str, str]] = set()

    for scenario in scenarios:
        print(f"\n[stress] scenario={scenario}")
        # Reconstruct the same four episode-indexed latent frames used by the
        # primary benchmark.  Only the stressed arms execute here; nominal
        # episode-3 endpoints are loaded from the primary seed envelopes.
        scenarios_by_seed: dict[int, dict[int, pd.DataFrame]] = {}
        baselines_by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for seed in seed_list:
            episode_frames: dict[int, pd.DataFrame] = {}
            for episode_index in range(4):
                scenario_seed = _stream_seed(
                    seed, scenario, episode_index, "scenario",
                )
                environment_seed = _stream_seed(
                    seed, scenario, episode_index, "environment",
                )
                episode_frames[episode_index] = apply_scenario(
                    df_base,
                    scenario,
                    Policy(),
                    np.random.default_rng(scenario_seed),
                    stoch=make_stochastic_layer(
                        np.random.default_rng(environment_seed),
                        stream_seed=environment_seed,
                    ),
                )
            scenarios_by_seed[seed] = episode_frames
            primary_cell, identity = _load_primary_nominal(scenario, seed)
            baselines_by_seed[seed] = {"agribrain": primary_cell}
            primary_identities.add(
                (identity["source_commit"], identity["run_tag"])
            )
        summary[scenario] = {"baseline_seed_list": list(seed_list),
                             "baseline_by_seed": baselines_by_seed}

        for stressor in stressors:
            print(f" [stress] stressor={stressor}")
            modes = stress_modes[stressor]
            stressed_by_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
            for seed in seed_list:
                stressed_frames: dict[int, pd.DataFrame] = {}
                for episode_index, scenario_df in scenarios_by_seed[seed].items():
                    # Independent stress dose by seed and episode; treatment
                    # remains identical across policy arms (H3 has one arm).
                    key = (
                        f"stress|{scenario}|{stressor}|{seed}|{episode_index}"
                    ).encode("utf-8")
                    cell_seed = int.from_bytes(
                        hashlib.sha256(key).digest()[:8], "big",
                    )
                    stressed_frames[episode_index] = _perturb_df(
                        scenario_df, stressor, np.random.default_rng(cell_seed),
                    )
                stressed_by_seed[seed] = _run_pair(
                    stressed_frames, scenario, seed=seed,
                    with_faults=(stressor in {"mcp_fault_injection", "compounded"}),
                    modes=modes,
                    ledger_condition=stressor,
                )
                for mode in modes:
                    nominal = baselines_by_seed[seed][mode]
                    stressed = stressed_by_seed[seed][mode]
                    if (
                        nominal["latent_environment_sha256"]
                        != stressed["latent_environment_sha256"]
                    ):
                        raise RuntimeError(
                            "H3 observation treatment changed latent truth: "
                            f"scenario={scenario}, stressor={stressor}, "
                            f"seed={seed}, mode={mode}"
                        )
                    expected_environment_id = _stream_id(
                        seed, scenario, 3, "environment",
                    )
                    expected_policy_id = _stream_id(
                        seed, scenario, 3, "policy",
                    )
                    if any((
                        nominal["environment_stream_id"]
                        != expected_environment_id,
                        stressed["environment_stream_id"]
                        != expected_environment_id,
                        nominal["stochastic_stream_id"]
                        != expected_environment_id,
                        stressed["stochastic_stream_id"]
                        != expected_environment_id,
                        nominal["policy_stream_id"] != expected_policy_id,
                        stressed["policy_stream_id"] != expected_policy_id,
                    )):
                        raise RuntimeError(
                            "H3 retained stream identity mismatch: "
                            f"scenario={scenario}, stressor={stressor}, "
                            f"seed={seed}, mode={mode}"
                        )
                    for prior_hash in (
                        "context_prior_sha256", "policy_theta_initial_sha256",
                    ):
                        if nominal[prior_hash] != stressed[prior_hash]:
                            raise RuntimeError(
                                "H3 stressed arm did not use the primary prior: "
                                f"scenario={scenario}, stressor={stressor}, "
                                f"seed={seed}, mode={mode}, field={prior_hash}"
                            )
                    if (
                        nominal["demand_observation_sha256"]
                        != stressed["demand_observation_sha256"]
                    ):
                        raise RuntimeError(
                            "H3 temperature/humidity/MCP stress changed the "
                            "seed-locked demand process: "
                            f"scenario={scenario}, stressor={stressor}, "
                            f"seed={seed}, mode={mode}"
                        )
                    if stressor != "mcp_fault_injection" and (
                        nominal["observed_policy_input_sha256"]
                        == stressed["observed_policy_input_sha256"]
                    ):
                        raise RuntimeError(
                            "H3 observation treatment was a no-op: "
                            f"scenario={scenario}, stressor={stressor}, "
                            f"seed={seed}, mode={mode}"
                        )
                    if stressor == "mcp_fault_injection" and (
                        nominal["observed_policy_input_sha256"]
                        != stressed["observed_policy_input_sha256"]
                    ):
                        raise RuntimeError(
                            "MCP-only fault injection changed the sensor/forecast "
                            "observation stream: "
                            f"scenario={scenario}, seed={seed}, mode={mode}"
                        )
            summary[scenario][stressor] = stressed_by_seed

            # Aggregate deltas across seeds: mean, std, and per-seed list.
            for mode in modes:
                deltas_list = []
                for seed in seed_list:
                    deltas_list.append(_degrade(
                        baselines_by_seed[seed][mode],
                        stressed_by_seed[seed][mode],
                    ))
                # Build aggregate row: mean and std across seeds for each
                # delta field, plus a 'pass' Clopper-Pearson CI.
                agg = {"Scenario": scenario, "Stressor": stressor, "Method": mode,
                       "n_seeds": n_seeds,
                       "Confirmatory_H3": bool(mode == "agribrain"),
                       "inferential_status": (
                           "confirmatory_h3" if mode == "agribrain"
                           else "descriptive_comparator"
                       )}
                for k in deltas_list[0]:
                    vals = np.array([d[k] for d in deltas_list], dtype=float)
                    agg[k] = float(np.mean(vals))
                    agg[k + "_std"] = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
                # Report the realised fault dose independently of the fixed
                # schedule.  These fields are zero for non-MCP stressors and
                # can be below the 28 scheduled opportunities when the tool
                # channel is unavailable in the scenario.
                for exposure_field in (
                    "fault_injection_scheduled_opportunity_steps",
                    "fault_injection_trigger_steps",
                    "fault_injected_tool_result_count",
                ):
                    exposure = np.asarray([
                        stressed_by_seed[seed][mode][exposure_field]
                        for seed in seed_list
                    ], dtype=float)
                    agg[f"{exposure_field}_mean"] = float(np.mean(exposure))
                    agg[f"{exposure_field}_min"] = float(np.min(exposure))
                    agg[f"{exposure_field}_max"] = float(np.max(exposure))
                agg["treatment_exposure_verified"] = (
                    _treatment_exposure_verified(
                        stressor,
                        [stressed_by_seed[seed][mode] for seed in seed_list],
                    )
                )
                stressed_ledger_binding = _ledger_set_binding(
                    {
                        seed: stressed_by_seed[seed][mode]
                        for seed in seed_list
                    },
                    seed_list,
                )
                nominal_ledger_binding = _ledger_set_binding(
                    {
                        seed: baselines_by_seed[seed][mode]
                        for seed in seed_list
                    },
                    seed_list,
                )
                agg.update({
                    "retained_stressed_decision_ledger_count": (
                        stressed_ledger_binding["count"]
                    ),
                    "retained_stressed_decision_count": (
                        stressed_ledger_binding["decision_count"]
                    ),
                    "retained_stressed_decision_ledger_set_sha256": (
                        stressed_ledger_binding["sha256"]
                    ),
                    "reused_nominal_decision_ledger_count": (
                        nominal_ledger_binding["count"]
                    ),
                    "reused_nominal_decision_count": (
                        nominal_ledger_binding["decision_count"]
                    ),
                    "reused_nominal_decision_ledger_set_sha256": (
                        nominal_ledger_binding["sha256"]
                    ),
                })
                ari_tost = _equivalence_tost(
                    [d["ari_delta"] for d in deltas_list],
                    STRESS_THRESHOLDS["ari_abs_delta_max"],
                )
                for key, value in ari_tost.items():
                    agg[f"ari_tost_{key}"] = value
                rows.append(agg)

    if len(primary_identities) != 1:
        raise RuntimeError(
            "H3 primary nominal envelopes have inconsistent source/run identity: "
            f"{sorted(primary_identities)}"
        )
    primary_source_commit, primary_run_tag = next(iter(primary_identities))
    requested_commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    requested_run_tag = os.environ.get("RUN_TAG", "").strip()
    if requested_commit and requested_commit != primary_source_commit:
        raise RuntimeError(
            "H3 source commit differs from reused primary benchmark: "
            f"{requested_commit} != {primary_source_commit}"
        )
    if requested_run_tag and requested_run_tag != primary_run_tag:
        raise RuntimeError(
            "H3 run tag differs from reused primary benchmark: "
            f"{requested_run_tag} != {primary_run_tag}"
        )

    out_payload = {
        "meta": {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "source_commit": primary_source_commit,
            "run_tag": primary_run_tag,
            "execution_provenance": execution_provenance,
            "nominal_reference": "reused_primary_benchmark_episode_3",
            "nominal_seed_directory": str(PRIMARY_SEEDS_DIR),
            "scenarios": scenarios,
            "max_rows": max_rows if max_rows > 0 else None,
            "thresholds": STRESS_THRESHOLDS,
            "adaptation_episodes_per_stressed_condition": 3,
            "frozen_evaluation_episodes_per_stressed_condition": 1,
            "adaptation_posture": (
                "the primary nominal endpoint is reused; each stressed arm "
                "adapts from the same declared priors on episodes 0-2 and "
                "retains a no-update frozen episode 3"
            ),
            "decision_history_posture": (
                "fresh in-memory decision history at every episode; only "
                "learner state persists within an arm; stressed JSONL audit "
                "outputs are partitioned by scenario, stressor, seed, and mode"
            ),
            "mcp_reliability_posture": os.environ.get(
                "MCP_RELIABILITY", "false",
            ).lower(),
            "state_design": (
                "policy routes only on *_policy_observed; every reported "
                "endpoint uses common *_outcome_environmental latent truth"
            ),
            "common_random_numbers": (
                "mode-independent episode-indexed environment and policy "
                "stream initialization; latent hash equality is enforced"
            ),
            "mcp_fault_dose": {
                "schedule_rule": (
                    "after successful dispatch, replace each invoked MCP tool "
                    "result when int(hour) % 11 == 0"
                ),
                "full_trace_scheduled_opportunity_steps": 28,
                "full_trace_total_steps": 288,
                "full_trace_scheduled_opportunity_fraction": 28 / 288,
                "observed_exposure_fields": {
                    "trigger_steps": "fault_injection_trigger_steps_*",
                    "tool_results_replaced": "fault_injected_tool_result_count_*",
                },
                "interpretation": (
                    "28 is the number of schedule opportunities on a complete "
                    "72-hour trace, not a guaranteed affected-step count. "
                    "Observed exposure is reported per cell because an MCP "
                    "channel can already be unavailable."
                ),
            },
            "retained_ledger_design": {
                "stressed_ledgers_per_scenario_task": (
                    len(stressors) * len(seed_list)
                ),
                "stressed_decisions_per_scenario_task": (
                    len(stressors) * len(seed_list) * 288
                ),
                "reused_primary_nominal_ledgers_per_scenario_task": len(seed_list),
                "newly_executed_nominal_episodes": 0,
                "canonical_stressed_ledger_root": (
                    f"decision_ledger_h3/{primary_run_tag}"
                ),
                "canonical_nominal_ledger_root": (
                    f"decision_ledger_per_seed/{primary_run_tag}"
                ),
            },
        },
        "results": summary,
    }
    output_stage = Path(tempfile.mkdtemp(
        prefix=f".{RESULTS_DIR.name}.partial.", dir=str(RESULTS_DIR.parent),
    ))
    (output_stage / "stress_summary.json").write_text(
        json.dumps(out_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    df = pd.DataFrame(rows)
    df.to_csv(output_stage / "stress_degradation.csv", index=False)
    # Formal pass/fail comes from TOST on the seed-level differences. The
    # simple mean-within-margin flag and per-seed within-margin fraction are
    # retained as descriptive diagnostics only; the latter receives an exact
    # Clopper-Pearson interval.
    pass_rows = []
    for _, r in df.iterrows():
        rec = r.to_dict()
        # Cross-mode comparison rows (added 2026-04) carry
        # `comparison_type == "cross_mode_under_stress"` and a synthetic
        # Method like `agribrain_minus_hybrid_rl_stressed`. They are
        # descriptive only — no pass/fail threshold — and so we skip
        # the per-mode pass-rate computation for them.
        if rec.get("comparison_type") == "cross_mode_under_stress":
            rec["Confirmatory_H3"] = False
            rec["H3_Pass"] = None
            rec["Pass_Mean"] = None
            rec["Pass"] = None  # descriptive only
            rec["Pass_Count"] = None
            rec["Pass_N"] = None
            rec["Pass_Rate"] = None
            rec["Pass_Rate_CI_Low"] = None
            rec["Pass_Rate_CI_High"] = None
            # Cross-mode rows have *_diff fields not *_Base/_Stressed;
            # set the canonical Base/Stressed columns to NaN so the
            # validator schema check passes without inventing numbers.
            for col in ("ARI_Base", "ARI_Stressed", "Waste_Base",
                        "Waste_Stressed", "SLCA_Base", "SLCA_Stressed"):
                rec[col] = None
            _CANONICAL_THRESHOLDS = {
                "ari_abs_delta_max":              "Threshold_ARI",
                "waste_delta_max":                "Threshold_Waste",
                "slca_delta_min":                 "Threshold_SLCA",
                "rle_delta_min":                  "Threshold_RLE",
                "carbon_delta_max":               "Threshold_Carbon",
                "equity_delta_min":               "Threshold_Equity",
                "constraint_violation_delta_max": "Threshold_CVR",
                "latency_ms_delta_max":           "Threshold_LatencyMs",
            }
            for k, col in _CANONICAL_THRESHOLDS.items():
                rec[col] = STRESS_THRESHOLDS[k]
            pass_rows.append(rec)
            continue

        rec["Pass_Mean"] = _within_margin(rec["ari_delta"])
        rec["Pass_Equivalence"] = bool(
            rec.get("ari_tost_equivalent_alpha_0p05", False)
        )
        rec["Pass"] = rec["Pass_Equivalence"]
        rec["Confirmatory_H3"] = bool(rec.get("Method") == "agribrain")
        rec["H3_Pass"] = (
            rec["Pass_Equivalence"] if rec["Confirmatory_H3"] else None
        )
        # Per-seed pass rate
        scen, stressor, mode = rec["Scenario"], rec["Stressor"], rec["Method"]
        per_seed_passes = []
        if scen in summary and stressor in summary[scen] and "baseline_by_seed" in summary[scen]:
            for seed in summary[scen]["baseline_seed_list"]:
                base_for_mode = summary[scen]["baseline_by_seed"][seed].get(mode)
                stressed_for_mode = summary[scen][stressor].get(seed, {}).get(mode)
                if base_for_mode is None or stressed_for_mode is None:
                    continue
                d_seed = _degrade(base_for_mode, stressed_for_mode)
                per_seed_passes.append(1 if _within_margin(d_seed["ari_delta"]) else 0)
        # Surface mean absolute levels alongside the paired mean deltas. The
        # old implementation copied only the first seed into these columns,
        # making the levels inconsistent with the 20-seed delta in the row.
        metric_columns = {
            "ari": "ARI", "waste": "Waste", "slca": "SLCA",
            "rle": "RLE", "carbon": "Carbon", "equity": "Equity",
            "constraint_violation_rate": "CVR",
            "decision_latency_ms": "LatencyMs",
        }
        for metric, label in metric_columns.items():
            base_values = []
            stressed_values = []
            for seed in summary[scen]["baseline_seed_list"]:
                base_for_mode = summary[scen]["baseline_by_seed"][seed].get(mode)
                stressed_for_mode = summary[scen][stressor].get(seed, {}).get(mode)
                if base_for_mode is None or stressed_for_mode is None:
                    continue
                base_values.append(float(base_for_mode[metric]))
                stressed_values.append(float(stressed_for_mode[metric]))
            if not base_values or len(base_values) != len(stressed_values):
                raise RuntimeError(
                    f"Incomplete absolute stress levels for "
                    f"{scen}/{stressor}/{mode}/{metric}"
                )
            rec[f"{label}_Base"] = float(np.mean(base_values))
            rec[f"{label}_Stressed"] = float(np.mean(stressed_values))
        if len(per_seed_passes) != n_seeds:
            raise RuntimeError(
                f"Incomplete paired stress panel for {scen}/{stressor}/{mode}: "
                f"expected {n_seeds}, found {len(per_seed_passes)}"
            )
        n_pass = sum(per_seed_passes)
        n_total = len(per_seed_passes)
        rec["Pass_Count"] = n_pass
        rec["Pass_N"] = n_total
        rec["Pass_Rate"] = n_pass / n_total
        # Clopper-Pearson 95 % CI on binomial proportion.
        from scipy.stats import beta as _beta
        alpha = 0.05
        lo = float(_beta.ppf(alpha / 2, n_pass, n_total - n_pass + 1)) if n_pass > 0 else 0.0
        hi = float(_beta.ppf(1 - alpha / 2, n_pass + 1, n_total - n_pass)) if n_pass < n_total else 1.0
        rec["Pass_Rate_CI_Low"] = lo
        rec["Pass_Rate_CI_High"] = hi
        rec["Pass_Rate_CI_Method"] = "Clopper-Pearson exact 95%"
        # Threshold columns. The publication validator pins exact
        # names: Threshold_ARI / Threshold_Waste / Threshold_SLCA /
        # Threshold_RLE / Threshold_Carbon / Threshold_Equity /
        # Threshold_CVR / Threshold_LatencyMs. Use the canonical names
        # rather than the title-cased delta keys.
        _CANONICAL_THRESHOLDS = {
            "ari_abs_delta_max":              "Threshold_ARI",
            "waste_delta_max":                "Threshold_Waste",
            "slca_delta_min":                 "Threshold_SLCA",
            "rle_delta_min":                  "Threshold_RLE",
            "carbon_delta_max":               "Threshold_Carbon",
            "equity_delta_min":               "Threshold_Equity",
            "constraint_violation_delta_max": "Threshold_CVR",
            "latency_ms_delta_max":           "Threshold_LatencyMs",
        }
        for k, col in _CANONICAL_THRESHOLDS.items():
            rec[col] = STRESS_THRESHOLDS[k]
        pass_rows.append(rec)
    pd.DataFrame(pass_rows).to_csv(output_stage / "stress_passfail.csv", index=False)
    h3_cells = [
        r for r in pass_rows
        if r.get("Confirmatory_H3") is True
        and r.get("Method") == "agribrain"
        and r.get("comparison_type") != "cross_mode_under_stress"
    ]
    expected_h3_keys = {
        (scenario, stressor) for scenario in scenarios for stressor in stressors
    }
    actual_h3_keys = {
        (str(r.get("Scenario")), str(r.get("Stressor"))) for r in h3_cells
    }
    if len(h3_cells) != len(expected_h3_keys) or actual_h3_keys != expected_h3_keys:
        raise RuntimeError(
            "Confirmatory H3 output is not the exact AGRI-BRAIN-only "
            f"scenario-stressor panel: missing={sorted(expected_h3_keys - actual_h3_keys)}, "
            f"unexpected={sorted(actual_h3_keys - expected_h3_keys)}"
        )
    h3_episode_accounting = build_h3_episode_accounting(
        n_seeds=n_seeds,
        n_scenarios=len(scenarios),
        n_stressors=len(stressors),
        episodes_per_condition=max(
            1, int(os.environ.get("STRESS_LEARNING_EPISODES", "4"))
        ),
        nominal_reference_reused=True,
    )
    h3_payload = {
        "source_commit": out_payload["meta"]["source_commit"],
        "run_tag": out_payload["meta"]["run_tag"],
        "execution_provenance": out_payload["meta"]["execution_provenance"],
        "hypothesis": (
            "H3: for every declared scenario-stressor cell, the mean paired "
            "seed-level AGRI-BRAIN ARI change is equivalent to zero within "
            "the ±0.01 margin."
        ),
        "test": "paired one-sample TOST on seed-level ARI differences",
        "alpha": 0.05,
        "equivalence_margin": STRESS_THRESHOLDS["ari_abs_delta_max"],
        "confirmatory_method": "agribrain",
        "expected_scenarios": list(scenarios),
        "expected_stressors": list(stressors),
        "expected_n_cells": len(scenarios) * len(stressors),
        "global_decision_rule": (
            "intersection-union: supported only when every prespecified "
            "AGRI-BRAIN scenario-stressor cell passes TOST and has verified "
            "nonzero treatment exposure"
        ),
        "one_sided_bound_rule": (
            "max(-one_sided_95_lower_bound, one_sided_95_upper_bound) < 0.01"
        ),
        "episode_accounting": h3_episode_accounting,
        "adaptation_episodes_per_stressed_condition": 3,
        "frozen_evaluation_episodes_per_stressed_condition": 1,
        "nominal_reference": "reused_primary_benchmark_episode_3",
        "adaptation_posture": out_payload["meta"]["adaptation_posture"],
        "mcp_fault_dose": out_payload["meta"]["mcp_fault_dose"],
        "n_cells": len(h3_cells),
        "n_cells_equivalent": sum(bool(r.get("Pass_Equivalence")) for r in h3_cells),
        "n_cells_with_verified_exposure": sum(
            bool(r.get("treatment_exposure_verified")) for r in h3_cells
        ),
        "retained_stressed_decision_ledger_count": sum(
            int(r["retained_stressed_decision_ledger_count"])
            for r in h3_cells
        ),
        "reused_nominal_decision_ledger_references": len(seed_list),
        "newly_executed_nominal_episodes": 0,
        "supported_all_cells": bool(h3_cells) and all(
            bool(r.get("Pass_Equivalence"))
            and bool(r.get("treatment_exposure_verified"))
            for r in h3_cells
        ),
        "cells": h3_cells,
    }
    (output_stage / "stress_h3_test.json").write_text(
        json.dumps(_json_safe(h3_payload), indent=2, allow_nan=False),
        encoding="utf-8"
    )
    expected_output_files = {
        "stress_summary.json", "stress_degradation.csv",
        "stress_passfail.csv", "stress_h3_test.json",
    }
    observed_output_files = {
        path.name for path in output_stage.iterdir() if path.is_file()
    }
    if observed_output_files != expected_output_files or any(
        not path.is_file() or path.is_symlink() for path in output_stage.iterdir()
    ):
        raise RuntimeError(
            "H3 transaction does not contain the exact four-file task output"
        )
    for name in ("stress_summary.json", "stress_h3_test.json"):
        json.loads((output_stage / name).read_text(encoding="utf-8"))
    for name in ("stress_degradation.csv", "stress_passfail.csv"):
        if pd.read_csv(output_stage / name).empty:
            raise RuntimeError(f"H3 transaction contains an empty table: {name}")
    os.replace(output_stage, RESULTS_DIR)
    print(f"Saved {RESULTS_DIR / 'stress_summary.json'}")
    print(f"Saved {RESULTS_DIR / 'stress_degradation.csv'}")
    print(f"Saved {RESULTS_DIR / 'stress_passfail.csv'}")
    print(f"Saved {RESULTS_DIR / 'stress_h3_test.json'}")


if __name__ == "__main__":
    main()
