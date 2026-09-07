#!/usr/bin/env python3
"""
AGRI-BRAIN Methodology-Aligned Simulation Orchestrator
=======================================================
Runs five scenarios over the seven locked primary modes and three one-factor
secondary ablations. Imported publication workers write only to the explicit
run-scoped paths supplied by those workers. Standalone execution is
development-only and writes under ``mvp/simulation/development_results``;
it cannot overwrite the canonical publication directory.

Uses an AgentCoordinator with four lifecycle decision owners (farm,
processor, distributor, recovery) and a non-owning cooperative overlay.

MCP/piR context injection is enabled for the full and single-channel
variants. ``no_context`` initializes the same bounded learning components but
bypasses both external context channels, so it provides the confirmatory
channel-ablation comparator.

Supply and demand forecast information (both point and residual-std
uncertainty) is represented as state features in phi(s) at indices
6-8, populated from ``query_yield`` (validation-selected persistence
supply proxy) and ``query_demand`` (validation-selected non-seasonal
Holt-linear demand forecast) and consumed by ``build_feature_vector``.

Standalone usage:
    cd mvp/simulation
    python generate_results.py

Callable from backend:
    from mvp.simulation.generate_results import run_all, get_summary_json

This module is a **Layer 3 orchestrator**.  All scientific models, equations,
and scoring functions live in the backend model files (Layer 1):

    src.models.spoilage           — Arrhenius decay, declared rational lag factor
    src.models.forecast           — Holt-linear demand forecast (confirmatory)
    src.models.persistence_forecast — persistence supply proxy (confirmatory)
    src.models.lstm_demand        — Numpy-only LSTM demand diagnostic
    src.models.yield_forecast     — Holt-linear supply diagnostic
    src.models.slca               — 4-component author-declared social-performance proxy
    src.models.policy             — Policy configuration
    src.models.waste              — Operational waste model
    src.models.carbon             — Modeled transport-emissions indicator + COP term
    src.models.resilience         — ARI, RLE, temporal proxy-stability metric
    src.models.reward             — Multi-objective reward function
    src.models.action_selection   — Softmax policy, feature vectors
    src.models.reverse_logistics  — Modeled route-circularity indicator
    src.agents.coordinator        — Four decision owners plus cooperative overlay
"""
from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# Ensure backend models are importable
# ---------------------------------------------------------------------------
_BACKEND_SRC = Path(__file__).resolve().parent.parent.parent / "agribrain" / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

import hashlib
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime
from functools import wraps

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Layer 1 imports — all scientific logic lives here
from pirag.mcp.tools.demand_query import query_demand

# Supply and demand forecasts are routed through the MCP tools so simulator
# and REST share a single forecasting code path; the underlying forecaster
# modules above remain importable for tests that exercise them directly.
from pirag.mcp.tools.yield_query import query_yield
from src.agents.coordinator import AgentCoordinator
from src.chain.decision_ledger import (
    DecisionLedger,
    decision_ledger_episode_scope,
    decision_ledger_output_scope,
    get_active_decision_ledger_output_dir,
)
from src.models.action_selection import (
    ACTION_KM_KEYS,
    ACTIONS,
    build_feature_vector,
    compute_slca_attenuation,
    compute_thermal_stress,
)
from src.models.carbon import compute_carbon_efficiency, compute_transport_carbon
from src.models.episode_evidence_contract import (
    build_episode_evidence_contract,
    reconstruct_episode_evidence,
)
from src.models.footprint import FootprintMeter
from src.models.forecast import yield_demand_forecast
from src.models.lstm_demand import lstm_demand_forecast
from src.models.mode_capabilities import (
    AGRIBRAIN_LOGIT_MODES,
    CONTEXT_INFRASTRUCTURE_MODES,
    MULTI_EPISODE_MODES,
    PUBLICATION_BENCHMARK_MODES,
    capabilities_for,
)
from src.models.mode_capabilities import (
    PRIMARY_MODES as LOCKED_PRIMARY_MODES,
)
from src.models.outcome_equation_contract import build_outcome_equation_contract
from src.models.policy import Policy
from src.models.policy_learner import PolicyLearner
from src.models.resilience import (
    RLETracker,
    compute_ari,
    compute_equity,
)
from src.models.reverse_logistics import compute_circular_economy_score, evaluate_recovery_options
from src.models.reward import compute_reward
from src.models.slca import slca_score
from src.models.spoilage import (
    advance_spoilage_risk_midpoint,
    arrhenius_k,
)
from src.models.pinn_residual import (
    MAX_RESIDUAL,
    build_residual_feature_row,
    load_frozen_checkpoint,
    predict_residual,
)
from src.models.synthetic_spoilage_dgp import (
    compute_spoilage_independent_synthetic_dgp,
)
from src.models.waste import (
    INV_BASELINE,
    WASTE_CAP,
    compute_save_factor,
    compute_waste_rate,
)

try:
    from .stochastic import _DISABLED as _STOCH_DISABLED
    from .stochastic import _is_deterministic, make_stochastic_layer
except ImportError:
    from stochastic import _DISABLED as _STOCH_DISABLED
    from stochastic import _is_deterministic, make_stochastic_layer
try:
    from .benchmarks.episode_archive import (
        canonical_json_sha256 as _archive_canonical_sha256,
    )
    from .benchmarks.episode_archive import (
        measure_episode_runtime,
        read_gzip_json,
        write_gzip_json_atomic,
    )
    from .benchmarks.episode_archive import (
        to_json_native as _archive_json_native,
    )
except ImportError:
    from benchmarks.episode_archive import (  # type: ignore[no-redef]
        canonical_json_sha256 as _archive_canonical_sha256,
    )
    from benchmarks.episode_archive import (
        measure_episode_runtime,
        read_gzip_json,
        write_gzip_json_atomic,
    )
    from benchmarks.episode_archive import (
        to_json_native as _archive_json_native,
    )

# Confirmatory defaults were selected by validation RMSE without using the test
# segment.  Alternative models remain available only when explicitly requested.
FORECAST_METHOD = os.environ.get("FORECAST_METHOD", "holt_linear")
SUPPLY_FORECAST_METHOD = os.environ.get(
    "SUPPLY_FORECAST_METHOD", "persistence",
)

# Online learning toggle (default: disabled to preserve deterministic results)
ONLINE_LEARNING = os.environ.get("ONLINE_LEARNING", "false").lower() == "true"

# Re-export for backward compat; prefer _is_deterministic() at call sites.
DETERMINISTIC_MODE = _is_deterministic()

TRACE_SCHEMA_VERSION = 5
EPISODE_EVIDENCE_SCHEMA_VERSION = 1


_EPISODE_ENVIRONMENT_FIELDS = (
    "APP_ENV",
    "FORECAST_METHOD",
    "SUPPLY_FORECAST_METHOD",
    "ONLINE_LEARNING",
    "LLM_PROVIDER",
    "DETERMINISTIC_MODE",
    "STOCH_TEMP_STD_C",
    "STOCH_RH_STD",
    "STOCH_DEMAND_FRAC_STD",
    "STOCH_INVENTORY_FRAC_STD",
    "STOCH_TRANSPORT_KM_STD",
    "STOCH_K_REF_STD",
    "STOCH_EA_R_STD",
    "STOCH_ONSET_JITTER_H",
    "STOCH_THETA_NOISE_STD",
    "STOCH_POLICY_TEMP_STD",
    "STOCH_DELAY_PROB",
    "FAILURE_INJECTION",
    "MCP_RELIABILITY",
    "MCP_QOS_ROUTING",
    "PIR_COUNTERFACTUAL",
    "PHYSICS_CONSISTENCY_GATE",
    "HETEROGENEOUS_PROFILES",
    "RESEARCH_METRICS",
    "DYNAMIC_KB_FEEDBACK",
    "MCP_RATE_LIMITS",
    "PROTOCOL_MAX_RECORDS",
    "STRICT_VALIDATION",
    "FULL_EVIDENCE_CAPTURE",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _full_evidence_capture_enabled() -> bool:
    """Return the explicit lossless-archive posture, rejecting typos."""

    value = os.environ.get("FULL_EVIDENCE_CAPTURE", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("FULL_EVIDENCE_CAPTURE must be exactly 0 or 1")
    return value == "1"


def _evidence_value(value: object, *, dataframe_cell: bool = False) -> object:
    """Convert evidence values without rounding or silent stringification."""

    if value is pd.NA:
        if dataframe_cell:
            return None
        raise ValueError("pd.NA is valid only inside an archived dataframe")
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if dataframe_cell:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            return None
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"episode evidence key is not a string: {key!r}")
            converted[key] = _evidence_value(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_evidence_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        converted = tolist()
        if converted is value:
            raise TypeError(f"tolist() did not convert {type(value).__name__}")
        return _evidence_value(converted, dataframe_cell=dataframe_cell)
    return _archive_json_native(value)


def _dataframe_evidence_payload(frame: pd.DataFrame) -> dict:
    """Retain the exact episode input frame, schema, index, and attributes."""

    rows = [
        [_evidence_value(value, dataframe_cell=True) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": None if frame.index.name is None else str(frame.index.name),
        "index": [
            _evidence_value(value, dataframe_cell=True) for value in frame.index
        ],
        "rows": rows,
        "attrs": _evidence_value(dict(frame.attrs)),
    }


def _sha256_file(path: Path) -> tuple[str, int]:
    literal = Path(path).read_bytes()
    return hashlib.sha256(literal).hexdigest(), len(literal)


def _learner_continuation_payload(state: dict) -> dict:
    """Project a checkpoint onto fields that must persist between episodes.

    ``theta_learner`` is a schema-v1 compatibility alias for the currently
    active role and can legitimately point at a different role after a fresh
    coordinator is constructed.  When the authoritative per-role mapping is
    present, continuity is therefore checked against that mapping instead.
    Freeze labels describe the episode boundary rather than learned weights.
    """

    projected = deepcopy(state)
    if projected.get("theta_learners"):
        projected.pop("theta_learner", None)
    for field in ("learners_frozen", "learner_phase", "freeze_reason"):
        projected.pop(field, None)
    return projected


def _episode_ledger_root() -> Path:
    scoped = get_active_decision_ledger_output_dir()
    if scoped is not None:
        return scoped
    return Path(os.environ.get(
        "DECISION_LEDGER_DIR",
        str(RESULTS_DIR / "decision_ledger"),
    ))


def _archive_episode_evidence(
    *,
    frame: pd.DataFrame,
    mode: str,
    scenario: str,
    benchmark_seed: int,
    episode_index: int,
    result: dict,
    ledger_root: Path,
) -> None:
    """Write one lossless, hash-bound archive immediately after an episode."""

    ledger_path = Path(str(result["decision_ledger_path"]))
    ledger_sha256, ledger_bytes = _sha256_file(ledger_path)
    try:
        ledger_relative = ledger_path.resolve().relative_to(
            ledger_root.resolve(),
        ).as_posix()
    except ValueError as exc:
        # A complete-episode archive is independently resumable only when its
        # external ledger lives beneath the same arm root.  Falling back to a
        # basename would create an archive that cannot later be validated and
        # would defer the failure until the end of an expensive HPC task.
        raise ValueError(
            f"decision ledger is outside its active evidence root: {ledger_path}"
        ) from exc

    before_state = result.get("_learner_state_before_evidence")
    after_state = result.get("_learner_state_after_evidence")
    protocol_recorder = result.get("_protocol_recorder")
    protocol_records = (
        protocol_recorder.get_records() if protocol_recorder is not None else []
    )
    trace_exporter = result.get("_trace_exporter")
    trace_exports = {}
    if trace_exporter is not None:
        trace_exports = {
            "raw_decision_traces": [
                asdict(trace) for trace in trace_exporter._traces
            ],
            "summary": trace_exporter.summary(),
            "role_comparison": trace_exporter.export_role_comparison_table(),
            "provenance_chains": trace_exporter.export_provenance_chains(),
            "interoperability_trace": trace_exporter.export_interoperability_trace(),
            "feature_heatmap": trace_exporter.export_feature_heatmap_data(),
        }

    public_result = {
        key: value for key, value in result.items() if not key.startswith("_")
    }
    public_result["decision_ledger_path"] = ledger_relative
    frame_payload = _dataframe_evidence_payload(frame)
    payload = {
        "schema_version": EPISODE_EVIDENCE_SCHEMA_VERSION,
        "archive_contract": {
            "numeric_rounding": "none",
            "compression": "deterministic_gzip_mtime_0_level_9",
            "json": "canonical_sorted_keys_utf8_allow_nan_false",
            "scope": (
                "complete executed episode: input frame, RNG identities, learner "
                "state continuity, public episode result, protocol traces, sampled "
                "explainability exports, and external Merkle ledger binding"
            ),
        },
        "identity": {
            "benchmark_seed": int(benchmark_seed),
            "scenario": scenario,
            "mode": mode,
            "episode_index": int(episode_index),
            "episode_phase": str(result["episode_phase"]),
            "learning_enabled": bool(result["learning_enabled"]),
            "source_commit": os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip(),
            "source_tree_sha256": os.environ.get(
                "AGRIBRAIN_SOURCE_TREE_SHA256", "",
            ).strip(),
            "run_tag": os.environ.get("RUN_TAG", "").strip(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "").strip(),
            "slurm_array_job_id": os.environ.get(
                "SLURM_ARRAY_JOB_ID", "",
            ).strip(),
            "slurm_array_task_id": os.environ.get(
                "SLURM_ARRAY_TASK_ID", "",
            ).strip(),
        },
        "rng": {
            "derivation": (
                "SHA256('agribrain-v3|benchmark_seed|scenario|episode_index|stream')"
                " first 8 bytes, unsigned big-endian"
            ),
            "scenario_seed": _stream_seed(
                benchmark_seed, scenario, episode_index, "scenario",
            ),
            "environment_seed": _stream_seed(
                benchmark_seed, scenario, episode_index, "environment",
            ),
            "policy_seed": _stream_seed(
                benchmark_seed, scenario, episode_index, "policy",
            ),
            "environment_stream_id": str(result["environment_stream_id"]),
            "policy_stream_id": str(result["policy_stream_id"]),
            "stochastic_stream_id": str(result["stochastic_stream_id"]),
        },
        "scientific_environment": {
            name: os.environ.get(name) for name in _EPISODE_ENVIRONMENT_FIELDS
        },
        "input_frame": frame_payload,
        "input_frame_sha256": _archive_canonical_sha256(frame_payload),
        "learner_state": {
            "before": before_state,
            "before_sha256": result["learner_state_before_sha256"],
            "continuation_before_sha256": result[
                "learner_continuation_before_sha256"
            ],
            "after": after_state,
            "after_sha256": result["learner_state_after_sha256"],
            "continuation_after_sha256": result[
                "learner_continuation_after_sha256"
            ],
        },
        "runtime": result["episode_runtime_receipt"],
        "protocol_records": protocol_records,
        "protocol_records_sha256": _archive_canonical_sha256(protocol_records),
        "trace_exports": trace_exports,
        "episode_result": public_result,
        "decision_ledger": {
            "relative_path": ledger_relative,
            "storage": result["decision_ledger_storage"],
            "literal_sha256": ledger_sha256,
            "literal_bytes": ledger_bytes,
            "merkle_root": str(result["decision_ledger_root"]),
            "n_records": int(result["decision_ledger_n"]),
        },
    }
    archive_path = (
        ledger_root / "complete_episode_evidence" / f"{mode}__{scenario}"
        / f"episode_{int(episode_index)}.json.gz"
    )
    if (
        archive_path.exists()
        and os.environ.get("STRICT_VALIDATION", "0") == "1"
    ):
        raise FileExistsError(
            f"refusing to overwrite complete episode evidence: {archive_path}"
        )
    receipt = write_gzip_json_atomic(archive_path, payload)
    result.update({
        "episode_evidence_schema_version": EPISODE_EVIDENCE_SCHEMA_VERSION,
        "episode_evidence_path": str(archive_path),
        "episode_evidence_literal_sha256": receipt.literal_sha256,
        "episode_evidence_literal_bytes": receipt.literal_bytes,
        "episode_evidence_canonical_sha256": receipt.canonical_json_sha256,
        "episode_evidence_canonical_bytes": receipt.canonical_json_bytes,
        "decision_ledger_sha256": ledger_sha256,
        "decision_ledger_bytes": ledger_bytes,
    })


def _resume_complete_episode_evidence(
    *,
    frame: pd.DataFrame,
    mode: str,
    scenario: str,
    benchmark_seed: int,
    episode_index: int,
    learning_enabled: bool,
    ledger_root: Path,
    learner_state_cache: dict | None,
) -> dict | None:
    """Resume one already complete episode without executing it again.

    Publication workers may be requeued after a node or downstream packaging
    failure.  A pre-existing archive is accepted only when its literal ledger,
    RNG identity, exact input frame, source/run identity, scientific
    environment, and learner boundary all validate.  Any mismatch blocks the
    retry; it never overwrites or silently reruns completed scientific work.
    """

    archive_path = (
        ledger_root / "complete_episode_evidence" / f"{mode}__{scenario}"
        / f"episode_{int(episode_index)}.json.gz"
    )
    if not archive_path.exists():
        return None
    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError(f"existing episode archive is unsafe: {archive_path}")
    try:
        payload, archive_receipt = read_gzip_json(archive_path)
        identity = payload.get("identity") or {}
        expected_phase = (
            "adaptation"
            if learning_enabled and capabilities_for(mode).learned
            else "frozen_evaluation"
            if capabilities_for(mode).learned
            else "fixed_evaluation"
        )
        expected_identity = {
            "benchmark_seed": int(benchmark_seed),
            "scenario": scenario,
            "mode": mode,
            "episode_index": int(episode_index),
            "episode_phase": expected_phase,
            "learning_enabled": bool(learning_enabled),
            "source_commit": os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip(),
            "source_tree_sha256": os.environ.get(
                "AGRIBRAIN_SOURCE_TREE_SHA256", "",
            ).strip(),
            "run_tag": os.environ.get("RUN_TAG", "").strip(),
        }
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                raise ValueError(
                    f"resume identity {field!r} differs: "
                    f"{identity.get(field)!r} != {expected!r}"
                )
        expected_environment = {
            name: os.environ.get(name) for name in _EPISODE_ENVIRONMENT_FIELDS
        }
        if payload.get("scientific_environment") != expected_environment:
            raise ValueError("scientific environment differs from archived episode")
        frame_payload = _dataframe_evidence_payload(frame)
        if (
            payload.get("input_frame_sha256")
            != _archive_canonical_sha256(frame_payload)
            or payload.get("input_frame") != frame_payload
        ):
            raise ValueError("episode input frame differs from archived evidence")

        # Reuse the same independent archive/ledger validator used by the task
        # completion gate.  The lazy import avoids a module-import cycle.
        from hpc.validate_complete_episode_evidence import _validate_archive

        validated_payload, record = _validate_archive(archive_path, ledger_root)
        if validated_payload != payload:
            raise ValueError("episode archive changed between independent reads")
        learner = payload.get("learner_state") or {}
        capabilities = capabilities_for(mode)
        cached_state = (
            learner_state_cache.get(mode)
            if learner_state_cache is not None else None
        )
        if capabilities.learned and int(episode_index) > 0 and cached_state is None:
            raise ValueError("learned-episode retry lacks its prior learner checkpoint")
        if cached_state is not None and _archive_canonical_sha256(
            _learner_continuation_payload(cached_state)
        ) != learner.get("continuation_before_sha256"):
            raise ValueError("learner checkpoint differs at resume boundary")

        ledger_relative = PurePosixPath(str(
            (payload.get("decision_ledger") or {}).get("relative_path", "")
        ))
        if ledger_relative.is_absolute() or not ledger_relative.parts or any(
            part in {"", ".", ".."} for part in ledger_relative.parts
        ):
            raise ValueError("archived decision-ledger path is unsafe")
        ledger_path = ledger_root.joinpath(*ledger_relative.parts).resolve()
        if not ledger_path.is_relative_to(ledger_root.resolve()):
            raise ValueError("archived decision-ledger path escapes its root")

        result = deepcopy(payload["episode_result"])
        result.update({
            "decision_ledger_path": str(ledger_path),
            "decision_ledger_sha256": record["ledger_literal_sha256"],
            "decision_ledger_bytes": record["ledger_literal_bytes"],
            "episode_evidence_schema_version": EPISODE_EVIDENCE_SCHEMA_VERSION,
            "episode_evidence_path": str(archive_path),
            "episode_evidence_literal_sha256": archive_receipt.literal_sha256,
            "episode_evidence_literal_bytes": archive_receipt.literal_bytes,
            "episode_evidence_canonical_sha256": (
                archive_receipt.canonical_json_sha256
            ),
            "episode_evidence_canonical_bytes": (
                archive_receipt.canonical_json_bytes
            ),
            "_resumed_from_complete_episode_evidence": True,
            "_archived_trace_exports": deepcopy(payload.get("trace_exports") or {}),
            "_archived_protocol_records": deepcopy(
                payload.get("protocol_records") or []
            ),
        })
        if learner_state_cache is not None and learning_enabled:
            learner_state_cache[mode] = deepcopy(learner.get("after") or {})
        return result
    except Exception as exc:
        raise RuntimeError(
            f"existing episode evidence cannot be safely resumed: {archive_path}: {exc}"
        ) from exc


def _canonical_sha256(payload: object) -> str:
    """Hash a strict, stable JSON representation of an evidence payload."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stream_seed(
    benchmark_seed: int, scenario: str, episode_index: int, stream: str,
) -> int:
    """Derive a stable, mode-independent common-random-number stream seed."""
    key = (
        f"agribrain-v3|{int(benchmark_seed)}|{scenario}|"
        f"{int(episode_index)}|{stream}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _stream_id(
    benchmark_seed: int, scenario: str, episode_index: int, stream: str,
) -> str:
    return (
        f"seed={int(benchmark_seed)};scenario={scenario};"
        f"episode={int(episode_index)};stream={stream}"
    )


def _policy_categorical_uniform(policy_stream_seed: int, step_index: int) -> float:
    """Return the locked portable u53 categorical variate for one step."""
    key = (
        "agribrain-policy-categorical-v1|"
        f"{int(policy_stream_seed)}|{int(step_index)}"
    ).encode("utf-8")
    # Take the leading 53 bits and divide by 2**53, yielding exactly a binary
    # floating-point value in [0, 1). No NumPy sampler implementation is part
    # of the publication action contract.
    u53 = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
    return float(u53 / (1 << 53))


def policy_theta_for_seed(theta: np.ndarray, benchmark_seed: int) -> np.ndarray:
    """Apply Source-7 policy-prior perturbation once for a benchmark seed."""
    prior_seed = _stream_seed(benchmark_seed, "all", -1, "policy-prior")
    layer = make_stochastic_layer(
        np.random.default_rng(prior_seed), stream_seed=prior_seed,
    )
    return np.asarray(
        layer.perturb_theta(np.asarray(theta, dtype=float), counter=0),
        dtype=float,
    )


def _paired_context_action_changed(
    live_action: int,
    counterfactual_action: int | None,
    counterfactual_probs,
) -> bool:
    """Score a paired pre-selection-RNG-state context intervention.

    ``counterfactual_probs`` is also the availability sentinel: the
    coordinator leaves it as ``None`` when the context-ablation call was not
    applicable or failed.  Comparing the live action with the paired
    context-ablated action, rather than with either policy's argmax, ensures
    that stochastic sampling alone cannot be misclassified as context
    influence. On the stochastic policy path both calls consume the same
    categorical variate; the declared probability-gap override may discard the live sampled
    action but does not skip the draw.
    """
    return bool(
        counterfactual_probs is not None
        and counterfactual_action is not None
        and int(live_action) != int(counterfactual_action)
    )


@contextmanager
def decision_ledger_scope(path: Path, *, reset: bool = False):
    """Assign one arm an exclusive filesystem ledger-output directory.

    The active in-memory episode ledger shadows this directory during routing,
    so audit history resets every episode while learner state may persist.
    ``reset`` removes only a stale final-episode file for legacy development
    runs. Full-evidence publication runs preserve all completed bytes so the
    validated episode-resume path can reuse them after a worker requeue.
    """
    scope = Path(path).resolve()
    scope.mkdir(parents=True, exist_ok=True)
    if reset and not _full_evidence_capture_enabled():
        for ledger in scope.glob("*.jsonl"):
            if ledger.is_file():
                ledger.unlink()
    with decision_ledger_output_scope(scope):
        yield scope

def _demand_forecast(df, horizon=1, **kwargs):
    """Dispatch to the explicitly configured demand forecaster.

    The ``holt_winters`` value of ``FORECAST_METHOD`` is retained as a
    legacy alias and selects ``yield_demand_forecast`` (Holt's linear
    level + trend, no seasonal indices); the actual implementation is
    not Holt-Winters seasonal smoothing.
    """
    if FORECAST_METHOD in {"holt_linear", "holt_winters"}:
        return yield_demand_forecast(df, horizon=horizon, **kwargs)
    if FORECAST_METHOD == "lstm":
        return lstm_demand_forecast(df, horizon=horizon, **kwargs)
    if FORECAST_METHOD == "persistence":
        from src.models.persistence_forecast import persistence_forecast
        return persistence_forecast(
            df, horizon=horizon, series_col="demand_units",
        )
    raise ValueError(f"unsupported demand forecast method: {FORECAST_METHOD!r}")


# ---------------------------------------------------------------------------
# Constants (orchestration-level only — no physics here)
# ---------------------------------------------------------------------------
SEED = 42

SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]

# Seven primary modes plus three one-factor secondary ablations.
PRIMARY_MODES = list(LOCKED_PRIMARY_MODES)

# The simulator exposes exactly the locked publication arms. Structural
# parameter sensitivity is a separate 100-point, 29-factor design under
# ``mvp/simulation/sensitivity`` and never appears as a benchmark mode.
MODES = list(PUBLICATION_BENCHMARK_MODES)
_CONTEXT_ENABLED_MODES = set(CONTEXT_INFRASTRUCTURE_MODES)
_AGRIBRAIN_LOGIT_MODES = set(AGRIBRAIN_LOGIT_MODES)

# Confirmatory learning budget. Every learned policy in the primary
# comparison receives the same number of within-scenario episodes. Static
# remains a one-pass non-learning reference. Learner state is reset between
# scenarios (see run_all), so the scenario order cannot influence outcomes.
_MULTI_EPISODE_MODES: dict = dict(MULTI_EPISODE_MODES)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_CSV = Path(os.environ.get("DATA_CSV", "")) if os.environ.get("DATA_CSV") else _BACKEND_SRC / "src" / "data_spinach.csv"


# ---------------------------------------------------------------------------
# Scenario perturbation — delegates to the backend pure-domain engine
# (src.models.scenario_engine), not the FastAPI router. The simulator
# is a Python subprocess; depending on the router would couple it to
# HTTP-coupled module-level state and complicate cross-platform reuse.
# ---------------------------------------------------------------------------
from src.models import scenario_engine as _scenario_engine
from src.models.scenario_engine import (  # noqa: F401
    SCENARIO_FUNCTIONS as _SCENARIO_FN,
)
from src.models.scenario_engine import (
    hours_from_start as _hours_from_start,
)


def apply_scenario(df: pd.DataFrame, name: str, policy: Policy,
                   rng: np.random.Generator, stoch=None,
                   intensity: float = 1.0) -> pd.DataFrame:
    """Apply scenario perturbation with optional onset-time jitter (Source 6).

    When stochastic mode is active, the scenario's event mask is shifted by a
    signed ``±onset_jitter_hours`` offset. Timestamps are never altered: doing
    so and then subtracting the shifted first timestamp would cancel the
    intended treatment.
    """
    intensity, _ = _scenario_engine.validate_scenario_controls(intensity, 0.0)
    if name == "baseline":
        # Explicit baseline: recompute derived columns against this policy.
        result = _scenario_engine.recompute_derived(df.copy(), policy)
        result["scenario_onset_offset_hours"] = 0.0
        result["scenario_intensity"] = intensity
        return result
    if name not in _SCENARIO_FN:
        raise ValueError(
            f"unknown scenario {name!r}; expected baseline or one of "
            f"{sorted(_SCENARIO_FN)}"
        )

    # Source 6: signed scenario-onset jitter. Positive means a later onset.
    # baseline and adaptive_pricing have no fixed onset, so skip jitter.
    jitter_h = 0.0
    if stoch is not None and stoch.enabled and name not in ("baseline", "adaptive_pricing"):
        jitter_h = float(stoch.jitter_onset_hour(0.0, counter=0))

    result = _scenario_engine.apply(
        name,
        df,
        policy=policy,
        intensity=intensity,
        onset_offset_hours=jitter_h,
        rng=rng,
    )
    result["scenario_onset_offset_hours"] = jitter_h
    result["scenario_intensity"] = intensity

    return result


# ---------------------------------------------------------------------------
# Single episode runner (orchestration only — calls Layer 1 models)
# ---------------------------------------------------------------------------
def run_episode(
    df: pd.DataFrame, mode: str, policy: Policy,
    rng: np.random.Generator, scenario: str = "baseline",
    stoch=None, seed: int = 0,
    learner_state_cache: dict | None = None,
    context_learner_overrides: dict | None = None,
    benchmark_seed: int | None = None,
    episode_index: int = 0,
    environment_stream_id: str = "",
    policy_stream_id: str = "",
    stochastic_stream_id: str = "",
    learning_enabled: bool = True,
) -> dict:
    """Run one isolated episode while retaining caller-managed learner state.

    The current episode's append-only ledger is exposed to ``chain_query``
    through a ContextVar.  An empty new episode therefore shadows every stale
    JSONL file, and each step can see only earlier decisions from this episode.
    """
    canonical_seed = int(seed if benchmark_seed is None else benchmark_seed)
    expected_environment_stream_id = _stream_id(
        canonical_seed, scenario, int(episode_index), "environment",
    )
    expected_policy_stream_id = _stream_id(
        canonical_seed, scenario, int(episode_index), "policy",
    )
    if environment_stream_id and (
        environment_stream_id != expected_environment_stream_id
    ):
        raise ValueError("environment_stream_id does not match episode identity")
    if policy_stream_id and policy_stream_id != expected_policy_stream_id:
        raise ValueError("policy_stream_id does not match episode identity")
    environment_stream_id = (
        environment_stream_id or expected_environment_stream_id
    )
    policy_stream_id = policy_stream_id or expected_policy_stream_id
    stochastic_stream_id = stochastic_stream_id or environment_stream_id
    observation_treatment = dict(df.attrs.get(
        "observation_treatment",
        {
            "stressor": "nominal",
            "n_steps": int(len(df)),
            "data_observation_treatment": False,
            "delay_steps": 0,
            "missing_count": 0,
        },
    ))
    decision_ledger = DecisionLedger(episode_metadata={
        "mode": mode,
        "scenario": scenario,
        "seed": canonical_seed,
        "benchmark_seed": canonical_seed,
        "episode_index": int(episode_index),
        "environment_stream_id": str(environment_stream_id),
        "policy_stream_id": str(policy_stream_id),
        "stochastic_stream_id": str(stochastic_stream_id),
        "learning_enabled": bool(learning_enabled),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        # H3 treatment provenance is part of the retained ledger file, not
        # merely a detached aggregate-summary assertion.  The literal-file
        # SHA-256 binds this header while the per-step exposure fields below
        # are covered by the Merkle root.
        "observation_treatment": observation_treatment,
    })
    ledger_root = _episode_ledger_root()
    if _full_evidence_capture_enabled():
        resumed = _resume_complete_episode_evidence(
            frame=df,
            mode=mode,
            scenario=scenario,
            benchmark_seed=canonical_seed,
            episode_index=int(episode_index),
            learning_enabled=bool(learning_enabled),
            ledger_root=ledger_root,
            learner_state_cache=learner_state_cache,
        )
        if resumed is not None:
            return resumed
    runtime_measurement = measure_episode_runtime()
    with decision_ledger_episode_scope(decision_ledger):
        with runtime_measurement:
            result = _run_episode_impl(
                df, mode, policy, rng, scenario=scenario, stoch=stoch,
                seed=canonical_seed,
                learner_state_cache=learner_state_cache,
                context_learner_overrides=context_learner_overrides,
                decision_ledger=decision_ledger,
                benchmark_seed=canonical_seed,
                episode_index=episode_index,
                environment_stream_id=environment_stream_id,
                policy_stream_id=policy_stream_id,
                stochastic_stream_id=stochastic_stream_id,
                learning_enabled=learning_enabled,
            )
    result["episode_runtime_receipt"] = runtime_measurement.receipt.as_dict()
    if _full_evidence_capture_enabled():
        _archive_episode_evidence(
            frame=df,
            mode=mode,
            scenario=scenario,
            benchmark_seed=canonical_seed,
            episode_index=int(episode_index),
            result=result,
            ledger_root=ledger_root,
        )
    result.pop("_learner_state_before_evidence", None)
    result.pop("_learner_state_after_evidence", None)
    return result


def _run_episode_impl(
    df: pd.DataFrame, mode: str, policy: Policy,
    rng: np.random.Generator, scenario: str = "baseline",
    stoch=None, seed: int = 0,
    learner_state_cache: dict | None = None,
    context_learner_overrides: dict | None = None,
    decision_ledger: DecisionLedger | None = None,
    benchmark_seed: int | None = None,
    episode_index: int = 0,
    environment_stream_id: str = "",
    policy_stream_id: str = "",
    stochastic_stream_id: str = "",
    learning_enabled: bool = True,
) -> dict:
    """Run one (mode, scenario) episode.

    Parameters
    ----------
    learner_state_cache : optional mode-keyed dict that persists learner
        state across repeated episodes of one scenario.
        When provided, the coordinator's learner state is restored from
        ``learner_state_cache[mode]`` after ``reset()`` (if present) and
        written back at the end of the episode. ``run_all`` creates a fresh
        cache for each scenario, so the policy-delta and context learners
        accumulate only within the declared equal-budget training block;
        fixed scenario order cannot become an implicit curriculum. Omit the
        argument to retain per-episode-reset semantics.
    """
    if stoch is None:
        stoch = _STOCH_DISABLED
    benchmark_seed = int(seed if benchmark_seed is None else benchmark_seed)
    expected_policy_stream_id = _stream_id(
        benchmark_seed, scenario, int(episode_index), "policy",
    )
    if policy_stream_id != expected_policy_stream_id:
        raise ValueError("policy stream identity changed before episode execution")
    policy_stream_seed = _stream_seed(
        benchmark_seed, scenario, int(episode_index), "policy",
    )
    n = len(df)
    hours = _hours_from_start(df)

    # --- Source 5: Spoilage model error (once per episode) ---
    # Perturb Arrhenius parameters to model batch-to-batch biological variability
    eff_k_ref = stoch.perturb_k_ref(policy.k_ref, counter=0)
    eff_ea_r = stoch.perturb_ea_r(policy.Ea_R, counter=0)
    decision_ledger.metadata["outcome_equation_contract"] = (
        build_outcome_equation_contract(
            policy,
            effective_k_ref=eff_k_ref,
            effective_ea_r=eff_ea_r,
            stochastic_layer=stoch,
        )
    )

    # Optional policy-temperature sensitivity. It is disabled in the
    # confirmatory benchmark by default; enabling it is an explicitly
    # labelled sensitivity analysis, never a device for targeting an
    # effect-size range.
    episode_policy_temp = stoch.policy_temperature(base=1.0, counter=0)

    # --- Multi-agent coordinator ---
    context_mode = mode in _CONTEXT_ENABLED_MODES
    coordinator = AgentCoordinator(
        context_enabled=context_mode,
        context_learner_overrides=context_learner_overrides,
        mode=mode,
    )
    coordinator.reset()

    # Within-scenario learner state persistence. ``coordinator.reset()``
    # wipes learner state by design, but a caller-provided cache restores the
    # state from the previous repeated episode of the same scenario. The
    # publication driver creates a fresh cache before each scenario, so no
    # cross-scenario training curriculum is introduced.
    if learner_state_cache is not None and mode in learner_state_cache:
        coordinator.load_learner_states(learner_state_cache[mode])
    # Capture the exact loaded checkpoint before the retained-episode freeze is
    # applied.  Episode 0 therefore records the declared initialized state;
    # episodes 1--3 record the exact continuation state from the prior episode.
    learner_state_before_evidence = deepcopy(coordinator.save_learner_states())

    # The public wrapper always supplies the active episode ledger.  Keeping a
    # defensive fallback makes direct internal calls fail safe rather than
    # dereferencing None, but it is not exposed to chain_query outside a scope.
    if decision_ledger is None:
        decision_ledger = DecisionLedger(episode_metadata={
            "mode": mode,
            "scenario": scenario,
            "seed": benchmark_seed,
            "benchmark_seed": benchmark_seed,
            "episode_index": int(episode_index),
            "environment_stream_id": str(environment_stream_id),
            "policy_stream_id": str(policy_stream_id),
            "stochastic_stream_id": str(stochastic_stream_id),
            "learning_enabled": bool(learning_enabled),
            "trace_schema_version": TRACE_SCHEMA_VERSION,
        })

    # --- Green AI footprint meter ---
    meter = FootprintMeter(
        measurement_scope=(
            "coordinator.step action-selection wall time only; excludes "
            "scenario construction, forecast preparation, outcome scoring, "
            "learner post-step updates, artifact I/O, and idle allocation"
        ),
        proxy_step_unit="standardized routing opportunity",
    )
    decision_ledger.metadata["episode_evidence_contract"] = (
        build_episode_evidence_contract(
            measurement_scope=meter.measurement_scope,
            proxy_step_unit=meter.proxy_step_unit,
            influence_threshold=0.10,
            assumed_active_power_w=meter.assumed_active_power_W,
            water_rate_l_per_server_second=meter.water_per_server_second_L,
            energy_per_step_proxy_j=meter.energy_per_step_proxy_J,
            water_per_step_proxy_l=meter.water_per_step_proxy_L,
        )
    )

    # --- PolicyLearner (optional, off by default) ---
    learner = PolicyLearner() if ONLINE_LEARNING else None
    if not learning_enabled:
        coordinator.freeze_learners(
            learner,
            reason=(
                "retained_episode_3"
                if int(episode_index) == 3 else "fixed_nonlearning_episode"
            ),
        )

    # --- Independent common latent DGP + policy-side PINN ablation ---
    # Every paired arm is scored against the same noise-free trajectory from
    # the declared independent synthetic DGP.  The DGP is not a PINN output.
    # ``no_pinn`` removes the frozen residual only from the policy-observed
    # estimator below; it must not redefine the world used for outcome scoring.
    # No target construction, training, or optimization occurs in an episode.
    effective_mode = "agribrain" if mode in _AGRIBRAIN_LOGIT_MODES else mode
    df = compute_spoilage_independent_synthetic_dgp(
        df, k_ref=eff_k_ref, Ea_R=eff_ea_r,
        T_ref_K=policy.T_ref_K, beta=policy.beta_humidity,
        lag_lambda=policy.lag_lambda,
    )
    decision_ledger.metadata["latent_spoilage_model"] = deepcopy(
        df.attrs["synthetic_spoilage_dgp"]
    )
    if capabilities_for(mode).spoilage_residual:
        frozen_pinn = load_frozen_checkpoint()
        decision_ledger.metadata["spoilage_estimator"] = deepcopy(
            {
                "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
                "checkpoint_sha256": frozen_pinn.checkpoint_sha256,
                "training_dataset_sha256": frozen_pinn.dataset_sha256,
                "training_target_origin": "independent_synthetic_dgp",
                "residual_bound_abs": MAX_RESIDUAL,
                "deployment_transform": (
                    "clip_quality_to_unit_interval_then_cumulative_minimum"
                ),
                "synthetic_only": True,
                "external_validation": False,
            }
        )
    else:
        frozen_pinn = None
        decision_ledger.metadata["spoilage_estimator"] = {
            "kind": "mechanistic_only_no_pinn",
            "checkpoint_sha256": None,
            "training_dataset_sha256": None,
            "training_target_origin": None,
            "residual_bound_abs": None,
            "deployment_transform": None,
            "synthetic_only": True,
            "external_validation": False,
        }

    ari_vals, waste_vals, slca_vals = [], [], []
    rle_tracker = RLETracker()
    carbon_total = 0.0
    # ``*_policy_observed`` values are the only state supplied to routing.
    # ``*_outcome_environmental`` values are the latent simulation truth used
    # by every scored endpoint.  Keeping both namespaces explicit prevents an
    # observation fault from silently changing the world being evaluated.
    rho_policy_observed_trace: list[float] = []
    rho_outcome_environmental_trace: list[float] = []
    action_trace, prob_trace = [], []
    reward_trace, carbon_trace, slca_component_trace = [], [], []
    simulated_dispatch_accounted_trace: list[bool] = []
    transport_multiplier_outcome_environmental_trace: list[float] = []
    # Per-step diagnostic flags emitted by the coordinator. They are retained
    # in the ledger but are not treated as proof that a fault was prevented.
    cooperative_veto_trace: list[int] = []
    fault_recovery_trace: list[int] = []
    fault_injected_result_count_trace: list[int] = []
    physics_gate_trace: list[int] = []
    circular_scores = []
    supply_hats = []
    decision_latency_ms = []
    temp_policy_observed_trace, rh_policy_observed_trace = [], []
    inventory_policy_observed_trace: list[float] = []
    demand_policy_observed_trace: list[float] = []
    demand_forecast_policy_observed_trace: list[float] = []
    demand_regime_flag_trace: list[float] = []
    price_signal_trace: list[float] = []
    supply_forecast_policy_observed_trace: list[float] = []
    temp_outcome_environmental_trace: list[float] = []
    rh_outcome_environmental_trace: list[float] = []
    inventory_outcome_environmental_trace: list[float] = []
    demand_outcome_environmental_trace: list[float] = []
    constraint_violation_steps = 0
    compliance_violation_steps = 0
    temperature_violation_steps = 0
    quality_violation_steps = 0
    # Operating-envelope violations are computed uniformly for every mode;
    # only MCP-enabled modes can route the resulting tool message into policy.
    operational_violation_steps = 0
    # Outcome-side disposition on the env-driven violation event set:
    # of the steps where (temp_violation OR quality_violation) fired,
    # what did the agent's chosen action do at that standardized opportunity?
    # cold_chain continues through the central route, local_redistribute enters
    # the declared short-dwell route, and recovery enters a non-food route. The
    # resulting rates are observed outputs, not validator-enforced rankings.
    # See compute_violation_disposition in resilience.py for the
    # canonical definition; counters below mirror it inline so the
    # episode summary can be emitted without an extra trace pass.
    violation_routed_to_cold_chain = 0
    violation_routed_to_local = 0
    violation_routed_to_recovery = 0
    violation_event_count_local = 0

    # Context-alignment counters: did the chosen action match the action that
    # the context layer most strongly recommended? Only counted for steps
    # where the modifier vector carries a meaningful signal (max abs above
    # CONTEXT_SIGNAL_THRESHOLD). Steps without context (no_context, static)
    # contribute zero to both counters and the rate is 0/0 by definition.
    # P4: we also track honor rate at three alternative thresholds so the
    # paper can report sensitivity of the metric to its single free
    # parameter. 0.10 remains the headline threshold in the main text.
    CONTEXT_SIGNAL_THRESHOLD = 0.10
    CONTEXT_SIGNAL_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
    context_active_steps = 0
    context_honored_steps = 0
    # 2026-05 apples-to-apples cross-mode denominator. Counts steps
    # where the context layer actually executed and emitted a non-empty
    # modifier vector (i.e. ``coordinator._step_context_modifier`` is
    # populated and non-empty). This is BEFORE the
    # CONTEXT_SIGNAL_THRESHOLD gate, BEFORE the retrieval guard, and
    # BEFORE the cooperative-overlay window check -- so it counts
    # "did the mode's context channel run on this step" regardless of
    # whether downstream gates zeroed the signal.
    #
    # Why we need it: the headline ``context_active_steps`` is
    # post-guard, so retrieval-dependent modes (agribrain, pirag_only)
    # report fewer active steps than guard-free modes (mcp_only) on
    # scenarios where the primary retrieval cannot pass the
    # cooperative-governance retrieval guard (heatwave + baseline).
    # The pre-2026-05 metric set therefore couldn't surface the fact
    # that all three modes DISPATCH context on every step -- only the
    # post-guard signal magnitudes differ. ``context_dispatch_attempt_steps``
    # makes this explicit; ``context_dispatch_influence_rate`` below
    # divides the influenced-step count by THIS denominator instead
    # of context_active_steps, giving a cross-mode-comparable paired
    # paired pre-selection-state action-change fraction without the asymmetric
    # activation-regime confound.
    context_dispatch_attempt_steps = 0
    # Companion counter for the ``context_influence_rate`` metric. A step is
    # "context-influenced" when the sampled live action differs from a
    # context-ablated action generated from the exact RNG state saved before
    # live sampling. Both stochastic calls consume the same categorical
    # variate, even if the live probability-gap override discards its sampled
    # action. The modifier is the only controlled difference. Thus a
    # random sample that differs from a policy argmax cannot count on its own.
    # The same active-step gate as the honor counter
    # (max(|modifier|) > 0.10) is used, so both rates share a denominator.
    #
    # See fig 9 panel-c docstring in generate_figures.py for the
    # paper-narrative framing.
    context_influenced_steps = 0
    context_ignored_per_recommendation = {0: 0, 1: 0, 2: 0}
    context_active_per_recommendation = {0: 0, 1: 0, 2: 0}
    # Per-threshold (active, honored, influenced) triples for P4
    # sensitivity table. The supplementary methods reports honor +
    # influence at each threshold so a reviewer can verify the
    # metric story is robust to the gating-threshold choice.
    context_threshold_counters = {
        thr: {"active": 0, "honored": 0, "influenced": 0}
        for thr in CONTEXT_SIGNAL_THRESHOLDS
    }
    previous_protocol_counts = {
        "protocol_interaction_count_step": 0,
        "protocol_jsonrpc_error_count_step": 0,
        "protocol_tool_iserror_count_step": 0,
        "protocol_real_tool_iserror_count_step": 0,
        "protocol_error_count_step": 0,
        "protocol_dropped_interaction_count_step": 0,
    }
    previous_protocol_method_counts = {
        "tools/call": 0,
        "prompts/get": 0,
    }
    previous_message_count = 0

    prev_temp_policy_observed = 0.0
    prev_rh_policy_observed = 0.0
    prev_rho_policy_mechanistic = 0.0
    prev_quality_policy_observed = 1.0
    canonical_temp_observed_history: list[float] = []
    canonical_rh_observed_history: list[float] = []
    h3_predelay_temp_history: list[float] = []
    h3_predelay_rh_history: list[float] = []
    h3_stressor = str(
        decision_ledger.metadata.get("observation_treatment", {}).get(
            "stressor", "nominal",
        )
    )

    for idx in range(n):
        row = df.iloc[idx]
        rho_outcome_environmental = float(
            row.get("spoilage_risk", 1.0 - row["shelf_left"])
        )
        inv_outcome_environmental = float(row.get("inventory_units", 100.0))
        temp_outcome_environmental = float(row["tempC"])
        rh_outcome_environmental = float(row["RH"])
        demand_outcome_environmental = float(row.get("demand_units", 100.0))
        # Source 2 is exogenous variability in the demand observation stream.
        # It is sampled before forecasting and is shared across policy arms by
        # the environment stream's semantic timestep counter. The old code
        # perturbed only the one-step forecast after fitting, leaving the
        # observed history, Bollinger regime, and price signal untouched.
        demand_policy_observed = stoch.perturb_demand(
            float(row.get(
                "demand_policy_observed", demand_outcome_environmental,
            )),
            counter=idx,
        )

        # Policy observations begin from the declared sensor stream. H3 writes
        # stressed values only to these columns; it never mutates latent truth.
        canonical_temp_policy_observed = stoch.perturb_temperature(float(
            row.get("temp_policy_observed", temp_outcome_environmental)
        ), counter=idx)
        canonical_rh_policy_observed = stoch.perturb_humidity(float(
            row.get("rh_policy_observed", rh_outcome_environmental)
        ), counter=idx)
        inv_policy_observed = stoch.perturb_inventory(float(
            row.get("inventory_policy_observed", inv_outcome_environmental)
        ), counter=idx)

        # Telemetry delay: carry over previous perturbed step's readings
        if idx > 0 and stoch.should_delay(counter=idx):
            canonical_temp_policy_observed = (
                canonical_temp_observed_history[-1]
            )
            canonical_rh_policy_observed = canonical_rh_observed_history[-1]
        canonical_temp_observed_history.append(
            float(canonical_temp_policy_observed)
        )
        canonical_rh_observed_history.append(
            float(canonical_rh_policy_observed)
        )

        # H3 is a separate observation-only layer applied after the canonical
        # stochastic sensing stream. This exact order is shared with the raw
        # H3 validator and prevents non-commuting missing/delay transforms from
        # being reconstructed against the wrong base stream.
        temp_policy_observed = float(canonical_temp_policy_observed)
        rh_policy_observed = float(canonical_rh_policy_observed)
        if h3_stressor in {"sensor_noise", "compounded"}:
            temp_policy_observed += float(row.get("h3_temp_noise_c", 0.0))
            rh_policy_observed = float(np.clip(
                rh_policy_observed
                + float(row.get("h3_rh_noise_pct", 0.0)),
                15.0,
                100.0,
            ))
        missing_observation = bool(
            row.get("h3_missing_observation", False)
        )
        if h3_stressor in {"missing_data", "compounded"} and (
            missing_observation
        ):
            if idx == 0:
                raise RuntimeError("H3 missing-data dose cannot mask step zero")
            temp_policy_observed = h3_predelay_temp_history[-1]
            rh_policy_observed = h3_predelay_rh_history[-1]
        h3_predelay_temp_history.append(float(temp_policy_observed))
        h3_predelay_rh_history.append(float(rh_policy_observed))
        if h3_stressor in {"telemetry_delay", "compounded"}:
            source_step = int(row.get(
                "h3_telemetry_source_step_index", max(idx - 4, 0),
            ))
            if source_step < 0 or source_step > idx:
                raise RuntimeError("H3 telemetry source step is outside history")
            temp_policy_observed = h3_predelay_temp_history[source_step]
            rh_policy_observed = h3_predelay_rh_history[source_step]

        # Advance the observed cumulative state with the same midpoint physics
        # as the latent trajectory. This preserves monotonicity even when
        # sensor noise or a delayed reading changes the instantaneous rate.
        if idx == 0:
            rho_policy_mechanistic = 0.0
        else:
            rho_policy_mechanistic = advance_spoilage_risk_midpoint(
                prev_rho_policy_mechanistic,
                previous_temp_C=prev_temp_policy_observed,
                current_temp_C=temp_policy_observed,
                previous_rh_pct=prev_rh_policy_observed,
                current_rh_pct=rh_policy_observed,
                previous_hour=float(hours[idx - 1]),
                current_hour=float(hours[idx]),
                k_ref=eff_k_ref,
                Ea_R=eff_ea_r,
                T_ref_K=policy.T_ref_K,
                beta=policy.beta_humidity,
                lag_lambda=policy.lag_lambda,
            )

        # Residual-enabled policies see the frozen PINN estimate evaluated on
        # their observation stream; the scored outcome above remains the
        # independent DGP trajectory. Keep a separate mechanistic state so the
        # additive correction is not recursively fed back into the ODE. The
        # no-PINN arm skips only this policy-side correction.
        if capabilities_for(mode).spoilage_residual:
            elapsed_h = float(hours[idx])
            rh_transient = 0.0
            if idx > 0:
                step_h = elapsed_h - float(hours[idx - 1])
                rh_transient = (
                    abs(rh_policy_observed - prev_rh_policy_observed) / step_h
                    if step_h > 0.0 else 0.0
                )
            online_features = build_residual_feature_row(
                time_h=elapsed_h,
                temp_c=temp_policy_observed,
                rh_pct=rh_policy_observed,
                shock_g=float(row.get("shockG", 0.0)),
                rh_transient_per_h=rh_transient,
                k_ref=eff_k_ref,
                ea_over_r=eff_ea_r,
            )
            delta_quality = float(predict_residual(
                online_features, frozen_pinn,
            )[0])
            quality_policy_observed = min(
                prev_quality_policy_observed,
                float(np.clip(1.0 - rho_policy_mechanistic + delta_quality, 0.0, 1.0)),
            )
            rho_policy_observed = 1.0 - quality_policy_observed
        else:
            rho_policy_observed = rho_policy_mechanistic
            quality_policy_observed = 1.0 - rho_policy_observed

        prev_temp_policy_observed = temp_policy_observed
        prev_rh_policy_observed = rh_policy_observed
        prev_rho_policy_mechanistic = rho_policy_mechanistic
        prev_quality_policy_observed = quality_policy_observed
        temp_policy_observed_trace.append(temp_policy_observed)
        rh_policy_observed_trace.append(rh_policy_observed)
        rho_policy_observed_trace.append(rho_policy_observed)
        inventory_policy_observed_trace.append(inv_policy_observed)
        demand_policy_observed_trace.append(demand_policy_observed)
        temp_outcome_environmental_trace.append(temp_outcome_environmental)
        rh_outcome_environmental_trace.append(rh_outcome_environmental)
        rho_outcome_environmental_trace.append(rho_outcome_environmental)
        inventory_outcome_environmental_trace.append(inv_outcome_environmental)
        demand_outcome_environmental_trace.append(demand_outcome_environmental)

        lookback = min(idx + 1, 48)
        hist_slice = df.iloc[max(0, idx + 1 - lookback):idx + 1]
        demand_history = demand_policy_observed_trace[-lookback:]
        inventory_history_col = (
            "inventory_policy_observed"
            if "inventory_policy_observed" in hist_slice.columns
            else "inventory_units"
        )
        # Demand forecast via the MCP demand_query tool. Residual std
        # feeds phi_8 (demand_uncertainty) (Hyndman & Athanasopoulos
        # 2018, Ch. 8.7). Both simulator and REST route through this
        # tool so the paper numerics and live inference share one path.
        yf = query_demand(
            demand_history=demand_history,
            horizon=1,
            method=FORECAST_METHOD,
        )
        y_hat = float(yf["forecast"][0]) if yf["forecast"] else 100.0
        demand_forecast_policy_observed_trace.append(y_hat)
        demand_std = float(yf.get("std", 0.0) or 0.0)

        # Yield/supply forecast via the MCP yield_query tool. ``std`` is
        # the matching residual-std prediction-uncertainty estimate used
        # for phi_7 (supply_uncertainty).
        sf = query_yield(
            inventory_history=hist_slice[inventory_history_col].astype(float).tolist(),
            horizon=1,
            method=SUPPLY_FORECAST_METHOD,
        )
        supply_hat = (
            float(sf["forecast"][0])
            if sf["forecast"] else inv_policy_observed
        )
        supply_std = float(sf.get("std", 0.0) or 0.0)
        supply_hats.append(supply_hat)
        supply_forecast_policy_observed_trace.append(supply_hat)

        # Routing sees the observed surplus; outcome models use latent surplus.
        surplus_ratio_policy_observed = max(
            0.0, inv_policy_observed / INV_BASELINE - 1.0,
        )
        surplus_ratio_outcome_environmental = max(
            0.0, inv_outcome_environmental / INV_BASELINE - 1.0,
        )

        # Price signal: Bollinger z-score of demand, clipped to [-1, 1].
        # Positive = demand above rolling mean (shortage / price up);
        # negative = demand below (oversupply / price down). This is the
        # same statistic the REST /decide path already uses for the
        # volatility trigger, exposed here as a continuous market-pressure
        # proxy feeding phi_9.
        _boll_window = int(getattr(policy, "boll_window", 16))
        _demand_slice = pd.Series(demand_history, dtype=float)
        if len(_demand_slice) > 0:
            _rm = _demand_slice.rolling(_boll_window, min_periods=1).mean().iloc[-1]
            _rs = _demand_slice.rolling(_boll_window, min_periods=1).std().fillna(0.0).iloc[-1]
            _price_z = (float(_demand_slice.iloc[-1]) - float(_rm)) / max(float(_rs), 1e-6)
            price_signal = float(np.clip(_price_z, -1.0, 1.0))
            tau = float(abs(float(_price_z)) > float(policy.boll_k))
        else:
            price_signal = 0.0
            tau = 0.0
        demand_regime_flag_trace.append(tau)
        price_signal_trace.append(price_signal)

        # Context-enabled modes retrieve piR evidence inside the coordinator.
        # Controls do not perform an unused retrieval pass.
        rag_context = None

        # Build env_state for the coordinator. Supply and demand point
        # forecasts and residual-std uncertainties all flow through
        # obs.raw into build_feature_vector as phi_6..phi_8. The older
        # ``supply_uncertainty`` key that populated the previous psi_5
        # context feature is no longer consumed (the supply-uncertainty
        # signal lives in phi now, not psi) but is left in env_state for
        # downstream tracing tools that already read it.
        _supply_cv = (
            float(min(max(supply_std / max(abs(supply_hat), 1.0), 0.0), 1.0))
            if supply_hat else 0.0
        )
        env_state = {
            "rho": rho_policy_observed,
            "inv": inv_policy_observed,
            "temp": temp_policy_observed,
            "rh": rh_policy_observed,
            "y_hat": y_hat,
            "tau": tau,
            "surplus_ratio": surplus_ratio_policy_observed,
            "supply_hat": supply_hat,
            "supply_std": supply_std,
            "demand_std": demand_std,
            "price_signal": price_signal,
            "supply_uncertainty": round(_supply_cv, 4),
            "inv_history": hist_slice[inventory_history_col].astype(float).tolist(),
            "policy_flags": {
                "enable_mcp_qos_routing": bool(getattr(policy, "enable_mcp_qos_routing", False)),
                "enable_mcp_reliability": bool(getattr(policy, "enable_mcp_reliability", False)),
                "enable_pirag_counterfactual_eval": bool(getattr(policy, "enable_pirag_counterfactual_eval", False)),
                "enable_physics_consistency_gate": bool(getattr(policy, "enable_physics_consistency_gate", False)),
                "enable_heterogeneous_profiles": bool(getattr(policy, "enable_heterogeneous_profiles", False)),
                "enable_temporal_retrieval_weighting": bool(getattr(policy, "enable_temporal_retrieval_weighting", True)),
                "enable_dynamic_knowledge_feedback": bool(getattr(policy, "enable_dynamic_knowledge_feedback", False)),
                "enable_failure_injection": bool(getattr(policy, "enable_failure_injection", False)),
                "enable_research_metrics": bool(getattr(policy, "enable_research_metrics", False)),
            },
        }

        # Action selection via AgentCoordinator
        # Pass the actual mode name so the coordinator can apply context_mode mapping
        context_log_count_before = len(coordinator.context_log)
        step_t0 = time.perf_counter()
        action_idx, probs, active_agent = coordinator.step(
            env_state, hours[idx], effective_mode if mode not in _CONTEXT_ENABLED_MODES else mode,
            policy, rng, scenario, rag_context=rag_context,
            policy_temperature=episode_policy_temp,
            policy_categorical_uniform=_policy_categorical_uniform(
                policy_stream_seed, idx,
            ),
        )
        # Latency is recorded as observed wall-clock time (descriptive
        # only across hardware-mixed seeds; treat as a profiling hint).
        # The deterministic complexity proxy is the count of MCP tool
        # invocations and piR queries the step issued — those are
        # bit-identical across machines for the same seed.
        decision_latency_ms.append((time.perf_counter() - step_t0) * 1000.0)
        action = ACTIONS[action_idx]

        # Context-honor scoring. The coordinator records the per-step context
        # modifier vector (THETA_CONTEXT @ psi); when it carries a meaningful
        # signal we ask whether the chosen action matches the action that the
        # context layer most strongly recommends. This is the "did the agent
        # honor the context" metric the MCP+piR robustness story requires;
        # protocol reliability alone does not answer it.
        _step_modifier = getattr(coordinator, "_step_context_modifier", None)
        # Influence is scored after coordinator.post_step(), when the paired
        # paired pre-selection-state context-ablation action is available. Retain the
        # activation gates from this live decision for that deferred score.
        _influence_headline_eligible = False
        _influence_thresholds: list[float] = []
        if _step_modifier is not None:
            _mod = np.asarray(_step_modifier)
            if _mod.size:
                # 2026-05 apples-to-apples dispatch counter. Increments
                # BEFORE any threshold / guard / window check, so it
                # counts "the context layer ran on this step" regardless
                # of whether either channel subsequently survives the
                # CONTEXT_SIGNAL_THRESHOLD gate. Retrieval guards zero only
                # the piR-derived term; separately computed MCP features
                # may keep the combined modifier non-zero. A zero vector still
                # enters this branch with _mod.size > 0, so dispatch attempts
                # remain comparable across context arms.
                context_dispatch_attempt_steps += 1
                _max_abs = float(np.max(np.abs(_mod)))
                _rec = int(np.argmax(_mod))
                _honored_this_step = _rec == int(action_idx)
                # Headline threshold counters (0.10)
                if _max_abs > CONTEXT_SIGNAL_THRESHOLD:
                    _influence_headline_eligible = True
                    context_active_steps += 1
                    context_active_per_recommendation[_rec] = (
                        context_active_per_recommendation.get(_rec, 0) + 1
                    )
                    if _honored_this_step:
                        context_honored_steps += 1
                    else:
                        context_ignored_per_recommendation[_rec] = (
                            context_ignored_per_recommendation.get(_rec, 0) + 1
                        )
                # P4: per-threshold counters
                for _thr in CONTEXT_SIGNAL_THRESHOLDS:
                    if _max_abs > _thr:
                        _influence_thresholds.append(_thr)
                        context_threshold_counters[_thr]["active"] += 1
                        if _honored_this_step:
                            context_threshold_counters[_thr]["honored"] += 1

        # Modeled transport-emissions indicator (Layer 1: carbon.py)
        # Source 4: Transport distance jitter (detours, traffic, loading delays)
        # Physical outcome model is mode-neutral: identical routes under
        # identical environmental conditions produce identical emissions.
        # Architectural effects can therefore arise only through the selected
        # route and its distance, not a mode label embedded in the metric.
        transport_multiplier = stoch.perturb_transport_multiplier(counter=idx)
        transport_multiplier_outcome_environmental_trace.append(
            transport_multiplier
        )
        km = float(getattr(policy, ACTION_KM_KEYS[action])) * transport_multiplier
        thermal_stress = compute_thermal_stress(temp_outcome_environmental)
        carbon = compute_transport_carbon(
            km, policy.carbon_per_km, thermal_stress,
            eff_factor=1.0,
        )

        # Author-declared social-performance proxy with stress attenuation
        slca_result = slca_score(
            carbon,
            action,
            w_c=policy.w_c,
            w_l=policy.w_l,
            w_r=policy.w_r,
            w_p=policy.w_p,
            carbon_cap=policy.carbon_cap,
        )
        slca_raw = slca_result["composite"]
        slca_quality = compute_slca_attenuation(
            thermal_stress, surplus_ratio_outcome_environmental,
        )
        slca_c = slca_raw * slca_quality

        # Waste computation (Layer 1: waste.py + spoilage.py)
        # Uses perturbed Arrhenius params (Source 5: spoilage model error)
        k_inst = arrhenius_k(
            temp_outcome_environmental, eff_k_ref, eff_ea_r,
                             policy.T_ref_K, rh_outcome_environmental / 100.0,
                             policy.beta_humidity)
        waste_raw = compute_waste_rate(
            k_inst, surplus_ratio_outcome_environmental,
        )

        # Declared operating-envelope check applied uniformly across modes.
        # Earlier the check was gated on MCP-active modes only, so non-
        # Non-MCP modes (Static, Hybrid RL, no_slca, no_context,
        # pirag_only) reported a structurally-zero compliance violation
        # rate and MCP-active modes (mcp_only, agribrain) reported the
        # only non-zero rates. That asymmetry made cross-mode
        # comparisons on this metric meaningless. The compliance check
        # itself is just a function of (temperature, humidity); MCP's
        # role is to ROUTE that result into the policy, not to GENERATE
        # it. Calling the same function for every mode produces a
        # mode-agnostic environmental metric that any reviewer can
        # compare across the table without an asymmetry footnote.
        # MCP-active policies can receive their own operating-envelope context
        # during action selection.  The direct evaluation below is outcome
        # instrumentation only and never changes the waste of a fixed action.
        from pirag.mcp.tools.compliance import check_compliance as _check_compliance
        _compliance_uniform = _check_compliance(
            temperature=temp_outcome_environmental,
            humidity=rh_outcome_environmental,
        )
        compliance_violation = not bool(_compliance_uniform.get("compliant", True))
        if compliance_violation:
            compliance_violation_steps += 1
        # This record is passed for API compatibility; the mode-neutral outcome
        # equation intentionally ignores context for a fixed action.
        save = compute_save_factor(
            action, mode, surplus_ratio_outcome_environmental,
            compliance_data=_compliance_uniform,
        )
        waste = float(waste_raw * (1.0 - save))

        temp_violation = (
            temp_outcome_environmental > float(policy.max_temp_c)
        )
        shelf_left = 1.0 - rho_outcome_environmental
        quality_violation = shelf_left < float(policy.min_shelf_expedite)
        if temp_violation:
            temperature_violation_steps += 1
        if quality_violation:
            quality_violation_steps += 1
        if temp_violation or quality_violation:
            operational_violation_steps += 1
            # Outcome-side: record what the policy chose to do with the
            # at-risk batch on this violation step. action is the
            # canonical name from ACTIONS so the equality checks below
            # cover every dispatch path; aliased equivalents would
            # already have resolved through ACTIONS earlier in the loop.
            violation_event_count_local += 1
            if action == "cold_chain":
                violation_routed_to_cold_chain += 1
            elif action == "local_redistribute":
                violation_routed_to_local += 1
            elif action == "recovery":
                violation_routed_to_recovery += 1
        # constraint_violation_steps counts ambient-driven benchmark
        # window breaches: ``temp_violation`` (cold-chain ceiling
        # exceeded) and ``quality_violation`` (shelf-fraction below
        # policy expedite floor). Both predicates are functions of the
        # *environment trajectory* — they fire on the temperature and
        # spoilage rho from the dataset row, not on the agent's chosen
        # action. As a result, ``constraint_violation_rate`` measures
        # how stress-laden a *scenario* is, not how good a *policy*
        # is, and it should be near-flat across modes within a given
        # scenario; the latent rho path is identical across paired modes.
        # The earlier definition included compliance, which made
        # MCP-active modes appear different from non-MCP modes solely
        # because only the former invoked the tool. That was a
        # metric-definition artefact, not a policy effect.
        # ``compliance_violation_rate`` is now computed *uniformly*
        # across all modes by calling ``check_compliance`` on every
        # step regardless of mode (see uniform check above), so the
        # MCP-vs-non-MCP asymmetry is fully eliminated. Reviewers
        # should read ``constraint_violation_rate`` as an
        # *environmental signature*, not a policy-quality score.
        if temp_violation or quality_violation:
            constraint_violation_steps += 1

        # Modeled route-circularity indicator (Layer 1: reverse_logistics.py)
        recovery_opts = evaluate_recovery_options(
            rho_outcome_environmental,
            inv_outcome_environmental,
            temp_outcome_environmental,
        )
        circular = compute_circular_economy_score(action, recovery_opts)
        circular_scores.append(circular)

        # Confirmatory ARI uses latent environmental risk. Policy differences
        # can affect modeled waste and the social-performance proxy through the selected action,
        # but cannot rewrite the common environmental trajectory.
        ari = compute_ari(waste, slca_c, rho_outcome_environmental)

        # RLE tracking (Layer 1: resilience.py). The tracker computes the one
        # canonical hierarchy-inspired, severity-weighted score.
        rle_tracker.update(rho_outcome_environmental, action)

        # Reward is evaluated against the same latent outcomes as the reported
        # endpoints. No route-conditioned freshness proxy is introduced.
        # No-SLCA is a one-factor objective ablation: social-proxy shaping is
        # still measured as an outcome, but it is withheld from the learning
        # signal and from the policy's SLCA logit terms.
        reward_slca = 0.0 if mode == "no_slca" else slca_c
        reward = compute_reward(
            reward_slca, waste, rho_outcome_environmental,
            eta=policy.eta, eta_rho=policy.eta_rho,
        )

        # Activity-based Green-AI estimate for the exact action-selection timer
        # above. Per-step constants remain separately labelled proxies; water
        # and energy estimates use measured elapsed seconds. The meter's scope
        # string explicitly lists the work outside this timer.
        meter.compute_footprint(
            steps=1,
            elapsed_seconds=decision_latency_ms[-1] / 1000.0,
        )

        # Per-decision explainability record: surface the psi vector, the
        # logit modifier, and the dominant context feature so that
        # mvp/simulation/analysis/explainability_metrics.py can compute
        # policy-trace coverage and sign-consistency
        # percentages without rerunning the policy. Fields are optional:
        # context-disabled modes (static, hybrid_rl, no_context) leave them as
        # None and the analysis script ignores those rows.
        _psi_vec = getattr(coordinator, "_step_context_features", None)
        _mod_vec = getattr(coordinator, "_step_context_modifier", None)
        _gov_override = bool(getattr(coordinator, "_step_override", False))
        psi_list = (
            [float(v) for v in np.asarray(_psi_vec).flatten()]
            if _psi_vec is not None else None
        )
        mod_list = (
            [float(v) for v in np.asarray(_mod_vec).flatten()]
            if _mod_vec is not None else None
        )

        # --- H2 conditional observed-state feature-group ingredients ---
        # Observer-only fields recorded by the instrumented policy
        # (action_selection.select_action ``out`` + coordinator per-channel
        # masks). They let aggregate_channel_attribution.py reconstruct modal
        # routing after algebraically retaining each feature group in the same
        # observed state. This is not a channel-disable experiment. Fields are
        # None on steps that bypass the modifier.
        def _veclist(name):
            v = getattr(coordinator, name, None)
            if v is None:
                return None
            return [float(x) for x in np.asarray(v).flatten()]

        _base_logits = getattr(coordinator, "_step_base_logits", None)
        _post_context_logits = getattr(
            coordinator, "_step_post_context_logits_pre_override", None
        )
        _slca_shaping = getattr(coordinator, "_step_slca_shaping", None)
        _slca_amp = getattr(coordinator, "_step_slca_amp", None)
        _policy_temp = getattr(coordinator, "_step_policy_temperature", None)
        _mod_mcp = _veclist("_step_modifier_mcp")
        _mod_pirag = _veclist("_step_modifier_pirag")
        _effective_theta_arr = getattr(
            coordinator, "_step_effective_context_theta", None
        )
        _effective_theta = (
            [[float(x) for x in row] for row in np.asarray(_effective_theta_arr)]
            if _effective_theta_arr is not None else None
        )
        _chosen_context_contrib = _veclist(
            "_step_chosen_action_context_contributions"
        )
        _context_feature_contrib_arr = getattr(
            coordinator, "_step_context_feature_contributions", None
        )
        _context_feature_contrib = (
            [[float(x) for x in row]
             for row in np.asarray(_context_feature_contrib_arr)]
            if _context_feature_contrib_arr is not None else None
        )
        _context_nonfeature_residual = _veclist(
            "_step_context_nonfeature_residual"
        )
        _context_jacobian_arr = getattr(
            coordinator, "_step_context_modifier_theta_jacobian", None
        )
        _context_modifier_theta_jacobian = (
            [[float(x) for x in row]
             for row in np.asarray(_context_jacobian_arr)]
            if _context_jacobian_arr is not None else None
        )
        _context_integration = getattr(
            coordinator, "_step_context_integration_trace", None
        )
        _chosen_context_residual = getattr(
            coordinator, "_step_chosen_action_context_residual", None
        )
        _context_attribution_scope = getattr(
            coordinator, "_step_context_attribution_scope", None
        )
        _phi = _veclist("_step_phi")
        _message_bias = _veclist("_step_message_bias")
        _rag_context = getattr(coordinator, "_step_rag_context", {}) or {}
        _retrieval_top_fused_score = float(
            _rag_context.get(
                "top_fused_score",
                _rag_context.get("top_citation_score", 0.0),
            ) or 0.0
        )
        _retrieval_top_rerank_score = float(
            _rag_context.get("top_rerank_score", 0.0) or 0.0
        )

        # Action-aware dominance uses the final feature allocation after
        # scaling, clipping, and cooperative composition. A declared fixed
        # cooperative adjustment is kept as a separate residual and can itself
        # be the largest recorded calculation component.
        if _chosen_context_contrib is not None:
            _max_feature = float(np.max(np.abs(_chosen_context_contrib)))
            if (abs(float(_chosen_context_residual or 0.0)) > _max_feature):
                dominant_psi_idx = None
                dominant_context_component = "nonfeature_residual"
            else:
                dominant_psi_idx = int(np.argmax(np.abs(_chosen_context_contrib)))
                dominant_context_component = f"psi_{dominant_psi_idx}"
        else:
            dominant_psi_idx = None
            dominant_context_component = None
        dominant_action_idx = (
            int(np.argmax(np.asarray(_mod_vec))) if _mod_vec is not None else None
        )

        # Snapshot every live-decision and effective-policy field before
        # post_step runs learner updates. The record is appended only after
        # post_step computes the paired pre-selection-state context ablation.
        # Because all arrays are converted to plain lists here, later learner
        # mutation cannot change the snapshotted policy quantities.
        decision_record = {
            "step_index": int(idx),
            "ts": int(hours[idx] * 3600),
            "hour": float(hours[idx]),
            "agent": str(active_agent.agent_id),
            "role": str(active_agent.role),
            "action": str(action),
            "action_idx": int(action_idx),
            "probs": [float(p) for p in probs],
            "policy_probs_pre_override": [
                float(p) for p in np.asarray(
                    getattr(
                        coordinator,
                        "_step_policy_probs_pre_override",
                        probs,
                    ),
                    dtype=float,
                )
            ],
            "policy_categorical_uniform": (
                float(coordinator._step_policy_categorical_uniform)
                if coordinator._step_policy_categorical_uniform is not None
                else None
            ),
            "sampled_action_pre_override": int(
                coordinator._step_sampled_action_pre_override
            ),
            # Raw measured timer value. Episode latency and Green-AI activity
            # estimates are reconstructed from these Merkle-covered samples.
            "decision_latency_ms": float(decision_latency_ms[-1]),
            "reward": float(reward),
            "waste": float(waste),
            # Legacy alias is explicitly the policy-observed state.
            "rho": float(rho_policy_observed),
            "rho_policy_observed": float(rho_policy_observed),
            "rho_outcome_environmental": float(rho_outcome_environmental),
            "temp_policy_observed": float(temp_policy_observed),
            "temp_outcome_environmental": float(temp_outcome_environmental),
            "rh_policy_observed": float(rh_policy_observed),
            "rh_outcome_environmental": float(rh_outcome_environmental),
            # Frozen PINN feature retained so both latent and observation
            # trajectories can be reconstructed independently from the ledger.
            "shock_g": float(row.get("shockG", 0.0)),
            "inventory_policy_observed": float(inv_policy_observed),
            "inventory_outcome_environmental": float(inv_outcome_environmental),
            "demand_policy_observed": float(demand_policy_observed),
            "demand_forecast_policy_observed": float(y_hat),
            "supply_forecast_policy_observed": float(supply_hat),
            "demand_forecast_std_policy_observed": float(demand_std),
            "supply_forecast_std_policy_observed": float(supply_std),
            "demand_outcome_environmental": float(demand_outcome_environmental),
            "bollinger_regime_flag": float(tau),
            "regime_logit_bias": (
                [
                    float(value) for value in
                    coordinator._step_regime_logit_bias
                ]
                if coordinator._step_regime_logit_bias is not None else None
            ),
            "price_signal": float(price_signal),
            "transport_multiplier_outcome_environmental": float(
                transport_multiplier
            ),
            # Every 15-minute row is one standardized simulated routing
            # opportunity. This flag makes the outcome-accounting denominator
            # explicit without implying a measured shipment at every row.
            "simulated_dispatch_accounted": True,
            "slca": float(slca_c),
            "ari": float(ari),
            "carbon_kg": float(carbon),
            "mode": str(mode),
            "scenario": str(scenario),
            "psi": psi_list,
            "phi": _phi,
            "peer_message_bias": _message_bias,
            "combined_role_bias": [
                float(value) for value in np.asarray(
                    coordinator._step_combined_role_bias, dtype=float,
                )
            ],
            "effective_theta_delta": [
                [float(value) for value in row_values]
                for row_values in np.asarray(
                    coordinator._step_theta_delta, dtype=float,
                )
            ],
            "effective_slca_bonus_delta": [
                float(value) for value in np.asarray(
                    coordinator._step_slca_bonus_delta, dtype=float,
                )
            ],
            "effective_slca_rho_delta": [
                float(value) for value in np.asarray(
                    coordinator._step_slca_rho_delta, dtype=float,
                )
            ],
            "effective_no_slca_offset_delta": [
                float(value) for value in np.asarray(
                    coordinator._step_no_slca_offset_delta, dtype=float,
                )
            ],
            "context_modifier": mod_list,
            "effective_context_theta": _effective_theta,
            "context_feature_contributions": _context_feature_contrib,
            "context_nonfeature_residual": _context_nonfeature_residual,
            # Attribution and differentiation are separate records. The former
            # reconstructs the clipped modifier; the latter is the exact
            # derivative consumed by the context-matrix learner.
            "context_modifier_theta_jacobian": (
                _context_modifier_theta_jacobian
            ),
            "context_integration": _context_integration,
            "chosen_action_context_contributions": _chosen_context_contrib,
            "chosen_action_context_residual": (
                float(_chosen_context_residual)
                if _chosen_context_residual is not None else None
            ),
            "context_attribution_basis": (
                "final_modifier_feature_allocation_plus_explicit_residual"
                if _chosen_context_contrib is not None else None
            ),
            "context_attribution_scope": _context_attribution_scope,
            "dominant_psi_idx": dominant_psi_idx,
            "dominant_context_component": dominant_context_component,
            "dominant_action_idx": dominant_action_idx,
            "governance_override": _gov_override,
            # H2 conditional feature-group masking ingredients (observer-only).
            "base_logits": (
                [float(v) for v in _base_logits] if _base_logits is not None else None
            ),
            "post_context_logits_pre_override": (
                [float(v) for v in _post_context_logits]
                if _post_context_logits is not None else None
            ),
            "slca_shaping": (
                [float(v) for v in _slca_shaping] if _slca_shaping is not None else None
            ),
            "slca_amp": (float(_slca_amp) if _slca_amp is not None else None),
            "policy_temperature": (
                float(_policy_temp) if _policy_temp is not None else None
            ),
            "modifier_mcp": _mod_mcp,
            "modifier_pirag": _mod_pirag,
            "retrieval_top_doc_id": str(_rag_context.get("top_doc_id", "")),
            # ``retrieval_top_score`` is retained as the legacy raw-RRF alias.
            # The explicit fields prevent the ordering score from being
            # mistaken for the calibrated policy/guard strength.
            "retrieval_top_score": _retrieval_top_fused_score,
            "retrieval_top_fused_score": _retrieval_top_fused_score,
            "retrieval_top_rerank_score": _retrieval_top_rerank_score,
            "retrieval_evidence_hashes": [
                str(v) for v in (_rag_context.get("evidence_hashes", []) or [])
            ],
        }

        # Post-step: update agent state and route messages
        obs = active_agent.observe(env_state, hours[idx])
        outcome = {
            "waste": waste,
            "rho": rho_outcome_environmental,
            "slca": slca_c,
            "carbon_kg": carbon,
        }
        coordinator.post_step(active_agent, action_idx, obs, outcome,
                              hour=hours[idx], reward=reward)

        # Preserve the raw execution/activity increments that feed the
        # episode-level context, protocol, message, and complexity scalars.
        # These are collected after post_step so calls or messages emitted by
        # either half of the declared decision boundary cannot be omitted.
        context_log = coordinator.context_log
        context_log_delta = len(context_log) - context_log_count_before
        if context_log_delta not in {0, 1}:
            raise RuntimeError(
                "context activity log emitted more than one row for a decision"
            )
        context_entry = context_log[-1] if context_log_delta else {}
        primary_mcp_tools_invoked_step = list(
            context_entry.get("primary_mcp_tools_invoked", [])
        )
        cooperative_mcp_tools_invoked_step = list(
            context_entry.get("cooperative_mcp_tools_invoked", [])
        )
        mcp_tool_call_count_step = (
            len(primary_mcp_tools_invoked_step)
            + len(cooperative_mcp_tools_invoked_step)
        )
        primary_pirag_query_attempted_step = bool(
            context_entry.get("primary_retrieval_attempted", False)
        )
        cooperative_pirag_query_attempted_step = bool(
            context_entry.get("cooperative_retrieval_attempted", False)
        )
        pirag_query_count_step = (
            int(primary_pirag_query_attempted_step)
            + int(cooperative_pirag_query_attempted_step)
        )
        if int(context_entry.get(
            "pirag_query_count", pirag_query_count_step,
        )) != pirag_query_count_step:
            raise RuntimeError(
                "context log retrieval accounting is internally inconsistent"
            )
        dispatcher_tool_failure_count_step = len(
            context_entry.get("mcp_tools_failed", [])
        )

        protocol_summary_step = (
            coordinator.protocol_recorder.summary()
            if coordinator.protocol_recorder is not None else {}
        )
        protocol_summary_mapping = {
            "protocol_interaction_count_step": "total_interactions",
            "protocol_jsonrpc_error_count_step": "jsonrpc_errors",
            "protocol_tool_iserror_count_step": "tool_iserror_responses",
            "protocol_real_tool_iserror_count_step": (
                "tool_iserror_responses_real"
            ),
            "protocol_error_count_step": "real_error_responses",
            "protocol_dropped_interaction_count_step": "dropped_interactions",
        }
        protocol_count_deltas = {}
        for ledger_field, summary_field in protocol_summary_mapping.items():
            current_count = int(protocol_summary_step.get(summary_field, 0))
            delta = current_count - previous_protocol_counts[ledger_field]
            if delta < 0:
                raise RuntimeError("protocol recorder counters decreased mid-episode")
            protocol_count_deltas[ledger_field] = delta
            previous_protocol_counts[ledger_field] = current_count
        protocol_methods = protocol_summary_step.get("methods", {}) or {}
        protocol_method_deltas = {}
        for method in previous_protocol_method_counts:
            current_count = int(protocol_methods.get(method, 0))
            delta = current_count - previous_protocol_method_counts[method]
            if delta < 0:
                raise RuntimeError(
                    "protocol recorder method counters decreased mid-episode"
                )
            protocol_method_deltas[method] = delta
            previous_protocol_method_counts[method] = current_count
        # The protocol recorder counts every dispatched tools/call, including
        # calls that then raised. ``mcp_tool_call_count_step`` deliberately
        # counts only invocations that returned, because it feeds the reported
        # per-episode call metric. Reconcile the two by adding the dispatcher's
        # failure count rather than by redefining either quantity; skipped
        # tools are never dispatched and so appear in neither.
        expected_protocol_tool_calls = (
            mcp_tool_call_count_step + dispatcher_tool_failure_count_step
        )
        if protocol_method_deltas["tools/call"] != expected_protocol_tool_calls:
            raise RuntimeError(
                "MCP tool activity does not match protocol tools/call traffic"
            )
        if protocol_method_deltas["prompts/get"] != pirag_query_count_step:
            raise RuntimeError(
                "retrieval activity does not match protocol prompts/get traffic"
            )
        if protocol_count_deltas["protocol_interaction_count_step"] != (
            protocol_method_deltas["tools/call"]
            + protocol_method_deltas["prompts/get"]
        ):
            raise RuntimeError(
                "protocol interaction total includes an undeclared method"
            )

        current_message_count = len(coordinator.message_log)
        inter_agent_message_count_step = (
            current_message_count - previous_message_count
        )
        if inter_agent_message_count_step < 0:
            raise RuntimeError("inter-agent message count decreased mid-episode")
        previous_message_count = current_message_count

        # Paired context-ablation score. ``post_step`` reconstructs the same
        # policy call with context removed and restores the RNG snapshot taken
        # immediately before live selection. Both stochastic calls consume the
        # same categorical variate; the live probability-gap override discards its
        # sampled action only after that draw. Ordinary softmax sampling away
        # from an argmax is therefore not, by itself, evidence that context
        # changed the action.
        _influenced_this_step = _paired_context_action_changed(
            action_idx,
            getattr(coordinator, "_step_counterfactual_action", None),
            getattr(coordinator, "_step_counterfactual_probs", None),
        )
        if _influenced_this_step:
            if _influence_headline_eligible:
                context_influenced_steps += 1
            for _thr in _influence_thresholds:
                context_threshold_counters[_thr]["influenced"] += 1

        _counterfactual_probs = getattr(
            coordinator, "_step_counterfactual_probs", None
        )
        _counterfactual_action = (
            int(coordinator._step_counterfactual_action)
            if _counterfactual_probs is not None else None
        )
        decision_record.update({
            # Exact, detached, hash-sealed evidence for the peer, primary MCP /
            # retrieval, and cooperative MCP / retrieval channels.  It is
            # attached after post_step so emitted messages are complete, and is
            # then covered by this decision record's Merkle leaf.
            "step_channel_evidence": deepcopy(
                getattr(coordinator, "_step_channel_evidence", {}),
            ),
            # Paired context ablation, computed with the pre-selection RNG
            # snapshot and the same pre-update policy state as the live call.
            "context_counterfactual_action_idx": _counterfactual_action,
            "context_counterfactual_action": (
                str(ACTIONS[_counterfactual_action])
                if _counterfactual_action is not None else None
            ),
            "context_counterfactual_probs": (
                [float(p) for p in _counterfactual_probs]
                if _counterfactual_probs is not None else None
            ),
            "context_counterfactual_categorical_uniform": (
                float(coordinator._step_counterfactual_categorical_uniform)
                if coordinator._step_counterfactual_categorical_uniform
                is not None else None
            ),
            "context_counterfactual_sampled_action_pre_override": (
                int(
                    coordinator
                    ._step_counterfactual_sampled_action_pre_override
                )
                if coordinator
                ._step_counterfactual_sampled_action_pre_override
                is not None else None
            ),
            "context_action_changed": (
                bool(_influenced_this_step)
                if _counterfactual_probs is not None else None
            ),
            "context_influence_active": bool(_influence_headline_eligible),
            "context_influence_counted": bool(
                _influence_headline_eligible and _influenced_this_step
            ),
            "context_influence_threshold": float(CONTEXT_SIGNAL_THRESHOLD),
            "mcp_tool_call_count_step": int(mcp_tool_call_count_step),
            "primary_mcp_tools_invoked_step": (
                primary_mcp_tools_invoked_step
            ),
            "cooperative_mcp_tools_invoked_step": (
                cooperative_mcp_tools_invoked_step
            ),
            "pirag_query_count_step": int(pirag_query_count_step),
            "primary_pirag_query_attempted_step": bool(
                primary_pirag_query_attempted_step
            ),
            "cooperative_pirag_query_attempted_step": bool(
                cooperative_pirag_query_attempted_step
            ),
            "dispatcher_tool_failure_count_step": int(
                dispatcher_tool_failure_count_step
            ),
            "inter_agent_message_count_step": int(
                inter_agent_message_count_step
            ),
            **protocol_count_deltas,
            "protocol_tools_call_count_step": int(
                protocol_method_deltas["tools/call"]
            ),
            "protocol_prompts_get_count_step": int(
                protocol_method_deltas["prompts/get"]
            ),
            # Publication H3 exposure accounting.  These values are recorded
            # for every mode (nominal runs carry zeros) so a validator can
            # reconstruct the retained episode's treatment dose directly
            # from Merkle-covered decision records.
            "h3_stressor": str(
                decision_ledger.metadata.get("observation_treatment", {}).get(
                    "stressor", "nominal",
                )
            ),
            "h3_data_observation_treatment": bool(
                decision_ledger.metadata.get("observation_treatment", {}).get(
                    "data_observation_treatment", False,
                )
            ),
            "h3_temp_noise_c": float(
                df["h3_temp_noise_c"].iloc[idx]
                if "h3_temp_noise_c" in df.columns else 0.0
            ),
            "h3_rh_noise_pct": float(
                df["h3_rh_noise_pct"].iloc[idx]
                if "h3_rh_noise_pct" in df.columns else 0.0
            ),
            "h3_missing_observation": bool(
                df["h3_missing_observation"].iloc[idx]
                if "h3_missing_observation" in df.columns else False
            ),
            "h3_telemetry_source_step_index": int(
                df["h3_telemetry_source_step_index"].iloc[idx]
                if "h3_telemetry_source_step_index" in df.columns else idx
            ),
            "h3_fault_injection_scheduled_opportunity": bool(
                getattr(policy, "enable_failure_injection", False)
                and int(float(hours[idx])) % 11 == 0
            ),
            "h3_fault_injection_triggered": bool(
                getattr(coordinator, "_step_fault_recovery", False)
            ),
            "h3_fault_injected_tool_result_count": int(
                getattr(coordinator, "_step_fault_injected_result_count", 0)
            ),
        })
        decision_ledger.append(decision_record)

        # PolicyLearner: record experience for optional online learning.
        # Must pass the same 10-dim phi the policy actually saw,
        # otherwise the learner's gradient is computed against the wrong
        # feature vector.
        if learner is not None:
            phi = build_feature_vector(
                rho_policy_observed,
                inv_policy_observed,
                y_hat,
                temp_policy_observed,
                supply_hat=supply_hat,
                supply_std=supply_std,
                demand_std=demand_std,
                price_signal=price_signal,
            )
            learner.record(phi, action_idx, reward)

        # Collect traces
        ari_vals.append(ari)
        waste_vals.append(waste)
        slca_vals.append(slca_c)
        carbon_total += carbon
        action_trace.append(action_idx)
        prob_trace.append(probs.tolist())
        reward_trace.append(reward)
        carbon_trace.append(carbon)
        simulated_dispatch_accounted_trace.append(True)
        # 2026-05 trace honesty: ``slca_result`` from ``slca_score`` is the
        # RAW (unattenuated) composite computed from the four pillars.
        # The headline ``slca`` scalar (and ``slca_trace[t]`` below) uses
        # the THERMAL-STRESS / SURPLUS-ATTENUATED form
        # ``slca_c = slca_raw * slca_quality``. The two diverge whenever
        # ``slca_quality < 1`` (heatwave, overproduction). To make this
        # explicit and verifiable from the trace alone, we extend each
        # entry with two extra keys:
        #   ``slca_quality``         -- the attenuation factor (0..1)
        #   ``composite_attenuated`` -- the exact per-step contribution to
        #                               the headline scalar
        # Identity that now holds:
        #   slca_episode == mean(s["composite_attenuated"]
        #                         for s in slca_component_trace)
        # Identity that still holds (raw decomposition, fig 3 panel D):
        #   composite == w_c*C + w_l*L + w_r*R + w_p*P
        slca_component_trace.append({
            **slca_result,
            "slca_quality": float(slca_quality),
            "composite_attenuated": float(slca_c),
        })
        # Read the coordinator's per-step diagnostic activation flags.
        # ``getattr`` defaults guard against runs that swap in a
        # coordinator without these attributes (e.g. older pickled
        # learner-state replays).
        cooperative_veto_trace.append(int(bool(
            getattr(coordinator, "_step_cooperative_veto", False),
        )))
        fault_recovery_trace.append(int(bool(
            getattr(coordinator, "_step_fault_recovery", False),
        )))
        fault_injected_result_count_trace.append(int(
            getattr(coordinator, "_step_fault_injected_result_count", 0),
        ))
        physics_gate_trace.append(int(bool(
            getattr(coordinator, "_step_physics_gate", False),
        )))

    # PolicyLearner: apply gradient update at episode end (disabled by default)
    if learner is not None:
        import src.models.action_selection as _as_module
        updated_theta = learner.update(_as_module.THETA.copy())
        delta_norm = np.linalg.norm(updated_theta - _as_module.THETA)
        _as_module.THETA = updated_theta  # persist update for next episode
        print(f"  Policy weights updated via REINFORCE (delta norm: {delta_norm:.6f})")

    # Episode-level metrics (Layer 1: resilience.py).
    # 2026-04 single-version-of-the-truth pass: per user mandate every
    # metric has exactly one formulation across the codebase. The
    # simulator emits one ARI (compute_ari, multiplicative), one RLE
    # (compute_rle, EU-hierarchy + severity-weighted, rho-conditional
    # with smooth transition), one equity (compute_equity, stability-
    # weighted mean) - no parallel "geometric-mean" / "uniform-
    # weights" / "Sen-welfare" companions. The earlier robustness
    # companions (ari_geom, rle_uniform, equity_sen) were retired
    # alongside their function definitions in resilience.py.
    rle = rle_tracker.rle
    equity = compute_equity(slca_vals)

    # Rolling equity (6-hour window = 24 steps at 15-min resolution)
    eq_window = 24
    equity_trace = []
    for idx in range(n):
        start = max(0, idx - eq_window + 1)
        window_slca = slca_vals[start:idx + 1]
        eq_val = compute_equity(window_slca)
        equity_trace.append(eq_val)

    latency_arr = np.array(decision_latency_ms, dtype=float) if decision_latency_ms else np.array([0.0])
    latency_penalty_usd = float(np.sum(np.maximum(latency_arr - 50.0, 0.0)) * 0.0002)
    from pirag.context_to_logits import THETA_CONTEXT as _THETA_CONTEXT_PRIOR
    context_prior = (
        np.asarray(context_learner_overrides["initial_theta"], dtype=float)
        if context_learner_overrides
        and "initial_theta" in context_learner_overrides
        else np.asarray(_THETA_CONTEXT_PRIOR, dtype=float)
    )
    import src.models.action_selection as _action_selection_module
    context_prior_sha256 = _canonical_sha256(context_prior.tolist())
    policy_theta_initial_sha256 = _canonical_sha256(
        np.asarray(_action_selection_module.THETA, dtype=float).tolist()
    )
    latent_environment_sha256 = _canonical_sha256({
        "hours": [float(value) for value in hours],
        "temp_outcome_environmental": temp_outcome_environmental_trace,
        "rh_outcome_environmental": rh_outcome_environmental_trace,
        "rho_outcome_environmental": rho_outcome_environmental_trace,
        "inventory_outcome_environmental": inventory_outcome_environmental_trace,
        "demand_outcome_environmental": demand_outcome_environmental_trace,
        "transport_multiplier_outcome_environmental": (
            transport_multiplier_outcome_environmental_trace
        ),
        "effective_k_ref": float(eff_k_ref),
        "effective_Ea_R": float(eff_ea_r),
        "scenario_onset_offset_hours": float(
            df["scenario_onset_offset_hours"].iloc[0]
            if "scenario_onset_offset_hours" in df.columns else 0.0
        ),
    })
    observed_policy_input_sha256 = _canonical_sha256({
        "hours": [float(value) for value in hours],
        "temp_policy_observed": temp_policy_observed_trace,
        "rh_policy_observed": rh_policy_observed_trace,
        "rho_policy_observed": rho_policy_observed_trace,
        "inventory_policy_observed": inventory_policy_observed_trace,
        "demand_forecast_policy_observed": (
            demand_forecast_policy_observed_trace
        ),
        "supply_forecast_policy_observed": (
            supply_forecast_policy_observed_trace
        ),
    })
    demand_observation_sha256 = _canonical_sha256({
        "hours": [float(value) for value in hours],
        "demand_policy_observed": demand_policy_observed_trace,
        "demand_forecast_policy_observed": (
            demand_forecast_policy_observed_trace
        ),
        "demand_regime_flag": demand_regime_flag_trace,
        "price_signal": price_signal_trace,
    })
    result = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "benchmark_seed": benchmark_seed,
        "episode_index": int(episode_index),
        "environment_stream_id": str(environment_stream_id),
        "policy_stream_id": str(policy_stream_id),
        "stochastic_stream_id": str(stochastic_stream_id),
        "learning_enabled": bool(learning_enabled),
        "episode_phase": (
            "adaptation"
            if learning_enabled and capabilities_for(mode).learned
            else "frozen_evaluation"
            if capabilities_for(mode).learned
            else "fixed_evaluation"
        ),
        "context_prior_sha256": context_prior_sha256,
        "policy_theta_initial_sha256": policy_theta_initial_sha256,
        "spoilage_estimator": deepcopy(
            decision_ledger.metadata["spoilage_estimator"]
        ),
        "latent_spoilage_model": deepcopy(
            decision_ledger.metadata["latent_spoilage_model"]
        ),
        "latent_environment_sha256": latent_environment_sha256,
        "observed_policy_input_sha256": observed_policy_input_sha256,
        "demand_observation_sha256": demand_observation_sha256,
        "demand_forecast_method": FORECAST_METHOD,
        "supply_forecast_method": SUPPLY_FORECAST_METHOD,
        "scenario_onset_offset_hours": float(
            df["scenario_onset_offset_hours"].iloc[0]
            if "scenario_onset_offset_hours" in df.columns else 0.0
        ),
        "effective_k_ref": float(eff_k_ref),
        "effective_Ea_R": float(eff_ea_r),
        "dispatch_opportunity_count": int(n),
        "dispatch_cadence_hours": float(
            np.median(np.diff(hours)) if len(hours) > 1 else 0.0
        ),
        "endpoint_unit": "standardized_routing_opportunity",
        "functional_unit": (
            "one standardized batch-routing opportunity per 15-minute row"
        ),
        "shipment_interpretation": (
            "synthetic activity unit; not evidence of 288 measured shipments"
        ),
        "waste_definition": "mean_modelled_fraction_per_routing_opportunity",
        "waste_cap_fraction_after_surplus_amplification": float(WASTE_CAP),
        "carbon_definition": "summed_standardized_action_distance_emissions_proxy_kgCO2e",
        "carbon_efficiency_definition": (
            "episode_mean_ari/episode_summed_modeled_transport_emissions_"
            "indicator_kgCO2e; no factor of 1000"
        ),
        "transport_carbon_model": {
            "equation": (
                "E_kgCO2e=route_km*stochastic_distance_multiplier*"
                "carbon_per_km*physical_efficiency_factor*"
                "(1+refrigeration_cop_penalty*thermal_stress)"
            ),
            "route_km_by_action_before_stochastic_multiplier": {
                action: float(getattr(policy, ACTION_KM_KEYS[action]))
                for action in ACTIONS
            },
            "carbon_per_km_kgCO2e": float(policy.carbon_per_km),
            "physical_efficiency_factor": 1.0,
            "physical_efficiency_factor_basis": (
                "held common across experimental modes"
            ),
            "experimental_mode_multiplier_present": False,
            "parameter_status": (
                "author-declared synthetic assumptions; not fleet inventory"
            ),
        },
        "slca_carbon_basis": (
            "per_routing_opportunity_action_emissions_proxy_kgCO2e"
        ),
        "slca_carbon_cap_kg_per_routing_opportunity": float(policy.carbon_cap),
        "ari": float(np.mean(ari_vals)), "rle": float(rle),
        "waste": float(np.mean(waste_vals)), "slca": float(np.mean(slca_vals)),
        "carbon": float(carbon_total), "equity": float(equity),
        "carbon_efficiency_ari_per_kgco2e_proxy": compute_carbon_efficiency(
            float(np.mean(ari_vals)), float(carbon_total),
        ),
        "circular_economy": float(np.mean(circular_scores)),
        "mean_supply_forecast": float(np.mean(supply_hats)),
        # Wall-clock latency is descriptive only (hardware-dependent;
        # not used for inferential CIs per docs/STATISTICAL_METHODS.md).
        # The reproducibility-friendly proxy is the deterministic
        # complexity counter further down (`mcp_calls_per_episode`,
        # `pirag_queries_per_episode`).
        "mean_decision_latency_ms": float(np.mean(latency_arr)),
        "mean_decision_latency_ms_descriptive_only": True,
        "p95_decision_latency_ms": float(np.percentile(latency_arr, 95)),
        "latency_penalty_usd": latency_penalty_usd,
        "latency_penalty_usd_descriptive_only": True,
        # ``constraint_violation_rate`` is the fraction of steps that
        # breach the cold-chain temperature ceiling OR the shelf-life
        # expedite floor. Both predicates are environmental — driven by
        # the scenario's ambient trajectory rather than the chosen
        # action — so this metric is best read as a *scenario stress
        # signature*, not as a policy quality score. Compliance
        # (``check_compliance``) is reported separately and is now
        # called uniformly across all modes, removing the former
        # MCP-vs-non-MCP definitional asymmetry.
        "constraint_violation_rate": float(constraint_violation_steps / max(n, 1)),
        "constraint_violation_rate_is_environmental": True,
        "compliance_violation_rate": float(compliance_violation_steps / max(n, 1)),
        "operating_envelope_violation_rate": float(
            compliance_violation_steps / max(n, 1)
        ),
        "temperature_violation_rate": float(temperature_violation_steps / max(n, 1)),
        "quality_violation_rate": float(quality_violation_steps / max(n, 1)),
        # P2: CVR split so cross-mode comparisons are honest. operational_cvr
        # is the OR of temperature and quality (comparable across every mode,
        # including static / hybrid_rl, which do not invoke the MCP tool).
        # The legacy regulatory_violation_rate key is retained for API
        # compatibility. It is an author-declared operating-envelope rate,
        # not a regulatory determination.
        "operational_violation_rate": float(operational_violation_steps / max(n, 1)),
        "regulatory_violation_rate": float(compliance_violation_steps / max(n, 1)),
        # Outcome-side disposition on the env-driven violation event set
        # (steps where temp_violation OR quality_violation fired). The
        # three rates are conditional on a violation event having
        # occurred and sum to 1.0 by construction whenever
        # violation_event_count > 0; when the episode had zero violation
        # events, all three are 0.0 by convention. Unlike
        # constraint_violation_rate / regulatory_violation_rate (env-
        # driven, near-flat across modes within a scenario by design),
        # these three ARE policy-driven by construction: they record
        # what the agent chose to do with the at-risk batch. Their direction
        # is read from results rather than encoded as a validation condition.
        "downstream_violation_rate": float(
            violation_routed_to_cold_chain / max(violation_event_count_local, 1)
        ) if violation_event_count_local > 0 else 0.0,
        "redistribute_violation_rate": float(
            violation_routed_to_local / max(violation_event_count_local, 1)
        ) if violation_event_count_local > 0 else 0.0,
        "contained_violation_rate": float(
            violation_routed_to_recovery / max(violation_event_count_local, 1)
        ) if violation_event_count_local > 0 else 0.0,
        "violation_event_count": int(violation_event_count_local),
        "context_active_steps": int(context_active_steps),
        "context_active_fraction": float(context_active_steps / max(n, 1)),
        # 2026-05 apples-to-apples cross-mode dispatch counter. See the
        # initialisation comment in the per-step loop for the full
        # rationale. ``dispatch_attempts`` counts every step where the
        # context layer ran (modifier vector emitted), regardless of
        # whether the signal then survived the CONTEXT_SIGNAL_THRESHOLD
        # gate. Equals 0 for context-disabled modes (static, hybrid_rl,
        # no_context) and equals n_steps for every context-enabled mode
        # on the canonical 72-hour episodes (288 steps).
        "context_dispatch_attempt_steps": int(context_dispatch_attempt_steps),
        "context_dispatch_attempt_fraction": float(
            context_dispatch_attempt_steps / max(n, 1)
        ),
        "context_honored_steps": int(context_honored_steps),
        "context_honor_rate": (
            float(context_honored_steps / context_active_steps)
            if context_active_steps else 0.0
        ),
        # Context-influence metric: companion to honor rate. Counts
        # context-active steps where live and context-ablated policy calls,
        # paired on the same saved pre-selection RNG state, select different
        # actions.
        # Headline metric for fig 9; honor rate is retained above as a
        # supplementary-methods companion.
        "context_influenced_steps": int(context_influenced_steps),
        "context_influence_rate": (
            float(context_influenced_steps / context_active_steps)
            if context_active_steps else 0.0
        ),
        # 2026-05 cross-mode-comparable influence rate: same numerator
        # as context_influence_rate but normalised by
        # ``context_dispatch_attempt_steps`` instead of
        # ``context_active_steps``. Reads as the paired pre-selection-state action-change
        # fraction per context-dispatch attempt, apples-to-apples across all
        # context-enabled modes regardless of whether their
        # primary-stage retrieval cleared the cooperative-window guard.
        # On heatwave for d33b8de this resolves the asymmetry in
        # context_active_steps (72 for agribrain/pirag_only vs ~168 for
        # mcp_only) by using a 288-step denominator for all three.
        "context_dispatch_influence_rate": (
            float(context_influenced_steps / context_dispatch_attempt_steps)
            if context_dispatch_attempt_steps else 0.0
        ),
        "context_active_per_recommendation": dict(context_active_per_recommendation),
        "context_ignored_per_recommendation": dict(context_ignored_per_recommendation),
        "context_threshold_counters": {
            f"{thr:.2f}": {
                "active": int(counters["active"]),
                "honored": int(counters["honored"]),
                "influenced": int(counters["influenced"]),
                "honor_rate": (
                    float(counters["honored"] / counters["active"])
                    if counters["active"] else 0.0
                ),
                "influence_rate": (
                    float(counters["influenced"] / counters["active"])
                    if counters["active"] else 0.0
                ),
            }
            for thr, counters in context_threshold_counters.items()
        },
        "ari_trace": ari_vals, "waste_trace": waste_vals,
        "action_trace": action_trace,
        "rho_policy_observed_trace": rho_policy_observed_trace,
        "rho_outcome_environmental_trace": rho_outcome_environmental_trace,
        "temp_policy_observed_trace": temp_policy_observed_trace,
        "temp_outcome_environmental_trace": temp_outcome_environmental_trace,
        "rh_policy_observed_trace": rh_policy_observed_trace,
        "rh_outcome_environmental_trace": rh_outcome_environmental_trace,
        "inventory_policy_observed_trace": inventory_policy_observed_trace,
        "inventory_outcome_environmental_trace": (
            inventory_outcome_environmental_trace
        ),
        "demand_policy_observed_trace": demand_policy_observed_trace,
        "demand_forecast_policy_observed_trace": (
            demand_forecast_policy_observed_trace
        ),
        "demand_regime_flag_trace": demand_regime_flag_trace,
        "price_signal_trace": price_signal_trace,
        "supply_forecast_policy_observed_trace": (
            supply_forecast_policy_observed_trace
        ),
        "demand_outcome_environmental_trace": (
            demand_outcome_environmental_trace
        ),
        "transport_multiplier_outcome_environmental_trace": (
            transport_multiplier_outcome_environmental_trace
        ),
        "simulated_dispatch_accounted_trace": (
            simulated_dispatch_accounted_trace
        ),
        # Legacy aliases are policy-observed by definition in schema v3.
        "rho_trace": rho_policy_observed_trace,
        "prob_trace": prob_trace, "reward_trace": reward_trace,
        "carbon_trace": carbon_trace,
        "slca_component_trace": slca_component_trace, "slca_trace": slca_vals,
        "equity_trace": equity_trace,
        "hours": hours.tolist(),
        # Per-step diagnostic mechanism-activation traces. Each entry is 0 or 1; for
        # static / hybrid_rl / no_context every entry is 0 by
        # construction (those modes skip the context channel where the
        # mechanisms live). These legacy-keyed traces are retained for
        # diagnostic analysis; the canonical Figure 4 does not plot them.
        "cooperative_veto_trace": cooperative_veto_trace,
        "fault_recovery_trace": fault_recovery_trace,
        "fault_injected_result_count_trace": fault_injected_result_count_trace,
        # Separate the deterministic schedule from observed treatment
        # exposure.  A scheduled opportunity need not produce a drop when the
        # tool channel is structurally unavailable; cyber_outage after hour 24
        # is the canonical example.
        "fault_injection_scheduled_opportunity_steps": int(
            sum(int(int(float(hour)) % 11 == 0) for hour in hours)
            if bool(getattr(policy, "enable_failure_injection", False)) else 0
        ),
        "fault_injection_trigger_steps": int(sum(fault_recovery_trace)),
        "fault_injected_tool_result_count": int(
            sum(fault_injected_result_count_trace)
        ),
        "physics_gate_trace": physics_gate_trace,
        "temp_trace": temp_policy_observed_trace,
        "rh_trace": rh_policy_observed_trace,
        "demand_trace": demand_forecast_policy_observed_trace,
        "inventory_trace": inventory_policy_observed_trace,
        "footprint": meter.summary(),
        "agent_summaries": coordinator.agent_summaries(),
        "message_count": len(coordinator.message_log),
        "learner_freeze_summary": coordinator.learner_freeze_summary(),
    }

    # Context diagnostics for context-enabled modes
    if context_mode:
        result["context_summary"] = coordinator.context_summary()
        result["learner_summary"] = coordinator.learner_summary()
        result["evaluator_summary"] = coordinator.evaluator_summary()
        # Deterministic complexity proxy for latency (hardware-independent).
        # Wall-clock decision_latency_ms varies 2-10x across machines;
        # these counters are bit-identical given the same seed and so
        # are the reproducibility-friendly latency surrogates.
        ctx_sum = result["context_summary"]
        result["mcp_calls_per_episode"] = int(ctx_sum.get("total_mcp_tool_calls", 0))
        result["pirag_queries_per_episode"] = int(
            ctx_sum.get("total_pirag_queries", 0)
        )
        result["dispatcher_tool_failure_count"] = int(
            ctx_sum.get("dispatcher_tool_failures", 0)
        )
        if (
            result["dispatcher_tool_failure_count"]
            and os.environ.get("STRICT_VALIDATION", "0") == "1"
        ):
            raise RuntimeError(
                "MCP dispatcher failure(s) were retained in a strict episode: "
                f"mode={mode}, scenario={scenario}, seed={seed}, "
                f"count={result['dispatcher_tool_failure_count']}"
            )

        # Finalize the real in-process JSON-RPC record after every episode,
        # including the non-retained learning episodes. A protocol/tool error
        # changes the context received by the policy, so canonical strict runs
        # must abort instead of silently treating that fallback as experimental
        # data. The H3 fault dose is applied after successful calls by replacing
        # values in ``mcp_results``; it therefore remains a declared treatment
        # and does not increment these protocol counters.
        protocol_recorder = coordinator.protocol_recorder
        if protocol_recorder is None:
            protocol_summary = {
                "recorder_available": False,
                "total_interactions": 0,
                "dropped_interactions": 0,
                "jsonrpc_errors": 0,
                "tool_iserror_responses": 0,
                "tool_iserror_responses_real": 0,
                "real_error_responses": 0,
                "has_real_errors": False,
            }
            if os.environ.get("STRICT_VALIDATION", "0") == "1":
                raise RuntimeError(
                    "MCP ProtocolRecorder unavailable for context-enabled "
                    f"episode mode={mode}, scenario={scenario}, seed={seed}"
                )
        else:
            protocol_summary = protocol_recorder.finalize_episode(
                strict_validation=(
                    os.environ.get("STRICT_VALIDATION", "0") == "1"
                ),
                episode_label=f"mode={mode}, scenario={scenario}, seed={seed}",
            )
            protocol_summary["recorder_available"] = True
        result["protocol_summary"] = protocol_summary
        result["protocol_interaction_count"] = int(
            protocol_summary.get("total_interactions", 0)
        )
        result["protocol_tools_call_count"] = int(
            (protocol_summary.get("methods", {}) or {}).get("tools/call", 0)
        )
        result["protocol_prompts_get_count"] = int(
            (protocol_summary.get("methods", {}) or {}).get("prompts/get", 0)
        )
        result["protocol_jsonrpc_error_count"] = int(
            protocol_summary.get("jsonrpc_errors", 0)
        )
        result["protocol_tool_iserror_count"] = int(
            protocol_summary.get("tool_iserror_responses", 0)
        )
        result["protocol_real_tool_iserror_count"] = int(
            protocol_summary.get("tool_iserror_responses_real", 0)
        )
        result["protocol_error_count"] = int(
            protocol_summary.get("real_error_responses", 0)
        )
        result["protocol_dropped_interaction_count"] = int(
            protocol_summary.get("dropped_interactions", 0)
        )
        result["context_execution_error_count"] = (
            result["dispatcher_tool_failure_count"]
            + result["protocol_error_count"]
        )
    else:
        # Non-context modes still report the counters as zero so the
        # field exists in every row of the aggregated tables.
        result["mcp_calls_per_episode"] = 0
        result["pirag_queries_per_episode"] = 0
        result["protocol_interaction_count"] = 0
        result["protocol_tools_call_count"] = 0
        result["protocol_prompts_get_count"] = 0
        result["protocol_jsonrpc_error_count"] = 0
        result["protocol_tool_iserror_count"] = 0
        result["protocol_real_tool_iserror_count"] = 0
        result["protocol_error_count"] = 0
        result["protocol_dropped_interaction_count"] = 0
        result["dispatcher_tool_failure_count"] = 0
        result["context_execution_error_count"] = 0

    # Policy-delta learner runs for every non-static mode, not just the
    # context-enabled ones, so its summary lives outside the context block.
    _theta_summary = coordinator.theta_learner_summary()
    if _theta_summary:
        result["theta_learner_summary"] = _theta_summary
    _rsl_summary = coordinator.reward_shaping_learner_summary()
    if _rsl_summary:
        result["reward_shaping_learner_summary"] = _rsl_summary

    learner_state_after_evidence = deepcopy(coordinator.save_learner_states())
    result["learner_state_before_sha256"] = _archive_canonical_sha256(
        learner_state_before_evidence,
    )
    result["learner_state_after_sha256"] = _archive_canonical_sha256(
        learner_state_after_evidence,
    )
    result["learner_continuation_before_sha256"] = _archive_canonical_sha256(
        _learner_continuation_payload(learner_state_before_evidence),
    )
    result["learner_continuation_after_sha256"] = _archive_canonical_sha256(
        _learner_continuation_payload(learner_state_after_evidence),
    )
    # The full values are internal until the wrapper has written the lossless
    # archive.  Only hashes remain in ordinary endpoint envelopes.
    result["_learner_state_before_evidence"] = learner_state_before_evidence
    result["_learner_state_after_evidence"] = learner_state_after_evidence

    # Trace export for paper evidence is independent of whether a reward-
    # shaping learner exists in this mode.
    if coordinator.trace_exporter is not None:
        result["trace_summary"] = coordinator.trace_exporter.summary()
        result["_trace_exporter"] = coordinator.trace_exporter

    # Protocol recorder for in-process MCP dispatcher traces (see
    # pirag/mcp/protocol_recorder.py docstring for the distinction between
    # dispatch traces and wire bytes).
    if coordinator.protocol_recorder is not None:
        result["_protocol_recorder"] = coordinator.protocol_recorder

    reconstructed_episode_evidence = reconstruct_episode_evidence(
        decision_ledger.recent_records(len(decision_ledger)),
        decision_ledger.metadata["episode_evidence_contract"],
        where=f"mode={mode},scenario={scenario},seed={seed}",
        contract_validated=True,
    )
    for evidence_field, reconstructed_value in (
        reconstructed_episode_evidence.items()
    ):
        if isinstance(reconstructed_value, dict):
            if result.get(evidence_field) != reconstructed_value:
                raise RuntimeError(
                    f"episode object {evidence_field!r} does not reconstruct "
                    "from per-decision evidence"
                )
            continue
        observed_value = result.get(evidence_field)
        if isinstance(reconstructed_value, int):
            consistent = observed_value == reconstructed_value
        else:
            consistent = math.isclose(
                float(observed_value), float(reconstructed_value),
                rel_tol=1e-12, abs_tol=1e-12,
            )
        if not consistent:
            raise RuntimeError(
                f"episode scalar {evidence_field!r} does not reconstruct from "
                "per-decision evidence"
            )

    # Finalise the per-episode decision ledger: compute the Merkle root,
    # write the JSONL artifact, and (optionally) anchor the root on-chain
    # when CHAIN_SUBMIT=1 and chain_cfg is provided via environment.
    ledger_dir = _episode_ledger_root()
    full_evidence = _full_evidence_capture_enabled()
    if full_evidence and int(episode_index) < 3:
        ledger_path = (
            ledger_dir / "adaptation_episode_ledgers" / f"{mode}__{scenario}"
            / f"episode_{int(episode_index)}.jsonl.gz"
        )
        ledger_storage = "deterministic_gzip_jsonl"
    else:
        # Episode 3 retains the historical canonical path consumed by every
        # publication validator and aggregator. Non-publication callers with
        # FULL_EVIDENCE_CAPTURE=0 retain the legacy behavior for compatibility.
        ledger_path = ledger_dir / f"{mode}__{scenario}.jsonl"
        ledger_storage = "plain_jsonl"
    decision_ledger.metadata.update({
        "benchmark_seed": benchmark_seed,
        "episode_index": int(episode_index),
        "environment_stream_id": str(environment_stream_id),
        "policy_stream_id": str(policy_stream_id),
        "stochastic_stream_id": str(stochastic_stream_id),
        "learning_enabled": bool(learning_enabled),
        "episode_phase": result["episode_phase"],
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "dispatch_opportunity_count": int(n),
        "dispatch_cadence_hours": result["dispatch_cadence_hours"],
        "context_prior_sha256": context_prior_sha256,
        "policy_theta_initial_sha256": policy_theta_initial_sha256,
        "latent_environment_sha256": latent_environment_sha256,
        "observed_policy_input_sha256": observed_policy_input_sha256,
        "demand_observation_sha256": demand_observation_sha256,
        "demand_forecast_method": FORECAST_METHOD,
        "supply_forecast_method": SUPPLY_FORECAST_METHOD,
        "effective_k_ref": float(eff_k_ref),
        "effective_Ea_R": float(eff_ea_r),
        "scenario_onset_offset_hours": result["scenario_onset_offset_hours"],
        "learner_state_before_sha256": result["learner_state_before_sha256"],
        "learner_state_after_sha256": result["learner_state_after_sha256"],
        "learner_continuation_before_sha256": result[
            "learner_continuation_before_sha256"
        ],
        "learner_continuation_after_sha256": result[
            "learner_continuation_after_sha256"
        ],
    })
    if ledger_storage == "deterministic_gzip_jsonl":
        compressed_receipt = decision_ledger.write_jsonl_gzip(ledger_path)
        result["decision_ledger_sha256"] = compressed_receipt.literal_sha256
        result["decision_ledger_bytes"] = compressed_receipt.literal_bytes
    else:
        decision_ledger.write_jsonl(ledger_path)
        ledger_sha256, ledger_bytes = _sha256_file(ledger_path)
        result["decision_ledger_sha256"] = ledger_sha256
        result["decision_ledger_bytes"] = ledger_bytes
    result["decision_ledger_path"] = str(ledger_path)
    result["decision_ledger_storage"] = ledger_storage
    result["decision_ledger_root"] = decision_ledger.merkle_root()
    result["decision_ledger_n"] = len(decision_ledger)
    if os.environ.get("CHAIN_SUBMIT", "0") == "1":
        chain_cfg_json = os.environ.get("CHAIN_CFG_JSON")
        if chain_cfg_json:
            # Default to best-effort during simulation so a single chain
            # failure does not abort a 20-seed HPC run, but emit a WARN
            # log via decision_ledger.submit_onchain so operators can
            # see how many submissions actually landed. Set
            # CHAIN_BEST_EFFORT=false to make submission failures fatal.
            os.environ.setdefault("CHAIN_BEST_EFFORT", "true")
            try:
                import json as _json
                chain_cfg = _json.loads(chain_cfg_json)
                tx = decision_ledger.submit_onchain(chain_cfg)
                if tx:
                    result["decision_ledger_tx"] = tx
                else:
                    result["decision_ledger_tx_status"] = "best_effort_skipped"
            except Exception as _exc:
                _log.warning("on-chain ledger submission skipped: %s", _exc)
                result["decision_ledger_tx_status"] = f"error:{type(_exc).__name__}"

    # Optional per-episode ProvenanceRegistry anchoring. The confirmatory
    # benchmark always writes the local JSONL ledger and Merkle root, while
    # contract submission is an explicitly requested deployment operation.
    if os.environ.get("CHAIN_SUBMIT", "0") == "1":
        try:
            from pirag.chain.client import anchor_root as _prov_anchor
            episode_tag = f"episode_{mode}_{scenario}_{seed}"
            prov_tx = _prov_anchor(
                decision_ledger.merkle_root(), policy_uri=episode_tag
            )
            if prov_tx:
                result["provenance_registry_tx"] = prov_tx
            else:
                result["provenance_registry_tx_status"] = "chain_not_configured"
        except Exception as _exc:  # noqa: BLE001
            _log.warning("provenance registry anchor skipped: %s", _exc)
            result["provenance_registry_tx_status"] = (
                f"error:{type(_exc).__name__}"
            )
    else:
        result["provenance_registry_tx_status"] = "not_requested"

    # Persist learner state for the next equal-budget iteration of this same
    # scenario. ``run_all`` discards the cache before the next scenario.
    if learner_state_cache is not None and learning_enabled:
        learner_state_cache[mode] = deepcopy(learner_state_after_evidence)

    return result


# ---------------------------------------------------------------------------
# Full run across all scenarios × modes
# ---------------------------------------------------------------------------
def _restore_policy_theta_after_call(function):
    """Restore the process-global base-policy matrix on every exit path.

    The benchmark temporarily applies a seed-level prior and several
    sensitivity priors. A failed local probe must not contaminate a later run
    in the same interpreter, so cleanup cannot depend on reaching the normal
    tail of :func:`run_all`.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        import src.models.action_selection as action_selection

        saved = action_selection.THETA.copy()
        try:
            return function(*args, **kwargs)
        finally:
            action_selection.THETA = saved

    return wrapped


@_restore_policy_theta_after_call
def run_all(seed: int = SEED) -> dict:
    policy = Policy()
    # Optional experiment toggles from environment.
    policy.enable_failure_injection = os.environ.get("FAILURE_INJECTION", "false").lower() == "true"
    policy.enable_mcp_reliability = os.environ.get("MCP_RELIABILITY", "false").lower() == "true"
    policy.enable_mcp_qos_routing = os.environ.get("MCP_QOS_ROUTING", "false").lower() == "true"
    policy.enable_pirag_counterfactual_eval = os.environ.get("PIR_COUNTERFACTUAL", "false").lower() == "true"
    policy.enable_physics_consistency_gate = os.environ.get("PHYSICS_CONSISTENCY_GATE", "false").lower() == "true"
    policy.enable_heterogeneous_profiles = os.environ.get("HETEROGENEOUS_PROFILES", "false").lower() == "true"
    policy.enable_research_metrics = os.environ.get("RESEARCH_METRICS", "false").lower() == "true"

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found: {DATA_CSV}")

    df_base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])

    results: dict[str, dict[str, dict]] = {}
    df_scenarios: dict[str, pd.DataFrame] = {}

    # --- Source 7: Policy weight perturbation (once per seed) ---
    import src.models.action_selection as _as_module
    _original_theta = _as_module.THETA.copy()
    _as_module.THETA = policy_theta_for_seed(_original_theta, seed)

    # Learning-trajectory cache for the locked adaptive modes. Keyed
    # by mode, each entry is a list of per-iteration diagnostics appended
    # in outer-loop order (scenario nested inside iteration).
    trajectory_cache: dict[str, list] = {}

    for scenario in SCENARIOS:
        results[scenario] = {}
        # State persists across the equal-budget iterations for this scenario
        # only. Resetting here prevents fixed scenario order from becoming an
        # unreported training curriculum.
        learner_state_cache: dict[str, dict] = {}
        # Build each episode's exogenous scenario frame once, outside the mode
        # loop.  This gives adaptation/evaluation episodes distinct stochastic
        # scenario draws while preserving exact common random numbers across
        # all paired policy arms.  The retained dataframe is episode 3.
        scenario_frames: dict[int, pd.DataFrame] = {}
        for episode_idx in range(4):
            scenario_seed = _stream_seed(
                seed, scenario, episode_idx, "scenario",
            )
            environment_seed = _stream_seed(
                seed, scenario, episode_idx, "environment",
            )
            scenario_frames[episode_idx] = apply_scenario(
                df_base,
                scenario,
                policy,
                np.random.default_rng(scenario_seed),
                stoch=make_stochastic_layer(
                    np.random.default_rng(environment_seed),
                    stream_seed=environment_seed,
                ),
            )
        df_scenarios[scenario] = scenario_frames[3]

        for mode in MODES:
            mode_capabilities = capabilities_for(mode)
            n_iter = mode_capabilities.episode_count

            episode = None
            # Learned arms adapt on episodes 0..2 and execute episode 3 with
            # every update path frozen. Fixed arms execute only the retained
            # episode 3, using the same environment/policy stream initialization
            # as every learned arm's reported final episode.
            episode_indices = range(4) if n_iter == 4 else (3,)
            for iter_idx in episode_indices:
                learning_enabled = bool(
                    mode_capabilities.learned
                    and iter_idx < mode_capabilities.adaptation_episode_count
                )
                environment_seed = _stream_seed(
                    seed, scenario, iter_idx, "environment",
                )
                policy_seed = _stream_seed(seed, scenario, iter_idx, "policy")
                environment_id = _stream_id(
                    seed, scenario, iter_idx, "environment",
                )
                policy_id = _stream_id(seed, scenario, iter_idx, "policy")
                mode_rng = np.random.default_rng(policy_seed)
                stoch = make_stochastic_layer(
                    np.random.default_rng(environment_seed),
                    stream_seed=environment_seed,
                )
                episode = run_episode(
                    scenario_frames[iter_idx], mode, policy, mode_rng, scenario,
                    stoch=stoch, seed=seed, benchmark_seed=seed,
                    episode_index=iter_idx,
                    environment_stream_id=environment_id,
                    policy_stream_id=policy_id,
                    stochastic_stream_id=environment_id,
                    learner_state_cache=learner_state_cache,
                    learning_enabled=learning_enabled,
                )
                if mode_capabilities.learned and iter_idx == 3:
                    freeze_summary = episode.get("learner_freeze_summary", {})
                    if not freeze_summary.get("learners_frozen", False):
                        raise RuntimeError(
                            f"retained evaluation learner freeze failed for "
                            f"{scenario}/{mode}/seed={seed}"
                        )
                # Record every iteration into the learning-trajectory cache.
                # The final frozen iteration is what goes into
                # results[scenario][mode].
                # NOTE: theta_change_norm, max_entry_change, and sign_preserved
                # live in episode["learner_summary"] (ContextMatrixLearner
                # summary), not episode["context_summary"] (coordinator's
                # per-step log). Reading the wrong dict was a silent bug in
                # the previous version: the trajectory file ended up with
                # theta_change_norm=0.0 across every iteration even though
                # the learner was actually moving.
                if n_iter > 1:
                    lrn_summary = episode.get("learner_summary", {}) or {}
                    trajectory_cache.setdefault(mode, []).append({
                        "scenario": scenario,
                        "iter": iter_idx,
                        "episode_phase": episode["episode_phase"],
                        "learning_enabled": episode["learning_enabled"],
                        "ari": episode["ari"],
                        "waste": episode["waste"],
                        "rle": episode["rle"],
                        "slca": episode["slca"],
                        "context_active_steps": episode.get("context_active_steps", 0),
                        "context_honored_steps": episode.get("context_honored_steps", 0),
                        "context_honor_rate": episode.get("context_honor_rate", 0.0),
                        "theta_change_norm": float(lrn_summary.get("theta_change_norm", 0.0)),
                        "max_entry_change": float(lrn_summary.get("max_entry_change", 0.0)),
                        "sign_preserved": bool(lrn_summary.get("sign_preserved", True)),
                    })
            assert episode is not None

            results[scenario][mode] = episode
            tag = f" ({n_iter}x)" if n_iter > 1 else ""
            print(f"  [{scenario:>20s}] [{mode:>17s}]{tag} ARI={episode['ari']:.3f}  "
                  f"waste={episode['waste']:.3f}  "
                  f"RLE={episode['rle']:.3f}  "
                  f"social_proxy={episode['slca']:.3f}  "
                  f"emissions_indicator={episode['carbon']:.0f}  "
                  f"temporal_proxy_stability={episode['equity']:.3f}  "
                  f"lat_ms={episode['mean_decision_latency_ms']:.2f}  "
                  f"cvr={episode['constraint_violation_rate']:.3f}")
            if mode in _CONTEXT_ENABLED_MODES and "context_summary" in episode:
                ctx = episode["context_summary"]
                evl = episode.get("evaluator_summary", {})
                lrn = episode.get("learner_summary", {})
                print(f"    Context: {ctx.get('total_mcp_tool_calls', 0)} MCP calls, "
                      f"{ctx.get('total_pirag_queries', 0)} piR queries, "
                      f"modifier nonzero {ctx.get('nonzero_modifier_steps', 0)}/{ctx.get('total_context_steps', 0)} steps, "
                      f"guard failures {ctx.get('guard_failures', 0)}, "
                      f"probability-gap overrides {ctx.get('governance_overrides', 0)}")
                if evl:
                    print(f"    Evaluator: action changed {evl.get('context_changed_action_count', 0)}/{evl.get('total_steps', 0)} steps")
                if mode == "agribrain" and lrn.get("final_theta"):
                    print(f"    Learned THETA_CONTEXT (change norm={lrn['theta_change_norm']:.4f}):")
                    for i, row_name in enumerate(["ColdChain", "Redist  ", "Recovery"]):
                        final = lrn["final_theta"][i]
                        initial = lrn["initial_theta"][i]
                        delta = [
                            f - ini for ini, f in zip(
                                initial, final, strict=True,
                            )
                        ]
                        print(f"      {row_name}: [{', '.join(f'{v:+.3f}' for v in final)}] "
                              f"(delta=[{', '.join(f'{d:+.3f}' for d in delta)}])")
                    print(f"    Social-proxy amp: {lrn['initial_slca_amp']:.3f} -> {lrn['final_slca_amp']:.3f}  "
                          f"Signs preserved: {lrn['sign_preserved']}")

            # Export traces for agribrain mode
            if mode == "agribrain" and "_trace_exporter" in episode:
                exporter = episode["_trace_exporter"]
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                trace_path = RESULTS_DIR / f"traces_{scenario}.json"
                exporter.export_json(str(trace_path))

                role_table = exporter.export_role_comparison_table()
                if role_table:
                    print("    Role context comparison:")
                    for row in role_table:
                        kw_str = ", ".join(row.get("top_keywords", [])[:3]) or "none"
                        print(f"      {row['role']:12s}: MCP={row['mcp_tools']}, "
                              f"KB={row['primary_kb_document']}, "
                              f"guidance={row['primary_guidance_type']}, "
                              f"keywords=[{kw_str}]")

                chains = exporter.export_provenance_chains()
                print(f"    Provenance: {len(chains)} local Merkle commitment records")

                if exporter._traces:
                    sample = exporter._traces[0]
                    if sample.explanation_summary:
                        print(f"    Sample explanation (hour {sample.hour}, {sample.role}):")
                        print(f"      {sample.explanation_summary[:120]}")

                # Save in-process project JSON-RPC/MCP-style dispatcher traces.
                # The legacy mcp_interop filename is retained for compatibility;
                # these records are not official MCP-conformance evidence.
                interop = exporter.export_interoperability_trace()
                if interop:
                    interop_path = RESULTS_DIR / f"mcp_interop_{scenario}.json"
                    with open(interop_path, "w") as f:
                        json.dump(interop, f, indent=2, default=str)

            # Export in-process project JSON-RPC/MCP-style dispatcher recordings.
            if mode == "agribrain" and "_protocol_recorder" in episode:
                proto = episode["_protocol_recorder"]
                proto_summary = proto.summary()
                if proto_summary.get("dropped_interactions", 0):
                    raise RuntimeError(
                        "Publication protocol trace is incomplete: "
                        f"{proto_summary['dropped_interactions']} interaction(s) "
                        "exceeded the recorder capacity"
                    )
                if proto_summary["total_interactions"] > 0:
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    proto_path = RESULTS_DIR / f"mcp_protocol_{scenario}.json"
                    proto.export_json(str(proto_path))
                    print(f"    Protocol: {proto_summary['total_interactions']} project MCP-style dispatcher interactions, "
                          f"methods={proto_summary['methods']}")

            # Export per-scenario context-alignment summary.
            #
            # Two consumer classes:
            #   1. The headline ``context_alignment_{scenario}.json``
            #      (agribrain only) is a fallback when
            #      ``benchmark_summary.json`` is missing. The production
            #      context-influence panel reads from
            #      ``benchmark_summary.json``, so the file is a
            #      fallback / single-seed-render evidence artifact.
            #   2. The per-mode ``context_alignment_{scenario}_{mode}.json``
            #      files cover the locked context-enabled comparators and
            #      secondary ablations. They let a reviewer verify per-mode
            #      honor/influence rates against benchmark-summary aggregates.
            #      The artifact manifest makes missing or altered files fail
            #      verification.
            if mode in _CONTEXT_ENABLED_MODES and mode != "no_context":
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                if mode == "agribrain":
                    alignment_path = RESULTS_DIR / f"context_alignment_{scenario}.json"
                else:
                    alignment_path = (
                        RESULTS_DIR / f"context_alignment_{scenario}_{mode}.json"
                    )
                with open(alignment_path, "w") as f:
                    json.dump({
                        "scenario": scenario,
                        "mode": mode,
                        "context_active_steps": episode["context_active_steps"],
                        "context_active_fraction": episode["context_active_fraction"],
                        # 2026-05 apples-to-apples cross-mode dispatch
                        # counter (n_steps for any context-enabled mode;
                        # zero for static / hybrid_rl / no_context).
                        # Carried through here so the supp-methods
                        # alignment table has the same denominator the
                        # benchmark JSON uses.
                        "context_dispatch_attempt_steps":
                            episode.get("context_dispatch_attempt_steps", 0),
                        "context_dispatch_attempt_fraction":
                            episode.get("context_dispatch_attempt_fraction", 0.0),
                        "context_honored_steps": episode["context_honored_steps"],
                        "context_honor_rate": episode["context_honor_rate"],
                        # 2026-05: paper headline switched from honor_rate to
                        # influence_rate (fig 9 panel c). Both rates emit on
                        # the same active-step denominator so a downstream
                        # reader can quote either off the same JSON.
                        "context_influenced_steps":
                            episode.get("context_influenced_steps", 0),
                        "context_influence_rate":
                            episode.get("context_influence_rate", 0.0),
                        # Apples-to-apples cross-mode influence rate
                        # (numerator unchanged, denominator switched from
                        # context_active_steps to
                        # context_dispatch_attempt_steps). Use this when
                        # comparing influence across modes whose
                        # activation regimes differ (heatwave + baseline).
                        "context_dispatch_influence_rate":
                            episode.get("context_dispatch_influence_rate", 0.0),
                        "context_active_per_recommendation":
                            episode["context_active_per_recommendation"],
                        "context_ignored_per_recommendation":
                            episode["context_ignored_per_recommendation"],
                        "signal_threshold": 0.10,
                        "honor_rate_by_threshold":
                            episode["context_threshold_counters"],
                        "null_baseline_random_honor_rate": 1.0 / len(ACTIONS),
                        "actions": list(ACTIONS),
                    }, f, indent=2)
                if mode == "agribrain":
                    print(f"    Context alignment: {episode['context_honored_steps']}/"
                          f"{episode['context_active_steps']} honored "
                          f"({100.0 * episode['context_honor_rate']:.1f}%)")

    # Restore original THETA after all episodes (Source 7 cleanup)
    _as_module.THETA = _original_theta

    # Primary comparison table. Secondary one-factor and structural-sensitivity
    # diagnostics are deliberately excluded.
    table1_methods = PRIMARY_MODES
    table1_rows = []
    for scenario in SCENARIOS:
        for method in table1_methods:
            ep = results[scenario][method]
            table1_rows.append({
                "Scenario": scenario, "Method": method,
                "ARI": round(ep["ari"], 3),
                # Single canonical RLE: EU-hierarchy + severity-weighted
                # form per resilience.py. The binary, match-quality, and
                # capacity-constrained variants were retired in 2026-04.
                "RLE": round(ep["rle"], 3),
                "Waste": round(ep["waste"], 3), "SLCA": round(ep["slca"], 3),
                "Carbon": round(ep["carbon"], 0), "Equity": round(ep["equity"], 3),
                "DecisionLatencyMs": round(ep["mean_decision_latency_ms"], 3),
                "ConstraintViolationRate": round(ep["constraint_violation_rate"], 4),
                "OperatingEnvelopeViolationRate": round(
                    ep["operating_envelope_violation_rate"], 4
                ),
                # Outcome-side disposition: of the env-driven violation
                # events, what did the policy do with the at-risk batch?
                # Conditional disposition on the common violation-event set.
                "DownstreamViolationRate": round(ep.get("downstream_violation_rate", 0.0), 4),
                "ContainedViolationRate": round(ep.get("contained_violation_rate", 0.0), 4),
            })
    table1 = pd.DataFrame(table1_rows)

    # Compact architectural ablation table used in the paper.  Channel-specific
    # modes remain in Table 1/H1-H2; prior-sensitivity modes have their own
    # supplementary artifacts.
    table2_modes = [
        "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
        "agribrain",
    ]
    table2_rows = []
    for scenario in SCENARIOS:
        for mode in table2_modes:
            ep = results[scenario][mode]
            table2_rows.append({
                "Scenario": scenario, "Variant": mode,
                "ARI": round(ep["ari"], 3),
                "RLE": round(ep["rle"], 3),
                "Waste": round(ep["waste"], 3), "SLCA": round(ep["slca"], 3),
                "Carbon": round(ep["carbon"], 0), "Equity": round(ep["equity"], 3),
                "DecisionLatencyMs": round(ep["mean_decision_latency_ms"], 3),
                "ConstraintViolationRate": round(ep["constraint_violation_rate"], 4),
                "DownstreamViolationRate": round(ep.get("downstream_violation_rate", 0.0), 4),
                "ContainedViolationRate": round(ep.get("contained_violation_rate", 0.0), 4),
            })
    table2 = pd.DataFrame(table2_rows)

    # Persist learning-trajectory data for the locked adaptive modes. Each
    # entry records iteration index, scenario, and key metrics for auditing
    # the three adaptation episodes and the retained frozen evaluation.
    for mode_name, traj in trajectory_cache.items():
        if not traj:
            continue
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        trajectory_path = RESULTS_DIR / f"learning_trajectory_{mode_name}.json"
        with open(trajectory_path, "w") as f:
            json.dump({
                "mode": mode_name,
                "seed": int(seed),
                "n_iterations_per_scenario": _MULTI_EPISODE_MODES.get(mode_name, 1),
                "trajectory": traj,
            }, f, indent=2)

    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "benchmark_seed": int(seed),
        "state_design": (
            "routing uses *_policy_observed; scored endpoints use "
            "*_outcome_environmental"
        ),
        "forecast_protocol": {
            "demand_method": FORECAST_METHOD,
            "supply_proxy_method": SUPPLY_FORECAST_METHOD,
            "selection": "minimum validation-segment rolling-origin RMSE",
            "test_used_for_selection": False,
        },
        "results": results,
        "table1": table1,
        "table2": table2,
        "df_scenarios": df_scenarios,
    }


def save_tables(table1: pd.DataFrame, table2: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t1_path = RESULTS_DIR / "table1_summary.csv"
    t2_path = RESULTS_DIR / "table2_ablation.csv"
    table1.to_csv(t1_path, index=False)
    table2.to_csv(t2_path, index=False)
    print(f"Saved {t1_path}")
    print(f"Saved {t2_path}")


def configure_standalone_development_output() -> Path:
    """Select an isolated output directory for direct CLI execution.

    ``generate_results.py`` is a one-seed development smoke, not the locked
    20-seed publication workflow.  The publication workers import this module
    and supply their own run-scoped paths; only the ``__main__`` path calls
    this helper.  An explicit path equal to the canonical ``results`` folder
    is rejected so a local smoke cannot masquerade as current evidence.
    """
    global RESULTS_DIR
    canonical = (Path(__file__).resolve().parent / "results").resolve()
    requested = os.environ.get("AGRIBRAIN_DEVELOPMENT_OUTPUT_DIR", "").strip()
    if requested:
        target = Path(requested).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        target = (
            Path(__file__).resolve().parent
            / "development_results"
            / f"cli_seed_{SEED}_{stamp}"
        ).resolve()
    if target == canonical:
        raise RuntimeError(
            "BLOCK: standalone generate_results.py is development-only and "
            "cannot write to mvp/simulation/results"
        )
    RESULTS_DIR = target
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    return RESULTS_DIR


def get_summary_json(run_data: dict | None = None) -> dict:
    if run_data is None:
        run_data = run_all()
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for mode in MODES:
            ep = run_data["results"][scenario][mode]
            summary[scenario][mode] = {
                "ari": round(ep["ari"], 4),
                # Single canonical RLE (EU-hierarchy + severity-weighted).
                "rle": round(ep["rle"], 4),
                "waste": round(ep["waste"], 4), "slca": round(ep["slca"], 4),
                "carbon": round(ep["carbon"], 2), "equity": round(ep["equity"], 4),
                "decision_latency_ms": round(ep["mean_decision_latency_ms"], 4),
                "constraint_violation_rate": round(ep["constraint_violation_rate"], 6),
            }
    return summary


if __name__ == "__main__":
    development_output = configure_standalone_development_output()
    print("=" * 70)
    print("AGRI-BRAIN development-only simulation")
    print("=" * 70)
    print("Publication evidence: NO")
    print(f"Isolated output: {development_output}")
    print(f"Seed: {SEED}")
    print(f"Deterministic mode: {_is_deterministic()}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Modes: {MODES}")
    print()

    data = run_all()
    save_tables(data["table1"], data["table2"])
    (RESULTS_DIR / "development_run_metadata.json").write_text(
        json.dumps({
            "evidence_status": "development_only",
            "publication_evidence": False,
            "seed": int(SEED),
            "scenario_count": len(SCENARIOS),
            "mode_count": len(MODES),
            "warning": (
                "This one-seed local output is not a publication benchmark. "
                "Use the clean-commit hpc/hpc_run.sh workflow."
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Table 1 — Summary (Scenario × Method)")
    print("=" * 70)
    print(data["table1"].to_string(index=False))

    print()
    print("=" * 70)
    print("Table 2 — Ablation (Scenario × Variant)")
    print("=" * 70)
    print(data["table2"].to_string(index=False))

    # Print context summaries for agribrain mode
    print()
    print("=" * 70)
    print("Context Integration Summary (agribrain mode)")
    print("=" * 70)
    for scenario in SCENARIOS:
        ep = data["results"][scenario].get("agribrain", {})
        ctx = ep.get("context_summary", {})
        lrn = ep.get("learner_summary", {})
        evl = ep.get("evaluator_summary", {})
        if ctx:
            print(f"\n  [{scenario}]")
            print(f"    MCP tool calls: {ctx.get('total_mcp_tool_calls', 0)}")
            print(f"    Mean modifier magnitude: {ctx.get('mean_modifier_magnitude', 0):.4f}")
            print(f"    Guard failures: {ctx.get('guard_failures', 0)}")
            print(f"    Probability-gap overrides: {ctx.get('governance_overrides', 0)}")
            if lrn:
                print(f"    Learner updates: {lrn.get('n_updates', 0)}")
                print(f"    Mean advantage: {lrn.get('mean_advantage', 0):.4f}")
            if evl:
                print(f"    Context change rate: {evl.get('context_change_rate', 0):.3f}")

    print()
    print("Done. Development outputs saved to", RESULTS_DIR)
