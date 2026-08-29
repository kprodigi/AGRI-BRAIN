"""Adversarial checks for the executable publication outcome contract."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "agribrain" / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import src.models.action_selection as action_selection  # noqa: E402
from src.chain.decision_ledger import merkle_root_hex  # noqa: E402
from src.models.outcome_equation_contract import (  # noqa: E402
    reconstruct_step_outcomes,
)

from hpc.validate_decision_ledgers import validate_ledger  # noqa: E402
from mvp.simulation.generate_results import (  # noqa: E402
    DATA_CSV,
    Policy,
    _canonical_sha256,
    _stream_id,
    _stream_seed,
    apply_scenario,
    decision_ledger_scope,
    policy_theta_for_seed,
    run_episode,
)
from mvp.simulation.stochastic import make_stochastic_layer  # noqa: E402


def _canonical_leaf(record: dict) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def _rewrite_rehashed(path: Path, header: dict, records: list[dict]) -> None:
    leaves = [_canonical_leaf(record) for record in records]
    header["merkle_root"] = merkle_root_hex(leaves)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, sort_keys=True, default=str) + "\n")
        for record, leaf in zip(records, leaves, strict=True):
            handle.write(json.dumps(
                {**record, "_leaf": leaf}, sort_keys=True, default=str,
            ) + "\n")


@pytest.fixture(scope="module")
def valid_ledger(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("outcome_ledger")
    seed = 17
    policy = Policy()
    base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    environment_seed = _stream_seed(seed, "baseline", 3, "environment")
    environment_layer = make_stochastic_layer(
        np.random.default_rng(environment_seed), stream_seed=environment_seed,
    )
    frame = apply_scenario(
        base,
        "baseline",
        policy,
        np.random.default_rng(_stream_seed(seed, "baseline", 3, "scenario")),
        stoch=environment_layer,
    )
    original_theta = np.asarray(action_selection.THETA, dtype=float).copy()
    action_selection.THETA = policy_theta_for_seed(
        np.asarray(action_selection.DECLARED_THETA, dtype=float), seed,
    )
    try:
        with decision_ledger_scope(output, reset=True):
            run_episode(
                frame,
                "static",
                policy,
                np.random.default_rng(_stream_seed(seed, "baseline", 3, "policy")),
                scenario="baseline",
                stoch=environment_layer,
                seed=seed,
                benchmark_seed=seed,
                episode_index=3,
                environment_stream_id=_stream_id(
                    seed, "baseline", 3, "environment",
                ),
                policy_stream_id=_stream_id(seed, "baseline", 3, "policy"),
                stochastic_stream_id=_stream_id(
                    seed, "baseline", 3, "environment",
                ),
                learning_enabled=False,
            )
    finally:
        action_selection.THETA = original_theta
    path = output / "static__baseline.jsonl"
    validate_ledger(
        path, mode="static", scenario="baseline", benchmark_seed=seed,
    )
    return path


def _load_ledger(path: Path) -> tuple[dict, list[dict]]:
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header = payloads[0]
    records = []
    for stored in payloads[1:]:
        record = dict(stored)
        record.pop("_leaf")
        records.append(record)
    return header, records


@pytest.mark.parametrize("scenario", ["heatwave", "adaptive_pricing"])
def test_source_replay_accepts_locked_nonbaseline_scenario_streams(
    tmp_path: Path, scenario: str,
) -> None:
    seed = 17
    policy = Policy()
    base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    environment_seed = _stream_seed(seed, scenario, 3, "environment")
    layer = make_stochastic_layer(
        np.random.default_rng(environment_seed), stream_seed=environment_seed,
    )
    frame = apply_scenario(
        base,
        scenario,
        policy,
        np.random.default_rng(_stream_seed(seed, scenario, 3, "scenario")),
        stoch=layer,
    )
    original_theta = np.asarray(action_selection.THETA, dtype=float).copy()
    action_selection.THETA = policy_theta_for_seed(
        np.asarray(action_selection.DECLARED_THETA, dtype=float), seed,
    )
    try:
        with decision_ledger_scope(tmp_path, reset=True):
            run_episode(
                frame,
                "static",
                policy,
                np.random.default_rng(_stream_seed(seed, scenario, 3, "policy")),
                scenario=scenario,
                stoch=layer,
                seed=seed,
                benchmark_seed=seed,
                episode_index=3,
                environment_stream_id=_stream_id(
                    seed, scenario, 3, "environment",
                ),
                policy_stream_id=_stream_id(seed, scenario, 3, "policy"),
                stochastic_stream_id=_stream_id(
                    seed, scenario, 3, "environment",
                ),
                learning_enabled=False,
            )
    finally:
        action_selection.THETA = original_theta
    summary = validate_ledger(
        tmp_path / f"static__{scenario}.jsonl",
        mode="static",
        scenario=scenario,
        benchmark_seed=seed,
    )
    assert summary["source_replay_hashes"][
        "latent_environment_sha256"
    ] == summary["latent_environment_sha256"]


def test_source_replay_applies_h3_dose_after_canonical_observation_stream(
    tmp_path: Path,
) -> None:
    from mvp.simulation.benchmarks import run_stress_suite as stress

    seed = 17
    scenario = "baseline"
    stressor = "compounded"
    policy = Policy()
    base = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    environment_seed = _stream_seed(seed, scenario, 3, "environment")
    layer = make_stochastic_layer(
        np.random.default_rng(environment_seed), stream_seed=environment_seed,
    )
    frame = apply_scenario(
        base,
        scenario,
        policy,
        np.random.default_rng(_stream_seed(seed, scenario, 3, "scenario")),
        stoch=layer,
    )
    stress_seed = int.from_bytes(hashlib.sha256(
        f"stress|{scenario}|{stressor}|{seed}|3".encode("utf-8")
    ).digest()[:8], "big")
    stressed = stress._perturb_df(
        frame, stressor, np.random.default_rng(stress_seed),
    )
    original_theta = np.asarray(action_selection.THETA, dtype=float).copy()
    action_selection.THETA = policy_theta_for_seed(
        np.asarray(action_selection.DECLARED_THETA, dtype=float), seed,
    )
    try:
        with decision_ledger_scope(tmp_path, reset=True):
            run_episode(
                stressed,
                "static",
                policy,
                np.random.default_rng(_stream_seed(seed, scenario, 3, "policy")),
                scenario=scenario,
                stoch=layer,
                seed=seed,
                benchmark_seed=seed,
                episode_index=3,
                environment_stream_id=_stream_id(
                    seed, scenario, 3, "environment",
                ),
                policy_stream_id=_stream_id(seed, scenario, 3, "policy"),
                stochastic_stream_id=_stream_id(
                    seed, scenario, 3, "environment",
                ),
                learning_enabled=False,
            )
    finally:
        action_selection.THETA = original_theta
    summary = validate_ledger(
        tmp_path / f"static__{scenario}.jsonl",
        mode="static",
        scenario=scenario,
        benchmark_seed=seed,
    )
    assert summary["source_replay_hashes"][
        "observed_policy_input_sha256"
    ] is not None


def test_rehashed_carbon_tamper_is_rejected_by_equation_reconstruction(
    valid_ledger: Path, tmp_path: Path,
) -> None:
    header, records = _load_ledger(valid_ledger)
    records[0]["carbon_kg"] = float(records[0]["carbon_kg"]) + 0.25
    tampered = tmp_path / "static__baseline.jsonl"
    _rewrite_rehashed(tampered, header, records)

    with pytest.raises(RuntimeError, match="carbon_kg violates the outcome equation"):
        validate_ledger(
            tampered, mode="static", scenario="baseline", benchmark_seed=17,
        )


def test_self_consistent_rehashed_parameter_substitution_is_rejected(
    valid_ledger: Path, tmp_path: Path,
) -> None:
    header, records = _load_ledger(valid_ledger)
    expected_contract = copy.deepcopy(header["metadata"]["outcome_equation_contract"])
    substituted = header["metadata"]["outcome_equation_contract"]
    substituted["carbon"]["route_km_by_action"]["cold_chain"] += 10.0
    for record in records:
        rebuilt = reconstruct_step_outcomes(record, substituted)
        for field in ("waste", "carbon_kg", "slca", "reward", "ari"):
            record[field] = rebuilt[field]
    tampered = tmp_path / "static__baseline.jsonl"
    _rewrite_rehashed(tampered, header, records)

    with pytest.raises(RuntimeError, match="route_km_by_action/cold_chain"):
        validate_ledger(
            tampered,
            mode="static",
            scenario="baseline",
            benchmark_seed=17,
            expected_outcome_equation_contract=expected_contract,
        )


def test_coherent_rehashed_environment_forgery_is_rejected_by_source_replay(
    valid_ledger: Path, tmp_path: Path,
) -> None:
    """Self-consistent hashes/equations cannot substitute another environment."""

    header, records = _load_ledger(valid_ledger)
    contract = header["metadata"]["outcome_equation_contract"]
    for record in records:
        record["transport_multiplier_outcome_environmental"] = (
            float(record["transport_multiplier_outcome_environmental"]) * 1.125
        )
        rebuilt = reconstruct_step_outcomes(record, contract)
        for field in ("waste", "carbon_kg", "slca", "reward", "ari"):
            record[field] = rebuilt[field]

    metadata = header["metadata"]
    metadata["latent_environment_sha256"] = _canonical_sha256({
        "hours": [float(record["hour"]) for record in records],
        "temp_outcome_environmental": [
            float(record["temp_outcome_environmental"]) for record in records
        ],
        "rh_outcome_environmental": [
            float(record["rh_outcome_environmental"]) for record in records
        ],
        "rho_outcome_environmental": [
            float(record["rho_outcome_environmental"]) for record in records
        ],
        "inventory_outcome_environmental": [
            float(record["inventory_outcome_environmental"])
            for record in records
        ],
        "demand_outcome_environmental": [
            float(record["demand_outcome_environmental"]) for record in records
        ],
        "transport_multiplier_outcome_environmental": [
            float(record["transport_multiplier_outcome_environmental"])
            for record in records
        ],
        "effective_k_ref": float(metadata["effective_k_ref"]),
        "effective_Ea_R": float(metadata["effective_Ea_R"]),
        "scenario_onset_offset_hours": float(
            metadata["scenario_onset_offset_hours"]
        ),
    })
    tampered = tmp_path / "static__baseline__coherent_forgery.jsonl"
    _rewrite_rehashed(tampered, header, records)

    with pytest.raises(
        RuntimeError,
        match=(
            "source-bound replay mismatch for "
            "transport_multiplier_outcome_environmental"
        ),
    ):
        validate_ledger(
            tampered,
            mode="static",
            scenario="baseline",
            benchmark_seed=17,
        )
