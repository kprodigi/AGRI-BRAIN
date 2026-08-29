#!/usr/bin/env python3
"""Fail closed when frozen synthetic PINN evidence drifts before HPC."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(repo_root: Path) -> list[str]:
    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)

    protocol_path = repo_root / "mvp/simulation/experiment_protocol.json"
    artifact_dir = repo_root / "mvp/simulation/pinn/artifacts"
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        spoilage = protocol["spoilage_model"]
        declaration = spoilage["training_evidence"]
    except Exception as exc:
        return [f"cannot load PINN protocol declaration: {exc}"]

    if spoilage.get("neural_residual_enabled") is not True:
        fail("locked protocol does not enable the frozen residual")
    if spoilage.get("no_pinn_publication_arm") is not True:
        fail("locked protocol does not declare the no_pinn arm")
    if spoilage.get("deployed_quality_transform") != (
        "clip C_mech + delta_C to [0,1], then cumulative minimum within trajectory"
    ):
        fail("locked protocol changes the deployed quality transform")
    if spoilage.get("raw_network_alone_claimed_physically_valid") is not False:
        fail("locked protocol incorrectly treats the raw network as physically valid")
    outcome_reference = spoilage.get("outcome_reference")
    if not isinstance(outcome_reference, dict):
        fail("locked protocol lacks the independent synthetic-DGP outcome reference")
        outcome_reference = {}
    expected_outcome_reference = {
        "kind": "independent_synthetic_dgp_v1",
        "state_equation": "dC_true/dt = -k_base*alpha*exp(u)*C_true",
        "stress_multiplier_equation": (
            "u = 0.44*(packaging_index-0.50)+0.80*handling_shock_G+"
            "0.0040*abs_dRH_dt"
        ),
        "packaging_index_default": 0.5,
        "observation_noise_applied_to_scored_outcome": False,
        "mode_invariant_within_paired_episode": True,
        "paired_outcome_hash_required": True,
        "synthetic_only": True,
        "external_validation": False,
        "empirical_claims_permitted": False,
    }
    for key, expected in expected_outcome_reference.items():
        if outcome_reference.get(key) != expected:
            fail(f"locked protocol changes outcome-reference field {key}")
    evaluation_input_path = repo_root / str(
        outcome_reference.get("evaluation_sensor_input_path", "")
    )
    if not evaluation_input_path.is_file():
        fail("independent DGP evaluation sensor input is missing")
    elif sha256(evaluation_input_path) != outcome_reference.get(
        "evaluation_sensor_input_sha256"
    ):
        fail("independent DGP evaluation sensor input SHA-256 mismatch")
    if outcome_reference.get("coefficient_source") != (
        "mvp/simulation/pinn/generate_synthetic_spoilage_data.py"
    ):
        fail("independent DGP coefficient source changed")
    no_pinn_difference = str(
        spoilage.get("no_pinn_difference_from_agribrain", "")
    )
    if "policy-observed spoilage estimate" not in no_pinn_difference or (
        "independent synthetic-DGP scored outcome" not in no_pinn_difference
    ):
        fail("locked protocol does not declare a policy-only no-PINN ablation")
    if declaration.get("synthetic_only") is not True:
        fail("protocol does not label PINN targets synthetic-only")
    if declaration.get("external_validation") is not False:
        fail("protocol incorrectly permits external PINN validation")
    if declaration.get("empirical_claims_permitted") is not False:
        fail("protocol incorrectly permits empirical PINN claims")

    dataset_manifest_path = repo_root / declaration.get("dataset_manifest_path", "")
    checkpoint_manifest_path = repo_root / declaration.get("checkpoint_manifest_path", "")
    try:
        dataset_manifest = json.loads(
            dataset_manifest_path.read_text(encoding="utf-8")
        )
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return failures + [f"cannot load PINN evidence manifests: {exc}"]

    for label, manifest in (
        ("dataset", dataset_manifest),
        ("checkpoint", checkpoint_manifest),
    ):
        if manifest.get("target_origin") != "independent_synthetic_dgp":
            fail(f"{label} manifest target origin changed")
        if manifest.get("synthetic_only") is not True:
            fail(f"{label} manifest lacks synthetic-only label")
        if manifest.get("external_validation") is not False:
            fail(f"{label} manifest claims external validation")
        if manifest.get("empirical_claims_permitted") is not False:
            fail(f"{label} manifest permits empirical claims")

    dataset_path = artifact_dir / str(dataset_manifest.get("dataset_file", ""))
    checkpoint_path = artifact_dir / str(checkpoint_manifest.get("checkpoint_file", ""))
    history_path = artifact_dir / str(checkpoint_manifest.get("training_history_file", ""))
    for label, path, expected in (
        ("dataset", dataset_path, dataset_manifest.get("dataset_sha256")),
        ("checkpoint", checkpoint_path, checkpoint_manifest.get("checkpoint_sha256")),
        ("training history", history_path, checkpoint_manifest.get("training_history_sha256")),
    ):
        if not path.is_file():
            fail(f"{label} artifact is missing: {path}")
        elif sha256(path) != expected:
            fail(f"{label} artifact SHA-256 mismatch")

    if dataset_manifest.get("dataset_sha256") != declaration.get("dataset_sha256"):
        fail("dataset SHA-256 disagrees with locked protocol")
    if checkpoint_manifest.get("checkpoint_sha256") != declaration.get(
        "checkpoint_sha256"
    ):
        fail("checkpoint SHA-256 disagrees with locked protocol")
    if checkpoint_manifest.get("training_history_sha256") != declaration.get(
        "training_history_sha256"
    ):
        fail("training-history SHA-256 disagrees with locked protocol")
    if checkpoint_manifest.get("dataset_sha256") != dataset_manifest.get(
        "dataset_sha256"
    ):
        fail("checkpoint is bound to the wrong training dataset")

    splits = dataset_manifest.get("trajectory_splits", {})
    split_sets = {
        name: set(map(str, splits.get(name, [])))
        for name in ("train", "validation", "test")
    }
    if tuple(map(len, split_sets.values())) != (24, 6, 6):
        fail("trajectory split counts are not 24/6/6")
    if any(
        split_sets[left].intersection(split_sets[right])
        for left, right in (
            ("train", "validation"), ("train", "test"),
            ("validation", "test"),
        )
    ):
        fail("PINN trajectory splits overlap")

    if dataset_path.is_file():
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "trajectory_id", "split", "observed_quality_fraction",
                "latent_quality_fraction", "mechanistic_quality_fraction",
                "latent_correction_fraction",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                fail("PINN dataset schema is incomplete")
            row_count = 0
            seen: dict[str, str] = {}
            correction_abs_max = 0.0
            for row in reader:
                row_count += 1
                trajectory = str(row["trajectory_id"])
                split = str(row["split"])
                if trajectory in seen and seen[trajectory] != split:
                    fail(f"trajectory {trajectory} crosses data splits")
                seen[trajectory] = split
                try:
                    quality = float(row["observed_quality_fraction"])
                    correction_abs_max = max(
                        correction_abs_max,
                        abs(float(row["latent_correction_fraction"])),
                    )
                except Exception:
                    fail("PINN dataset contains a nonnumeric target")
                    break
                if not 0.0 <= quality <= 1.0:
                    fail("PINN observed-quality target leaves [0,1]")
                    break
            if row_count != dataset_manifest.get("n_rows"):
                fail("PINN dataset row count changed")
            if correction_abs_max > float(dataset_manifest.get("correction_bound", -1)):
                fail("latent synthetic correction exceeds declared bound")

    metrics = checkpoint_manifest.get("metrics", {}).get("test", {})
    if not (
        isinstance(metrics.get("latent_rmse"), (int, float))
        and isinstance(metrics.get("mechanistic_latent_rmse"), (int, float))
        and metrics["latent_rmse"] < metrics["mechanistic_latent_rmse"]
    ):
        fail("held-out synthetic PINN improvement is absent")
    if metrics.get("deployed_unit_interval_violation_count") != 0:
        fail("deployed PINN has unit-interval violations")
    if metrics.get("deployed_monotonicity_violation_count") != 0:
        fail("deployed PINN has monotonicity violations")
    if float(metrics.get("residual_abs_max", 1.0)) > 0.08 + 1e-12:
        fail("held-out PINN residual exceeds +/-0.08")
    protocol_diagnostics = spoilage.get("test_diagnostics", {})
    diagnostic_bindings = {
        "raw_monotonicity_violation_count": "monotonicity_violation_count",
        "raw_unit_interval_violation_count": "unit_interval_violation_count",
        "deployed_monotonicity_violation_count": (
            "deployed_monotonicity_violation_count"
        ),
        "deployed_unit_interval_violation_count": (
            "deployed_unit_interval_violation_count"
        ),
        "raw_latent_rmse": "latent_rmse",
        "deployed_latent_rmse": "deployed_latent_rmse",
        "mechanistic_latent_rmse": "mechanistic_latent_rmse",
    }
    for protocol_key, manifest_key in diagnostic_bindings.items():
        if protocol_diagnostics.get(protocol_key) != metrics.get(manifest_key):
            fail(
                f"protocol diagnostic {protocol_key} disagrees with checkpoint manifest"
            )

    history = None
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"training history is invalid JSON: {exc}")
    if history is not None:
        if history.get("test_split_used_for_selection") is not False:
            fail("PINN test split was used for checkpoint selection")
        if history.get("training_seeds") != [104729, 130363, 155921]:
            fail("PINN initialization seeds changed")

    # Standard-library source check keeps the login-node gate usable before
    # backend installation. Runtime tests additionally compare dataclass fields.
    mode_source = (
        repo_root / "agribrain/backend/src/models/mode_capabilities.py"
    ).read_text(encoding="utf-8")
    if '"no_pinn": _caps(' not in mode_source or "spoilage_residual=False" not in mode_source:
        fail("no_pinn is not wired as a residual-only capability ablation")
    return failures


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(argv[0]).resolve() if argv else Path.cwd().resolve()
    failures = validate(repo_root)
    if failures:
        for failure in failures:
            print(f"BLOCK: {failure}")
        return 1
    print("Frozen synthetic PINN evidence contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
