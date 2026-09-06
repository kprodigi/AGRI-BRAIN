"""Regression tests for fresh-run integrity and retired repair pathways."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from hpc.core_submission_receipt import SNAPSHOT_MODE, build_receipt
from hpc.capture_failed_publisher_accounting import (
    capture_failed_publisher_accounting,
)
from hpc.preserved_raw_manifest import build_manifest as build_raw_manifest
from hpc.publication_recovery_receipt import (
    RECOVERY_REASON_CODE,
    create_recovery_receipt,
)
from mvp.simulation.validation import validate_publication_artifacts as semantic_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = _load(
    "aggregate_seeds_publication_repair",
    "mvp/simulation/benchmarks/aggregate_seeds.py",
)
manifest_builder = _load(
    "manifest_builder_publication_repair",
    "mvp/simulation/analysis/build_artifact_manifest.py",
)
archive_builder = _load(
    "archive_builder_publication_repair",
    "mvp/simulation/analysis/build_publication_archive.py",
)
manifest_verifier = _load(
    "manifest_verifier_single_provenance",
    "mvp/simulation/analysis/verify_manifest.py",
)
recovery_gate = _load(
    "recovery_gate_publication_repair",
    "mvp/simulation/analysis/recovery_provenance.py",
)

SIMULATION_COMMIT = "a" * 40
PUBLICATION_COMMIT = "b" * 40
SOURCE_TREE_SHA256 = "d" * 64
RUN_TAG = "aaaaaaa_20260829_105800"


def _record(mean: float, low: float, high: float) -> dict:
    return {
        "mean": mean,
        "std": 0.1,
        "ci_low": low,
        "ci_high": high,
        "ci_method": "BCa",
        "n_seeds": 20,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _recovery_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    """Create one fully valid canonical core-recovery provenance set."""

    results = tmp_path / "mvp" / "simulation" / "results"
    results.mkdir(parents=True)
    original = results / "core_submission_receipts" / f"{RUN_TAG}.json"
    _write_json(original, build_receipt(
        run_tag=RUN_TAG,
        source_commit=SIMULATION_COMMIT,
        partition="compute",
        seed_job_id="101",
        stress_job_id="102",
        publisher_job_id="103",
        source_snapshot_mode=SNAPSHOT_MODE,
        source_tree_sha256=SOURCE_TREE_SHA256,
    ))

    raw_root = tmp_path / "preserved_raw"
    raw_root.mkdir()
    (raw_root / "seed_42.json").write_bytes(b'{"complete":true}\n')
    raw_path = results / "preserved_raw_manifests" / f"{RUN_TAG}.json"
    _write_json(raw_path, build_raw_manifest(
        kind="core",
        run_tag=RUN_TAG,
        simulation_commit=SIMULATION_COMMIT,
        simulation_source_tree_sha256=SOURCE_TREE_SHA256,
        roots=[("simulation_workers", raw_root)],
        files=[],
    ))

    accounting = tmp_path / "failed_accounting.json"
    values = (
        "103", "103", "agribrain-publish", "FAILED", "1:0",
        "2026-08-29T11:00:00", "2026-08-29T11:00:01",
        "2026-08-29T11:00:02", "2026-08-29T11:00:03", "1",
        "node001", "compute", "cluster",
    )
    accounting_stdout = "|".join(values) + "|\n"

    def accounting_runner(command, **_kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, "slurm 24.05\n", "",
            )
        return subprocess.CompletedProcess(command, 0, accounting_stdout, "")

    _write_json(accounting, capture_failed_publisher_accounting(
        job_id="103", runner=accounting_runner,
    ))
    logs = tmp_path / "logs"
    logs.mkdir()
    stdout = logs / "publish_103.out"
    stderr = logs / "publish_103.err"
    stdout.write_bytes(b"publisher failed after simulation completion\n")
    stderr.write_bytes(b"ValueError: accounting validation defect\n")
    receipt = (
        results / "publication_recovery_receipts" / f"{RUN_TAG}.json"
    )
    monkeypatch.setattr(
        "hpc.publication_recovery_receipt._git_clean_identity",
        lambda _root: (PUBLICATION_COMMIT, "c" * 40),
    )
    create_recovery_receipt(
        output=receipt,
        repo_root=tmp_path,
        kind="core",
        run_tag=RUN_TAG,
        simulation_commit=SIMULATION_COMMIT,
        original_receipt_path=original,
        failed_accounting_record_path=accounting,
        failed_stdout_path=stdout,
        failed_stderr_path=stderr,
        raw_output_manifest_path=raw_path,
        held_recovery_publisher_job_id="301",
        reason_code=RECOVERY_REASON_CODE,
        expected_publication_commit=PUBLICATION_COMMIT,
    )
    authorization = recovery_gate.validate_recovery_context(
        receipt,
        results_dir=results,
        run_tag=RUN_TAG,
        simulation_commit=SIMULATION_COMMIT,
        publication_commit=PUBLICATION_COMMIT,
    )
    return results, receipt, authorization


def test_resampling_identity_pins_seed_algorithm_and_observation_order():
    identity = aggregate._resampling_identity([42, 1337, 2024])
    assert identity["generator"] == "numpy.random.Generator"
    assert identity["bit_generator"] == "PCG64"
    assert identity["seed_derivation"] == {
        "algorithm": "BLAKE2b",
        "digest_size_bytes": 4,
        "key_hex": "",
        "salt_hex": "",
        "personalization_hex": "",
        "payload_encoding": "UTF-8",
        "payload_template": "scope::cell_key[0]::cell_key[1]::...",
        "integer_conversion": "unsigned big-endian",
    }
    assert identity["observation_order"]["seeds"] == [42, 1337, 2024]
    assert identity["example"]["derived_seed"] == 732674068
    assert aggregate._cell_seed(
        "bootstrap_ci", ("heatwave", "agribrain", "ari")
    ) == 732674068


def test_publication_tables_use_canonical_constraint_record(tmp_path: Path):
    bucket = {
        key: _record(0.5, 0.4, 0.6)
        for key, _display in set(aggregate._TABLE1_COLUMNS) | set(aggregate._TABLE2_COLUMNS)
    }
    bucket["constraint_violation_rate"] = _record(0.5, 0.45, 0.55)
    # The legacy alias deliberately differs here.  The public table must ignore
    # it and remain a direct rounded projection of the canonical summary key.
    bucket["operational_violation_rate"] = _record(0.5, 0.40, 0.60)
    summary = {"heatwave": {"static": bucket}}

    aggregate._rewrite_stochastic_csvs(tmp_path, summary)

    for filename in ("table1_summary.csv", "table2_ablation.csv"):
        with (tmp_path / filename).open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        assert row["ConstraintViolationRate"] == "0.5000"
        assert row["ConstraintViolationRate_ci_low"] == "0.4500"
        assert row["ConstraintViolationRate_ci_high"] == "0.5500"


def test_manifest_primary_hash_is_literal_bytes(tmp_path: Path):
    payload = tmp_path / "table.csv"
    payload.write_bytes(b"a,b\r\n1,2\r\n")
    raw = hashlib.sha256(payload.read_bytes()).hexdigest()
    normalized = hashlib.sha256(payload.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert raw != normalized
    assert manifest_builder._sha256(payload) == raw


def _strict_manifest(tmp_path: Path) -> Path:
    artifact = tmp_path / "payload.json"
    artifact.write_bytes(b'{"fresh":true}\n')
    commit = "1" * 40
    manifest = {
        "schema_version": 2,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "dual_provenance": False,
        "git_dirty": False,
        "artifact_count": 1,
        "artifacts": [{
            "file": artifact.name,
            "bytes": len(artifact.read_bytes()),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }],
    }
    path = tmp_path / "artifact_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_strict_manifest_verifier_accepts_only_clean_single_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _strict_manifest(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["verify_manifest.py", "--strict-commit", "--manifest", str(path)],
    )
    assert manifest_verifier.main() == 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"publication_code_commit": "2" * 40},
        {"simulation_source_commit": "2" * 40},
        {"dual_provenance": True},
        {"git_dirty": True},
    ],
)
def test_strict_manifest_verifier_rejects_repaired_or_dirty_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict,
) -> None:
    path = _strict_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["verify_manifest.py", "--strict-commit", "--manifest", str(path)],
    )
    assert manifest_verifier.main() == 1


def test_commit_overrides_are_rejected_without_complete_recovery_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires AGRIBRAIN_RECOVERY_RECEIPT"):
        recovery_gate.recovery_context_from_environment(
            results_dir=tmp_path,
            repo_root=tmp_path,
            environ={
                "AGRIBRAIN_SIMULATION_COMMIT": SIMULATION_COMMIT,
                "AGRIBRAIN_PUBLICATION_CODE_COMMIT": PUBLICATION_COMMIT,
                "RUN_TAG": RUN_TAG,
            },
        )


def test_recovery_inventory_is_never_admitted_to_a_fresh_manifest() -> None:
    for directory in (
        "publication_recovery_receipts", "preserved_raw_manifests",
    ):
        name = f"{directory}/{RUN_TAG}.json"
        assert not manifest_builder._is_canonical_path(
            name, include_raw=True, run_tag=RUN_TAG,
        )
        assert manifest_builder._is_canonical_path(
            name, include_raw=True, run_tag=RUN_TAG, include_recovery=True,
        )


def test_strict_manifest_accepts_validated_dual_provenance_only_with_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, receipt, authorization = _recovery_results(tmp_path, monkeypatch)
    payload_file = results / "payload.json"
    payload_file.write_bytes(b'{"repaired":true}\n')
    artifact_paths = [
        payload_file,
        results / authorization["receipt_file"],
        results / authorization["preserved_raw_manifest_file"],
        results / authorization["original_submission_receipt_file"],
    ]
    records = [{
        "file": path.relative_to(results).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in artifact_paths]
    manifest_path = results / "artifact_manifest.json"
    _write_json(manifest_path, {
        "schema_version": 2,
        "git_commit": SIMULATION_COMMIT,
        "simulation_source_commit": SIMULATION_COMMIT,
        "publication_code_commit": PUBLICATION_COMMIT,
        "dual_provenance": True,
        "git_dirty": False,
        "artifact_run_tag": RUN_TAG,
        "recovery_authorization": authorization,
        "artifact_count": len(records),
        "artifacts": records,
    })

    monkeypatch.setattr("sys.argv", [
        "verify_manifest.py", "--strict-commit", "--manifest", str(manifest_path),
    ])
    assert manifest_verifier.main() == 1
    monkeypatch.setattr("sys.argv", [
        "verify_manifest.py", "--strict-commit", "--manifest", str(manifest_path),
        "--recovery-receipt", str(receipt),
    ])
    assert manifest_verifier.main() == 0


def test_authorized_recovery_flows_manifest_to_semantic_receipt_to_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete provenance hand-off without faking its receipt."""

    results, recovery_receipt, authorization = _recovery_results(
        tmp_path, monkeypatch,
    )
    payload_file = results / "publication_environment.json"
    payload_file.write_bytes(b'{"validated":true}\n')
    artifact_paths = [
        payload_file,
        results / authorization["receipt_file"],
        results / authorization["preserved_raw_manifest_file"],
        results / authorization["original_submission_receipt_file"],
    ]

    def record(path: Path) -> dict:
        data = path.read_bytes()
        return {
            "file": path.relative_to(results).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    manifest_path = results / "artifact_manifest.json"
    manifest = {
        "schema_version": 2,
        "git_commit": SIMULATION_COMMIT,
        "simulation_source_commit": SIMULATION_COMMIT,
        "publication_code_commit": PUBLICATION_COMMIT,
        "dual_provenance": True,
        "git_dirty": False,
        "includes_raw_run_artifacts": True,
        "artifact_run_tag": RUN_TAG,
        "recovery_authorization": authorization,
        "artifacts": [record(path) for path in artifact_paths],
    }
    manifest["artifact_count"] = len(manifest["artifacts"])
    _write_json(manifest_path, manifest)

    inventory = {
        "top_level_artifacts_excluding_receipt": 1,
        "benchmark_seed_envelopes": 20,
        "primary_retained_decision_ledgers": 1_100,
        "h3_retained_stressed_decision_ledgers": 500,
        "raw_stress_task_files": 20,
        "core_slurm_submission_receipts": 1,
        "publication_recovery_receipts": 1,
        "preserved_raw_manifests": 1,
    }
    monkeypatch.setattr(semantic_validator, "RESULTS_DIR", results)
    monkeypatch.setattr(semantic_validator, "REPO_ROOT", REPO_ROOT)
    monkeypatch.setattr(
        semantic_validator, "RECOVERY_RECEIPT_PATH", recovery_receipt,
    )
    monkeypatch.setattr(
        semantic_validator,
        "_validate_manifest_inventory",
        lambda _manifest, *, receipt_expected, recovery_authorization=None: inventory,
    )
    semantic_validator._write_publication_validation_receipt()
    semantic_receipt = results / "publication_validation_receipt.json"
    manifest["artifacts"].append(record(semantic_receipt))
    manifest["artifact_count"] = len(manifest["artifacts"])
    _write_json(manifest_path, manifest)
    semantic_validator.validate_publication_validation_receipt(
        results,
        repo_root=REPO_ROOT,
        recovery_receipt=recovery_receipt,
    )

    validator_identity = {
        "head_commit": PUBLICATION_COMMIT,
        "source_tree_clean_outside_exact_evidence_paths": True,
        "status_includes_untracked_files": True,
        "allowed_evidence_path_count": len(manifest["artifacts"]) + 1,
        "allowed_evidence_path_set_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        "mvp.simulation.analysis.recovery_provenance.current_checkout_commit",
        lambda _root: PUBLICATION_COMMIT,
    )
    monkeypatch.setattr(
        "mvp.simulation.analysis.recovery_provenance.current_checkout_tree",
        lambda _root: "c" * 40,
    )
    monkeypatch.setattr(
        archive_builder,
        "validate_clean_validator_checkout",
        lambda expected_commit, **_kwargs: (
            validator_identity
            if expected_commit == PUBLICATION_COMMIT
            else pytest.fail("archive validator used the simulation commit")
        ),
    )

    def validate_full(results_dir, *, repo_root, recovery_receipt=None):
        assert Path(recovery_receipt) == Path(recovery_receipt_path)
        semantic_validator.validate_publication_validation_receipt(
            results_dir,
            repo_root=repo_root,
            recovery_receipt=Path(recovery_receipt),
        )

    recovery_receipt_path = recovery_receipt
    monkeypatch.setattr(
        archive_builder, "validate_full_publication_release", validate_full,
    )
    bundle = tmp_path / "publication_bundle"
    archive = bundle / "evidence.tar.gz"
    archive_receipt = bundle / "archive_receipt.json"
    monkeypatch.setattr("sys.argv", [
        "build_publication_archive.py",
        "--results-dir", str(results),
        "--output", str(archive),
        "--receipt", str(archive_receipt),
        "--recovery-receipt", str(recovery_receipt),
    ])
    assert archive_builder.main() == 0
    packaged = json.loads(archive_receipt.read_text(encoding="utf-8"))
    assert packaged["simulation_rerun"] is False
    assert packaged["recovery_authorization"] == authorization


def test_archive_round_trip_verifies_final_literal_bytes(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    payload = results / "table.csv"
    payload.write_bytes(b"a,b\r\n1,2\r\n")
    record = {
        "file": "table.csv",
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": 2,
        "git_commit": "1" * 40,
        "simulation_source_commit": "1" * 40,
        "publication_code_commit": "2" * 40,
        "dual_provenance": True,
        "artifact_count": 1,
        "artifacts": [record],
    }
    manifest_path = results / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, records = archive_builder._load_manifest(manifest_path)
    assert loaded["dual_provenance"] is True
    archive_builder._verify_files(results, records)

    archive = tmp_path / "publication.tar.gz"
    archive_builder._write_archive(
        archive, results, manifest_path.read_bytes(), records, epoch=0
    )
    archive_builder._verify_archive(
        archive, manifest_path.read_bytes(), records
    )


def test_archive_receipt_distinguishes_fresh_run_from_publication_repair():
    fresh = {
        "simulation_source_commit": "1" * 40,
        "publication_code_commit": "1" * 40,
        "dual_provenance": False,
    }
    repair = {
        "simulation_source_commit": "1" * 40,
        "publication_code_commit": "2" * 40,
        "dual_provenance": True,
    }

    assert archive_builder._derivation_metadata(
        fresh, semantic_receipt_validated=True,
    ) == {
        "derivation_type": "fresh stochastic simulation and publication build",
        "simulation_rerun": True,
    }
    assert archive_builder._derivation_metadata(fresh) == {
        "derivation_type": "unknown: equal commits do not prove execution",
        "simulation_rerun": None,
    }
    with pytest.raises(ValueError, match="requires a validated"):
        archive_builder._derivation_metadata(repair)
    assert archive_builder._derivation_metadata(
        repair, recovery_receipt_validated=True,
    ) == {
        "derivation_type": "publication-only deterministic recovery",
        "simulation_rerun": False,
    }


def _minimal_fresh_archive_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    results = tmp_path / "results"
    results.mkdir()
    artifact = results / "publication_validation_receipt.json"
    artifact.write_text('{"validation_status":"PASS"}\n', encoding="utf-8")
    record = {
        "file": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    commit = "1" * 40
    manifest = {
        "schema_version": 2,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "dual_provenance": False,
        "git_dirty": False,
        "artifact_count": 1,
        "artifacts": [record],
    }
    (results / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    return results, bundle, bundle / "evidence.tar.gz", bundle / "receipt.json"


def test_archive_builder_atomically_promotes_complete_ready_bundle(
    tmp_path, monkeypatch,
):
    results, bundle, archive, receipt = _minimal_fresh_archive_inputs(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_publication_archive.py",
            "--results-dir", str(results),
            "--output", str(archive),
            "--receipt", str(receipt),
        ],
    )
    monkeypatch.setattr(
        archive_builder, "validate_full_publication_release", lambda *_args, **_kwargs: None,
    )
    validator_identity = {
        "head_commit": "1" * 40,
        "source_tree_clean_outside_exact_evidence_paths": True,
        "status_includes_untracked_files": True,
        "allowed_evidence_path_count": 2,
        "allowed_evidence_path_set_sha256": "a" * 64,
    }
    identity_calls = []

    def validate_identity(expected_commit, **kwargs):
        identity_calls.append((expected_commit, kwargs))
        return validator_identity

    monkeypatch.setattr(
        archive_builder, "validate_clean_validator_checkout", validate_identity,
    )

    assert archive_builder.main() == 0
    assert archive.is_file()
    assert receipt.is_file()
    ready = json.loads((bundle / "READY.json").read_text(encoding="utf-8"))
    assert ready["status"] == "READY"
    assert ready["archive"]["sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    archive_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert archive_receipt["validator_source_identity"] == validator_identity
    assert archive_receipt["validation"][
        "validator_checkout_same_clean_commit_outside_exact_evidence"
    ] == "PASS"
    assert len(identity_calls) == 3
    assert all(call[0] == "1" * 40 for call in identity_calls)


@pytest.mark.parametrize(
    "failure",
    [
        "validator checkout HEAD differs from the evidence source commit",
        "validator checkout has changes outside the exact evidence allowlist",
    ],
)
def test_archive_builder_rejects_wrong_or_dirty_validator_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    results, bundle, archive, receipt = _minimal_fresh_archive_inputs(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "build_publication_archive.py",
        "--results-dir", str(results),
        "--output", str(archive),
        "--receipt", str(receipt),
    ])
    semantic_calls = []
    monkeypatch.setattr(
        archive_builder,
        "validate_full_publication_release",
        lambda *_args, **_kwargs: semantic_calls.append(True),
    )
    monkeypatch.setattr(
        archive_builder,
        "validate_clean_validator_checkout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(failure)),
    )

    with pytest.raises(ValueError, match=failure):
        archive_builder.main()
    assert semantic_calls == []
    assert not bundle.exists()


def test_archive_builder_rechecks_validator_after_semantic_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, bundle, archive, receipt = _minimal_fresh_archive_inputs(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "build_publication_archive.py",
        "--results-dir", str(results),
        "--output", str(archive),
        "--receipt", str(receipt),
    ])
    identity = {
        "head_commit": "1" * 40,
        "source_tree_clean_outside_exact_evidence_paths": True,
        "status_includes_untracked_files": True,
        "allowed_evidence_path_count": 2,
        "allowed_evidence_path_set_sha256": "b" * 64,
    }
    calls = 0

    def identity_then_dirty(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("validator source changed during validation")
        return identity

    monkeypatch.setattr(
        archive_builder, "validate_clean_validator_checkout", identity_then_dirty,
    )
    monkeypatch.setattr(
        archive_builder, "validate_full_publication_release", lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="source changed during validation"):
        archive_builder.main()
    assert not bundle.exists()


def test_archive_builder_rejects_dual_provenance_even_with_parent_hash(
    tmp_path, monkeypatch,
):
    results = tmp_path / "results"
    results.mkdir()
    artifact = results / "publication_validation_receipt.json"
    artifact.write_text('{"validation_status":"PASS"}\n', encoding="utf-8")
    record = {
        "file": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    (results / "artifact_manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "simulation_source_commit": "1" * 40,
        "publication_code_commit": "2" * 40,
        "dual_provenance": True,
        "artifact_count": 1,
        "artifacts": [record],
    }), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "build_publication_archive.py",
        "--results-dir", str(results),
        "--output", str(tmp_path / "bundle" / "evidence.tar.gz"),
        "--receipt", str(tmp_path / "bundle" / "receipt.json"),
        "--parent-archive-sha256", "f" * 64,
    ])
    with pytest.raises(ValueError, match="parent-archive-sha256 is retired"):
        archive_builder.main()
