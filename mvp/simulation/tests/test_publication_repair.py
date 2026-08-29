"""Regression tests for fresh-run integrity and retired repair pathways."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

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


def _record(mean: float, low: float, high: float) -> dict:
    return {
        "mean": mean,
        "std": 0.1,
        "ci_low": low,
        "ci_high": high,
        "ci_method": "BCa",
        "n_seeds": 20,
    }


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
    with pytest.raises(ValueError, match="repair packaging is retired"):
        archive_builder._derivation_metadata(repair)


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
