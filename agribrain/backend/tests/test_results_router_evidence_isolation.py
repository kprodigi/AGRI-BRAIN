from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from src.routers import results

_REAL_CANONICAL_RELEASE_VALIDATOR = results._validate_canonical_release_contract


@pytest.fixture(autouse=True)
def _stub_full_release_validator(monkeypatch):
    """Most router tests isolate byte/provenance behavior from 1,541 files."""

    results._clear_publication_verification_cache()
    monkeypatch.setattr(
        results, "_validate_canonical_release_contract", lambda: None,
    )


class _Table:
    def __init__(self, payload: str):
        self.payload = payload

    def to_csv(self, path, index=False):
        assert index is False
        path.write_text(self.payload, encoding="utf-8")


def _write_valid_release(tmp_path, commit: str, artifact, **changes):
    manifest = {
        "schema_version": 2,
        "git_dirty": False,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "dual_provenance": False,
        "artifact_run_tag": "fresh_test_run",
        "artifacts": [{
            "file": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }],
    }
    manifest.update(changes)
    protocol_path = results._SIM_DIR / "experiment_protocol.json"
    protocol = protocol_path.read_bytes()
    receipt = {
        "schema_version": 1,
        "validation_status": "PASS",
        "validation_scope": "core_publication_evidence",
        "fresh_single_commit_run": True,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "run_tag": "fresh_test_run",
        "protocol": {
            "file": "mvp/simulation/experiment_protocol.json",
            "bytes": len(protocol),
            "sha256": hashlib.sha256(protocol).hexdigest(),
        },
        "semantic_artifact_set": {
            "artifact_count_excluding_receipt": len(manifest["artifacts"]),
            "merkle_root": results._artifact_set_root(manifest["artifacts"]),
        },
        "locked_accounting": {
            "core_unique_retained_cells": 1_600,
            "core_executed_episodes": 6_100,
            "core_simulated_steps": 1_756_800,
            "h1_directional_tests": 5,
            "h2_directional_tests": 20,
            "h3_equivalence_cells": 25,
        },
        "structural_sensitivity": {
            "included_in_core_receipt": False,
            "required_for_full_submission_evidence": True,
        },
    }
    receipt_path = tmp_path / results._VALIDATION_RECEIPT
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["artifacts"].append({
        "file": receipt_path.name,
        "bytes": receipt_path.stat().st_size,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    })
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return manifest, receipt_path


def test_development_run_cannot_overwrite_publication_artifacts(tmp_path, monkeypatch):
    publication = tmp_path / "publication"
    development = tmp_path / "development"
    publication.mkdir()
    locked = publication / "table1_summary.csv"
    locked.write_text("validated-publication", encoding="utf-8")
    monkeypatch.setattr(results, "_RESULTS_DIR", publication)
    monkeypatch.setattr(results, "_DEVELOPMENT_RESULTS_DIR", development)

    names = results._save_development_artifacts(
        {"table1": _Table("local-table-1"), "table2": _Table("local-table-2")},
        {"baseline": {}},
        7,
    )

    assert locked.read_text(encoding="utf-8") == "validated-publication"
    assert all(name.startswith("development_") for name in names.values())
    envelope = json.loads((development / names["summary"]).read_text(encoding="utf-8"))
    assert envelope["publication_evidence"] is False
    assert envelope["evidence_status"] == "development_only"
    assert envelope["mode_design"]["total"] == 11
    assert len(envelope["mode_design"]["primary"]) == 8
    assert len(envelope["mode_design"]["secondary"]) == 3


def test_development_download_returns_named_file_with_nonpublication_header(
    tmp_path, monkeypatch,
):
    development = tmp_path / "development"
    development.mkdir()
    artifact = development / "development_summary.json"
    artifact.write_text('{"publication_evidence": false}', encoding="utf-8")
    monkeypatch.setattr(results, "_DEVELOPMENT_RESULTS_DIR", development)

    response = results.get_development_artifact(artifact.name)

    assert isinstance(response, FileResponse)
    assert response.path == str(artifact)
    assert response.media_type == "application/json"
    assert response.headers["x-agribrain-evidence-status"] == "development-only"


def test_background_development_run_redirects_every_simulator_write(
    tmp_path, monkeypatch,
):
    publication = tmp_path / "publication"
    development = tmp_path / "development"
    publication.mkdir()
    sentinel = publication / "benchmark_summary.json"
    sentinel.write_text("untouched", encoding="utf-8")

    original_module_output = publication
    fake = types.ModuleType("generate_results")
    fake.RESULTS_DIR = original_module_output

    def run_all(seed=None):
        fake.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (fake.RESULTS_DIR / "traces_baseline.json").write_text(
            f"seed={seed}", encoding="utf-8",
        )
        ledger = fake.RESULTS_DIR / "decision_ledger"
        ledger.mkdir()
        (ledger / "agribrain__baseline.jsonl").write_text(
            "development", encoding="utf-8",
        )
        return {
            "table1": _Table("a,b\n1,2\n"),
            "table2": _Table("a,b\n3,4\n"),
        }

    fake.run_all = run_all
    fake.get_summary_json = lambda _data: {"baseline": {}}
    monkeypatch.setitem(sys.modules, "generate_results", fake)
    monkeypatch.setattr(results, "_RESULTS_DIR", publication)
    monkeypatch.setattr(results, "_DEVELOPMENT_RESULTS_DIR", development)

    results._run_in_background(seed=7)

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert fake.RESULTS_DIR == original_module_output
    run_dirs = list((development / "runs").glob("development_*_seed_7"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "traces_baseline.json").is_file()
    assert (run_dirs[0] / "decision_ledger" / "agribrain__baseline.jsonl").is_file()
    assert results._JOB["error"] is None


def test_publication_endpoint_requires_manifested_matching_bytes(tmp_path, monkeypatch):
    commit = "a" * 40
    payload = b"canonical"
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(payload)
    _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)

    verified = results._publication_artifact(artifact.name)
    assert verified.content == payload
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()
    artifact.write_bytes(b"tampered")
    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503


def test_publication_response_uses_the_already_verified_immutable_bytes(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    original = b'{"canonical":true}'
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(original)
    _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)
    verifier = results._publication_artifact

    def verify_then_replace(filename):
        captured = verifier(filename)
        artifact.write_bytes(b'{"tampered":true}')
        return captured

    monkeypatch.setattr(results, "_publication_artifact", verify_then_replace)
    response = results.get_figure(artifact.name)
    assert response.body == original
    assert response.headers["etag"] == (
        f'"{hashlib.sha256(original).hexdigest()}"'
    )


def test_full_release_audit_is_cached_but_any_payload_metadata_change_invalidates(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"canonical")
    _manifest, receipt = _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)
    calls = []
    monkeypatch.setattr(
        results,
        "_validate_canonical_release_contract",
        lambda: calls.append("full-audit"),
    )

    assert results._publication_artifact(artifact.name).content == b"canonical"
    assert results._publication_artifact(artifact.name).content == b"canonical"
    assert calls == ["full-audit"]

    receipt.write_bytes(receipt.read_bytes() + b"\n")
    with pytest.raises(HTTPException, match="integrity verification"):
        results._publication_artifact(artifact.name)
    assert calls == ["full-audit"]


def test_publication_endpoint_rejects_partial_self_consistent_release(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"canonical")
    _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)
    monkeypatch.setattr(
        results,
        "_validate_canonical_release_contract",
        _REAL_CANONICAL_RELEASE_VALIDATOR,
    )

    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503
    assert "Canonical publication evidence contract failed" in exc.value.detail


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_run_tag": None},
        {"dual_provenance": True, "publication_code_commit": "b" * 40},
        {"simulation_source_commit": "b" * 40},
    ],
)
def test_publication_endpoint_rejects_nonfresh_or_split_provenance(
    tmp_path, monkeypatch, change
):
    commit = "a" * 40
    payload = b"canonical"
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(payload)
    _write_valid_release(tmp_path, commit, artifact, **change)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)

    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503


def test_publication_endpoint_rejects_results_from_other_commit(tmp_path, monkeypatch):
    commit = "a" * 40
    payload = b"canonical"
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(payload)
    _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: "b" * 40)

    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503


def test_publication_endpoint_requires_hash_bound_semantic_receipt(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"canonical")
    manifest, receipt = _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)

    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503

    receipt.unlink()
    manifest["artifacts"] = [
        item for item in manifest["artifacts"]
        if item["file"] != results._VALIDATION_RECEIPT
    ]
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503


def test_current_source_identity_rejects_dirty_code_but_ignores_result_bytes(
    monkeypatch,
):
    commit = "a" * 40
    monkeypatch.delenv("AGRIBRAIN_GIT_COMMIT", raising=False)

    def dirty_code(command, **_kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return commit
        return " M README.md\0"

    monkeypatch.setattr(results.subprocess, "check_output", dirty_code)
    with pytest.raises(HTTPException, match="uncommitted non-result"):
        results._current_source_commit()

    def result_only(command, **_kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return commit
        return " M mvp/simulation/results/benchmark_summary.json\0"

    monkeypatch.setattr(results.subprocess, "check_output", result_only)
    assert results._current_source_commit() == commit


def test_current_source_identity_is_anchored_to_executing_router_repo(
    tmp_path, monkeypatch,
):
    commit = "a" * 40
    observed_cwds = []
    monkeypatch.delenv("AGRIBRAIN_GIT_COMMIT", raising=False)
    monkeypatch.setattr(results, "_SIM_DIR", tmp_path / "untrusted" / "simulation")

    def checked(command, **kwargs):
        observed_cwds.append(kwargs["cwd"])
        if command[1:3] == ["rev-parse", "HEAD"]:
            return commit
        return ""

    monkeypatch.setattr(results.subprocess, "check_output", checked)
    assert results._current_source_commit() == commit
    assert observed_cwds
    assert all(
        path == results._TRUSTED_REPO_ROOT.resolve() for path in observed_cwds
    )


def test_summary_never_falls_back_to_publication_named_tables(tmp_path, monkeypatch):
    (tmp_path / "table1_summary.csv").write_text("publication", encoding="utf-8")
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setitem(results._JOB, "summary", None)
    response = results.results_summary()
    assert response["ok"] is False
    assert response["publication_evidence"] is False
    assert "tables" not in response
