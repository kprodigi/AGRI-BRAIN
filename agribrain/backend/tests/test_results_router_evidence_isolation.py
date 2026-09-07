from __future__ import annotations

import hashlib
import json
import subprocess
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
        results, "_validate_canonical_release_contract", lambda *_args, **_kwargs: None,
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


def _write_valid_recovery_release(tmp_path, artifact):
    simulation_commit = "a" * 40
    publication_commit = "b" * 40
    run_tag = "aaaaaaa_20260829_105800"
    expected = {
        "receipt_file": f"publication_recovery_receipts/{run_tag}.json",
        "preserved_raw_manifest_file": f"preserved_raw_manifests/{run_tag}.json",
        "original_submission_receipt_file": f"core_submission_receipts/{run_tag}.json",
    }
    payloads = {
        expected["receipt_file"]: b'{"authorized":true}\n',
        expected["preserved_raw_manifest_file"]: b'{"preserved":true}\n',
        expected["original_submission_receipt_file"]: b'{"submitted":true}\n',
    }
    for relative, payload in payloads.items():
        path = tmp_path.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    authorization = {
        **expected,
        "receipt_literal_sha256": hashlib.sha256(
            payloads[expected["receipt_file"]]
        ).hexdigest(),
        "receipt_self_hash": "c" * 64,
        "preserved_raw_manifest_literal_sha256": hashlib.sha256(
            payloads[expected["preserved_raw_manifest_file"]]
        ).hexdigest(),
        "preserved_raw_payload_merkle_root": "d" * 64,
        "simulation_rerun": False,
        "publication_repair_tree": "e" * 40,
        "validated": True,
    }

    def record(path):
        payload = path.read_bytes()
        return {
            "file": path.relative_to(tmp_path).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    evidence = [artifact] + [
        tmp_path.joinpath(*relative.split("/")) for relative in expected.values()
    ]
    manifest = {
        "schema_version": 2,
        "git_dirty": False,
        "git_commit": simulation_commit,
        "simulation_source_commit": simulation_commit,
        "publication_code_commit": publication_commit,
        "dual_provenance": True,
        "artifact_run_tag": run_tag,
        "recovery_authorization": authorization,
        "artifacts": [record(path) for path in evidence],
    }
    protocol_path = results._SIM_DIR / "experiment_protocol.json"
    protocol = protocol_path.read_bytes()
    semantic_receipt = {
        "schema_version": 1,
        "validation_status": "PASS",
        "validation_scope": "core_publication_evidence",
        "fresh_single_commit_run": False,
        "authorized_deterministic_recovery": True,
        "simulation_rerun": False,
        "git_commit": simulation_commit,
        "simulation_source_commit": simulation_commit,
        "publication_code_commit": publication_commit,
        "run_tag": run_tag,
        "recovery_authorization": authorization,
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
    semantic_path = tmp_path / results._VALIDATION_RECEIPT
    semantic_path.write_text(json.dumps(semantic_receipt), encoding="utf-8")
    manifest["artifacts"].append(record(semantic_path))
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return manifest, authorization


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


def test_publication_endpoint_accepts_only_fully_authorized_core_recovery(
    tmp_path, monkeypatch,
):
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"recovered-publication")
    manifest, authorization = _write_valid_recovery_release(tmp_path, artifact)
    recovery_receipt = tmp_path.joinpath(
        *authorization["receipt_file"].split("/")
    )
    observed = {}

    def validate_context(receipt, **kwargs):
        observed["context_receipt"] = receipt
        observed["context"] = kwargs
        return authorization

    def validate_release(receipt=None):
        observed["release_receipt"] = receipt

    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        results, "_current_source_commit",
        lambda: manifest["publication_code_commit"],
    )
    monkeypatch.setattr(
        "mvp.simulation.analysis.recovery_provenance.validate_recovery_context",
        validate_context,
    )
    monkeypatch.setattr(
        results, "_validate_canonical_release_contract", validate_release,
    )

    verified = results._publication_artifact(artifact.name)

    assert verified.content == b"recovered-publication"
    assert observed["context_receipt"] == recovery_receipt
    assert observed["context"]["simulation_commit"] == (
        manifest["simulation_source_commit"]
    )
    assert observed["context"]["publication_commit"] == (
        manifest["publication_code_commit"]
    )
    assert observed["context"]["expected_kind"] == "core"
    assert observed["release_receipt"] == recovery_receipt


@pytest.mark.parametrize("failure", ["missing", "tampered", "symlink"])
def test_recovery_endpoint_rejects_missing_tampered_or_linked_authorization_receipt(
    tmp_path, monkeypatch, failure,
):
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"recovered-publication")
    manifest, authorization = _write_valid_recovery_release(tmp_path, artifact)
    receipt = tmp_path.joinpath(*authorization["receipt_file"].split("/"))
    if failure == "missing":
        receipt.unlink()
    elif failure == "tampered":
        receipt.write_bytes(b'{"authorized":false}\n')
    else:
        target = tmp_path / "outside-recovery-receipt.json"
        target.write_bytes(receipt.read_bytes())
        receipt.unlink()
        try:
            receipt.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable on this platform")

    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        results, "_current_source_commit",
        lambda: manifest["publication_code_commit"],
    )

    with pytest.raises(HTTPException) as exc:
        results._publication_artifact(artifact.name)
    assert exc.value.status_code == 503


def test_recovery_endpoint_serves_from_a_later_serving_commit(tmp_path, monkeypatch):
    """The serving checkout may move past the commit that published the evidence.

    The evidence records the commit that re-aggregated it, which ran on a
    machine whose history was never published; requiring ``HEAD`` to equal that
    commit made the endpoint unsatisfiable for every clone.  Provenance stays in
    the manifest and is still validated against the recovery receipt; it is no
    longer a claim about which code is running.
    """
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"recovered-publication")
    manifest, authorization = _write_valid_recovery_release(tmp_path, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        "mvp.simulation.analysis.recovery_provenance.validate_recovery_context",
        lambda receipt, **kwargs: authorization,
    )
    monkeypatch.setattr(
        results, "_validate_canonical_release_contract", lambda receipt=None: None,
    )

    assert manifest["publication_code_commit"] != "c" * 40
    verified = results._publication_artifact(artifact.name)

    assert verified.content == b"recovered-publication"


def test_committed_evidence_passes_the_audit():
    """The real manifest is tracked at HEAD and unmodified, so it audits clean."""
    results._require_published_evidence("artifact_manifest.json")


def test_evidence_missing_from_the_serving_commit_is_rejected():
    with pytest.raises(HTTPException, match="not committed in the serving checkout"):
        results._require_published_evidence("not_a_tracked_artifact.json")


def test_evidence_outside_the_repository_is_not_audited(tmp_path, monkeypatch):
    """Git cannot speak for a results root mounted outside the checkout.

    Skipping is what lets a packaged deployment serve evidence at all; the
    manifest hash chain is the guarantee there, and every test above that points
    ``_RESULTS_DIR`` at ``tmp_path`` depends on this branch.
    """
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)

    results._require_published_evidence("anything_at_all.json")


@pytest.mark.parametrize("dual", [False, True])
def test_publication_endpoint_rejects_partial_recovery_hints(
    tmp_path, monkeypatch, dual,
):
    commit = "a" * 40
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(b"canonical")
    changes = {"recovery_authorization": {"validated": True}}
    if dual:
        changes.update({
            "dual_provenance": True,
            "publication_code_commit": "b" * 40,
        })
    _write_valid_release(tmp_path, commit, artifact, **changes)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: commit)

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
        lambda receipt=None: calls.append("full-audit"),
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


def test_publication_endpoint_serves_a_fresh_release_from_a_later_commit(
    tmp_path, monkeypatch,
):
    """A single-commit release is served after the checkout moves on too.

    Pinning ``HEAD`` to the commit that produced the evidence broke the endpoint
    on the next commit of any kind, including one that touched nothing the
    evidence depends on.
    """
    commit = "a" * 40
    payload = b"canonical"
    artifact = tmp_path / "benchmark_significance.json"
    artifact.write_bytes(payload)
    _write_valid_release(tmp_path, commit, artifact)
    monkeypatch.setattr(results, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "_current_source_commit", lambda: "b" * 40)
    monkeypatch.setattr(
        results, "_validate_canonical_release_contract", lambda receipt=None: None,
    )

    assert results._publication_artifact(artifact.name).content == payload


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


def test_repository_subset_serves_committed_evidence_and_defers_the_rest(monkeypatch):
    """The committed evidence serves; the deposit-only remainder answers 404.

    This runs against the real results tree rather than a fixture, because the
    behaviour under test is a property of that tree: ``results/README.md``
    commits the tables, statistics and receipts so the paper's values can be
    checked against a clone, and leaves the 1,600 per-seed ledgers and the run's
    own figure renders to the evidence deposit.  Demanding all 1,684 payloads
    before serving any of them made every clone fail on the first one missing.

    The serving commit is stood in for so the assertion is about evidence rather
    than about whether this particular checkout happens to be clean.
    """
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=results._TRUSTED_REPO_ROOT,
        text=True,
    ).strip()
    monkeypatch.setattr(results, "_current_source_commit", lambda: head)

    for name in ("table1_summary.csv", "benchmark_summary.json"):
        assert results._publication_artifact(name).content

    with pytest.raises(HTTPException, match="evidence deposit") as exc:
        results._publication_artifact("ablation.pdf")
    assert exc.value.status_code == 404
