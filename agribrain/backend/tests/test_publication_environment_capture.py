"""Unit checks for the exact publication-environment artifact."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_PATH = REPO_ROOT / "hpc" / "capture_publication_environment.py"
VALIDATOR_PATH = (
    REPO_ROOT / "mvp" / "simulation" / "validation"
    / "validate_publication_artifacts.py"
)


def test_capture_script_tracks_both_canonical_sources():
    spec = importlib.util.spec_from_file_location("capture_publication_environment", CAPTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LOCK == REPO_ROOT / "agribrain" / "backend" / "requirements-lock.txt"
    assert module.PYPROJECT == REPO_ROOT / "agribrain" / "backend" / "pyproject.toml"
    assert module.ENV_SCRIPT == REPO_ROOT / "hpc" / "publication_env.sh"
    assert module.LOCK.is_file()
    assert module.ENV_SCRIPT.is_file()


def test_distribution_inventory_is_versioned_and_path_free():
    spec = importlib.util.spec_from_file_location("capture_publication_environment", CAPTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = module._installed_distributions()
    assert inventory
    assert all("==" in item for item in inventory)
    assert all("file://" not in item.lower() for item in inventory)
    assert all("\\" not in item and "/" not in item for item in inventory)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_publication_artifacts_for_environment_test", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_environment_fixture(root: Path, *, environment_run_tag: str = "run_1") -> Path:
    results = root / "mvp" / "simulation" / "results"
    lock = root / "agribrain" / "backend" / "requirements-lock.txt"
    pyproject = root / "agribrain" / "backend" / "pyproject.toml"
    env_script = root / "hpc" / "publication_env.sh"
    results.mkdir(parents=True)
    lock.parent.mkdir(parents=True)
    env_script.parent.mkdir(parents=True)
    lock.write_text("demo==1.0\n", encoding="utf-8")
    pyproject.write_text(
        "[project]\nname = 'demo'\nversion = '1.0'\n", encoding="utf-8"
    )
    env_script.write_text("export DEMO=1\n", encoding="utf-8")
    commit = "a" * 40
    manifest_run_tag = "run_1"
    (results / "artifact_manifest.json").write_text(json.dumps({
        "git_commit": commit,
        "artifact_run_tag": manifest_run_tag,
    }), encoding="utf-8")

    def file_record(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    (results / "publication_environment.json").write_text(json.dumps({
        "schema_version": 2,
        "run_tag": environment_run_tag,
        "git_commit": commit,
        "environment_scope": "version_resolved_runtime_inventory",
        "binary_reproducibility": {
            "byte_identical_environment_claimed": False,
            "distribution_artifact_hashes_recorded": False,
            "container_image_digest_recorded": False,
            "interpretation": "version-resolved, not byte-identical",
        },
        "python": {"version": "3.11.9"},
        "virtual_environment": {
            "run_scoped": True,
            "path_id": f".publication_venvs/{environment_run_tag}",
            "isolated_from_base_prefix": True,
        },
        "installed_package_count": 1,
        "installed_distributions": ["demo==1.0"],
        "distribution_validation": {
            "unique_normalized_names": True,
            "lock_versions_match": True,
            "core_version_match": True,
            "core_distribution": "demo==1.0",
            "locked_distribution_count": 1,
            "applicable_lock_distributions": ["demo==1.0"],
            "unexpected_distributions": [],
            "allowed_bootstrap_distributions": [],
        },
        "requirements_lock": file_record(lock),
        "backend_project": file_record(pyproject),
        "publication_environment_script": file_record(env_script),
    }), encoding="utf-8")
    return results


def test_publication_environment_validator_pins_run_and_source_hashes(
    tmp_path, monkeypatch
):
    validator = _load_validator()
    results = _write_environment_fixture(tmp_path)
    monkeypatch.setattr(validator, "RESULTS_DIR", results)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    validator._validate_publication_environment()


def test_publication_environment_validator_rejects_wrong_run_tag(
    tmp_path, monkeypatch
):
    validator = _load_validator()
    results = _write_environment_fixture(tmp_path, environment_run_tag="other_run")
    monkeypatch.setattr(validator, "RESULTS_DIR", results)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        validator._validate_publication_environment()
