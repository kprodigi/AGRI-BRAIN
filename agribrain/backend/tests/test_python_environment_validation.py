"""Focused checks for exact publication virtual-environment validation."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "hpc" / "capture_publication_environment.py"
SPEC = importlib.util.spec_from_file_location("capture_publication_environment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
VALIDATOR_PATH = (
    REPO_ROOT / "mvp" / "simulation" / "validation" / "validate_publication_artifacts.py"
)


def _load_artifact_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_publication_artifacts_environment_unit", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PythonEnvironmentValidationTests(unittest.TestCase):
    def test_exact_lock_core_and_bootstrap_set_is_accepted(self):
        validation, errors = MODULE._validate_distribution_set(
            [("Demo_Pkg", "1.0"), ("Agri.Brain_Backend", "1.2.0"), ("pip", "25.0")],
            {"demo-pkg": "1.0"},
            ("agri-brain-backend", "1.2.0"),
        )
        self.assertEqual(errors, [])
        self.assertTrue(validation["unique_normalized_names"])
        self.assertTrue(validation["lock_versions_match"])
        self.assertTrue(validation["core_version_match"])
        self.assertEqual(validation["unexpected_distributions"], [])
        self.assertEqual(
            validation["applicable_lock_distributions"], ["demo-pkg==1.0"]
        )

    def test_normalized_duplicate_is_not_masked(self):
        validation, errors = MODULE._validate_distribution_set(
            [("Demo_Pkg", "1.0"), ("demo-pkg", "1.0"), ("core", "2.0")],
            {"demo-pkg": "1.0"},
            ("core", "2.0"),
        )
        self.assertFalse(validation["unique_normalized_names"])
        self.assertTrue(any("duplicate normalized" in error for error in errors))

    def test_missing_mismatched_and_unexpected_packages_are_rejected(self):
        validation, errors = MODULE._validate_distribution_set(
            [("demo", "9.0"), ("core", "2.0"), ("contaminant", "1")],
            {"demo": "1.0", "missing": "3.0"},
            ("core", "2.0"),
        )
        joined = "; ".join(errors)
        self.assertIn("missing locked/core", joined)
        self.assertIn("version mismatches", joined)
        self.assertIn("unexpected distributions", joined)
        self.assertFalse(validation["lock_versions_match"])

    def test_lock_parser_requires_exact_unique_normalized_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "requirements.txt"
            lock.write_text("Demo_Pkg==1.0\ndemo-pkg==1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate normalized"):
                MODULE._locked_versions(lock)
            lock.write_text("demo>=1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not one exact version pin"):
                MODULE._locked_versions(lock)

    def test_virtual_environment_must_be_run_scoped_and_active(self):
        tag = "abc1234_20260819_120000"
        expected = MODULE.PUBLICATION_VENV_ROOT / tag
        record, errors = MODULE._validate_virtual_environment(
            {"RUN_TAG": tag, "AGRIBRAIN_VENV": f".publication_venvs/{tag}"},
            prefix=expected,
            base_prefix=MODULE.ROOT / "base-python",
        )
        self.assertEqual(errors, [])
        self.assertTrue(record["run_scoped"])
        _, wrong_errors = MODULE._validate_virtual_environment(
            {"RUN_TAG": tag, "AGRIBRAIN_VENV": ".venv"},
            prefix=MODULE.ROOT / ".venv",
            base_prefix=MODULE.ROOT / "base-python",
        )
        self.assertTrue(wrong_errors)

    def _write_environment_artifact(self, root: Path, packages: list[str]) -> Path:
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
        tag = "run_1"
        (results / "artifact_manifest.json").write_text(json.dumps({
            "git_commit": commit,
            "artifact_run_tag": tag,
        }), encoding="utf-8")

        def source_record(path: Path) -> dict:
            return {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        (results / "publication_environment.json").write_text(json.dumps({
            "schema_version": 2,
            "run_tag": tag,
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
                "path_id": f".publication_venvs/{tag}",
                "isolated_from_base_prefix": True,
            },
            "installed_package_count": len(packages),
            "installed_distributions": packages,
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
            "requirements_lock": source_record(lock),
            "backend_project": source_record(pyproject),
            "publication_environment_script": source_record(env_script),
        }), encoding="utf-8")
        return results

    def test_artifact_validator_accepts_exact_environment_record(self):
        validator = _load_artifact_validator()
        with tempfile.TemporaryDirectory() as temp:
            results = self._write_environment_artifact(Path(temp), ["demo==1.0"])
            with patch.object(validator, "RESULTS_DIR", results), patch.object(
                validator, "REPO_ROOT", results.parents[2],
            ):
                validator._validate_publication_environment()

    def test_artifact_validator_rejects_duplicate_normalized_inventory(self):
        validator = _load_artifact_validator()
        with tempfile.TemporaryDirectory() as temp:
            results = self._write_environment_artifact(
                Path(temp), ["demo==1.0", "demo==1.0"]
            )
            with patch.object(validator, "RESULTS_DIR", results), patch.object(
                validator, "REPO_ROOT", results.parents[2],
            ):
                with self.assertRaises(SystemExit):
                    validator._validate_publication_environment()


if __name__ == "__main__":
    unittest.main()
