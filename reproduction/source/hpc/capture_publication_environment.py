#!/usr/bin/env python3
"""Validate and record the version-resolved publication Python environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "mvp" / "simulation" / "results"
LOCK = ROOT / "agribrain" / "backend" / "requirements-lock.txt"
PYPROJECT = ROOT / "agribrain" / "backend" / "pyproject.toml"
ENV_SCRIPT = ROOT / "hpc" / "publication_env.sh"
PUBLICATION_VENV_ROOT = ROOT / ".publication_venvs"
BOOTSTRAP_DISTRIBUTIONS = frozenset({"pip", "setuptools", "wheel"})
_NORMALIZE_RE = re.compile(r"[-_.]+")
sys.path.insert(0, str(ROOT / "hpc"))
from validate_publication_env import (  # noqa: E402
    EXPECTED, MUST_BE_UNSET, errors_for_environment, interpreter_error,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_distribution_name(name: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return _NORMALIZE_RE.sub("-", str(name).strip()).lower()


def _installed_distribution_pairs() -> list[tuple[str, str]]:
    """Return every installed distribution without masking duplicates."""
    pairs = []
    for dist in importlib.metadata.distributions():
        name = str(dist.metadata.get("Name") or dist.name or "").strip()
        version = str(dist.version or "").strip()
        pairs.append((name, version))
    return pairs


def _installed_distributions() -> list[str]:
    """Return a normalized, versioned, path-free distribution inventory."""
    return sorted(
        f"{_normalize_distribution_name(name)}=={version}"
        for name, version in _installed_distribution_pairs()
    )


def _locked_versions(lock: Path = LOCK) -> dict[str, str]:
    """Parse applicable exact pins from the tracked requirements lock."""
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:  # pragma: no cover - publication lock includes it
        raise RuntimeError("packaging is required to validate requirements-lock.txt") from exc

    versions: dict[str, str] = {}
    for line_number, raw in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except Exception as exc:
            raise RuntimeError(f"invalid lock requirement at line {line_number}: {line!r}") from exc
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if requirement.url is not None:
            raise RuntimeError(f"lock requirement must not use a URL: {line!r}")
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise RuntimeError(f"lock requirement is not one exact version pin: {line!r}")
        normalized = _normalize_distribution_name(requirement.name)
        version = specifiers[0].version
        if normalized in versions:
            raise RuntimeError(f"duplicate normalized lock distribution: {normalized}")
        versions[normalized] = version
    if not versions:
        raise RuntimeError("requirements lock contains no applicable exact pins")
    return versions


def _core_identity(pyproject: Path = PYPROJECT) -> tuple[str, str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
    name = _normalize_distribution_name(project.get("name", ""))
    version = str(project.get("version", "")).strip()
    if not name or not version:
        raise RuntimeError("pyproject.toml lacks project name/version")
    return name, version


def _validate_distribution_set(
    installed_pairs: list[tuple[str, str]],
    locked_versions: dict[str, str],
    core_identity: tuple[str, str],
) -> tuple[dict, list[str]]:
    """Validate uniqueness, every lock pin, the core package, and extras."""
    errors: list[str] = []
    installed: dict[str, list[str]] = {}
    for raw_name, raw_version in installed_pairs:
        name = _normalize_distribution_name(raw_name)
        version = str(raw_version).strip()
        if not name or not version:
            errors.append(f"installed distribution lacks name/version: {(raw_name, raw_version)!r}")
            continue
        installed.setdefault(name, []).append(version)

    duplicates = {
        name: versions for name, versions in installed.items() if len(versions) != 1
    }
    if duplicates:
        errors.append(
            "duplicate normalized installed distributions: "
            + ", ".join(
                f"{name}={versions}" for name, versions in sorted(duplicates.items())
            )
        )

    core_name, core_version = core_identity
    expected = dict(locked_versions)
    if core_name in expected and expected[core_name] != core_version:
        errors.append(
            f"core distribution {core_name} conflicts with lock: "
            f"pyproject={core_version}, lock={expected[core_name]}"
        )
    expected[core_name] = core_version

    missing = []
    mismatched = []
    for name, wanted in sorted(expected.items()):
        actual = installed.get(name, [])
        if not actual:
            missing.append(name)
        elif len(actual) == 1 and actual[0] != wanted:
            mismatched.append(f"{name}: expected {wanted}, got {actual[0]}")
    if missing:
        errors.append("missing locked/core distributions: " + ", ".join(missing))
    if mismatched:
        errors.append("distribution version mismatches: " + "; ".join(mismatched))

    unexpected = sorted(set(installed).difference(expected).difference(BOOTSTRAP_DISTRIBUTIONS))
    if unexpected:
        errors.append("unexpected distributions outside lock/core: " + ", ".join(unexpected))

    core_actual = installed.get(core_name, [])
    validation = {
        "normalization": "PEP 503 (lowercase; runs of '-', '_', '.' become '-')",
        "unique_normalized_names": not duplicates,
        "lock_versions_match": not missing and not mismatched,
        "core_version_match": core_actual == [core_version],
        "core_distribution": f"{core_name}=={core_version}",
        "locked_distribution_count": len(locked_versions),
        "applicable_lock_distributions": [
            f"{name}=={version}"
            for name, version in sorted(locked_versions.items())
        ],
        "unexpected_distributions": unexpected,
        "allowed_bootstrap_distributions": sorted(
            name for name in installed if name in BOOTSTRAP_DISTRIBUTIONS
        ),
    }
    return validation, errors


def _validate_virtual_environment(
    environ: dict[str, str],
    *,
    prefix: str | Path | None = None,
    base_prefix: str | Path | None = None,
) -> tuple[dict, list[str]]:
    """Require the run-scoped venv exported by the Slurm orchestrator."""
    errors: list[str] = []
    raw = environ.get("AGRIBRAIN_VENV", "").strip()
    run_tag = environ.get("RUN_TAG", "").strip()
    expected_path: Path | None = None
    if not raw:
        errors.append("AGRIBRAIN_VENV is not set")
    else:
        candidate = Path(raw)
        expected_path = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
        if expected_path.parent != PUBLICATION_VENV_ROOT.resolve():
            errors.append(
                f"AGRIBRAIN_VENV must be a direct child of {PUBLICATION_VENV_ROOT}"
            )
        if run_tag and expected_path.name != run_tag:
            errors.append(
                f"AGRIBRAIN_VENV name {expected_path.name!r} does not match RUN_TAG {run_tag!r}"
            )
    if not run_tag:
        errors.append("RUN_TAG is not set")

    actual_prefix = Path(prefix or sys.prefix).resolve()
    actual_base = Path(base_prefix or sys.base_prefix).resolve()
    if actual_prefix == actual_base:
        errors.append("publication Python is not running inside a virtual environment")
    if expected_path is not None and actual_prefix != expected_path:
        errors.append(
            f"active Python prefix {actual_prefix} does not equal AGRIBRAIN_VENV {expected_path}"
        )

    record = {
        "run_scoped": bool(expected_path and run_tag and expected_path.name == run_tag),
        "path_id": (
            f".publication_venvs/{expected_path.name}" if expected_path is not None else None
        ),
        "isolated_from_base_prefix": actual_prefix != actual_base,
    }
    return record, errors


def _validated_snapshot() -> tuple[dict, list[str]]:
    errors = errors_for_environment(dict(os.environ))
    version_error = interpreter_error()
    if version_error:
        errors.append(version_error)
    for path, label in (
        (LOCK, "requirements lock"),
        (PYPROJECT, "backend pyproject"),
        (ENV_SCRIPT, "publication environment script"),
    ):
        if not path.is_file():
            errors.append(f"{label} missing: {path}")

    commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        errors.append("AGRIBRAIN_GIT_COMMIT must be a full lowercase SHA-1")

    venv_record, venv_errors = _validate_virtual_environment(dict(os.environ))
    errors.extend(venv_errors)

    distribution_validation: dict = {}
    distributions: list[str] = []
    if LOCK.is_file() and PYPROJECT.is_file():
        try:
            pairs = _installed_distribution_pairs()
            distributions = sorted(
                f"{_normalize_distribution_name(name)}=={version}"
                for name, version in pairs
            )
            locked = _locked_versions()
            core = _core_identity()
            distribution_validation, distribution_errors = _validate_distribution_set(
                pairs, locked, core
            )
            errors.extend(distribution_errors)
        except Exception as exc:  # noqa: BLE001 - publication check fails closed
            errors.append(f"cannot validate installed distributions: {exc}")
    if not distributions:
        errors.append("installed distribution inventory is empty")

    snapshot = {
        "commit": commit,
        "virtual_environment": venv_record,
        "distribution_validation": distribution_validation,
        "installed_distributions": distributions,
    }
    return snapshot, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the active run-scoped venv without writing an artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write the receipt to this path instead of the core results tree. "
            "Structural-sensitivity runs use an external run directory."
        ),
    )
    args = parser.parse_args(argv)
    snapshot, errors = _validated_snapshot()
    if errors:
        raise RuntimeError("non-canonical Python environment: " + "; ".join(errors))
    if args.validate_only:
        print("Canonical locked Python environment OK")
        return 0

    distributions = snapshot["installed_distributions"]
    payload = {
        "schema_version": 2,
        "environment_scope": "version_resolved_runtime_inventory",
        "binary_reproducibility": {
            "byte_identical_environment_claimed": False,
            "distribution_artifact_hashes_recorded": False,
            "container_image_digest_recorded": False,
            "interpretation": (
                "The receipt pins Python 3.11, source files, environment "
                "variables, and installed distribution versions; it does not "
                "claim byte-identical wheels, BLAS binaries, or a container image."
            ),
        },
        "run_tag": os.environ.get("RUN_TAG", "").strip(),
        "git_commit": snapshot["commit"],
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "version_detail": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "virtual_environment": snapshot["virtual_environment"],
        "canonical_environment": {
            name: os.environ.get(name) for name in sorted(EXPECTED)
        },
        "verified_unset_variables": sorted(MUST_BE_UNSET),
        "requirements_lock": {
            "path": LOCK.relative_to(ROOT).as_posix(),
            "bytes": LOCK.stat().st_size,
            "sha256": _sha256(LOCK),
        },
        "backend_project": {
            "path": PYPROJECT.relative_to(ROOT).as_posix(),
            "bytes": PYPROJECT.stat().st_size,
            "sha256": _sha256(PYPROJECT),
        },
        "publication_environment_script": {
            "path": ENV_SCRIPT.relative_to(ROOT).as_posix(),
            "bytes": ENV_SCRIPT.stat().st_size,
            "sha256": _sha256(ENV_SCRIPT),
        },
        "distribution_validation": snapshot["distribution_validation"],
        "installed_package_count": len(distributions),
        "installed_distributions": distributions,
    }
    out = args.output.resolve() if args.output else RESULTS_DIR / "publication_environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(out)
    print(f"Saved publication environment: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
