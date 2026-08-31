#!/usr/bin/env python3
"""Shared fail-closed gate for deterministic publication recovery.

Fresh stochastic publication does not enter this module.  Recovery is active
only when its explicit authorization receipt is supplied, and it is accepted
only when that receipt, the original submission receipt, and the preserved raw
manifest occupy their canonical run-scoped paths and remain byte-identical to
the receipt bindings.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
RECOVERY_RECEIPT_ENV = "AGRIBRAIN_RECOVERY_RECEIPT"
SIMULATION_COMMIT_ENV = "AGRIBRAIN_SIMULATION_COMMIT"
PUBLICATION_COMMIT_ENV = "AGRIBRAIN_PUBLICATION_CODE_COMMIT"


def _read_stable_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read: {path}")
    if not payload:
        raise ValueError(f"{label} must not be empty: {path}")
    return payload


def _require_commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA-1")
    return value


def _exact_run_path(results_dir: Path, directory: str, run_tag: str) -> Path:
    return results_dir / directory / f"{run_tag}.json"


def _require_exact_path(actual: Path, expected: Path, *, label: str) -> Path:
    actual_unresolved = actual.absolute()
    expected_unresolved = expected.absolute()
    if actual_unresolved.is_symlink() or expected_unresolved.is_symlink():
        raise ValueError(f"{label} must use a non-symlink canonical path")
    try:
        resolved = actual_unresolved.resolve(strict=True)
        expected_resolved = expected_unresolved.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist at its canonical path") from exc
    if resolved != expected_resolved:
        raise ValueError(f"{label} must use the canonical run-scoped path")
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def current_checkout_commit(repo_root: Path) -> str:
    """Return the executing checkout's full commit or fail closed."""

    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root.resolve(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot establish the publication-recovery checkout commit"
        ) from exc
    return _require_commit(value, label="publication checkout commit")


def current_checkout_tree(repo_root: Path) -> str:
    """Return the executing checkout's exact HEAD tree or fail closed."""

    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repo_root.resolve(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot establish the publication-recovery checkout tree"
        ) from exc
    return _require_commit(value, label="publication checkout tree")


def validate_recovery_context(
    receipt_path: Path,
    *,
    results_dir: Path,
    run_tag: str,
    simulation_commit: str,
    publication_commit: str,
    expected_kind: str = "core",
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Validate and byte-bind one recovery authorization to its artifacts."""

    simulation_commit = _require_commit(
        simulation_commit, label="simulation source commit"
    )
    publication_commit = _require_commit(
        publication_commit, label="publication code commit"
    )
    if simulation_commit == publication_commit:
        raise ValueError(
            "recovery mode requires distinct simulation and publication commits"
        )
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("recovery mode requires a non-empty run tag")

    results_dir = results_dir.resolve(strict=True)
    canonical_receipt = _exact_run_path(
        results_dir, "publication_recovery_receipts", run_tag,
    )
    receipt = _require_exact_path(
        receipt_path, canonical_receipt,
        label="publication-recovery receipt",
    )
    original = _require_exact_path(
        _exact_run_path(results_dir, "core_submission_receipts", run_tag),
        _exact_run_path(results_dir, "core_submission_receipts", run_tag),
        label="original submission receipt",
    )
    raw_manifest = _require_exact_path(
        _exact_run_path(results_dir, "preserved_raw_manifests", run_tag),
        _exact_run_path(results_dir, "preserved_raw_manifests", run_tag),
        label="preserved raw-output manifest",
    )

    # Import lazily so the normal fresh path retains its existing dependency
    # surface.  The authoritative validator checks the complete receipt schema,
    # self-hash, failed-publisher evidence, original receipt, and commit/run
    # bindings; this wrapper adds the canonical artifact-path and literal-byte
    # checks needed by manifest consumers.
    from hpc.publication_recovery_receipt import (  # pylint: disable=import-outside-toplevel
        validate_recovery_receipt_file,
    )

    validated = validate_recovery_receipt_file(
        receipt,
        original_receipt_path=original,
        expected_kind=expected_kind,
        expected_run_tag=run_tag,
        expected_simulation_commit=simulation_commit,
        expected_publication_commit=publication_commit,
    )
    if validated.get("simulation_rerun") is not False:
        raise ValueError(
            "recovery receipt must explicitly state simulation_rerun=false"
        )

    receipt_bytes = _read_stable_regular(
        receipt, label="publication-recovery receipt"
    )
    raw_bytes = _read_stable_regular(
        raw_manifest, label="preserved raw-output manifest"
    )
    raw_binding = validated.get("preserved_raw_outputs")
    if not isinstance(raw_binding, dict):
        raise ValueError("recovery receipt lacks the preserved-raw binding")
    if raw_binding.get("file") != raw_manifest.name:
        raise ValueError("recovery receipt names the wrong preserved raw manifest")
    if raw_binding.get("bytes") != len(raw_bytes) or raw_binding.get(
        "literal_sha256"
    ) != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("preserved raw manifest differs from recovery receipt")

    source_identity = validated.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("recovery receipt lacks publication source identity")
    publication_tree = _require_commit(
        source_identity.get("publication_repair_tree"),
        label="authorized publication-repair tree",
    )
    if repo_root is not None:
        head = current_checkout_commit(repo_root)
        if head != publication_commit:
            raise ValueError(
                "publication recovery commit differs from the executing checkout"
            )
        if current_checkout_tree(repo_root) != publication_tree:
            raise ValueError(
                "publication recovery tree differs from the executing checkout"
            )

    return {
        "receipt_file": receipt.relative_to(results_dir).as_posix(),
        "receipt_literal_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_self_hash": validated.get("receipt_sha256"),
        "preserved_raw_manifest_file": raw_manifest.relative_to(
            results_dir
        ).as_posix(),
        "preserved_raw_manifest_literal_sha256": hashlib.sha256(
            raw_bytes
        ).hexdigest(),
        "preserved_raw_payload_merkle_root": raw_binding.get(
            "payload_merkle_root"
        ),
        "original_submission_receipt_file": original.relative_to(
            results_dir
        ).as_posix(),
        "simulation_rerun": False,
        "publication_repair_tree": publication_tree,
        "validated": True,
    }


def recovery_context_from_environment(
    *,
    results_dir: Path,
    repo_root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    """Resolve recovery only from a complete explicit environment contract."""

    env = os.environ if environ is None else environ
    receipt_raw = str(env.get(RECOVERY_RECEIPT_ENV, "")).strip()
    simulation_commit = str(env.get(SIMULATION_COMMIT_ENV, "")).strip()
    publication_commit = str(env.get(PUBLICATION_COMMIT_ENV, "")).strip()
    requested = bool(receipt_raw or simulation_commit or publication_commit)
    if not requested:
        return None
    if not (receipt_raw and simulation_commit and publication_commit):
        raise ValueError(
            "recovery mode requires AGRIBRAIN_RECOVERY_RECEIPT, "
            "AGRIBRAIN_SIMULATION_COMMIT, and "
            "AGRIBRAIN_PUBLICATION_CODE_COMMIT together"
        )
    run_tag = str(
        env.get("ARTIFACT_RUN_TAG", "") or env.get("RUN_TAG", "")
    ).strip()
    artifact_run_tag = str(env.get("ARTIFACT_RUN_TAG", "")).strip()
    publisher_run_tag = str(env.get("RUN_TAG", "")).strip()
    if artifact_run_tag and publisher_run_tag and artifact_run_tag != publisher_run_tag:
        raise ValueError("ARTIFACT_RUN_TAG and RUN_TAG disagree in recovery mode")
    legacy_commit = str(env.get("AGRIBRAIN_GIT_COMMIT", "")).strip()
    if legacy_commit and legacy_commit != simulation_commit:
        raise ValueError(
            "AGRIBRAIN_GIT_COMMIT must remain the simulation commit in recovery mode"
        )
    # The recovery launcher exports the receipt path relative to the
    # repository root, while some callers (build_artifact_manifest,
    # export stages) run from mvp/simulation; resolve against repo_root
    # so the canonical-path comparison cannot depend on the caller's
    # working directory.
    receipt_arg = Path(receipt_raw)
    if not receipt_arg.is_absolute():
        receipt_arg = repo_root / receipt_arg
    return {
        "simulation_source_commit": simulation_commit,
        "publication_code_commit": publication_commit,
        "dual_provenance": True,
        "recovery_authorization": validate_recovery_context(
            receipt_arg,
            results_dir=results_dir,
            run_tag=run_tag,
            simulation_commit=simulation_commit,
            publication_commit=publication_commit,
            expected_kind="core",
            repo_root=repo_root,
        ),
    }
