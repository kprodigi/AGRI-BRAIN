# backend/src/routers/results.py
"""
API endpoints for running simulations and serving generated figures.

POST /results/generate  — kicks off simulation in background, returns job ID
GET  /results/status     — poll for completion
GET  /results/summary    — fetch last completed summary
GET  /results/figures/{filename} — serves a generated figure file
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from ..models.mode_capabilities import (
    PRIMARY_MODES,
)
from ..models.mode_capabilities import (
    PUBLICATION_BENCHMARK_MODES as PUBLICATION_MODES,
)
from ..models.mode_capabilities import (
    SECONDARY_ABLATION_MODES as SECONDARY_MODES,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Ensure the simulation module is importable
# ---------------------------------------------------------------------------
# Validator authority is anchored to the source tree that contains this router.
# SIM_DIR may be present for legacy deployments, but it is accepted only when
# it resolves to this same canonical simulation directory; it can never select
# a second repository whose validators would then bless that repository.
_TRUSTED_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_CANONICAL_SIM_DIR = _TRUSTED_REPO_ROOT / "mvp" / "simulation"
_configured_sim_dir = os.environ.get("SIM_DIR", "").strip()
if _configured_sim_dir:
    try:
        if Path(_configured_sim_dir).resolve(strict=True) != _CANONICAL_SIM_DIR.resolve(
            strict=True
        ):
            raise RuntimeError(
                "SIM_DIR must resolve to the simulation directory in the "
                "currently executing AGRI-BRAIN source tree"
            )
    except OSError as exc:
        raise RuntimeError("SIM_DIR cannot be resolved safely") from exc
_SIM_DIR = _CANONICAL_SIM_DIR
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

_RESULTS_DIR = _SIM_DIR / "results"
_DEVELOPMENT_RESULTS_DIR = _SIM_DIR / "development_results"

# Background job state (guarded by _JOB_LOCK for thread safety)
_JOB_LOCK = threading.Lock()
_JOB = {"running": False, "started_at": None, "finished_at": None,
        "error": None, "summary": None, "artifacts": None}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_TAG = re.compile(r"^([0-9a-f]{7})_[0-9]{8}_[0-9]{6}$")
_VALIDATION_RECEIPT = "publication_validation_receipt.json"


@dataclass(frozen=True)
class _VerifiedPublicationArtifact:
    """Immutable response bytes captured by the integrity check itself."""

    content: bytes
    sha256: str


@dataclass(frozen=True)
class _PublicationVerificationCache:
    results_root: str
    manifest_sha256: str
    source_commit: str
    payload_metadata: tuple[tuple[object, ...], ...]


_PUBLICATION_CACHE_LOCK = threading.RLock()
_PUBLICATION_CACHE: _PublicationVerificationCache | None = None


def _clear_publication_verification_cache() -> None:
    """Test/startup hook; a normal cache invalidates from file metadata."""

    global _PUBLICATION_CACHE
    with _PUBLICATION_CACHE_LOCK:
        _PUBLICATION_CACHE = None


def _safe_manifest_payload(filename: str) -> Path:
    if not filename or "\\" in filename:
        raise HTTPException(status_code=503, detail="Publication manifest contains an unsafe path")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise HTTPException(status_code=503, detail="Publication manifest contains an unsafe path")
    base = _RESULTS_DIR.resolve()
    source = _RESULTS_DIR.joinpath(*relative.parts)
    cursor = source
    while cursor != _RESULTS_DIR:
        if cursor.is_symlink():
            raise HTTPException(
                status_code=503,
                detail=f"Manifested artifact traverses a symlink: {filename}",
            )
        cursor = cursor.parent
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"Manifested artifact is missing: {filename}",
        ) from exc
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Manifested artifact escapes results or is irregular: {filename}",
        )
    return resolved


def _manifest_payload_metadata(
    records: dict[str, dict],
    published: frozenset[str],
) -> tuple[tuple[object, ...], ...]:
    """Cheap all-payload mutation fingerprint used after one full audit.

    Size, inode/device, mode, mtime, and ctime are included. Requested bytes
    are still SHA-256 checked on every response; any metadata change anywhere
    invalidates the cache and triggers the complete semantic audit again.
    """

    snapshot: list[tuple[object, ...]] = []
    for filename in sorted(published):
        path = _safe_manifest_payload(filename)
        info = path.stat(follow_symlinks=False)
        snapshot.append((
            filename,
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_mode),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(info.st_ctime_ns),
        ))
    return tuple(snapshot)


def _manifest_payload_snapshot(
    records: dict[str, dict],
    published: frozenset[str],
) -> dict[str, tuple[int, str]]:
    """Hash every published payload through the shared safe-path boundary."""

    snapshot: dict[str, tuple[int, str]] = {}
    for filename in sorted(published):
        record = records.get(filename)
        if not isinstance(filename, str) or not isinstance(record, dict):
            raise HTTPException(status_code=503, detail="Publication manifest record is invalid")
        expected_size = record.get("bytes")
        expected_digest = record.get("sha256")
        if (
            not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_digest, str)
            or not _HEX64.fullmatch(expected_digest)
        ):
            raise HTTPException(status_code=503, detail="Publication manifest record is invalid")
        source = _safe_manifest_payload(filename)
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        actual = (size, digest.hexdigest())
        if actual != (expected_size, expected_digest):
            raise HTTPException(
                status_code=503,
                detail=f"Publication artifact failed integrity verification: {filename}",
            )
        snapshot[filename] = actual
    return snapshot


def _current_source_commit() -> str:
    """Return the identity of the code currently serving publication files.

    A packaged deployment without ``.git`` must set ``AGRIBRAIN_GIT_COMMIT``.
    When git is available, an environment override is accepted only if it
    agrees with ``HEAD``.  Evidence paths are exempt from the cleanliness scan
    below so that a local pipeline run does not disable the endpoint;
    :func:`_require_published_evidence` audits the specific files a response
    depends on instead, which is what stops an unaudited results directory from
    being served merely because it is internally consistent.
    """
    declared = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip().lower()
    if declared and not _HEX40.fullmatch(declared):
        raise HTTPException(
            status_code=503,
            detail="AGRIBRAIN_GIT_COMMIT is not a concrete 40-character commit",
        )

    repo_root = _TRUSTED_REPO_ROOT.resolve()
    checked_out = ""
    try:
        checked_out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().lower()
    except (OSError, subprocess.SubprocessError):
        pass
    if checked_out and not _HEX40.fullmatch(checked_out):
        checked_out = ""
    if declared and checked_out and declared != checked_out:
        raise HTTPException(
            status_code=503,
            detail="Declared AGRIBRAIN_GIT_COMMIT differs from the checked-out code",
        )
    if checked_out:
        try:
            porcelain = subprocess.check_output(
                [
                    "git", "status", "--porcelain=v1", "-z",
                    "--untracked-files=all",
                ],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Cannot verify that the serving source checkout is clean",
            ) from exc
        tokens = porcelain.split("\0")
        dirty_paths: list[str] = []
        index = 0
        while index < len(tokens):
            entry = tokens[index]
            index += 1
            if not entry:
                continue
            if len(entry) < 4:
                dirty_paths.append(entry)
                continue
            status = entry[:2]
            paths = [entry[3:].replace("\\", "/")]
            if ("R" in status or "C" in status) and index < len(tokens):
                paths.append(tokens[index].replace("\\", "/"))
                index += 1
            if any(
                path != "mvp/simulation/results"
                and not path.startswith("mvp/simulation/results/")
                for path in paths
            ):
                dirty_paths.extend(paths)
        if dirty_paths:
            raise HTTPException(
                status_code=503,
                detail="Serving source checkout has uncommitted non-result changes",
            )
    current = checked_out or declared
    if not current:
        raise HTTPException(
            status_code=503,
            detail=(
                "Current source identity is unavailable; set AGRIBRAIN_GIT_COMMIT "
                "for deployments without git metadata"
            ),
        )
    return current


def _require_published_evidence(*relative_names: str) -> None:
    """Require each evidence file to be committed at ``HEAD``, unmodified.

    ``_current_source_commit`` exempts ``mvp/simulation/results`` from its
    cleanliness scan, because a local pipeline run writes there and must not
    take the endpoint down.  That exemption is also the one route by which an
    unaudited tree could reach this point: a loose but self-consistent results
    directory dropped into an otherwise clean checkout.  Requiring the manifest,
    the receipts and the requested payload to be tracked at ``HEAD`` with
    exactly their committed bytes closes that route per file, and -- unlike
    pinning the checkout to the commit that produced the evidence -- remains
    satisfiable once the source tree moves on.

    Git can only speak for files inside the repository it is asked about, so
    two situations skip the audit rather than fail it: a deployment packaged
    without git metadata, and one that mounts its evidence outside the serving
    checkout.  In both the manifest hash chain is the whole guarantee, as it
    already is for the ``AGRIBRAIN_GIT_COMMIT`` identity path.
    """
    repo_root = _TRUSTED_REPO_ROOT.resolve()
    try:
        evidence_reldir = _RESULTS_DIR.resolve().relative_to(repo_root).as_posix()
    except (OSError, ValueError):
        return

    def _git(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *argv],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        if _git("rev-parse", "--git-dir").returncode != 0:
            return
    except (OSError, subprocess.SubprocessError):
        return

    for name in relative_names:
        path = f"{evidence_reldir}/{name}"
        try:
            listed = _git("ls-tree", "-z", "HEAD", "--", path)
            modified = _git("diff", "--quiet", "HEAD", "--", path)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot verify that publication evidence is committed: {name}",
            ) from exc
        if listed.returncode != 0 or not listed.stdout.strip():
            raise HTTPException(
                status_code=503,
                detail=f"Publication evidence is not committed in the serving checkout: {name}",
            )
        # ``--quiet`` exits 1 on a difference; anything else is a failed audit.
        if modified.returncode != 0:
            raise HTTPException(
                status_code=503,
                detail=f"Publication evidence differs from the serving commit: {name}",
            )


def _published_evidence_subset(records: dict[str, dict]) -> frozenset[str]:
    """The manifested artifacts this checkout actually publishes.

    Tracking, not mere presence, defines the subset: a committed artifact must
    be on disk and must hash-match, while one that was deliberately left to the
    deposit is simply not servable here.  Deleting a committed artifact
    therefore fails the audit rather than quietly shrinking the set.

    Where git cannot answer -- a packaged deployment, or evidence mounted
    outside the checkout -- presence on disk defines it instead, which is the
    same tolerance ``verify_manifest.py --allow-missing`` applies.
    """
    repo_root = _TRUSTED_REPO_ROOT.resolve()
    try:
        evidence_reldir = _RESULTS_DIR.resolve().relative_to(repo_root).as_posix()
    except (OSError, ValueError):
        evidence_reldir = ""

    tracked: set[str] | None = None
    if evidence_reldir:
        try:
            listed = subprocess.run(
                ["git", "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", evidence_reldir],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            listed = None
        if listed is not None and listed.returncode == 0:
            prefix = evidence_reldir + "/"
            tracked = {
                entry[len(prefix):]
                for entry in listed.stdout.split("\x00")
                if entry.startswith(prefix)
            }

    if tracked is not None:
        return frozenset(name for name in records if name in tracked)
    return frozenset(
        name for name in records if (_RESULTS_DIR / PurePosixPath(name)).is_file()
    )


def _artifact_set_root(records: list[dict]) -> str:
    leaves = [
        hashlib.sha256(json.dumps(
            {
                "file": str(record["file"]),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).digest()
        for record in sorted(records, key=lambda item: str(item["file"]))
        if record.get("file") != _VALIDATION_RECEIPT
    ]
    if not leaves:
        return "0" * 64
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def _canonical_recovery_receipt_path(
    manifest: dict,
    records: dict[str, dict],
    *,
    run_tag: str,
    simulation_commit: str,
    publication_commit: str,
) -> Path:
    """Resolve only the run-scoped evidence for an authorized core recovery."""

    match = _RUN_TAG.fullmatch(run_tag)
    if match is None or match.group(1) != simulation_commit[:7]:
        raise HTTPException(
            status_code=503,
            detail="Recovery publication run tag is unsafe or not simulation-bound",
        )
    if simulation_commit == publication_commit:
        raise HTTPException(
            status_code=503,
            detail="Recovery publication requires distinct simulation and publication commits",
        )
    authorization = manifest.get("recovery_authorization")
    if not isinstance(authorization, dict):
        raise HTTPException(
            status_code=503,
            detail="Recovery publication lacks a complete authorization",
        )
    expected = {
        "receipt_file": f"publication_recovery_receipts/{run_tag}.json",
        "preserved_raw_manifest_file": f"preserved_raw_manifests/{run_tag}.json",
        "original_submission_receipt_file": f"core_submission_receipts/{run_tag}.json",
    }
    if (
        authorization.get("validated") is not True
        or authorization.get("simulation_rerun") is not False
        or any(authorization.get(key) != value for key, value in expected.items())
    ):
        raise HTTPException(
            status_code=503,
            detail="Recovery publication authorization is incomplete or noncanonical",
        )
    for label, relative in expected.items():
        if relative not in records:
            raise HTTPException(
                status_code=503,
                detail=f"Recovery publication does not manifest {label}",
            )
        # The relative name is constructed from the validated run tag rather
        # than accepted from the authorization.  _safe_manifest_payload then
        # rejects a symlink at the leaf or in any parent and confines the file
        # to the canonical results root.
        _safe_manifest_payload(relative)
    return _safe_manifest_payload(expected["receipt_file"])


def _validate_recovery_authorization(
    manifest: dict,
    recovery_receipt: Path,
    *,
    run_tag: str,
    simulation_commit: str,
    publication_commit: str,
) -> dict[str, object]:
    """Independently reproduce and exactly match the manifest authorization."""

    try:
        repo_root = _TRUSTED_REPO_ROOT.resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from mvp.simulation.analysis.recovery_provenance import (
            validate_recovery_context,
        )

        validated = validate_recovery_context(
            recovery_receipt,
            results_dir=_RESULTS_DIR,
            run_tag=run_tag,
            simulation_commit=simulation_commit,
            publication_commit=publication_commit,
            expected_kind="core",
        )
    except (ImportError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Recovery publication authorization failed: {exc}",
        ) from exc
    if (
        validated != manifest.get("recovery_authorization")
        or validated.get("validated") is not True
        or validated.get("simulation_rerun") is not False
    ):
        raise HTTPException(
            status_code=503,
            detail="Recovery publication authorization does not match its evidence",
        )
    return validated


def _validate_canonical_release_contract(
    recovery_receipt: Path | None = None,
) -> None:
    """Run the same exact-inventory receipt verifier used by packaging."""

    repo_root = _TRUSTED_REPO_ROOT.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from mvp.simulation.validation.validate_publication_artifacts import (
            validate_full_publication_release,
        )

        validate_full_publication_release(
            _RESULTS_DIR.resolve(),
            repo_root=repo_root,
            recovery_receipt=recovery_receipt,
        )
    except (ImportError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Canonical publication evidence contract failed: {exc}",
        ) from exc


def _validate_semantic_receipt(
    manifest: dict,
    records: dict[str, dict],
    simulation_commit: str,
    publication_commit: str,
    run_tag: str,
    *,
    recovery_authorization: dict[str, object] | None = None,
) -> None:
    record = records.get(_VALIDATION_RECEIPT)
    if record is None:
        raise HTTPException(
            status_code=503,
            detail="Hash-bound semantic publication validation is unavailable",
        )
    path = _safe_manifest_payload(_VALIDATION_RECEIPT)
    payload = path.read_bytes()
    if (
        record.get("bytes") != len(payload)
        or record.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise HTTPException(
            status_code=503,
            detail="Semantic validation receipt failed manifest integrity",
        )
    try:
        receipt = json.loads(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail="Semantic validation receipt is invalid",
        ) from exc
    common_invalid = (
        receipt.get("schema_version") != 1
        or receipt.get("validation_status") != "PASS"
        or receipt.get("validation_scope") != "core_publication_evidence"
        or receipt.get("git_commit") != simulation_commit
        or receipt.get("simulation_source_commit") != simulation_commit
        or receipt.get("publication_code_commit") != publication_commit
        or receipt.get("run_tag") != run_tag
    )
    if recovery_authorization is None:
        provenance_invalid = (
            receipt.get("fresh_single_commit_run") is not True
            or simulation_commit != publication_commit
        )
    else:
        provenance_invalid = (
            receipt.get("fresh_single_commit_run") is not False
            or receipt.get("authorized_deterministic_recovery") is not True
            or receipt.get("simulation_rerun") is not False
            or receipt.get("recovery_authorization") != recovery_authorization
            or simulation_commit == publication_commit
        )
    if common_invalid or provenance_invalid:
        raise HTTPException(
            status_code=503,
            detail="Semantic validation receipt does not match this release",
        )
    protocol_path = _SIM_DIR / "experiment_protocol.json"
    protocol = receipt.get("protocol")
    if not protocol_path.is_file() or not isinstance(protocol, dict):
        raise HTTPException(status_code=503, detail="Locked protocol is unavailable")
    protocol_bytes = protocol_path.read_bytes()
    if (
        protocol.get("file") != "mvp/simulation/experiment_protocol.json"
        or protocol.get("bytes") != len(protocol_bytes)
        or protocol.get("sha256") != hashlib.sha256(protocol_bytes).hexdigest()
    ):
        raise HTTPException(
            status_code=503,
            detail="Semantic validation receipt does not match the locked protocol",
        )
    artifact_set = receipt.get("semantic_artifact_set")
    semantic_records = [
        item for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("file") != _VALIDATION_RECEIPT
    ]
    if (
        not isinstance(artifact_set, dict)
        or artifact_set.get("artifact_count_excluding_receipt")
        != len(semantic_records)
        or artifact_set.get("merkle_root")
        != _artifact_set_root(manifest.get("artifacts", []))
    ):
        raise HTTPException(
            status_code=503,
            detail="Semantic validation receipt does not bind this artifact set",
        )
    if receipt.get("locked_accounting") != {
        "core_unique_retained_cells": 1_600,
        "core_executed_episodes": 6_100,
        "core_simulated_steps": 1_756_800,
        "h1_directional_tests": 5,
        "h2_directional_tests": 20,
        "h3_equivalence_cells": 25,
    }:
        raise HTTPException(status_code=503, detail="Semantic accounting is invalid")
    structural = receipt.get("structural_sensitivity")
    if not isinstance(structural, dict) or (
        structural.get("included_in_core_receipt") is not False
        or structural.get("required_for_full_submission_evidence") is not True
    ):
        raise HTTPException(
            status_code=503,
            detail="Core evidence scope does not disclose structural sensitivity",
        )


def _publication_artifact(filename: str) -> _VerifiedPublicationArtifact:
    """Resolve only byte-verified artifacts inventoried by the manifest.

    The public endpoint deliberately fails closed.  A file merely existing in
    ``results/`` does not make it validated publication evidence.
    """
    global _PUBLICATION_CACHE

    if _RESULTS_DIR.is_symlink() or not _RESULTS_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail="Publication results root is unavailable or unsafe",
        )
    manifest_path = _RESULTS_DIR / "artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise HTTPException(status_code=503, detail="Validated publication manifest is unavailable")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Publication manifest is invalid") from exc
    if manifest.get("schema_version") != 2 or manifest.get("git_dirty") is not False:
        raise HTTPException(status_code=503, detail="Publication manifest is not a clean validated release")
    run_tag = manifest.get("artifact_run_tag")
    identity_values = {
        key: str(manifest.get(key, "")).strip().lower()
        for key in ("git_commit", "simulation_source_commit", "publication_code_commit")
    }
    identities = set(identity_values.values())
    simulation_commit = identity_values["simulation_source_commit"]
    publication_commit = identity_values["publication_code_commit"]
    dual_provenance = manifest.get("dual_provenance")
    if (
        not isinstance(run_tag, str)
        or not run_tag.strip()
        or any(not _HEX40.fullmatch(value) for value in identity_values.values())
    ):
        raise HTTPException(
            status_code=503,
            detail="Publication manifest has invalid source provenance",
        )
    if dual_provenance is False:
        if len(identities) != 1 or manifest.get("recovery_authorization") is not None:
            raise HTTPException(
                status_code=503,
                detail="Publication manifest is not a fresh single-commit evidence run",
            )
    elif dual_provenance is True:
        if (
            identity_values["git_commit"] != simulation_commit
            or simulation_commit == publication_commit
        ):
            raise HTTPException(
                status_code=503,
                detail="Recovery publication has inconsistent source provenance",
            )
    else:
        raise HTTPException(
            status_code=503,
            detail="Publication manifest has an invalid provenance mode",
        )
    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, list) or any(
        not isinstance(record, dict) or not isinstance(record.get("file"), str)
        for record in artifact_records
    ):
        raise HTTPException(status_code=503, detail="Publication manifest records are invalid")
    names = [str(record["file"]) for record in artifact_records]
    if len(names) != len(set(names)):
        raise HTTPException(status_code=503, detail="Publication manifest repeats a payload")
    records = {str(record["file"]): record for record in artifact_records}
    rec = records.get(filename)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Artifact is not in the publication manifest: {filename}")
    published = _published_evidence_subset(records)
    if filename not in published:
        raise HTTPException(
            status_code=404,
            detail=(
                "Artifact is certified but not committed to this repository; "
                f"it is in the evidence deposit: {filename}"
            ),
        )

    recovery_receipt: Path | None = None
    if dual_provenance is True:
        recovery_receipt = _canonical_recovery_receipt_path(
            manifest,
            records,
            run_tag=run_tag,
            simulation_commit=simulation_commit,
            publication_commit=publication_commit,
        )

    serving_commit = _current_source_commit()
    audited = ["artifact_manifest.json", _VALIDATION_RECEIPT, filename]
    if recovery_receipt is not None:
        audited.append(recovery_receipt.relative_to(_RESULTS_DIR).as_posix())
    _require_published_evidence(*dict.fromkeys(audited))

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    results_root = str(_RESULTS_DIR.resolve(strict=True))
    with _PUBLICATION_CACHE_LOCK:
        metadata_before = _manifest_payload_metadata(records, published)
        expected_cache = _PublicationVerificationCache(
            results_root=results_root,
            manifest_sha256=manifest_sha256,
            source_commit=serving_commit,
            payload_metadata=metadata_before,
        )
        if _PUBLICATION_CACHE != expected_cache:
            # Hash every payload before opening even the semantic receipt. This
            # rejects symlinks, path swaps, and coherent receipt edits before
            # the expensive validator is allowed to trust any result file.
            before = _manifest_payload_snapshot(records, published)
            recovery_authorization = None
            if recovery_receipt is not None:
                recovery_authorization = _validate_recovery_authorization(
                    manifest,
                    recovery_receipt,
                    run_tag=run_tag,
                    simulation_commit=simulation_commit,
                    publication_commit=publication_commit,
                )
            _validate_semantic_receipt(
                manifest,
                records,
                simulation_commit,
                publication_commit,
                run_tag,
                recovery_authorization=recovery_authorization,
            )
            # The full-tree contract replays ledger, figure, environment and
            # DAG gates that only mean anything when every manifested artifact
            # is present. A repository checkout publishes a subset on purpose --
            # the tables, statistics and receipts, so the paper's values can be
            # checked against a clone -- and leaves the 1,600 per-seed ledgers
            # and the run's own figure renders to the evidence deposit. That
            # contract is therefore unreachable here rather than being skipped:
            # what governs a subset is what CI runs on every push, which is
            # every published payload hashed against the manifest, each one
            # tracked at the serving commit, and the hash-bound semantic
            # receipt validated above.
            if len(published) == len(records):
                _validate_canonical_release_contract(recovery_receipt)
            if manifest_path.read_bytes() != manifest_bytes:
                raise HTTPException(
                    status_code=503,
                    detail="Publication manifest changed during semantic validation",
                )
            after = _manifest_payload_snapshot(records, published)
            metadata_after = _manifest_payload_metadata(records, published)
            if after != before or metadata_after != metadata_before:
                raise HTTPException(
                    status_code=503,
                    detail="Publication payload set changed during semantic validation",
                )
            _PUBLICATION_CACHE = expected_cache

    # Every response still captures and hashes the requested bytes after the
    # release-wide audit/cache decision. No later filesystem read is needed by
    # the response object, closing the path-to-response replacement race.
    path = _safe_manifest_payload(filename)
    stat_before = path.stat(follow_symlinks=False)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != rec.get("bytes") or digest != rec.get("sha256"):
        raise HTTPException(status_code=503, detail=f"Publication artifact failed integrity verification: {filename}")
    path_after = _safe_manifest_payload(filename)
    stat_after = path_after.stat(follow_symlinks=False)
    if (
        path_after != path
        or (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
            stat_before.st_ctime_ns,
        )
        != (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
            stat_after.st_ctime_ns,
        )
    ):
        raise HTTPException(
            status_code=503,
            detail=f"Publication artifact changed while being read: {filename}",
        )
    if manifest_path.read_bytes() != manifest_bytes:
        raise HTTPException(
            status_code=503,
            detail="Publication manifest changed while serving an artifact",
        )
    return _VerifiedPublicationArtifact(content=payload, sha256=digest)


def _save_development_artifacts(
    data: dict,
    summary: dict,
    seed: int | None,
    *,
    run_id: str | None = None,
) -> dict:
    """Write a local exploratory run without touching publication artifacts."""
    _DEVELOPMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = (
            f"development_{int(time.time() * 1000)}_seed_"
            f"{seed if seed is not None else 'default'}"
        )
    summary_path = _DEVELOPMENT_RESULTS_DIR / f"{run_id}_summary.json"
    table1_path = _DEVELOPMENT_RESULTS_DIR / f"{run_id}_table1.csv"
    table2_path = _DEVELOPMENT_RESULTS_DIR / f"{run_id}_table2.csv"
    envelope = {
        "evidence_status": "development_only",
        "publication_evidence": False,
        "seed": seed,
        "mode_design": {
            "total": len(PUBLICATION_MODES), "primary": list(PRIMARY_MODES),
            "secondary": list(SECONDARY_MODES),
        },
        "summary": summary,
    }
    summary_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    data["table1"].to_csv(table1_path, index=False)
    data["table2"].to_csv(table2_path, index=False)
    return {"summary": summary_path.name, "table1": table1_path.name, "table2": table2_path.name}


def _run_in_background(seed: int | None = None):
    """Worker: run simulation and save tables."""
    simulation_module = None
    original_results_dir = None
    try:
        import generate_results as simulation_module

        run_id = (
            f"development_{int(time.time() * 1000)}_seed_"
            f"{seed if seed is not None else 'default'}"
        )
        run_dir = (_DEVELOPMENT_RESULTS_DIR / "runs" / run_id).resolve()
        canonical = _RESULTS_DIR.resolve()
        if run_dir == canonical or run_dir.is_relative_to(canonical):
            raise RuntimeError(
                "development simulation output resolved inside publication results"
            )
        run_dir.mkdir(parents=True, exist_ok=False)

        # run_all writes traces, protocol records, learning trajectories,
        # tables, and decision ledgers through its module-level RESULTS_DIR.
        # Redirect that complete side-effect surface for this single worker,
        # then restore the imported module even when the run fails.
        original_results_dir = simulation_module.RESULTS_DIR
        simulation_module.RESULTS_DIR = run_dir
        data = (
            simulation_module.run_all(seed=seed)
            if seed is not None else simulation_module.run_all()
        )
        summary = simulation_module.get_summary_json(data)
        artifacts = _save_development_artifacts(
            data, summary, seed, run_id=run_id,
        )
        with _JOB_LOCK:
            _JOB["summary"] = summary
            _JOB["artifacts"] = artifacts
            _JOB["error"] = None
    except Exception as exc:
        with _JOB_LOCK:
            _JOB["error"] = str(exc)
            _JOB["summary"] = None
            _JOB["artifacts"] = None
    finally:
        if simulation_module is not None and original_results_dir is not None:
            simulation_module.RESULTS_DIR = original_results_dir
        with _JOB_LOCK:
            _JOB["running"] = False
            _JOB["finished_at"] = time.time()


# ---------------------------------------------------------------------------
# POST /results/generate — non-blocking: kicks off background job
# ---------------------------------------------------------------------------
@router.post("/generate")
def generate_results(seed: int | None = None):
    """Start one development seed: 55 endpoints, 205 episodes, 59,040 steps.

    Static executes only retained episode 3. Each of the nine learned modes
    executes adaptation episodes 0--2 plus frozen retained episode 3 in every
    scenario. This endpoint never creates publication evidence.

    Returns immediately with a job status. Poll GET /results/status for
    completion. This avoids HTTP timeouts for long-running simulations.
    """
    with _JOB_LOCK:
        if _JOB["running"]:
            elapsed = time.time() - (_JOB["started_at"] or time.time())
            return {"ok": True, "status": "running", "elapsed_s": round(elapsed, 1),
                    "evidence_status": "development_only", "publication_evidence": False}

    try:
        from generate_results import run_all  # noqa: F401 — verify import
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot import simulation module: {exc}. "
                   f"Ensure mvp/simulation/ exists relative to the backend.",
        ) from exc

    with _JOB_LOCK:
        if _JOB["running"]:
            return {"ok": True, "status": "running",
                    "evidence_status": "development_only", "publication_evidence": False}
        _JOB["running"] = True
        _JOB["started_at"] = time.time()
        _JOB["finished_at"] = None
        _JOB["error"] = None
        _JOB["summary"] = None
        _JOB["artifacts"] = None

    t = threading.Thread(target=_run_in_background, kwargs={"seed": seed}, daemon=True)
    t.start()

    return {
        "ok": True, "status": "started", "seed": seed,
        "evidence_status": "development_only", "publication_evidence": False,
        "mode_count": len(PUBLICATION_MODES),
        "primary_mode_count": len(PRIMARY_MODES),
        "secondary_mode_count": len(SECONDARY_MODES),
    }


@router.get("/status")
def results_status():
    """Poll simulation job status."""
    with _JOB_LOCK:
        if _JOB["running"]:
            elapsed = time.time() - (_JOB["started_at"] or time.time())
            return {"status": "running", "elapsed_s": round(elapsed, 1),
                    "evidence_status": "development_only", "publication_evidence": False}
        if _JOB["finished_at"]:
            duration = round((_JOB["finished_at"] - (_JOB["started_at"] or _JOB["finished_at"])), 1)
            if _JOB["error"]:
                return {"status": "error", "error": _JOB["error"], "duration_s": duration,
                        "evidence_status": "development_only", "publication_evidence": False}
            return {"status": "complete", "duration_s": duration,
                    "evidence_status": "development_only", "publication_evidence": False}
        return {"status": "idle"}


@router.get("/summary")
def results_summary():
    """Return the last local run, explicitly labelled as development-only."""
    if _JOB["summary"]:
        return {"ok": True, "evidence_status": "development_only",
                "publication_evidence": False, "development_summary": _JOB["summary"],
                "development_artifacts": _JOB["artifacts"]}
    return {"ok": False, "evidence_status": "development_only",
            "publication_evidence": False,
            "error": "No development run is available. Validated publication evidence is served separately."}


# ---------------------------------------------------------------------------
# GET /results/figures/{filename}
# ---------------------------------------------------------------------------
@router.get("/figures/{filename}")
def get_figure(filename: str):
    """Serve a generated figure file (PNG or PDF)."""
    # Sanitise: only allow filenames, no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    artifact = _publication_artifact(filename)

    _MIME = {
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }
    media = _MIME.get(Path(filename).suffix.lower(), "application/octet-stream")
    return Response(
        content=artifact.content,
        media_type=media,
        headers={
            "ETag": f'"{artifact.sha256}"',
            "X-AGRIBRAIN-Evidence-Status": "validated-publication",
        },
    )


@router.get("/development/{filename}")
def get_development_artifact(filename: str):
    """Serve explicitly named local-run artifacts; never publication evidence."""
    if "/" in filename or "\\" in filename or ".." in filename or not filename.startswith("development_"):
        raise HTTPException(status_code=400, detail="Invalid development artifact name")
    path = _DEVELOPMENT_RESULTS_DIR / filename
    if not path.resolve().is_relative_to(_DEVELOPMENT_RESULTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid development artifact name")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Development artifact not found: {filename}")
    media = {".csv": "text/csv", ".json": "application/json"}.get(path.suffix.lower())
    if media is None:
        raise HTTPException(status_code=400, detail="Unsupported development artifact type")
    return FileResponse(str(path), media_type=media,
                        headers={"X-AGRIBRAIN-Evidence-Status": "development-only"})
