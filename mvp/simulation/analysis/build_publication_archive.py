#!/usr/bin/env python3
"""Build and verify a deterministic publication archive from its manifest.

The manifest is finalized before packaging.  Every listed payload is checked
against its literal-byte length and SHA-256, the archive is written to a
temporary path, and the completed archive is reopened and checked again before
it is atomically promoted.  The shippable path accepts only one clean commit
with a fully validated fresh stochastic run; historical publication-only
repair packaging is deliberately retired.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mvp.simulation.validation.validate_publication_artifacts import (
    validate_full_publication_release,
)
from mvp.simulation.validation.validator_source_identity import (
    validate_clean_validator_checkout,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _safe_name(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"invalid empty/non-string manifest path: {raw!r}")
    if "\\" in raw:
        raise ValueError(f"manifest path contains a backslash: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe manifest path: {raw!r}")
    return path.as_posix()


def _load_manifest_bytes(payload: bytes) -> tuple[dict, list[dict]]:
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("artifact manifest is not a JSON object")
    records = data.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("artifact manifest has no artifact records")

    seen: set[str] = set()
    normalized: list[dict] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"artifact record {index} is not an object")
        name = _safe_name(record.get("file"))
        if name in seen or name == "artifact_manifest.json":
            raise ValueError(f"duplicate or self-referential artifact path: {name}")
        seen.add(name)
        digest = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise ValueError(f"invalid SHA-256 for {name}: {digest!r}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid byte count for {name}: {size!r}")
        normalized.append({"file": name, "bytes": size, "sha256": digest})

    declared_count = data.get("artifact_count")
    if declared_count != len(normalized):
        raise ValueError(
            f"manifest count mismatch: declared={declared_count!r}, actual={len(normalized)}"
        )
    for key in (
        "git_commit",
        "simulation_source_commit",
        "publication_code_commit",
    ):
        value = data.get(key)
        if not isinstance(value, str) or not _HEX40.fullmatch(value):
            raise ValueError(f"manifest {key} is not a full Git SHA-1: {value!r}")
    expected_dual = data["simulation_source_commit"] != data["publication_code_commit"]
    if data.get("dual_provenance") is not expected_dual:
        raise ValueError("manifest dual_provenance disagrees with its two commits")
    return data, normalized


def _load_manifest(path: Path) -> tuple[dict, list[dict]]:
    """Compatibility wrapper; production code snapshots the bytes itself."""

    return _load_manifest_bytes(path.read_bytes())


def _derivation_metadata(
    manifest: dict, *, semantic_receipt_validated: bool = False,
) -> dict[str, object]:
    """Describe derivation only from independently validated evidence."""
    if manifest["dual_provenance"]:
        raise ValueError(
            "publication-only repair packaging is retired; run the complete "
            "simulation and publication pipeline from one clean commit"
        )
    if semantic_receipt_validated:
        return {
            "derivation_type": "fresh stochastic simulation and publication build",
            "simulation_rerun": True,
        }
    return {
        "derivation_type": "unknown: equal commits do not prove execution",
        "simulation_rerun": None,
    }


def _safe_source_path(results_dir: Path, name: str) -> Path:
    source = results_dir / Path(name)
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"manifest payload is missing: {name}") from exc
    if not resolved.is_relative_to(results_dir):
        raise ValueError(f"manifest payload resolves outside results: {name}")
    cursor = source
    while cursor != results_dir:
        if cursor.is_symlink():
            raise ValueError(f"manifest payload traverses a symlink: {name}")
        cursor = cursor.parent
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest payload is not a regular file: {name}")
    return resolved


def _verify_files(results_dir: Path, records: list[dict]) -> None:
    for record in records:
        source = _safe_source_path(results_dir, record["file"])
        size = source.stat().st_size
        if size != record["bytes"]:
            raise ValueError(
                f"byte-count mismatch for {record['file']}: "
                f"manifest={record['bytes']}, actual={size}"
            )
        digest = _sha256_file(source)
        if digest != record["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {record['file']}: "
                f"manifest={record['sha256']}, actual={digest}"
            )


def _tar_info(name: str, size: int, epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = epoch
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(
    archive_path: Path,
    results_dir: Path,
    manifest_bytes: bytes,
    records: list[dict],
    epoch: int,
) -> None:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                tar.addfile(
                    _tar_info("artifact_manifest.json", len(manifest_bytes), epoch),
                    io.BytesIO(manifest_bytes),
                )
                for record in sorted(records, key=lambda item: item["file"]):
                    source = _safe_source_path(results_dir, record["file"])
                    with source.open("rb") as stream:
                        tar.addfile(
                            _tar_info(record["file"], record["bytes"], epoch),
                            stream,
                        )


def _verify_archive(
    archive_path: Path,
    manifest_bytes: bytes,
    records: list[dict],
) -> None:
    expected = {record["file"]: record for record in records}
    expected_names = {"artifact_manifest.json", *expected}
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar:
            name = _safe_name(member.name)
            if name in seen:
                raise ValueError(f"duplicate archive member: {name}")
            seen.add(name)
            if not member.isfile():
                raise ValueError(f"non-regular archive member: {name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member: {name}")
            if name == "artifact_manifest.json":
                archived_manifest = stream.read()
                if archived_manifest != manifest_bytes:
                    raise ValueError("archived manifest bytes differ from finalized manifest")
                continue
            record = expected.get(name)
            if record is None:
                raise ValueError(f"undeclared archive payload: {name}")
            if member.size != record["bytes"]:
                raise ValueError(f"archived byte-count mismatch: {name}")
            digest = _sha256_stream(stream)
            if digest != record["sha256"]:
                raise ValueError(f"archived SHA-256 mismatch: {name}")
    if seen != expected_names:
        raise ValueError(
            "archive membership differs from manifest: "
            f"missing={sorted(expected_names - seen)}, extra={sorted(seen - expected_names)}"
        )


def _payload_merkle_root(records: list[dict]) -> str:
    leaves = [
        hashlib.sha256(
            f"{record['file']}\0{record['bytes']}\0{record['sha256']}".encode("utf-8")
        ).digest()
        for record in sorted(records, key=lambda item: item["file"])
    ]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--parent-archive-sha256")
    parser.add_argument("--source-date-epoch", type=int, default=0)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    manifest_path = results_dir / "artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("artifact manifest must be a regular non-symlink file")
    if output.parent != receipt.parent:
        raise ValueError("archive and receipt must share one atomic bundle directory")
    bundle_dir = output.parent
    if bundle_dir.exists():
        raise FileExistsError(
            "refusing to overwrite an existing publication bundle directory"
        )
    if not bundle_dir.parent.is_dir():
        raise FileNotFoundError("publication bundle parent directory does not exist")
    if args.parent_archive_sha256:
        raise ValueError(
            "--parent-archive-sha256 is retired; a shippable release requires "
            "one clean fresh-run archive"
        )

    manifest_bytes = manifest_path.read_bytes()
    manifest, records = _load_manifest_bytes(manifest_bytes)
    if manifest["dual_provenance"]:
        raise ValueError(
            "dual-provenance publication repair archives are not shippable; "
            "regenerate all evidence from one clean commit"
        )
    source_commits = {
        manifest["git_commit"],
        manifest["simulation_source_commit"],
        manifest["publication_code_commit"],
    }
    if len(source_commits) != 1:
        raise ValueError(
            "fresh publication evidence must use one identical Git commit"
        )
    if manifest.get("git_dirty") is not False:
        raise ValueError("fresh publication evidence cannot carry a dirty Git stamp")
    source_commit = manifest["simulation_source_commit"]
    if not any(
        record.get("file") == "publication_validation_receipt.json"
        for record in records
    ):
        raise ValueError(
            "a fresh-run archive requires the manifested semantic validation receipt"
        )

    # Only the finalized manifest and the exact files it names may differ from
    # the checked-out commit.  This check is against this module's fixed
    # repository root, never a caller-supplied path.  It therefore binds the
    # validator code that is actually executing to the commit being certified.
    evidence_paths = [
        manifest_path,
        *(_safe_source_path(results_dir, record["file"]) for record in records),
    ]
    validator_source_identity = validate_clean_validator_checkout(
        source_commit,
        repo_root=_REPO_ROOT,
        allowed_dirty_paths=evidence_paths,
    )
    _verify_files(results_dir, records)
    validate_full_publication_release(
        results_dir, repo_root=_REPO_ROOT,
    )
    semantic_receipt_validated = True
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("artifact manifest changed during semantic validation")
    _verify_files(results_dir, records)
    if validate_clean_validator_checkout(
        source_commit,
        repo_root=_REPO_ROOT,
        allowed_dirty_paths=evidence_paths,
    ) != validator_source_identity:
        raise ValueError("validator source identity changed during semantic validation")

    receipt_payload: dict[str, object]
    with tempfile.TemporaryDirectory(
        prefix=f".{bundle_dir.name}.", dir=bundle_dir.parent,
    ) as temp_name:
        temp_bundle = Path(temp_name)
        temp_archive = temp_bundle / output.name
        temp_receipt = temp_bundle / receipt.name
        _write_archive(
            temp_archive,
            results_dir,
            manifest_bytes,
            records,
            max(0, args.source_date_epoch),
        )
        _verify_archive(temp_archive, manifest_bytes, records)
        archive_sha256 = _sha256_file(temp_archive)
        archive_bytes = temp_archive.stat().st_size

        receipt_payload = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **_derivation_metadata(
                manifest,
                semantic_receipt_validated=semantic_receipt_validated,
            ),
            "simulation_source_commit": manifest["simulation_source_commit"],
            "publication_code_commit": manifest["publication_code_commit"],
            "run_tag": manifest.get("artifact_run_tag"),
            "parent_archive_sha256": None,
            "validator_source_identity": validator_source_identity,
            "archive": {
                "file": output.name,
                "bytes": archive_bytes,
                "sha256": archive_sha256,
                "member_count": len(records) + 1,
            },
            "manifest": {
                "bytes": len(manifest_bytes),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "artifact_count": len(records),
                "payload_merkle_root": _payload_merkle_root(records),
                "hash_semantics": "literal bytes",
            },
            "validation": {
                "prearchive_payload_hashes": "PASS",
                "postarchive_payload_hashes": "PASS",
                "exact_manifest_membership": "PASS",
                "safe_regular_members_only": "PASS",
                "semantic_validation_receipt_manifested_and_verified": (
                    "PASS" if semantic_receipt_validated else "NOT_APPLICABLE"
                ),
                "validator_checkout_same_clean_commit_outside_exact_evidence": (
                    "PASS"
                ),
            },
            "evidence_scope": {
                "core_publication_evidence": True,
                "structural_sensitivity_included": False,
                "full_submission_requires_separate_structural_receipt": True,
            },
        }
        temp_receipt.write_text(
            json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # A final READY record is the consumer-visible completion marker.  The
        # whole temporary directory is renamed only after both payloads exist
        # and have been verified, making partial archive/receipt pairs
        # impossible at the published path.
        ready_payload = {
            "schema_version": 1,
            "status": "READY",
            "archive": {
                "file": output.name,
                "sha256": archive_sha256,
            },
            "receipt": {
                "file": receipt.name,
                "sha256": _sha256_file(temp_receipt),
            },
        }
        (temp_bundle / "READY.json").write_text(
            json.dumps(ready_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Re-read every final temporary payload before the atomic directory
        # promotion; this also catches a failed/short receipt write.
        json.loads(temp_receipt.read_text(encoding="utf-8"))
        json.loads((temp_bundle / "READY.json").read_text(encoding="utf-8"))
        _verify_archive(temp_archive, manifest_bytes, records)
        # Include only the three exact staging files in the final Git-status
        # exception.  Any code edit, new validator module, or unrelated result
        # appearing during validation or packaging fails before publication.
        validate_clean_validator_checkout(
            source_commit,
            repo_root=_REPO_ROOT,
            allowed_dirty_paths=[
                *evidence_paths,
                temp_archive,
                temp_receipt,
                temp_bundle / "READY.json",
            ],
        )
        os.replace(temp_bundle, bundle_dir)

    print(json.dumps(receipt_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
