#!/usr/bin/env python3
"""Hash-bind immutable simulation outputs before publication-only recovery.

The normal fresh-run publishers do not use this module.  It exists solely for
the fail-closed recovery path after simulations completed but their declared
publisher failed.  A manifest contains logical, path-portable names and
literal-byte hashes; callers supply the same NAME=PATH bindings when validating
it.  Validation compares the *complete* inventory of every bound directory, so
changed, missing, or newly added input files are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
RECEIPT_TYPE = "preserved_simulation_raw_output_manifest"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_TAG = re.compile(r"(?:sensitivity_)?([0-9a-f]{7})_[0-9]{8}_[0-9]{6}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_logical_name(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"invalid logical input name: {raw!r}")
    value = PurePosixPath(raw)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"unsafe logical input name: {raw!r}")
    return value.as_posix()


def _unresolved_safe_path(path: Path, *, label: str) -> Path:
    candidate = path.absolute()
    cursor = candidate
    while True:
        if cursor.is_symlink():
            raise ValueError(f"{label} traverses a symbolic link: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return candidate


def _parse_binding(raw: str, *, directory: bool) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"input binding must be NAME=PATH: {raw!r}")
    name, raw_path = raw.split("=", 1)
    logical = _safe_logical_name(name)
    unresolved = _unresolved_safe_path(
        Path(raw_path), label="input binding"
    )
    path = unresolved.resolve(strict=True)
    if directory and not path.is_dir():
        raise ValueError(f"input directory is not a directory: {path}")
    if not directory and not path.is_file():
        raise ValueError(f"input file is not a regular file: {path}")
    return logical, path


def _normalize_bindings(
    roots: Sequence[tuple[str, Path]], files: Sequence[tuple[str, Path]],
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    normalized_roots: list[tuple[str, Path]] = []
    normalized_files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    def add(destination: str, raw_path: Path, *, directory: bool) -> None:
        name = _safe_logical_name(destination)
        if name in seen:
            raise ValueError(f"duplicate logical input name: {name}")
        if any(
            name.startswith(f"{prior}/") or prior.startswith(f"{name}/")
            for prior in seen
        ):
            raise ValueError(f"overlapping logical input name: {name}")
        unresolved = _unresolved_safe_path(raw_path, label="input binding")
        path = unresolved.resolve(strict=True)
        if directory and not path.is_dir():
            raise ValueError(f"input directory is not a directory: {path}")
        if not directory and not path.is_file():
            raise ValueError(f"input file is not a regular file: {path}")
        seen.add(name)
        target = normalized_roots if directory else normalized_files
        target.append((name, path))
    for destination, raw_path in roots:
        add(destination, raw_path, directory=True)
    for destination, raw_path in files:
        add(destination, raw_path, directory=False)
    return sorted(normalized_roots), sorted(normalized_files)


def _collect_records(
    roots: Sequence[tuple[str, Path]], files: Sequence[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    binding_records: list[dict[str, Any]] = []
    for logical, root in roots:
        root_file_count = 0
        root_bytes = 0
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_symlink():
                raise ValueError(f"input directory contains a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"input directory contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            name = _safe_logical_name(f"{logical}/{relative}")
            size = path.stat().st_size
            records.append({
                "path": name,
                "bytes": size,
                "sha256": _file_sha256(path),
            })
            root_file_count += 1
            root_bytes += size
        binding_records.append({
            "name": logical,
            "type": "directory",
            "file_count": root_file_count,
            "bytes": root_bytes,
        })
    for logical, path in files:
        size = path.stat().st_size
        records.append({
            "path": logical,
            "bytes": size,
            "sha256": _file_sha256(path),
        })
        binding_records.append({
            "name": logical,
            "type": "file",
            "file_count": 1,
            "bytes": size,
        })
    records.sort(key=lambda item: item["path"])
    names = [str(item["path"]) for item in records]
    if len(names) != len(set(names)):
        raise ValueError("logical input bindings produce duplicate file paths")
    binding_records.sort(key=lambda item: item["name"])
    return records, binding_records


def _payload_merkle_root(records: Iterable[Mapping[str, Any]]) -> str:
    level = [
        hashlib.sha256(_canonical_bytes({
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        })).digest()
        for record in records
    ]
    if not level:
        return hashlib.sha256(b"").hexdigest()
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _validate_identity(
    *, kind: str, run_tag: str, simulation_commit: str,
    simulation_source_tree_sha256: str,
) -> None:
    if kind not in {"core", "structural"}:
        raise ValueError("raw manifest kind must be core or structural")
    match = _RUN_TAG.fullmatch(run_tag)
    if match is None or match.group(1) != simulation_commit[:7]:
        raise ValueError("raw manifest run tag is not simulation-commit-bound")
    if not _HEX40.fullmatch(simulation_commit):
        raise ValueError("simulation commit must be a full lowercase Git SHA-1")
    if not _HEX64.fullmatch(simulation_source_tree_sha256):
        raise ValueError("simulation source-tree identity must be a lowercase SHA-256")


def build_manifest(
    *, kind: str, run_tag: str, simulation_commit: str,
    simulation_source_tree_sha256: str,
    roots: Sequence[tuple[str, Path]], files: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    _validate_identity(
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
        simulation_source_tree_sha256=simulation_source_tree_sha256,
    )
    normalized_roots, normalized_files = _normalize_bindings(roots, files)
    records, bindings = _collect_records(normalized_roots, normalized_files)
    if not records:
        raise ValueError("preserved raw-output manifest cannot be empty")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "run_tag": run_tag,
        "simulation_source_commit": simulation_commit,
        "simulation_source_tree_sha256": simulation_source_tree_sha256,
        "hash_semantics": "SHA-256 of literal file bytes",
        "bindings": bindings,
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "payload_merkle_root": _payload_merkle_root(records),
        "files": records,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def validate_manifest_payload(
    payload: Mapping[str, Any], *, kind: str, run_tag: str,
    simulation_commit: str, simulation_source_tree_sha256: str,
    roots: Sequence[tuple[str, Path]], files: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    validated = validate_manifest_document(
        payload,
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
        simulation_source_tree_sha256=simulation_source_tree_sha256,
    )
    normalized_roots, normalized_files = _normalize_bindings(roots, files)
    records, bindings = _collect_records(normalized_roots, normalized_files)
    if validated.get("bindings") != bindings:
        raise ValueError("preserved raw-output binding inventory changed")
    if validated.get("files") != records:
        raise ValueError("preserved raw-output literal file inventory changed")
    return validated


def validate_manifest_document(
    payload: Mapping[str, Any], *, kind: str, run_tag: str,
    simulation_commit: str, simulation_source_tree_sha256: str,
) -> dict[str, Any]:
    """Validate the portable manifest document without resolving live inputs.

    Recovery authorization uses this to bind already-generated manifest bytes.
    The publisher must additionally call :func:`validate_manifest_payload`
    with live NAME=PATH bindings immediately before and after publication.
    """

    _validate_identity(
        kind=kind,
        run_tag=run_tag,
        simulation_commit=simulation_commit,
        simulation_source_tree_sha256=simulation_source_tree_sha256,
    )
    expected_keys = {
        "schema_version", "receipt_type", "created_at_utc", "kind", "run_tag",
        "simulation_source_commit", "simulation_source_tree_sha256",
        "hash_semantics", "bindings", "file_count", "total_bytes",
        "payload_merkle_root", "files", "manifest_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("preserved raw-output manifest fields are ambiguous")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("receipt_type") != RECEIPT_TYPE
        or payload.get("kind") != kind
        or payload.get("run_tag") != run_tag
        or payload.get("simulation_source_commit") != simulation_commit
        or payload.get("simulation_source_tree_sha256")
        != simulation_source_tree_sha256
        or payload.get("hash_semantics") != "SHA-256 of literal file bytes"
    ):
        raise ValueError("preserved raw-output manifest identity is inconsistent")
    timestamp = payload.get("created_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("preserved raw-output manifest timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("preserved raw-output manifest timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("preserved raw-output manifest timestamp lacks a timezone")
    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError("preserved raw-output manifest self-hash is invalid")

    raw_records = payload.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("preserved raw-output manifest has no file records")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ValueError(f"preserved raw-output file record {index} is malformed")
        name = _safe_logical_name(raw.get("path"))
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not _HEX64.fullmatch(digest)
        ):
            raise ValueError(f"preserved raw-output file record {name!r} is invalid")
        records.append({"path": name, "bytes": size, "sha256": digest})
    if records != sorted(records, key=lambda item: item["path"]):
        raise ValueError("preserved raw-output file records are not sorted")
    names = [record["path"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("preserved raw-output file records contain duplicates")

    bindings = payload.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("preserved raw-output manifest has no bindings")
    normalized_bindings: list[dict[str, Any]] = []
    covered: set[str] = set()
    logical_names: set[str] = set()
    for index, raw in enumerate(bindings):
        if not isinstance(raw, dict) or set(raw) != {
            "name", "type", "file_count", "bytes",
        }:
            raise ValueError(f"preserved raw-output binding {index} is malformed")
        name = _safe_logical_name(raw.get("name"))
        kind_value = raw.get("type")
        if kind_value not in {"directory", "file"} or name in logical_names:
            raise ValueError("preserved raw-output binding name/type is invalid")
        if any(
            name.startswith(f"{prior}/") or prior.startswith(f"{name}/")
            for prior in logical_names
        ):
            raise ValueError("preserved raw-output bindings overlap")
        logical_names.add(name)
        members = [
            record for record in records
            if (
                record["path"].startswith(f"{name}/")
                if kind_value == "directory"
                else record["path"] == name
            )
        ]
        expected_count = len(members)
        expected_bytes = sum(int(record["bytes"]) for record in members)
        if (
            raw.get("file_count") != expected_count
            or raw.get("bytes") != expected_bytes
            or (kind_value == "file" and expected_count != 1)
        ):
            raise ValueError("preserved raw-output binding summary is inconsistent")
        covered.update(str(record["path"]) for record in members)
        normalized_bindings.append({
            "name": name,
            "type": kind_value,
            "file_count": expected_count,
            "bytes": expected_bytes,
        })
    if normalized_bindings != sorted(
        normalized_bindings, key=lambda item: item["name"]
    ) or covered != set(names):
        raise ValueError("preserved raw-output binding coverage is inconsistent")
    if (
        payload.get("file_count") != len(records)
        or payload.get("total_bytes")
        != sum(int(record["bytes"]) for record in records)
        or payload.get("payload_merkle_root") != _payload_merkle_root(records)
    ):
        raise ValueError("preserved raw-output manifest summary is inconsistent")
    return dict(payload)


def _load_json(path: Path) -> dict[str, Any]:
    unresolved = _unresolved_safe_path(path, label="raw manifest")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"raw manifest is not a regular file: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("raw manifest must contain one JSON object")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = _unresolved_safe_path(path, label="raw manifest output")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite raw manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--kind", choices=("core", "structural"), required=True)
        command.add_argument("--run-tag", required=True)
        command.add_argument("--simulation-commit", required=True)
        command.add_argument("--simulation-source-tree-sha256", required=True)
        command.add_argument("--input-root", action="append", default=[])
        command.add_argument("--input-file", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = [_parse_binding(raw, directory=True) for raw in args.input_root]
    files = [_parse_binding(raw, directory=False) for raw in args.input_file]
    identity = {
        "kind": args.kind,
        "run_tag": args.run_tag,
        "simulation_commit": args.simulation_commit,
        "simulation_source_tree_sha256": args.simulation_source_tree_sha256,
        "roots": roots,
        "files": files,
    }
    if args.command == "create":
        payload = build_manifest(**identity)
        output = _unresolved_safe_path(
            args.manifest, label="raw manifest output"
        )
        for _name, input_path in (*roots, *files):
            if output == input_path or (
                input_path.is_dir() and output.is_relative_to(input_path)
            ):
                raise ValueError("raw manifest output must be outside preserved inputs")
        _write_new_json(output, payload)
    else:
        payload = validate_manifest_payload(_load_json(args.manifest), **identity)
    print(json.dumps({
        "status": "VALID",
        "kind": payload["kind"],
        "file_count": payload["file_count"],
        "total_bytes": payload["total_bytes"],
        "payload_merkle_root": payload["payload_merkle_root"],
        "manifest_sha256": payload["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
