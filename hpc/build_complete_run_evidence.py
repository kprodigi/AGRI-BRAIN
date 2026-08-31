#!/usr/bin/env python3
"""Build a separate, lossless post-HPC archive for future reanalysis."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

# This script is executed as `python hpc/build_complete_run_evidence.py`
# from the snapshot root with PYTHONPATH force-unset by publication_env.sh,
# so the repository root must be bootstrapped onto sys.path before the
# package imports below can resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc.capture_slurm_accounting import validate_accounting_payload
from mvp.simulation.benchmarks.episode_archive import (
    canonical_json_bytes,
    canonical_json_sha256,
)


def _parse_binding(raw: str, *, directory: bool) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"input binding must be NAME=PATH: {raw!r}")
    name, path_text = raw.split("=", 1)
    path = Path(path_text).resolve()
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe archive input name: {name!r}")
    if path.is_symlink() or (not path.is_dir() if directory else not path.is_file()):
        kind = "directory" if directory else "file"
        raise ValueError(f"input {kind} is missing or unsafe: {path}")
    return pure.as_posix(), path


def _binding(path: Path) -> tuple[str, int]:
    # The finished complete-evidence archive spans tens of gigabytes, far
    # beyond the publisher's memory allocation; hash in streamed chunks so
    # the binding never materializes the whole file in memory.
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _collect(
    roots: list[tuple[str, Path]], files: list[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    collected: list[tuple[str, Path]] = []
    names: set[str] = set()
    for prefix, root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"archive input contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            name = f"inputs/{prefix}/{relative}"
            if name in names:
                raise ValueError(f"duplicate archive member: {name}")
            names.add(name)
            collected.append((name, path))
    for name, path in files:
        member = f"inputs/{name}"
        if member in names:
            raise ValueError(f"duplicate archive member: {member}")
        names.add(member)
        collected.append((member, path))
    return sorted(collected)


def _validate_episode_manifests(
    members: list[tuple[str, Path]],
    *,
    expected_manifests: int,
    expected_groups: int,
    expected_episodes: int,
    expected_adaptation_ledgers: int,
    expected_final_ledgers: int,
) -> dict[str, int]:
    paths = [
        path for name, path in members
        if name.endswith("/complete_episode_evidence_manifest.json")
    ]
    if len(paths) != expected_manifests:
        raise ValueError(
            f"complete-episode manifest count is {len(paths)}, "
            f"expected {expected_manifests}"
        )
    totals = {
        "episode_groups": 0,
        "executed_episode_archives": 0,
        "adaptation_episode_ledgers": 0,
        "final_episode_ledgers": 0,
        "decision_records": 0,
    }
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise ValueError(f"episode manifest is not complete: {path}")
        claimed = payload.pop("manifest_sha256", None)
        if claimed != canonical_json_sha256(payload):
            raise ValueError(f"episode manifest self-hash mismatch: {path}")
        counts = payload.get("counts") or {}
        for field in totals:
            value = counts.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid {field} count in {path}")
            totals[field] += value
    expected = {
        "episode_groups": expected_groups,
        "executed_episode_archives": expected_episodes,
        "adaptation_episode_ledgers": expected_adaptation_ledgers,
        "final_episode_ledgers": expected_final_ledgers,
    }
    for field, value in expected.items():
        if totals[field] != value:
            raise ValueError(
                f"aggregate {field} is {totals[field]}, expected {value}"
            )
    return totals


def _validate_runtime_receipts(
    members: list[tuple[str, Path]],
    *,
    expected_count: int,
    source_commit: str,
    source_tree_sha256: str,
    run_tag: str,
) -> dict[str, float | int]:
    paths = [
        path for name, path in members
        if "/runtime_receipts/" in name
    ]
    wall_total = 0.0
    cpu_total = 0.0
    successful = 0
    failed = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("receipt_sha256", None)
        if claimed != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
            raise ValueError(f"runtime receipt self-hash mismatch: {path}")
        identity = payload.get("run_identity") or {}
        resource_delta = payload.get("resource_child_delta_or_peak") or {}
        if (
            payload.get("resource_available") is not True
            or identity.get("source_commit") != source_commit
            or identity.get("source_tree_sha256") != source_tree_sha256
            or identity.get("run_tag") != run_tag
        ):
            raise ValueError(f"runtime receipt identity/completion mismatch: {path}")
        wall = payload.get("wall_seconds")
        if not isinstance(wall, (int, float)) or float(wall) < 0:
            raise ValueError(f"runtime receipt has invalid wall time: {path}")
        if payload.get("returncode") == 0:
            successful += 1
            wall_total += float(wall)
            cpu_total += float(resource_delta.get("ru_utime", 0.0)) + float(
                resource_delta.get("ru_stime", 0.0)
            )
        else:
            failed += 1
    if successful != expected_count:
        raise ValueError(
            f"successful runtime receipt count is {successful}, expected {expected_count}"
        )
    return {
        "receipt_count": len(paths),
        "successful_receipt_count": successful,
        "failed_attempt_receipt_count": failed,
        "summed_task_wall_seconds_nonconcurrent": wall_total,
        "summed_child_cpu_seconds": cpu_total,
    }


def _validate_scheduler_accounting(
    members: list[tuple[str, Path]],
    *,
    expected_task_count: int,
    source_commit: str,
    source_tree_sha256: str,
    run_tag: str,
) -> dict[str, Any]:
    paths = [
        path for name, path in members
        if name.endswith("/slurm_simulation_accounting.json")
    ]
    if len(paths) != 1:
        raise ValueError(
            f"complete core evidence requires one scheduler accounting file, "
            f"found {len(paths)}"
        )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    return validate_accounting_payload(
        payload,
        kind="core",
        run_tag=run_tag,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        expected_task_count=expected_task_count,
    )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(
    archive: Path, manifest_bytes: bytes, members: list[tuple[str, Path]],
) -> None:
    with archive.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, compresslevel=9, mtime=0,
        ) as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="w") as tar:
                tar.addfile(
                    _tar_info("COMPLETE_RUN_EVIDENCE_MANIFEST.json", len(manifest_bytes)),
                    io.BytesIO(manifest_bytes),
                )
                for name, path in members:
                    size = path.stat().st_size
                    with path.open("rb") as stream:
                        tar.addfile(_tar_info(name, size), stream)


def _verify_archive(
    archive: Path, manifest: dict[str, Any],
) -> None:
    expected = {
        record["path"]: (record["sha256"], record["bytes"])
        for record in manifest["artifacts"]
    }
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if (
            len(names) != len(set(names))
            or names[0] != "COMPLETE_RUN_EVIDENCE_MANIFEST.json"
            or set(names[1:]) != set(expected)
            or any(not member.isfile() for member in members)
        ):
            raise ValueError("complete-run archive inventory is invalid")
        manifest_stream = tar.extractfile(members[0])
        if manifest_stream is None or manifest_stream.read() != canonical_json_bytes(manifest):
            raise ValueError("archived complete-run manifest differs from source")
        for member in members[1:]:
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(
                lambda stream=stream: stream.read(1024 * 1024), b"",
            ):
                digest.update(block)
                size += len(block)
            if (digest.hexdigest(), size) != expected[member.name]:
                raise ValueError(f"archive member binding mismatch: {member.name}")


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    roots = [_parse_binding(raw, directory=True) for raw in args.input_root]
    files = [_parse_binding(raw, directory=False) for raw in args.input_file]
    members = _collect(roots, files)
    totals = _validate_episode_manifests(
        members,
        expected_manifests=args.expected_manifests,
        expected_groups=args.expected_groups,
        expected_episodes=args.expected_episodes,
        expected_adaptation_ledgers=args.expected_adaptation_ledgers,
        expected_final_ledgers=args.expected_final_ledgers,
    )
    runtime_totals = _validate_runtime_receipts(
        members,
        expected_count=args.expected_runtime_receipts,
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        run_tag=args.run_tag,
    )
    scheduler_accounting = _validate_scheduler_accounting(
        members,
        expected_task_count=args.expected_scheduler_tasks,
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        run_tag=args.run_tag,
    )
    artifact_records = []
    for name, path in members:
        sha256, size = _binding(path)
        artifact_records.append({"path": name, "sha256": sha256, "bytes": size})
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scope": (
            "lossless raw core and H3 execution evidence for future analyses "
            "that do not alter the model, interventions, or experimental design"
        ),
        "limitation": (
            "No archive can answer a future question that requires a new model, "
            "new intervention, new data, or a different experimental design."
        ),
        "source_commit": args.source_commit,
        "source_tree_sha256": args.source_tree_sha256,
        "run_tag": args.run_tag,
        "episode_totals": totals,
        "runtime_receipt_totals": runtime_totals,
        "scheduler_accounting": scheduler_accounting,
        "artifact_count": len(artifact_records),
        "artifact_bytes": sum(record["bytes"] for record in artifact_records),
        "artifacts": artifact_records,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_bytes = canonical_json_bytes(manifest)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite complete evidence bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        archive = stage / f"complete_run_evidence_{args.run_tag}.tar.gz"
        _write_archive(archive, manifest_bytes, members)
        _verify_archive(archive, manifest)
        archive_sha256, archive_bytes = _binding(archive)
        receipt = {
            "schema_version": 1,
            "status": "VERIFIED",
            "run_tag": args.run_tag,
            "source_commit": args.source_commit,
            "manifest_sha256": manifest["manifest_sha256"],
            "archive": archive.name,
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
            "artifact_count": len(artifact_records),
            "episode_totals": totals,
            "runtime_receipt_totals": runtime_totals,
            "scheduler_accounting": scheduler_accounting,
        }
        receipt["receipt_sha256"] = canonical_json_sha256(receipt)
        (stage / "COMPLETE_RUN_EVIDENCE_MANIFEST.json").write_bytes(
            manifest_bytes + b"\n"
        )
        (stage / "RECEIPT.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
        (stage / "READY.json").write_bytes(canonical_json_bytes({
            "status": "READY",
            "archive_sha256": archive_sha256,
            "receipt_sha256": receipt["receipt_sha256"],
        }) + b"\n")
        os.replace(stage, output)
        stage = None
        return receipt
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", default=[], required=True)
    parser.add_argument("--input-file", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--expected-manifests", type=int, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-adaptation-ledgers", type=int, required=True)
    parser.add_argument("--expected-final-ledgers", type=int, required=True)
    parser.add_argument("--expected-runtime-receipts", type=int, required=True)
    parser.add_argument("--expected-scheduler-tasks", type=int, required=True)
    args = parser.parse_args()
    receipt = build_bundle(args)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
