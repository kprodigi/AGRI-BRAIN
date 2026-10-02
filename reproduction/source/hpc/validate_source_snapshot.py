#!/usr/bin/env python3
"""Validate the immutable, run-scoped source snapshot used by HPC workers.

The digest covers every tracked regular file outside the publication-results
tree, including its relative path, executable posture, length, and literal
bytes. Results are excluded because workers necessarily create and replace
evidence there; they are independently manifest-bound by the publication
pipeline. Every other tracked file must have all write bits removed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PREFIX = "mvp/simulation/results/"
SNAPSHOT_MODE = "detached_readonly_git_worktree_v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"git {' '.join(arguments)} failed")
    return completed.stdout


def tracked_source_paths(root: Path) -> list[tuple[str, Path]]:
    """Return the exact safe tracked-source inventory in Git order."""

    raw_names = _git(root, "ls-files", "-z").split(b"\0")
    records: list[tuple[str, Path]] = []
    resolved_root = root.resolve(strict=True)
    for raw_name in raw_names:
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="strict")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in name
        ):
            raise RuntimeError(f"unsafe tracked source path: {name!r}")
        if name.startswith(RESULTS_PREFIX):
            continue
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"tracked source is not a regular file: {name}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise RuntimeError(f"tracked source escapes the snapshot: {name}")
        records.append((name, resolved))
    return records


def tracked_source_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    records = tracked_source_paths(root)
    index_modes: dict[str, str] = {}
    for raw_record in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_name = raw_record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0].decode("ascii")
            name = raw_name.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("cannot parse tracked Git index modes") from exc
        index_modes[name] = mode
    for name, path in records:
        payload = path.read_bytes()
        mode = index_modes.get(name)
        if mode not in {"100644", "100755"}:
            raise RuntimeError(f"tracked source has unsupported Git mode: {name}")
        executable = mode == "100755"
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(b"\x01" if executable else b"\x00")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(records)


def validation_errors(
    environ: dict[str, str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    require_expected_digest: bool = True,
) -> list[str]:
    env = dict(os.environ if environ is None else environ)
    errors: list[str] = []
    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        return [f"cannot resolve source snapshot: {exc}"]
    declared_root = env.get("AGRIBRAIN_SOURCE_SNAPSHOT", "").strip()
    try:
        declared_resolved = Path(declared_root).resolve(strict=True)
    except OSError as exc:
        errors.append(f"AGRIBRAIN_SOURCE_SNAPSHOT cannot be resolved: {exc}")
        declared_resolved = None
    if declared_resolved != root:
        errors.append(
            "executing repository is not AGRIBRAIN_SOURCE_SNAPSHOT"
        )
    if env.get("AGRIBRAIN_SOURCE_SNAPSHOT_MODE") != SNAPSHOT_MODE:
        errors.append("AGRIBRAIN_SOURCE_SNAPSHOT_MODE is missing or incorrect")
    expected_commit = env.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if not _HEX40.fullmatch(expected_commit):
        errors.append("AGRIBRAIN_GIT_COMMIT is not a full lowercase Git SHA-1")
    try:
        head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
        if head != expected_commit:
            errors.append("source snapshot HEAD differs from AGRIBRAIN_GIT_COMMIT")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot resolve source snapshot HEAD: {exc}")

    try:
        digest, count = tracked_source_digest(root)
        if count <= 0:
            errors.append("source snapshot has no tracked source files")
        expected_digest = env.get("AGRIBRAIN_SOURCE_TREE_SHA256", "").strip()
        if require_expected_digest:
            if not _HEX64.fullmatch(expected_digest):
                errors.append("AGRIBRAIN_SOURCE_TREE_SHA256 is missing or invalid")
            elif digest != expected_digest:
                errors.append("source snapshot literal-byte digest changed")
        for name, path in tracked_source_paths(root):
            if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                errors.append(f"tracked source remains writable: {name}")
                if len(errors) >= 20:
                    break
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot hash the tracked source snapshot: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-digest",
        action="store_true",
        help="Print the validated tracked-source digest for initial export.",
    )
    args = parser.parse_args(argv)
    errors = validation_errors(require_expected_digest=not args.print_digest)
    if errors:
        print("BLOCK: source snapshot is not immutable and commit-exact:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    digest, _count = tracked_source_digest(REPO_ROOT)
    if args.print_digest:
        print(digest)
    else:
        print(f"Publication source snapshot OK: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
