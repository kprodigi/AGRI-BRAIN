#!/usr/bin/env python3
"""Fail fast unless an HPC task is running the declared clean source snapshot.

Publication jobs execute from one run-scoped detached worktree. Every task
therefore verifies that Git is available, ``AGRIBRAIN_GIT_COMMIT`` is the full
SHA of ``HEAD``, and no source file differs from that commit. Worker and
publisher jobs may use ``--allow-run-artifacts`` because parallel tasks
necessarily create files below ``mvp/simulation/results/``; no path outside
that output tree is then tolerated.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ARTIFACT_PREFIX = "mvp/simulation/results/"
_FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def _git_output(git: str, repo_root: Path, args: list[str]) -> bytes:
    proc = subprocess.run(
        [git, "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"git {' '.join(args)} exited {proc.returncode}")
    return proc.stdout


def _parse_porcelain_z(raw: bytes) -> list[tuple[str, tuple[str, ...]]]:
    """Parse ``git status --porcelain=v1 -z`` without path quoting loss."""
    records = raw.split(b"\0")
    entries: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("malformed NUL-delimited Git status record")
        status = record[:2].decode("ascii", errors="strict")
        paths = [record[3:].decode("utf-8", errors="surrogateescape")]
        # With -z, rename/copy records carry the second pathname as the next
        # NUL-delimited field.  Check both paths so a move across the permitted
        # results boundary cannot be mistaken for a run-artifact-only change.
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise ValueError("rename/copy Git status record lacks its second path")
            paths.append(records[index].decode("utf-8", errors="surrogateescape"))
            index += 1
        entries.append((status, tuple(paths)))
    return entries


def _is_run_artifact_path(path: str) -> bool:
    # Git emits repository-relative POSIX paths under --porcelain -z.  Refuse
    # absolute/traversal forms before applying the narrow output-tree prefix.
    if not path or path.startswith(("/", "\\")) or "\\" in path:
        return False
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    return path.startswith(RUN_ARTIFACT_PREFIX)


def validation_errors(
    environ: dict[str, str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    allow_run_artifacts: bool = False,
) -> list[str]:
    """Return every checkout-identity/cleanliness violation."""
    env = dict(os.environ if environ is None else environ)
    errors: list[str] = []
    git = shutil.which("git")
    if git is None:
        return ["git executable is unavailable on PATH"]

    expected = env.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if not _FULL_SHA1.fullmatch(expected):
        errors.append(
            "AGRIBRAIN_GIT_COMMIT must be a full lowercase 40-character Git SHA-1"
        )

    try:
        top_level = Path(
            _git_output(git, repo_root, ["rev-parse", "--show-toplevel"])
            .decode("utf-8", errors="strict")
            .strip()
        ).resolve()
        if top_level != repo_root.resolve():
            errors.append(
                f"checkout root mismatch: git reports {top_level}, expected {repo_root.resolve()}"
            )
    except Exception as exc:  # noqa: BLE001 - converted to a fail-closed error
        errors.append(f"cannot resolve Git checkout root: {exc}")

    try:
        head = (
            _git_output(git, repo_root, ["rev-parse", "HEAD"])
            .decode("ascii", errors="strict")
            .strip()
        )
        if not _FULL_SHA1.fullmatch(head):
            errors.append(f"Git HEAD is not a full lowercase SHA-1: {head!r}")
        elif _FULL_SHA1.fullmatch(expected) and head != expected:
            errors.append(
                f"AGRIBRAIN_GIT_COMMIT ({expected}) does not equal checkout HEAD ({head})"
            )
    except Exception as exc:  # noqa: BLE001 - converted to a fail-closed error
        errors.append(f"cannot resolve Git HEAD: {exc}")

    try:
        status_raw = _git_output(
            git,
            repo_root,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        )
        entries = _parse_porcelain_z(status_raw)
        dirty = []
        for status, paths in entries:
            if allow_run_artifacts and all(_is_run_artifact_path(path) for path in paths):
                continue
            dirty.append(f"{status} {' -> '.join(paths)}")
        if dirty:
            preview = "; ".join(dirty[:10])
            suffix = f"; plus {len(dirty) - 10} more" if len(dirty) > 10 else ""
            errors.append(f"checkout has uncommitted non-output changes: {preview}{suffix}")
    except Exception as exc:  # noqa: BLE001 - converted to a fail-closed error
        errors.append(f"cannot verify clean Git status: {exc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-run-artifacts",
        action="store_true",
        help=(
            "Allow Git-status entries only below mvp/simulation/results/. "
            "Use for worker/publisher jobs after parallel outputs may exist."
        ),
    )
    args = parser.parse_args(argv)
    errors = validation_errors(allow_run_artifacts=args.allow_run_artifacts)
    if errors:
        print("BLOCK: source checkout is not publication-clean:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print("Publication source checkout OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
