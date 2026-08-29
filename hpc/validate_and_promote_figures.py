#!/usr/bin/env python3
"""Validate a fresh staged figure set and transactionally promote its files."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mvp.simulation.validation.figure_artifacts import (
    EXPECTED_FIGURE_FILES,
    PROVENANCE_NAME,
    sha256_file,
    validate_figure_directory,
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def promote(
    staging_dir: Path,
    results_dir: Path,
    *,
    source_commit: str,
    run_tag: str,
) -> None:
    staging_dir = staging_dir.resolve(strict=True)
    results_dir = results_dir.resolve(strict=True)
    if staging_dir == results_dir or _is_relative_to(staging_dir, results_dir):
        raise ValueError("figure staging directory must be outside canonical results")
    if staging_dir.is_symlink() or results_dir.is_symlink():
        raise ValueError("staging/results directories must not be symlinks")

    validate_figure_directory(
        staging_dir,
        source_commit=source_commit,
        run_tag=run_tag,
        staging_only=True,
    )
    unexpected_existing = {
        path.name
        for path in results_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".pdf"}
        and path.name not in EXPECTED_FIGURE_FILES
    }
    if unexpected_existing:
        raise ValueError(
            "canonical results contain undeclared image residue: "
            f"{sorted(unexpected_existing)}"
        )

    targets = [*EXPECTED_FIGURE_FILES, PROVENANCE_NAME]
    transaction_parent = results_dir.parent
    with tempfile.TemporaryDirectory(
        prefix=f".figure_incoming_{run_tag}_", dir=transaction_parent,
    ) as incoming_name, tempfile.TemporaryDirectory(
        prefix=f".figure_backup_{run_tag}_", dir=transaction_parent,
    ) as backup_name:
        incoming = Path(incoming_name)
        backup = Path(backup_name)
        for name in targets:
            shutil.copy2(staging_dir / name, incoming / name)
            if sha256_file(incoming / name) != sha256_file(staging_dir / name):
                raise ValueError(f"staged figure copy changed bytes: {name}")
        validate_figure_directory(
            incoming,
            source_commit=source_commit,
            run_tag=run_tag,
            staging_only=True,
        )

        replaced: list[str] = []
        installed: list[str] = []
        try:
            for name in targets:
                destination = results_dir / name
                if destination.exists():
                    if not destination.is_file() or destination.is_symlink():
                        raise ValueError(f"unsafe existing figure target: {destination}")
                    os.replace(destination, backup / name)
                    replaced.append(name)
                os.replace(incoming / name, destination)
                installed.append(name)
            validate_figure_directory(
                results_dir,
                source_commit=source_commit,
                run_tag=run_tag,
            )
        except Exception:
            for name in reversed(installed):
                destination = results_dir / name
                if destination.exists() and destination.is_file():
                    destination.unlink()
            for name in reversed(replaced):
                os.replace(backup / name, results_dir / name)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args(argv)
    promote(
        args.staging_dir,
        args.results_dir,
        source_commit=args.source_commit,
        run_tag=args.run_tag,
    )
    print("[PASS] fresh decoded figure set promoted as one validated transaction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
