#!/usr/bin/env python3
"""Reject symlinked or ambiguous components without resolving the input path."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path


def validate_lexical_path(path: Path, *, kind: str) -> Path:
    """Validate one absolute lexical path, including dangling symlinks."""
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        raise ValueError(f"path must be absolute: {raw}")
    drive, tail = os.path.splitdrive(raw)
    separator = os.sep
    lexical_parts = tail.split(separator)
    if any(part in {".", ".."} for part in lexical_parts):
        raise ValueError(f"path contains an ambiguous component: {raw}")

    current = Path(drive + separator)
    components = [item for item in lexical_parts if item]
    if not components:
        mode = os.lstat(current).st_mode
        if kind == "absent":
            raise ValueError(f"path already exists: {raw}")
        if kind == "file" or not stat.S_ISDIR(mode):
            raise ValueError(f"path has the wrong type: {raw}")
        return path
    existing_leaf = False
    for index, part in enumerate(components):
        current = current / part
        is_leaf = index == len(components) - 1
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if kind == "absent":
                if is_leaf:
                    return path
                continue
            raise ValueError(f"path component does not exist: {current}") from None
        if stat.S_ISLNK(mode):
            raise ValueError(f"path has a symlink component: {current}")
        if not is_leaf and not stat.S_ISDIR(mode):
            raise ValueError(f"path parent is not a directory: {current}")
        if is_leaf:
            existing_leaf = True
            if kind == "file" and not stat.S_ISREG(mode):
                raise ValueError(f"path is not a regular file: {current}")
            if kind == "directory" and not stat.S_ISDIR(mode):
                raise ValueError(f"path is not a directory: {current}")

    if kind == "absent" and existing_leaf:
        raise ValueError(f"path already exists: {raw}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-file", action="append", type=Path, default=[])
    parser.add_argument("--require-directory", action="append", type=Path, default=[])
    parser.add_argument("--require-absent", action="append", type=Path, default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        for kind, paths in (
            ("file", args.require_file),
            ("directory", args.require_directory),
            ("absent", args.require_absent),
        ):
            for path in paths:
                validate_lexical_path(path, kind=kind)
    except (OSError, ValueError) as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
