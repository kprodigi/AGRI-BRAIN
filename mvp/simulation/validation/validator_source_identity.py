"""Fail-closed identity gate for code that validates publication evidence."""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def validate_clean_validator_checkout(
    expected_commit: object,
    *,
    repo_root: Path,
    allowed_dirty_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Require the executing checkout to match the evidence source commit.

    ``allowed_dirty_paths`` is deliberately an exact file allowlist, not a
    directory exclusion.  The archive builder uses it for the manifested run
    products that necessarily differ from the committed checkout.  Validators
    consuming an already-built archive leave it empty and therefore require a
    literally empty tracked-and-untracked porcelain status.
    """

    if not isinstance(expected_commit, str) or not _HEX40.fullmatch(
        expected_commit
    ):
        raise ValueError("expected validator source commit must be full lowercase hex")
    root = repo_root.resolve()

    def git(*arguments: str, strip: bool = True) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "cannot establish the publication validator git identity"
            ) from exc
        return completed.stdout.strip() if strip else completed.stdout

    if git("rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("publication validator is not running in a Git worktree")
    try:
        git_root = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    except OSError as exc:
        raise ValueError("publication validator Git root cannot be resolved") from exc
    if git_root != root:
        raise ValueError(
            "validator repo_root is not the executing Git worktree root"
        )

    head = git("rev-parse", "HEAD")
    if not _HEX40.fullmatch(head):
        raise ValueError("validator checkout HEAD is not a full lowercase commit")
    if head != expected_commit:
        raise ValueError(
            "validator checkout HEAD differs from the evidence source commit"
        )
    allowed_relative: set[str] = set()
    for raw_path in allowed_dirty_paths:
        candidate = Path(raw_path)
        try:
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError(
                "validator status allowlist path is outside the Git worktree"
            ) from exc
        if not relative or relative == "." or relative.startswith(".git/"):
            raise ValueError("invalid validator status allowlist path")
        allowed_relative.add(relative)

    porcelain = git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False
    )
    entries = porcelain.split("\0")
    unexpected: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            raise ValueError("cannot parse validator Git porcelain status")
        status = entry[:2]
        paths = [entry[3:]]
        if "R" in status or "C" in status:
            if index >= len(entries) or not entries[index]:
                raise ValueError("truncated rename/copy in validator Git status")
            paths.append(entries[index])
            index += 1
        if any(path not in allowed_relative for path in paths):
            unexpected.extend(paths)

    if unexpected:
        raise ValueError(
            "validator checkout has changes outside the exact evidence "
            f"allowlist; refuse semantic PASS attribution: {sorted(set(unexpected))[:5]}"
        )
    if not allowed_relative:
        return {
            "head_commit": head,
            "source_tree_clean": True,
            "tracked_and_untracked_status_empty": True,
        }
    encoded_allowlist = "\n".join(sorted(allowed_relative)).encode("utf-8")
    return {
        "head_commit": head,
        "source_tree_clean_outside_exact_evidence_paths": True,
        "status_includes_untracked_files": True,
        "allowed_evidence_path_count": len(allowed_relative),
        "allowed_evidence_path_set_sha256": hashlib.sha256(
            encoded_allowlist
        ).hexdigest(),
    }
