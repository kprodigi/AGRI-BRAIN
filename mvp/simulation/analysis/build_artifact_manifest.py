#!/usr/bin/env python3
"""Create a reproducibility manifest with hashes for paper artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

_log = logging.getLogger(__name__)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS_DIR / "artifact_manifest.json"


_TEXT_EXTS = {".json", ".csv", ".txt", ".md", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    """SHA-256 of *path*, with line-ending normalisation for text files.

    Same canonicalisation as ``verify_manifest._sha256``: for text
    artifacts (JSON/CSV/TXT/MD/YAML) we normalise CRLF to LF before
    hashing so the recorded SHA is stable across Windows and Linux
    working-tree checkouts. Binary files are hashed as-is.
    """
    h = hashlib.sha256()
    if path.suffix.lower() in _TEXT_EXTS:
        data = path.read_bytes().replace(b"\r\n", b"\n")
        h.update(data)
    else:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    include = []
    # Recursive glob so the manifest also covers per-seed dumps
    # (benchmark_seeds/*/seed_*.json) and per-step audit-trail
    # ledgers (decision_ledger/*.jsonl). The earlier non-recursive
    # ``RESULTS_DIR.glob("*")`` skipped both subdirectories
    # entirely, so verify_manifest.py --strict-commit could not
    # detect tampering of the per-seed evidence the manuscript
    # cites. Filter out hidden files and any files inside
    # ``__pycache__`` regardless of nesting depth.
    for p in sorted(RESULTS_DIR.rglob("*")):
        if not p.is_file():
            continue
        # Skip dotfiles (.gitkeep etc.) and any path component that
        # starts with "." (covers .ipynb_checkpoints / .pytest_cache).
        if any(part.startswith(".") for part in p.relative_to(RESULTS_DIR).parts):
            continue
        if "__pycache__" in p.parts:
            continue
        # Manifest path is the relative POSIX path so the JSON is
        # platform-stable (Windows backslashes never reach the
        # serialized output). The verify_manifest.py reader joins
        # this against RESULTS_DIR using Path() which is
        # cross-platform.
        rel = p.relative_to(RESULTS_DIR).as_posix()
        include.append(
            {
                "file": rel,
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
            }
        )
    # Resolution order for the manifest's git_commit:
    #   1. AGRIBRAIN_GIT_COMMIT env var (HPC pipelines export this so the
    #      manifest stamp survives slurm contexts where git is not in PATH);
    #   2. git rev-parse HEAD subprocess (the local-dev path);
    #   3. "unknown" sentinel (last-resort fallback; verify_manifest.py
    #      with --strict-commit will reject this, by design).
    #
    # Reproducibility hardening (post-2026-04 audit fix):
    #   - When the env-var override is supplied, we cross-check it against
    #     ``git rev-parse HEAD`` (when git is available) and reject the
    #     stamping if they disagree. A user could otherwise export an old
    #     SHA (deliberately or accidentally) and the manifest would record
    #     a commit that does not match the working tree the simulator
    #     actually ran on.
    #   - When the working tree is dirty (``git status --porcelain``
    #     non-empty), we record ``dirty=True`` and append "+dirty" to the
    #     stamped commit so verify_manifest.py --strict-commit (which
    #     rejects non-clean commits by default) catches the mismatch.
    #     This prevents the silent "the run matches commit X" claim being
    #     true-modulo-uncommitted-changes which is the canonical
    #     non-reproducible-stamping failure mode.
    #   - ``AGRIBRAIN_ALLOW_DIRTY=1`` overrides the dirty rejection for
    #     contexts where the user has explicitly accepted the dirty stamp
    #     (e.g. local exploratory runs where reproducibility is not the
    #     goal). The override still appends "+dirty" to the SHA so the
    #     resulting manifest is honestly labeled.
    git_root = str(RESULTS_DIR.parent.parent.parent)
    head_sha = ""
    is_dirty = False
    git_available = True
    try:
        head_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=git_root,
                stderr=subprocess.PIPE,
            ).decode("utf-8").strip()
        )
    except Exception:
        git_available = False
    try:
        if git_available:
            porcelain = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=git_root,
                    stderr=subprocess.PIPE,
                ).decode("utf-8").strip()
            )
            # Filter out changes inside ``mvp/simulation/results/``
            # before deciding "dirty". The HPC pipeline regenerates
            # every figure / CSV / JSON in that directory by design,
            # so those paths are EXPECTED to differ from their
            # committed versions at manifest-build time. The
            # dirty-tree refusal exists to catch uncommitted CODE
            # changes (which would silently break the
            # commit-SHA-stamps-the-actual-code reproducibility
            # guarantee), not run-artifact regeneration. Without
            # this filter, the manifest builder would have refused
            # every HPC run because the post-aggregation tree
            # always carries modified figures/tables relative to
            # the committed snapshot.
            non_artifact_lines = []
            for raw_line in porcelain.split("\n"):
                if not raw_line.strip():
                    continue
                # `git status --porcelain` lines are "XX path", with
                # rename targets formatted "XX old -> new".
                path = raw_line[3:].strip() if len(raw_line) >= 4 else ""
                if " -> " in path:
                    path = path.split(" -> ", 1)[1].strip()
                if path.startswith("mvp/simulation/results/"):
                    continue
                non_artifact_lines.append(raw_line)
            is_dirty = bool(non_artifact_lines)
    except Exception:
        # If status fails, assume dirty to avoid stamping a falsely-clean
        # SHA — better to fail loud than silent.
        is_dirty = True

    env_override = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if env_override:
        if git_available and head_sha and env_override != head_sha:
            raise RuntimeError(
                f"AGRIBRAIN_GIT_COMMIT env var ({env_override!r}) does not "
                f"match the working tree's HEAD ({head_sha!r}). Stamping a "
                f"manifest with a SHA that does not reflect the actual "
                f"checked-out code is exactly the silent-non-reproducibility "
                f"failure mode this check exists to prevent. To fix: unset "
                f"AGRIBRAIN_GIT_COMMIT or re-export it as "
                f"$(git rev-parse HEAD)."
            )
        commit = env_override
    elif head_sha:
        commit = head_sha
    else:
        _log.warning(
            "build_artifact_manifest: git rev-parse HEAD failed and "
            "AGRIBRAIN_GIT_COMMIT env var not set; manifest will record "
            "'unknown' which verify_manifest.py --strict-commit will "
            "reject. To fix this on HPC, export AGRIBRAIN_GIT_COMMIT="
            "$(git rev-parse HEAD) before invoking the aggregate job."
        )
        commit = "unknown"

    if is_dirty:
        allow_dirty = os.environ.get("AGRIBRAIN_ALLOW_DIRTY", "").strip() == "1"
        if not allow_dirty:
            raise RuntimeError(
                "build_artifact_manifest: working tree is dirty (uncommitted "
                "changes present). Stamping the manifest now would record a "
                "commit SHA that does not capture the actual code that ran. "
                "To proceed, either commit the changes (recommended) or set "
                "AGRIBRAIN_ALLOW_DIRTY=1 to explicitly accept a dirty stamp; "
                "the latter will append '+dirty' to the SHA so reviewers can "
                "see the stamp is not reproducible."
            )
        # Honest labeling: caller has acknowledged the dirty stamp.
        commit = f"{commit}+dirty"

    payload = {
        "git_commit": commit,
        "git_dirty": bool(is_dirty),
        "artifact_count": len(include),
        "artifacts": include,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved artifact manifest: {OUT}")


if __name__ == "__main__":
    main()
