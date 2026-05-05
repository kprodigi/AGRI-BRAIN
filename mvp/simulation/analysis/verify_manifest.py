#!/usr/bin/env python3
"""Verify mvp/simulation/results/artifact_manifest.json.

Re-hashes every artifact listed in the manifest and asserts the
SHA-256 matches what is recorded. Optionally also asserts that the
recorded `git_commit` is a non-empty 40-hex-char string. Exits 0 on
clean verification, 1 on any mismatch or missing artifact.

Usage::

    python mvp/simulation/analysis/verify_manifest.py
    python mvp/simulation/analysis/verify_manifest.py --strict-commit

This should be run on a fresh clone after `reproduce_core.py`
to confirm the published artifacts' integrity. The 2026-04 cleanup
flagged that the manifest was produced but never verified anywhere;
this script closes that gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "mvp" / "simulation" / "results"
MANIFEST_PATH = RESULTS_DIR / "artifact_manifest.json"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


_TEXT_EXTS = {".json", ".csv", ".txt", ".md", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    """SHA-256 of *path*, with line-ending normalisation for text files.

    For text artifacts (JSON/CSV/TXT/MD/YAML) the working-tree bytes
    differ between Windows (CRLF) and Linux (LF) checkouts even when
    the git blob is identical. Normalising CRLF to LF before hashing
    keeps the recorded SHA stable across platforms so the manifest
    written on Windows verifies cleanly on a Linux CI runner. Binary
    files (PNG/PDF/etc.) are hashed as-is.
    """
    h = hashlib.sha256()
    if path.suffix.lower() in _TEXT_EXTS:
        # Read whole file (small enough; text artifacts in this manifest
        # top out around a few MB) and normalise line endings.
        data = path.read_bytes().replace(b"\r\n", b"\n")
        h.update(data)
    else:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--strict-commit",
        action="store_true",
        help="Fail when manifest.git_commit is missing or not a 40-hex string.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Treat artifacts listed in the manifest but absent from the "
            "working tree as a non-fatal warning instead of an error. "
            "Use this on CI checkouts where gitignored artifacts (e.g. "
            "mcp_interop_*.json, traces_*.json) are not in the repo. "
            "On HPC delivery runs, omit this flag so a genuinely missing "
            "artifact still fails the gate."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Path to artifact_manifest.json (default: %(default)s).",
    )
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help=(
            "When combined with --allow-missing, hard-fail on any "
            "missing artifact that matches the git-tracked allowlist "
            "(see TRACKED_PATTERNS below). Untracked / gitignored "
            "artifacts (mcp_interop_*, traces_*, learning_trajectory_*, "
            "context_alignment_*, benchmark_seeds/*) still soft-warn. "
            "This tightens the CI gate so a tracked-figure deletion "
            "cannot pass while gitignored artifacts (which are not in "
            "a fresh CI checkout) still warn cleanly. Recommended for "
            "the artifact-validation CI job."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}")
        return 1

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    git_commit = payload.get("git_commit")
    if args.strict_commit:
        if not isinstance(git_commit, str) or not _HEX40.match(git_commit):
            print(
                f"FAIL: manifest.git_commit is missing or not a "
                f"40-hex SHA: {git_commit!r}"
            )
            return 1

    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        print("FAIL: manifest.artifacts is empty or not a list")
        return 1

    # Skip self-referential and known-volatile entries:
    #  - artifact_manifest.json hashes itself, which is a chicken-and-egg
    #    paradox by construction (the written file's hash would have to
    #    contain its own value).
    #  - validation_report.json is rewritten on every validator run and
    #    is intentionally not pinned by SHA.
    SKIP = {"artifact_manifest.json", "validation_report.json"}

    # File-name patterns that ARE tracked by git (the allowlist in
    # .gitignore under ``mvp/simulation/results/``). When --require-tracked
    # is set together with --allow-missing, a missing artifact that
    # matches one of these patterns is a hard failure rather than a
    # warning. Keep this list in lockstep with the .gitignore allowlist
    # (see top-level .gitignore around line 82).
    import fnmatch as _fnmatch_v
    _TRACKED_PATTERNS = (
        "fig*.png", "fig*.pdf",
        "table*.csv",
        "benchmark_summary.json", "benchmark_significance.json",
        "stress_summary.json", "stress_degradation.csv",
        "feature_heatmap_data.json",
    )

    def _is_tracked(name: str) -> bool:
        """Return True if ``name`` matches a tracked file pattern.

        ``name`` is the manifest's relative POSIX path (e.g.
        ``"fig9_robustness.png"`` or
        ``"benchmark_seeds/seed_42.json"``); the patterns target the
        BASENAME so subdirectory paths (benchmark_seeds/*) correctly
        return False.
        """
        base = name.split("/")[-1]
        return any(_fnmatch_v.fnmatch(base, p) for p in _TRACKED_PATTERNS)

    errors = 0
    checked = 0
    skipped = 0
    missing_warnings = 0
    for rec in artifacts:
        name = rec.get("file")
        recorded_sha = rec.get("sha256")
        if not name or not recorded_sha:
            print(f"FAIL: manifest entry missing file or sha256: {rec!r}")
            errors += 1
            continue
        if name in SKIP:
            skipped += 1
            continue
        path = manifest_path.parent / name
        if not path.exists():
            # Tracked artifacts MUST exist on a fresh checkout (they
            # are in the .gitignore allowlist). Untracked artifacts
            # (mcp_interop_*, traces_*, etc.) only exist after a sim
            # run; --allow-missing is for them.
            if args.require_tracked and _is_tracked(name):
                print(
                    f"FAIL: missing tracked artifact (require-tracked): "
                    f"{name}"
                )
                errors += 1
                continue
            if args.allow_missing:
                print(f"WARN: missing artifact (allow-missing): {path.name}")
                missing_warnings += 1
                continue
            print(f"FAIL: missing artifact: {path.name}")
            errors += 1
            continue
        actual = _sha256(path)
        if actual != recorded_sha:
            print(
                f"FAIL: SHA-256 mismatch for {path.name}: "
                f"manifest={recorded_sha} actual={actual}"
            )
            errors += 1
        else:
            checked += 1

    print(
        f"verify_manifest: checked {checked} files, skipped {skipped} "
        f"(self-ref / volatile), missing_warnings {missing_warnings} "
        f"(allow-missing), errors {errors}, "
        f"git_commit={git_commit!r}"
    )
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
