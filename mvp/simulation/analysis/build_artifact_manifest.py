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
    for p in sorted(RESULTS_DIR.glob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        include.append(
            {
                "file": p.name,
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
    # Step 2's failure is logged as a WARNING (not debug) so the user
    # sees it even when env-var override is unavailable, rather than a
    # silent "unknown" reaching the JSON and breaking CI 30 seconds
    # later in the next stage.
    commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if not commit:
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(RESULTS_DIR.parent.parent.parent),
                    stderr=subprocess.PIPE,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception as _exc:
            _log.warning(
                "build_artifact_manifest: git rev-parse HEAD failed (%s) and "
                "AGRIBRAIN_GIT_COMMIT env var not set; manifest will record "
                "'unknown' which verify_manifest.py --strict-commit will "
                "reject. To fix this on HPC, export AGRIBRAIN_GIT_COMMIT="
                "$(git rev-parse HEAD) before invoking the aggregate job.",
                _exc,
            )
            commit = "unknown"
    payload = {
        "git_commit": commit,
        "artifact_count": len(include),
        "artifacts": include,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved artifact manifest: {OUT}")


if __name__ == "__main__":
    main()
