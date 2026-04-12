#!/usr/bin/env python3
"""Create a reproducibility manifest with hashes for paper artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS_DIR / "artifact_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
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
    commit = "unknown"
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(RESULTS_DIR.parent.parent.parent))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass
    payload = {
        "git_commit": commit,
        "artifact_count": len(include),
        "artifacts": include,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved artifact manifest: {OUT}")


if __name__ == "__main__":
    main()
