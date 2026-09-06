"""The divergent legacy experiment path must never emit benchmark artifacts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_legacy_experiment_runner_fails_before_writing_results(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_experiments.py"
    )
    output_dir = tmp_path / "publication_like_output"

    completed = subprocess.run(
        [sys.executable, str(script), "--out", str(output_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "legacy equations do not match the publication methodology" in (
        completed.stderr
    )
    assert "No results were generated" in completed.stderr
    assert not output_dir.exists()
