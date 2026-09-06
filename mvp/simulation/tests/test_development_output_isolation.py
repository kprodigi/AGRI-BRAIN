"""Standalone simulation output must never overwrite publication evidence."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mvp.simulation import generate_results as gr


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_standalone_cli_rejects_publication_results_directory(monkeypatch):
    canonical = Path(gr.__file__).resolve().parent / "results"
    monkeypatch.setenv("AGRIBRAIN_DEVELOPMENT_OUTPUT_DIR", str(canonical))
    with pytest.raises(RuntimeError, match="development-only"):
        gr.configure_standalone_development_output()


def test_standalone_cli_selects_new_isolated_directory(tmp_path, monkeypatch):
    target = tmp_path / "development_run"
    monkeypatch.setenv("AGRIBRAIN_DEVELOPMENT_OUTPUT_DIR", str(target))
    previous = gr.RESULTS_DIR
    try:
        selected = gr.configure_standalone_development_output()
        assert selected == target.resolve()
        assert selected.is_dir()
        assert selected.name == "development_run"
    finally:
        gr.RESULTS_DIR = previous


@pytest.mark.parametrize(
    "script",
    (
        "mvp/simulation/benchmarks/aggregate_seeds.py",
        "mvp/simulation/benchmarks/aggregate_stress_outputs.py",
    ),
)
def test_confirmatory_aggregators_require_explicit_input_and_output(script):
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / script)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "required" in completed.stderr


@pytest.mark.parametrize(
    ("script", "input_flag"),
    (
        ("mvp/simulation/benchmarks/aggregate_seeds.py", "--seed-root"),
        ("mvp/simulation/benchmarks/aggregate_stress_outputs.py", "--input-root"),
    ),
)
def test_direct_aggregators_cannot_overwrite_canonical_results(
    tmp_path, script, input_flag,
):
    input_root = tmp_path / "input"
    input_root.mkdir()
    canonical = REPO_ROOT / "mvp" / "simulation" / "results"
    env = dict(os.environ)
    env.pop("AGRIBRAIN_PUBLICATION_AGGREGATION", None)
    env.pop("STRICT_VALIDATION", None)
    completed = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / script),
            input_flag, str(input_root), "--output-dir", str(canonical),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "restricted to the locked HPC publisher" in completed.stderr
