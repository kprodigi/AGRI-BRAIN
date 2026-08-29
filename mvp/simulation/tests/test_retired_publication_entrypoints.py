"""Fail-closed contracts for obsolete publication entry points."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("relative_path", "required_references"),
    (
        (
            "mvp/simulation/focapo_figures.py",
            (
                "hpc/hpc_run.sh",
                "mvp/simulation/benchmarks/aggregate_seeds.py",
                "mvp/simulation/generate_figures.py",
            ),
        ),
        (
            "mvp/simulation/reproduce_core.py",
            ("hpc/hpc_run.sh", "seed -> stress -> publish"),
        ),
        (
            "mvp/simulation/_run_h2_seed.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "normal per-seed"),
        ),
        (
            "mvp/simulation/_run_h2_all.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "normal per-seed"),
        ),
        (
            "mvp/simulation/_run_h2_stability.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "normal per-seed"),
        ),
        (
            "mvp/simulation/benchmarks/run_over_steer_ablation.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "episode-indexed"),
        ),
        (
            "mvp/simulation/benchmarks/run_temporal_stability.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "episode-indexed"),
        ),
        (
            "mvp/simulation/_h2_spec_curve.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "decision_ledger_per_seed"),
        ),
        (
            "mvp/simulation/_h2_stability_compare.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "decision_ledger_per_seed"),
        ),
        (
            "mvp/simulation/regenerate_figures_from_seeds.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "generate_figures.py"),
        ),
        (
            "mvp/simulation/tests/stochastic_rank_check.py",
            ("hpc/hpc_run.sh", "hpc/hpc_publish.sh", "3-adaptation"),
        ),
        (
            "agribrain/backend/scripts/calibrate_governance.py",
            ("hpc/hpc_sensitivity_run.sh", "hpc/hpc_sensitivity_publish.sh"),
        ),
        (
            "agribrain/backend/scripts/sweep_price_sensitivity.py",
            ("hpc/hpc_sensitivity_run.sh", "hpc/hpc_sensitivity_publish.sh"),
        ),
    ),
)
def test_retired_entrypoint_fails_without_emitting_outputs(
    tmp_path: Path,
    relative_path: str,
    required_references: tuple[str, ...],
) -> None:
    script = REPO_ROOT / relative_path
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "RETIRED" in completed.stderr
    for reference in required_references:
        assert reference in completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "relative_path",
    (
        "mvp/simulation/focapo_figures.py",
        "mvp/simulation/reproduce_core.py",
        "mvp/simulation/_run_h2_seed.py",
        "mvp/simulation/_run_h2_all.py",
        "mvp/simulation/_run_h2_stability.py",
        "mvp/simulation/benchmarks/run_over_steer_ablation.py",
        "mvp/simulation/benchmarks/run_temporal_stability.py",
        "mvp/simulation/_h2_spec_curve.py",
        "mvp/simulation/_h2_stability_compare.py",
        "mvp/simulation/regenerate_figures_from_seeds.py",
        "mvp/simulation/tests/stochastic_rank_check.py",
        "agribrain/backend/scripts/calibrate_governance.py",
        "agribrain/backend/scripts/sweep_price_sensitivity.py",
    ),
)
def test_retired_module_import_is_inert(relative_path: str) -> None:
    script = REPO_ROOT / relative_path
    module_name = "_retired_" + script.stem
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.RETIRED is True
    assert module.main([]) == module.EXIT_RETIRED == 2


def test_retired_sources_contain_no_generation_or_process_stack() -> None:
    forbidden_tokens = (
        "import matplotlib",
        "import numpy",
        "import pandas",
        "import subprocess",
        ".savefig(",
        ".write_text(",
        ".write_bytes(",
        "subprocess.run(",
        "subprocess.Popen(",
    )
    for relative_path in (
        "mvp/simulation/focapo_figures.py",
        "mvp/simulation/reproduce_core.py",
        "mvp/simulation/_run_h2_seed.py",
        "mvp/simulation/_run_h2_all.py",
        "mvp/simulation/_run_h2_stability.py",
        "mvp/simulation/benchmarks/run_over_steer_ablation.py",
        "mvp/simulation/benchmarks/run_temporal_stability.py",
        "mvp/simulation/_h2_spec_curve.py",
        "mvp/simulation/_h2_stability_compare.py",
        "mvp/simulation/regenerate_figures_from_seeds.py",
        "mvp/simulation/tests/stochastic_rank_check.py",
        "agribrain/backend/scripts/calibrate_governance.py",
        "agribrain/backend/scripts/sweep_price_sensitivity.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source
