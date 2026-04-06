#!/usr/bin/env python3
"""One-shot reproducibility runner for core research outputs.

Respects DETERMINISTIC_MODE (default: false / stochastic).
In stochastic mode the regression guard is skipped (exact-value checks
are meaningless under seeded perturbation noise).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = ROOT / "mvp" / "simulation"

# Read mode (same logic as stochastic.py)
_DETERMINISTIC = os.environ.get("DETERMINISTIC_MODE", "false").lower() == "true"


def _timeout_for(stage_name: str, default_s: int) -> int:
    key = f"REPRO_TIMEOUT_{stage_name.upper()}_S"
    raw = os.environ.get(key, str(default_s)).strip()
    try:
        return max(int(raw), 1)
    except ValueError:
        return default_s


def _run(stage_name: str, cmd: list[str], timeout_s: int) -> None:
    print(f"\n[{stage_name}] timeout={timeout_s}s")
    print(">", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[{stage_name}] TIMEOUT after {timeout_s}s")
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    print(f"[{stage_name}] PASS")


def _benchmark_seed_list() -> list[int]:
    raw = os.environ.get("BENCHMARK_SEEDS", "42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515").strip()
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out or [42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
                   606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515]


def main() -> None:
    mode_label = "DETERMINISTIC" if _DETERMINISTIC else "STOCHASTIC"
    print(f"Reproducibility pipeline — mode: {mode_label}")

    # Timeouts sized for long simulation episodes.
    stages = [
        ("generate_results", [sys.executable, str(SIM_DIR / "generate_results.py")], _timeout_for("generate_results", 36000)),
        ("validate_results", [sys.executable, str(SIM_DIR / "validate_results.py")], _timeout_for("validate_results", 300)),
        ("run_regression_guard", [sys.executable, str(SIM_DIR / "run_regression_guard.py")], _timeout_for("run_regression_guard", 300)),
        ("run_stress_suite", [sys.executable, str(SIM_DIR / "run_stress_suite.py")], _timeout_for("run_stress_suite", 36000)),
        ("run_external_validity", [sys.executable, str(SIM_DIR / "run_external_validity.py")], _timeout_for("run_external_validity", 18000)),
    ]
    for name, cmd, timeout_s in stages:
        _run(name, cmd, timeout_s)
    for seed in _benchmark_seed_list():
        _run(
            f"run_single_seed_{seed}",
            [sys.executable, str(SIM_DIR / "run_single_seed.py"), str(seed)],
            _timeout_for("run_single_seed", 36000),
        )
    _run("aggregate_seeds", [sys.executable, str(SIM_DIR / "aggregate_seeds.py")], _timeout_for("aggregate_seeds", 600))
    _run("generate_figures", [sys.executable, str(SIM_DIR / "generate_figures.py")], _timeout_for("generate_figures", 1800))
    _run("export_paper_evidence", [sys.executable, str(SIM_DIR / "export_paper_evidence.py")], _timeout_for("export_paper_evidence", 600))
    _run("build_artifact_manifest", [sys.executable, str(SIM_DIR / "build_artifact_manifest.py")], _timeout_for("build_artifact_manifest", 120))

    # Optional, non-canonical context-only benchmark export.
    if os.environ.get("REPRO_RUN_CONTEXT_BENCHMARK", "false").lower() == "true":
        _run(
            "run_benchmark_suite",
            [sys.executable, str(SIM_DIR / "run_benchmark_suite.py")],
            _timeout_for("run_benchmark_suite", 180000),
        )
    print(f"Core reproducibility pipeline complete ({mode_label} mode).")


if __name__ == "__main__":
    main()

