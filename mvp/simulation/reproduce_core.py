#!/usr/bin/env python3
"""One-shot reproducibility runner for core research outputs."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = ROOT / "mvp" / "simulation"


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


def main() -> None:
    stages = [
        ("generate_results", [sys.executable, str(SIM_DIR / "generate_results.py")], _timeout_for("generate_results", 2400)),
        ("validate_results", [sys.executable, str(SIM_DIR / "validate_results.py")], _timeout_for("validate_results", 120)),
        ("run_regression_guard", [sys.executable, str(SIM_DIR / "run_regression_guard.py")], _timeout_for("run_regression_guard", 120)),
        ("run_benchmark_suite", [sys.executable, str(SIM_DIR / "run_benchmark_suite.py")], _timeout_for("run_benchmark_suite", 2400)),
        ("run_stress_suite", [sys.executable, str(SIM_DIR / "run_stress_suite.py")], _timeout_for("run_stress_suite", 2400)),
        ("generate_figures", [sys.executable, str(SIM_DIR / "generate_figures.py")], _timeout_for("generate_figures", 1800)),
        ("export_paper_evidence", [sys.executable, str(SIM_DIR / "export_paper_evidence.py")], _timeout_for("export_paper_evidence", 600)),
        ("build_artifact_manifest", [sys.executable, str(SIM_DIR / "build_artifact_manifest.py")], _timeout_for("build_artifact_manifest", 120)),
    ]
    for name, cmd, timeout_s in stages:
        _run(name, cmd, timeout_s)
    print("Core reproducibility pipeline complete.")


if __name__ == "__main__":
    main()

