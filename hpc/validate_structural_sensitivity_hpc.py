#!/usr/bin/env python3
"""Validate the isolated Slurm identity for structural sensitivity.

This is deliberately separate from the core publication-output boundary.  A
structural run must use the same clean source commit and locked Python posture,
but its plan, task outputs, analysis, and archive live in an explicitly chosen
directory outside the repository.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
_FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_RUN_TAG = re.compile(r"^sensitivity_([0-9a-f]{7})_([0-9]{8}_[0-9]{6})$")
EXPECTED_TASKS = 3_000
EXPECTED_PARAMETERS = 29
EXPECTED_TOTAL = {
    "retained_cells": 6_500,
    "executed_episodes": 24_500,
    "simulated_steps": 7_056_000,
}


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validation_errors(
    environ: dict[str, str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    require_plan: bool = True,
) -> list[str]:
    """Return every structural-run identity or isolation violation."""

    env = dict(os.environ if environ is None else environ)
    errors: list[str] = []
    repo = repo_root.resolve()

    source_commit = env.get("AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT", "").strip()
    publication_commit = env.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    if not _FULL_SHA1.fullmatch(source_commit):
        errors.append(
            "AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT must be a full lowercase Git SHA-1"
        )
    if publication_commit != source_commit:
        errors.append(
            "AGRIBRAIN_GIT_COMMIT must equal AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT"
        )

    run_tag = env.get("RUN_TAG", "").strip()
    match = _RUN_TAG.fullmatch(run_tag)
    if match is None:
        errors.append(
            "RUN_TAG must match sensitivity_<7-char-commit>_<YYYYMMDD_HHMMSS>"
        )
    elif _FULL_SHA1.fullmatch(source_commit) and match.group(1) != source_commit[:7]:
        errors.append("RUN_TAG commit prefix does not match the sensitivity source commit")

    root_raw = env.get("AGRIBRAIN_SENSITIVITY_ROOT", "").strip()
    run_raw = env.get("SENSITIVITY_RUN_DIR", "").strip()
    plan_raw = env.get("SENSITIVITY_RUN_PLAN", "").strip()
    root = Path(root_raw) if root_raw else None
    run_dir = Path(run_raw) if run_raw else None
    plan_path = Path(plan_raw) if plan_raw else None

    if root is None or not root.is_absolute():
        errors.append("AGRIBRAIN_SENSITIVITY_ROOT must be an absolute path")
    if run_dir is None or not run_dir.is_absolute():
        errors.append("SENSITIVITY_RUN_DIR must be an absolute path")
    if plan_path is None or not plan_path.is_absolute():
        errors.append("SENSITIVITY_RUN_PLAN must be an absolute path")

    resolved_root = root.resolve() if root is not None and root.is_absolute() else None
    resolved_run = (
        run_dir.resolve() if run_dir is not None and run_dir.is_absolute() else None
    )
    resolved_plan = (
        plan_path.resolve() if plan_path is not None and plan_path.is_absolute() else None
    )
    if resolved_root is not None and _inside(resolved_root, repo):
        errors.append("AGRIBRAIN_SENSITIVITY_ROOT must be outside the repository")
    if resolved_run is not None and _inside(resolved_run, repo):
        errors.append("SENSITIVITY_RUN_DIR must be outside the repository")
    if resolved_root is not None and resolved_run is not None and run_tag:
        if resolved_run != resolved_root / run_tag:
            errors.append("SENSITIVITY_RUN_DIR must be ROOT/RUN_TAG exactly")
    if resolved_run is not None and resolved_plan is not None:
        if resolved_plan != resolved_run / "run_plan.json":
            errors.append("SENSITIVITY_RUN_PLAN must be RUN_DIR/run_plan.json exactly")

    if resolved_root is not None and not resolved_root.is_dir():
        errors.append("AGRIBRAIN_SENSITIVITY_ROOT does not exist as a directory")
    if resolved_run is not None and not resolved_run.is_dir():
        errors.append("SENSITIVITY_RUN_DIR does not exist as a directory")
    if run_dir is not None and run_dir.is_symlink():
        errors.append("SENSITIVITY_RUN_DIR must not be a symbolic link")

    expected_venv = f".publication_venvs/{run_tag}" if run_tag else ""
    if env.get("AGRIBRAIN_VENV", "").strip().replace("\\", "/") != expected_venv:
        errors.append("AGRIBRAIN_VENV must be .publication_venvs/RUN_TAG")

    if require_plan and resolved_plan is not None:
        if not resolved_plan.is_file():
            errors.append("SENSITIVITY_RUN_PLAN does not exist")
        else:
            try:
                from mvp.simulation.sensitivity.parameters import PARAMETERS
                from mvp.simulation.sensitivity.run_structural_sensitivity import (
                    _load_plan_bundle,
                )

                plan, _protocol, _design, manifest = _load_plan_bundle(resolved_plan)
                if plan.get("analysis_label") != "structural sensitivity":
                    errors.append("run plan is not labelled structural sensitivity")
                if plan.get("execution_scope") != "structural_sensitivity_only":
                    errors.append("run plan lacks the structural-only execution boundary")
                if plan.get("probability_interpretation") is not False:
                    errors.append("run plan incorrectly permits probability interpretation")
                if plan.get("run_tag") != run_tag:
                    errors.append("run plan RUN_TAG does not match the Slurm identity")
                if plan.get("source_commit") != source_commit:
                    errors.append("run plan source commit does not match the Slurm identity")
                if plan.get("source_tree_clean_at_generation") is not True:
                    errors.append("run plan was not generated from a clean source tree")
                if plan.get("development_only_dirty_plan") is not False:
                    errors.append("development-only dirty plan cannot run on Slurm")
                if Path(str(plan.get("protocol", {}).get("path", ""))).is_absolute():
                    errors.append("run plan protocol reference must be bundle-relative")
                if int(manifest.get("n_tasks", -1)) != EXPECTED_TASKS:
                    errors.append("run plan must contain exactly 3,000 tasks")
                if manifest.get("accounting", {}).get("total") != EXPECTED_TOTAL:
                    errors.append("run plan accounting does not match 6,500/24,500/7,056,000")
                if len(PARAMETERS) != EXPECTED_PARAMETERS:
                    errors.append("structural registry must contain exactly 29 active factors")
                if "slca_carbon_cap" not in {item.key for item in PARAMETERS}:
                    errors.append("structural registry omits slca_carbon_cap")
            except Exception as exc:  # noqa: BLE001 - fail closed at the queue boundary
                errors.append(f"cannot validate structural run plan: {exc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-missing-plan",
        action="store_true",
        help="validate paths and identity before the immutable plan is generated",
    )
    args = parser.parse_args(argv)
    errors = validation_errors(require_plan=not args.allow_missing_plan)
    if errors:
        print("BLOCK: invalid structural-sensitivity Slurm identity:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print("Structural-sensitivity Slurm identity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
