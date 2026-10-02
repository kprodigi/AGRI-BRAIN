#!/usr/bin/env python3
"""Fail-closed static preflight for publication launch/evidence wiring."""
from __future__ import annotations

import argparse
from pathlib import Path

CORE_CONTRACT = {
    "hpc/publication_env.sh": ("export FULL_EVIDENCE_CAPTURE=1",),
    "hpc/validate_publication_env.py": ('"FULL_EVIDENCE_CAPTURE": "1"',),
    "hpc/hpc_run.sh": (
        "hpc/validate_launch_preflight.py --workflow core",
        "hpc/validate_pinn_artifacts.py",
        "hpc/hpc_seed.sh",
        "hpc/hpc_stress.sh",
        "hpc/hpc_publish.sh",
        "--dependency=afterok:${SEED_JOB}:${STRESS_JOB}",
    ),
    "hpc/hpc_seed.sh": (
        "source hpc/publication_env.sh",
        "python hpc/validate_pinn_artifacts.py",
        "python hpc/run_with_resource_receipt.py",
        "python hpc/validate_complete_episode_evidence.py",
        "complete_episode_evidence_manifest.json",
    ),
    "hpc/hpc_stress.sh": (
        "source hpc/publication_env.sh",
        "python hpc/run_with_resource_receipt.py",
        "python hpc/validate_complete_episode_evidence.py",
        "complete_episode_evidence_manifest.json",
    ),
    "hpc/hpc_publish.sh": (
        "source hpc/publication_env.sh",
        "python hpc/capture_slurm_accounting.py",
        "python hpc/validate_raw_publication_inputs.py",
        "python hpc/validate_decision_ledgers.py",
        "python hpc/build_complete_run_evidence.py",
        "--expected-episodes 6100",
        "--expected-runtime-receipts 25",
        "--expected-scheduler-tasks 25",
    ),
}

STRUCTURAL_CONTRACT = {
    "hpc/publication_env.sh": ("export FULL_EVIDENCE_CAPTURE=1",),
    "hpc/validate_publication_env.py": ('"FULL_EVIDENCE_CAPTURE": "1"',),
    "hpc/hpc_sensitivity_run.sh": (
        '"$PUBLICATION_PYTHON_BIN" hpc/validate_launch_preflight.py --workflow structural',
        '"$PUBLICATION_PYTHON_BIN" hpc/validate_pinn_artifacts.py',
        '"$PUBLICATION_PYTHON_BIN" hpc/validate_source_checkout.py',
        "hpc/hpc_sensitivity_task.sh",
        "hpc/hpc_sensitivity_publish.sh",
        '--dependency="afterok:${PREVIOUS_JOB}"',
    ),
    "hpc/hpc_sensitivity_task.sh": (
        "source hpc/publication_env.sh",
        "python hpc/validate_pinn_artifacts.py",
        "python hpc/run_with_resource_receipt.py",
        "run_structural_sensitivity run-task",
        "--resume",
    ),
    "hpc/hpc_sensitivity_publish.sh": (
        "source hpc/publication_env.sh",
        "python hpc/capture_slurm_accounting.py",
        "run_structural_sensitivity status",
        "finalize_structural_sensitivity",
    ),
    "mvp/simulation/sensitivity/finalize_structural_sensitivity.py": (
        "complete_episode_evidence",
        "runtime_receipts",
        "scheduler_accounting",
    ),
}


def validate_contract(repo_root: Path, workflow: str) -> list[str]:
    contracts = []
    if workflow in {"core", "all"}:
        contracts.append(("core", CORE_CONTRACT))
    if workflow in {"structural", "all"}:
        contracts.append(("structural", STRUCTURAL_CONTRACT))
    failures: list[str] = []
    for label, contract in contracts:
        for relative, required_fragments in contract.items():
            path = repo_root / relative
            if not path.is_file() or path.is_symlink():
                failures.append(f"{label}: required regular file missing: {relative}")
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in required_fragments:
                if fragment not in text:
                    failures.append(
                        f"{label}: {relative} lacks required wiring: {fragment!r}"
                    )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=("core", "structural", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    failures = validate_contract(args.repo_root.resolve(), args.workflow)
    if failures:
        for failure in failures:
            print(f"BLOCK: {failure}")
        return 1
    print(f"Launch preflight OK: {args.workflow} complete-evidence wiring is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
