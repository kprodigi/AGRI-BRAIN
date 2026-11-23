"""Rerun the revised No-context arm, optionally with paired main comparators.

Run as a module from a clean committed checkout after sourcing
``hpc/publication_env.sh``. This produces a separate evidence set, not an
11-mode publication bundle. See docs/NO_CONTEXT_RERUN.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from hpc.slurm_execution_provenance import CORE_SEEDS, CORE_SCENARIOS

ROOT = Path(__file__).resolve().parents[3]
MODE_DEFINITION = "no_context_without_tools_retrieval_or_peers_v2"
COMPARISON_DEFINITION = "agribrain_standard_rag_vs_no_injection_v1"


def selected_modes(include_comparators: bool, include_agribrain: bool = False) -> list[str]:
    if include_comparators and include_agribrain:
        raise ValueError("Select the learned pair or all three comparators, not both")
    if include_comparators:
        return ["static", "no_context", "agribrain"]
    return ["no_context", "agribrain"] if include_agribrain else ["no_context"]


def reserve_output(output_root: Path, seed: int) -> Path:
    """Atomically reserve a fresh seed directory; never reuse old evidence."""
    root = output_root.resolve()
    if root == ROOT or ROOT in root.parents:
        raise ValueError("Use an output root outside the source checkout")
    target = root / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def build_envelope(data: dict, *, seed: int, modes: list[str], metadata: dict) -> dict:
    from mvp.simulation.benchmarks.episode_archive import to_json_native
    from mvp.simulation.benchmarks.trace_contract import TRACE_FIELDS, validate_trace_cell

    if set(data["results"]) != set(CORE_SCENARIOS):
        raise ValueError("Incomplete scenario set")
    scenarios, traces = {}, {}
    for scenario, arms in data["results"].items():
        if set(arms) != set(modes):
            raise ValueError(f"Unexpected modes for {scenario}")
        scenarios[scenario], traces[scenario] = {}, {}
        for mode, episode in arms.items():
            trace = to_json_native({field: episode[field] for field in TRACE_FIELDS})
            validate_trace_cell(trace, where=f"seed={seed}/{scenario}/{mode}")
            traces[scenario][mode] = trace
            scenarios[scenario][mode] = to_json_native({
                key: value for key, value in episode.items()
                if not key.startswith("_") and key not in TRACE_FIELDS
            })
    return {
        "_meta": metadata,
        "seed": seed,
        "trace_schema_version": data["trace_schema_version"],
        "state_design": data["state_design"],
        "forecast_protocol": data["forecast_protocol"],
        "scenarios": scenarios,
        "traces": traces,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=CORE_SEEDS)
    parser.add_argument("--output-root", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--include-comparators", action="store_true",
                        help="Also rerun Static and full AGRI-BRAIN on the same streams")
    scope.add_argument("--include-agribrain", action="store_true",
                       help="Rerun No-context and standard-RAG AGRI-BRAIN; reuse Static")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    # Check the canonical environment before assigning the seed-specific ledger
    # path or importing the simulator (its forecast settings are import-time).
    from hpc.capture_publication_environment import main as capture_environment
    from hpc.validate_source_checkout import validation_errors
    from hpc.validate_source_snapshot import tracked_source_digest
    from hpc.validate_pinn_artifacts import validate as validate_pinn

    errors = validation_errors() + validate_pinn(ROOT)
    root = args.output_root.resolve()
    if root == ROOT or ROOT in root.parents:
        errors.append("output root must be outside the source checkout")
    if (root / f"seed_{args.seed}").exists():
        errors.append(f"seed_{args.seed} already exists; choose a fresh output root")
    if errors:
        raise RuntimeError("Rerun preflight failed: " + "; ".join(errors))
    capture_environment(["--validate-only"])
    if args.check_only:
        print("Revised No-context rerun preflight passed; no simulation executed")
        return 0

    out = reserve_output(root, args.seed)
    capture_environment(["--output", str(out / "environment.json")])
    source_digest, source_file_count = tracked_source_digest(ROOT)
    # Do not inherit a tree digest from an older publication shell session.
    os.environ["AGRIBRAIN_SOURCE_TREE_SHA256"] = source_digest
    started = datetime.now(timezone.utc).isoformat()
    ledger_root = out / "decision_ledgers"
    os.environ["DECISION_LEDGER_DIR"] = str(ledger_root)
    modes = selected_modes(args.include_comparators, args.include_agribrain)

    from mvp.simulation import generate_results
    from src.models.mode_capabilities import capabilities_for
    from hpc.validate_complete_episode_evidence import validate_complete_evidence

    old_results = generate_results.RESULTS_DIR
    try:
        generate_results.RESULTS_DIR = out / "auxiliary"
        data = generate_results.run_all(args.seed, modes=modes)
    finally:
        generate_results.RESULTS_DIR = old_results
        os.environ.pop("DECISION_LEDGER_DIR", None)

    groups = len(CORE_SCENARIOS) * len(modes)
    episodes = len(CORE_SCENARIOS) * sum(capabilities_for(m).episode_count for m in modes)
    evidence = validate_complete_evidence(
        ledger_root, expected_groups=groups, expected_episodes=episodes,
        expected_adaptation_ledgers=episodes - groups,
        expected_final_ledgers=groups,
        manifest_path=out / "evidence_manifest.json",
    )
    errors = validation_errors()
    if tracked_source_digest(ROOT) != (source_digest, source_file_count):
        errors.append("tracked source byte digest changed")
    if errors:
        raise RuntimeError("Source changed during rerun: " + "; ".join(errors))
    payload = build_envelope(data, seed=args.seed, modes=modes, metadata={
        "source_commit": os.environ["AGRIBRAIN_GIT_COMMIT"],
        "source_tree_sha256": source_digest,
        "source_snapshot_scope": "clean committed checkout checked before and after execution",
        "run_tag": os.environ["RUN_TAG"],
        "mode_definition": MODE_DEFINITION,
        "comparison_definition": COMPARISON_DEFINITION,
        "mode_capabilities": {mode: asdict(capabilities_for(mode)) for mode in modes},
        "scope": "revised comparator rerun; not the full publication bundle",
        "historical_no_context_results_compatible": False,
        "episode_scope": "episode 3 frozen evaluation; episodes 0-2 adapt learned arms",
        "evidence_counts": evidence["counts"],
        "evidence_manifest_sha256": evidence["manifest_sha256"],
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": {key: os.environ[key] for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        ) if key in os.environ},
    })
    # Only a successfully validated seed gets its final envelope. Failed runs
    # retain diagnostic evidence and require a fresh output location to retry.
    target = out / f"seed_{args.seed}.json"
    temp = out / "seed.pending.json"
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(target)
    print(f"COMPLETE: {target} ({episodes} episodes, {groups} retained evaluations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
