"""Check a complete 20-seed learned-pair run before downloading its evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc.slurm_execution_provenance import CORE_SEEDS, CORE_SCENARIOS
from mvp.simulation.benchmarks.rerun_no_context import COMPARISON_DEFINITION
from mvp.simulation.benchmarks.trace_contract import validate_trace_cell


def verify(root: Path) -> dict:
    identities = set()
    modes = {"no_context", "agribrain"}
    for seed in CORE_SEEDS:
        path = root / f"seed_{seed}" / f"seed_{seed}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["_meta"]
        if data["seed"] != seed or meta["comparison_definition"] != COMPARISON_DEFINITION:
            raise ValueError(f"Wrong seed or treatment identity: {path}")
        identities.add((meta["source_commit"], meta["source_tree_sha256"], meta["run_tag"]))
        caps = meta["mode_capabilities"]
        if set(caps) != modes or caps["agribrain"]["retrieval_kind"] != "standard":
            raise ValueError(f"Expected two learned modes and standard RAG: {path}")
        if caps["no_context"]["peer_messages"] or caps["no_context"]["context_kind"] is not None:
            raise ValueError(f"No-context injection is not disabled: {path}")
        for key in ("scenarios", "traces"):
            if set(data[key]) != set(CORE_SCENARIOS):
                raise ValueError(f"Incomplete scenario panel: {path}")
            for scenario in CORE_SCENARIOS:
                if set(data[key][scenario]) != modes:
                    raise ValueError(f"Incomplete mode panel: {path}/{scenario}")
        for scenario in CORE_SCENARIOS:
            for mode in modes:
                validate_trace_cell(data["traces"][scenario][mode], where=f"{path}/{scenario}/{mode}")
                if not data["scenarios"][scenario][mode]["learner_freeze_summary"]["learners_frozen"]:
                    raise ValueError(f"Unfrozen evaluation: {path}/{scenario}/{mode}")
    if len(identities) != 1:
        raise ValueError("Seed outputs mix source commits, source bytes, or run tags")
    from hpc.validate_complete_episode_evidence import validate_complete_evidence
    result = validate_complete_evidence(
        root, expected_groups=200, expected_episodes=800,
        expected_adaptation_ledgers=600, expected_final_ledgers=200,
        manifest_path=root / "COMPLETE_EVIDENCE_MANIFEST.json",
    )
    print("COMPLETE: 20 seeds, 800 episodes, 200 frozen evaluations, 57,600 evaluation routing choices")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    verify(parser.parse_args().output_root.resolve())


if __name__ == "__main__":
    main()
