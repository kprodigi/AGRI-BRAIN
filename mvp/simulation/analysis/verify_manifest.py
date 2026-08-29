#!/usr/bin/env python3
"""Verify mvp/simulation/results/artifact_manifest.json.

Re-hashes every artifact listed in the manifest and asserts the
SHA-256 matches what is recorded. In strict fresh mode it also requires one
clean, identical source commit for simulation, aggregation, and publication.
Dual provenance is accepted only with a separately validated deterministic-
recovery receipt that explicitly records ``simulation_rerun=false``. Exits 0
on clean verification, 1 on any mismatch or missing artifact.

Usage::

    python mvp/simulation/analysis/verify_manifest.py
    python mvp/simulation/analysis/verify_manifest.py --strict-commit

The canonical ``hpc/hpc_run.sh`` workflow invokes this verifier from its
dependent ``hpc/hpc_publish.sh`` stage. It can also be run against a
transferred publication artifact set to reconfirm the published bytes. The
2026-04 cleanup flagged that the manifest was produced but never verified
anywhere; this script closes that gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.simulation.analysis.recovery_provenance import (
    validate_recovery_context,
)

RESULTS_DIR = REPO_ROOT / "mvp" / "simulation" / "results"
MANIFEST_PATH = RESULTS_DIR / "artifact_manifest.json"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    """Return SHA-256 over the literal file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--strict-commit",
        action="store_true",
        help=(
            "Require a clean full-commit identity: one equal commit for a "
            "fresh run, or two receipt-authorized commits for deterministic "
            "recovery."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Treat artifacts listed in the manifest but absent from the "
            "working tree as a non-fatal warning instead of an error. "
            "Use this on CI checkouts where gitignored artifacts (e.g. "
            "mcp_interop_*.json, traces_*.json) are not in the repo. "
            "On HPC delivery runs, omit this flag so a genuinely missing "
            "artifact still fails the gate."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Path to artifact_manifest.json (default: %(default)s).",
    )
    parser.add_argument(
        "--recovery-receipt",
        type=Path,
        help=(
            "Explicit run-scoped publication-recovery authorization. Required "
            "for a dual-provenance manifest and rejected for a fresh manifest."
        ),
    )
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help=(
            "When combined with --allow-missing, hard-fail on any "
            "missing artifact that matches the git-tracked allowlist "
            "(see TRACKED_PATTERNS below). Untracked / gitignored "
            "artifacts (mcp_interop_*, traces_*, learning_trajectory_*, "
            "context_alignment_*, benchmark_seeds/*) still soft-warn. "
            "This tightens the CI gate so a tracked-figure deletion "
            "cannot pass while gitignored artifacts (which are not in "
            "a fresh CI checkout) still warn cleanly. Recommended for "
            "the artifact-validation CI job."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}")
        return 1

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    git_commit = payload.get("git_commit")
    simulation_commit = payload.get("simulation_source_commit", git_commit)
    publication_commit = payload.get("publication_code_commit", git_commit)
    dual_provenance = payload.get("dual_provenance")

    if dual_provenance is True:
        for label, value in (
            ("git_commit", git_commit),
            ("simulation_source_commit", simulation_commit),
            ("publication_code_commit", publication_commit),
        ):
            if not isinstance(value, str) or not _HEX40.fullmatch(value):
                print(f"FAIL: manifest.{label} is not a 40-hex SHA: {value!r}")
                return 1
        if git_commit != simulation_commit or simulation_commit == publication_commit:
            print(
                "FAIL: a recovery manifest must keep git_commit bound to its "
                "distinct simulation source commit"
            )
            return 1
        if payload.get("git_dirty") is not False:
            print("FAIL: a recovery manifest cannot carry a dirty Git stamp")
            return 1
        if args.recovery_receipt is None:
            print(
                "FAIL: dual-provenance manifest requires --recovery-receipt"
            )
            return 1
        try:
            recovery_authorization = validate_recovery_context(
                args.recovery_receipt,
                results_dir=manifest_path.parent,
                run_tag=payload.get("artifact_run_tag"),
                simulation_commit=simulation_commit,
                publication_commit=publication_commit,
                expected_kind="core",
            )
        except (OSError, ValueError) as exc:
            print(f"FAIL: invalid publication-recovery authorization: {exc}")
            return 1
        if payload.get("recovery_authorization") != recovery_authorization:
            print(
                "FAIL: manifest recovery_authorization differs from the "
                "validated receipt and preserved-input binding"
            )
            return 1
        recovery_records = {
            record.get("file"): record
            for record in payload.get("artifacts", [])
            if isinstance(record, dict)
        }
        for path_key, digest_key in (
            ("receipt_file", "receipt_literal_sha256"),
            (
                "preserved_raw_manifest_file",
                "preserved_raw_manifest_literal_sha256",
            ),
            ("original_submission_receipt_file", None),
        ):
            name = recovery_authorization[path_key]
            record = recovery_records.get(name)
            if record is None:
                print(f"FAIL: recovery evidence is not manifested: {name}")
                return 1
            if digest_key is not None and record.get(
                "sha256"
            ) != recovery_authorization[digest_key]:
                print(f"FAIL: manifested recovery-evidence hash changed: {name}")
                return 1
    elif args.recovery_receipt is not None:
        print("FAIL: --recovery-receipt is invalid for single-provenance evidence")
        return 1

    if args.strict_commit:
        if not isinstance(git_commit, str) or not _HEX40.match(git_commit):
            print(
                f"FAIL: manifest.git_commit is missing or not a "
                f"40-hex SHA: {git_commit!r}"
            )
            return 1
        for label, value in (
            ("simulation_source_commit", simulation_commit),
            ("publication_code_commit", publication_commit),
        ):
            if not isinstance(value, str) or not _HEX40.match(value):
                print(f"FAIL: manifest.{label} is not a 40-hex SHA: {value!r}")
                return 1
        if dual_provenance is not True:
            if not git_commit == simulation_commit == publication_commit:
                print(
                    "FAIL: strict fresh publication evidence must use one "
                    "identical git/simulation/publication commit"
                )
                return 1
            if dual_provenance is not False:
                print(
                    "FAIL: manifest.dual_provenance must be false in strict "
                    "fresh mode"
                )
                return 1
        if payload.get("git_dirty") is not False:
            print("FAIL: manifest.git_dirty must be false in strict mode")
            return 1

    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        print("FAIL: manifest.artifacts is empty or not a list")
        return 1

    # Skip self-referential and known-volatile entries:
    #  - artifact_manifest.json hashes itself, which is a chicken-and-egg
    #    paradox by construction (the written file's hash would have to
    #    contain its own value).
    #  - validation_report.json is rewritten on every validator run and
    #    is intentionally not pinned by SHA.
    SKIP = {"artifact_manifest.json", "validation_report.json"}

    # File-name patterns that ARE tracked by git (the allowlist in
    # .gitignore under ``mvp/simulation/results/``). When --require-tracked
    # is set together with --allow-missing, a missing artifact that
    # matches one of these patterns is a hard failure rather than a
    # warning. Keep this list in lockstep with the .gitignore allowlist
    # (see top-level .gitignore around line 82).
    #
    # IMPORTANT: glob patterns must NOT over-match. The pre-2026-05
    # ``table*.csv`` glob matched both the canonical
    # ``table1_summary.csv`` (tracked) and the single-seed companion
    # ``table1_summary_seed42.csv`` (gitignored, HPC-side only),
    # causing CI's --require-tracked to hard-fail on missing
    # ``table1_summary_seed42.csv`` even though that file is not
    # in the .gitignore allowlist. Listing the table files
    # explicitly removes the ambiguity.
    import fnmatch as _fnmatch_v
    _TRACKED_PATTERNS = (
        "fig*.png", "fig*.pdf",
        # 2026-05 manuscript figure renames dropped the ``fig`` prefix
        # (see the .gitignore results allowlist); list them explicitly so
        # the --require-tracked gate still protects them. Keep in lockstep
        # with the .gitignore allowlist.
        "heatwave.png", "heatwave.pdf",
        "overproduction.png", "overproduction.pdf",
        "cyber_outage.png", "cyber_outage.pdf",
        "adaptive_pricing.png", "adaptive_pricing.pdf",
        "cross_scenario.png", "cross_scenario.pdf",
        "ablation.png", "ablation.pdf",
        "transport_emissions.png", "transport_emissions.pdf",
        # 2026-06 H1/H2/H3 paper figures (generate_figures.py fig11/fig12/fig13):
        # performance_efficiency (H1), context_value (H2), stress_robustness (H3).
        "context_value.png", "context_value.pdf",
        "performance_efficiency.png", "performance_efficiency.pdf",
        "stress_robustness.png", "stress_robustness.pdf",
        "table1_summary.csv", "table2_ablation.csv",
        "benchmark_summary.json", "benchmark_significance.json",
        "h2_directional_evidence.csv",
        "secondary_ablation_analysis.json", "secondary_ablation_analysis.csv",
        "stress_summary.json", "stress_degradation.csv",
        # Paper-evidence artefacts added to the gitignore allowlist in
        # 2026-05 so reviewers cloning the repo can verify the cited
        # evidence without downloading the HPC tar.gz archive.
        "paper_benchmark_table.json",
        "stress_passfail.csv",
        "stress_h3_test.json", "explainability_metrics.json",
        "forecast_validation_summary.json",
        "forecast_validation_predictions.csv",
        # 2026-06 decision-level channel analysis (§5.8 / Fig 14)
        "channel_attribution_aggregate.json",
        "channel_complementarity_test.json",
        "channel_saturation_analysis.json",
    )

    def _is_tracked(name: str) -> bool:
        """Return True if ``name`` is a tracked file at top-level.

        ``name`` is the manifest's relative POSIX path (e.g.
        ``"heatwave.png"`` or
        ``"benchmark_seeds/seed_42.json"``). The .gitignore allowlist
        targets ONLY top-level files in ``mvp/simulation/results/`` --
        ``benchmark_seeds/seed_42.json`` and ``preview/fig9.png`` are
        gitignored even though their basenames (``seed_42.json``,
        ``fig9.png``) might match a tracked pattern. Hence the
        top-level guard: a path with any '/' separator is NOT
        tracked, regardless of basename.
        """
        if "/" in name:
            return False
        return any(_fnmatch_v.fnmatch(name, p) for p in _TRACKED_PATTERNS)

    errors = 0
    checked = 0
    skipped = 0
    missing_warnings = 0
    for rec in artifacts:
        name = rec.get("file")
        recorded_sha = rec.get("sha256")
        if not name or not recorded_sha:
            print(f"FAIL: manifest entry missing file or sha256: {rec!r}")
            errors += 1
            continue
        if name in SKIP:
            skipped += 1
            continue
        path = manifest_path.parent / name
        if not path.exists():
            # Tracked artifacts MUST exist on a fresh checkout (they
            # are in the .gitignore allowlist). Untracked artifacts
            # (mcp_interop_*, traces_*, etc.) only exist after a sim
            # run; --allow-missing is for them.
            if args.require_tracked and _is_tracked(name):
                print(
                    f"FAIL: missing tracked artifact (require-tracked): "
                    f"{name}"
                )
                errors += 1
                continue
            if args.allow_missing:
                print(f"WARN: missing artifact (allow-missing): {path.name}")
                missing_warnings += 1
                continue
            print(f"FAIL: missing artifact: {path.name}")
            errors += 1
            continue
        actual = _sha256(path)
        if actual != recorded_sha:
            print(
                f"FAIL: SHA-256 mismatch for {path.name}: "
                f"manifest={recorded_sha} actual={actual}"
            )
            errors += 1
        else:
            checked += 1

    print(
        f"verify_manifest: checked {checked} files, skipped {skipped} "
        f"(self-ref / volatile), missing_warnings {missing_warnings} "
        f"(allow-missing), errors {errors}, "
        f"git_commit={git_commit!r}"
    )
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
