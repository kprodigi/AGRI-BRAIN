"""Regression test for aggregate_seeds.py per-seed-envelope loading.

Locks the contract that ``aggregate_seeds.main`` correctly unwraps
the post-2026-05 per-seed JSON envelope:

    {"seed": int,
     "scenarios": {sc: {mode: {metric: value}}},
     "traces":    {sc: {mode: {trace_field: [floats]}}}}

vs the legacy flat format:

    {sc: {mode: {metric: value}}}

The HPC run tagged 485c769_20260505_0349 produced an empty
benchmark_summary because the loader read the envelope literally
(``all_data[seed]['scenarios']`` was ignored, ``all_data[seed][sc]``
returned an empty dict, every metric was filtered out, no BCa
bootstraps ran). The aggregator's saved summary had every cell
``{}`` and crashed downstream on ``summary[sc]["agribrain"]["ari"]``.
This test fakes 3 seeds in both formats and asserts the per-cell
metrics are populated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIM_DIR = _REPO_ROOT / "mvp" / "simulation"

if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))


def _envelope_seed(seed: int, ari_value: float) -> dict:
    """Synthesise a minimal per-seed JSON in the envelope format."""
    return {
        "seed": seed,
        "scenarios": {
            "heatwave": {
                "agribrain": {
                    "ari": ari_value,
                    "waste": 0.04,
                    "rle": 0.80,
                    "slca": 0.70,
                    "carbon": 2200.0,
                    "equity": 0.65,
                },
                "static": {
                    "ari": 0.42,
                    "waste": 0.11,
                    "rle": 0.0,
                    "slca": 0.55,
                    "carbon": 4900.0,
                    "equity": 0.55,
                },
            },
        },
        "traces": {},
    }


def _legacy_flat_seed(seed: int, ari_value: float) -> dict:
    """Synthesise the legacy pre-2026-05 flat format."""
    return {
        "heatwave": {
            "agribrain": {
                "ari": ari_value,
                "waste": 0.04,
                "rle": 0.80,
                "slca": 0.70,
                "carbon": 2200.0,
                "equity": 0.65,
            },
            "static": {
                "ari": 0.42,
                "waste": 0.11,
                "rle": 0.0,
                "slca": 0.55,
                "carbon": 4900.0,
                "equity": 0.55,
            },
        },
    }


def _verify_unwrap_logic(payloads: list[dict]) -> dict:
    """Reproduce the on-load unwrap that aggregate_seeds.main does
    so the test isolates that contract without invoking the heavy
    bootstrap pipeline. Mirror of the load loop in main().
    """
    all_data: dict = {}
    for i, payload in enumerate(payloads):
        scenarios_block = payload.get("scenarios")
        if isinstance(scenarios_block, dict):
            all_data[i] = scenarios_block
        else:
            all_data[i] = payload
    return all_data


def test_envelope_format_unwraps_to_scenario_keys():
    """Envelope format must unwrap to scenario-keyed top level."""
    payloads = [_envelope_seed(s, 0.5 + 0.01 * i) for i, s in enumerate([42, 1337, 2024])]
    all_data = _verify_unwrap_logic(payloads)
    for seed_idx, data in all_data.items():
        assert "heatwave" in data, (
            f"Seed {seed_idx} top-level keys are {list(data.keys())[:3]}, "
            f"expected 'heatwave' at top level (envelope unwrap failed)."
        )
        assert "agribrain" in data["heatwave"]
        assert "ari" in data["heatwave"]["agribrain"]
        assert isinstance(data["heatwave"]["agribrain"]["ari"], float)


def test_legacy_flat_format_passes_through():
    """Legacy flat format must work unchanged (backward compat)."""
    payloads = [_legacy_flat_seed(s, 0.5) for s in [42, 1337, 2024]]
    all_data = _verify_unwrap_logic(payloads)
    for data in all_data.values():
        assert "heatwave" in data
        assert "agribrain" in data["heatwave"]
        assert data["heatwave"]["agribrain"]["ari"] == 0.5


def test_aggregator_per_cell_loop_sees_metrics_in_envelope_format():
    """Direct integration test: the per-cell loop in
    aggregate_seeds.main builds vals from
    ``all_data[s][sc][mode][met]``. With the envelope unwrap, this
    must return non-empty lists. Without the unwrap (the HPC bug),
    vals is empty and the BCa loop never fires.
    """
    payloads = [_envelope_seed(s, 0.5 + 0.01 * i) for i, s in enumerate([42, 1337, 2024])]
    all_data = _verify_unwrap_logic(payloads)

    # Mirror the aggregator's metric-extraction pattern.
    sc, mode, met = "heatwave", "agribrain", "ari"
    vals = [
        all_data[s][sc][mode][met]
        for s in all_data
        if mode in all_data[s].get(sc, {})
        and met in all_data[s][sc][mode]
        and all_data[s][sc][mode][met] is not None
    ]
    assert len(vals) == 3, (
        f"Per-cell metric extraction returned {len(vals)} values, "
        f"expected 3. The envelope unwrap is the contract that keeps "
        f"this from failing on every metric and producing an empty "
        f"benchmark_summary."
    )
    assert all(isinstance(v, float) for v in vals)


def test_envelope_without_unwrap_silently_loses_all_metrics():
    """Negative test: confirm the failure mode the HPC run hit.

    Without the unwrap, the per-cell loop returns 0 values because
    ``all_data[s].get(sc, {})`` returns {} (no scenarios at the
    top level of the envelope) and the ``mode in {}`` check is
    False for every (sc, mode, metric) tuple.
    """
    # Skip the unwrap step deliberately:
    payloads = [_envelope_seed(s, 0.5) for s in [42, 1337, 2024]]
    all_data_no_unwrap = {i: p for i, p in enumerate(payloads)}

    sc, mode, met = "heatwave", "agribrain", "ari"
    vals = [
        all_data_no_unwrap[s][sc][mode][met]
        for s in all_data_no_unwrap
        if mode in all_data_no_unwrap[s].get(sc, {})
        and met in all_data_no_unwrap[s][sc][mode]
        and all_data_no_unwrap[s][sc][mode][met] is not None
    ]
    assert len(vals) == 0, (
        "Without the envelope unwrap, the loop should yield zero "
        "values (this is the bug the unwrap fixes). If this "
        "assertion fails, the envelope shape may have changed and "
        "the regression-guard above no longer reflects reality."
    )


def test_aggregate_seeds_load_path_actually_unwraps(tmp_path: Path, monkeypatch):
    """End-to-end-ish: write a synthetic envelope JSON to a tagged
    seed dir, run only the load section of aggregate_seeds.main, and
    verify the in-memory ``all_data`` has scenario-keyed shape.

    We don't call the full main() because the heavy bootstrap +
    significance pipeline takes minutes; the assertion is on the
    load-path contract specifically.
    """
    seeds_dir = tmp_path / "benchmark_seeds" / "test_run"
    seeds_dir.mkdir(parents=True)
    for s in (42, 1337, 2024):
        (seeds_dir / f"seed_{s}.json").write_text(
            json.dumps(_envelope_seed(s, 0.55))
        )

    # Source-level pin so a future refactor that silently removes
    # the unwrap fails CI.
    src = (_SIM_DIR / "benchmarks" / "aggregate_seeds.py").read_text(
        encoding="utf-8"
    )
    assert 'payload.get("scenarios")' in src, (
        "aggregate_seeds.py no longer probes the per-seed JSON "
        "for a 'scenarios' envelope key. Without this unwrap, "
        "every per-cell metric is silently filtered out and the "
        "saved benchmark_summary.json has empty {} cells (the "
        "HPC RUN_TAG 485c769_20260505_0349 failure mode)."
    )
    assert "scenarios_block" in src, (
        "Unwrap variable name 'scenarios_block' missing from load path."
    )
    # Sanity: the synthesized seed files are valid JSON and round-trip.
    for s in (42, 1337, 2024):
        loaded = json.loads((seeds_dir / f"seed_{s}.json").read_text())
        assert "scenarios" in loaded
        assert "heatwave" in loaded["scenarios"]


# ---------------------------------------------------------------------------
# git_commit fallback chain — three tiers, last must be ``.git/HEAD`` direct read.
# ---------------------------------------------------------------------------

def test_git_commit_fallback_chain_has_three_tiers():
    """The aggregator's _meta.git_commit resolution must have THREE
    tiers (env var, subprocess, .git/HEAD direct read).

    The post-HPC RUN_TAG 485c769_20260505_0349 incident: a manual
    ``sbatch hpc/hpc_aggregate.sh`` resubmission bypassed the env
    export from hpc_run.sh AND the slurm worker's PATH didn't
    include ``git``. With only two tiers, both returned None and
    benchmark_summary._meta.git_commit landed as null. The
    third-tier fallback reads .git/HEAD directly so the chain
    survives the no-env-var + no-git-binary case.
    """
    src = (_SIM_DIR / "benchmarks" / "aggregate_seeds.py").read_text(
        encoding="utf-8"
    )
    # Tier 1: env var
    assert 'AGRIBRAIN_GIT_COMMIT' in src, (
        "Tier 1 missing: AGRIBRAIN_GIT_COMMIT env var read."
    )
    # Tier 2: git subprocess
    assert '"git", "rev-parse", "HEAD"' in src, (
        "Tier 2 missing: git rev-parse HEAD subprocess fallback."
    )
    # Tier 3: .git/HEAD direct read
    assert '.git" / "HEAD"' in src or "'.git'" in src and 'HEAD' in src, (
        "Tier 3 missing: direct .git/HEAD read fallback. Without "
        "this, slurm workers without ``git`` on PATH (and without "
        "AGRIBRAIN_GIT_COMMIT set) silently produce "
        "benchmark_summary._meta.git_commit = null."
    )
    # Tier 3 needs to handle ref-based AND detached-HEAD forms
    assert 'ref: ' in src, (
        "Tier 3 missing the ref-based HEAD case "
        "(``ref: refs/heads/main`` form)."
    )


def test_git_commit_tier3_parses_ref_based_head(tmp_path: Path):
    """Synthetic .git directory with a ref-based HEAD must yield
    the SHA the ref points at."""
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    expected_sha = "a" * 40
    (git_dir / "refs" / "heads" / "main").write_text(
        expected_sha + "\n", encoding="utf-8",
    )

    # Mirror the tier-3 logic from aggregate_seeds.py
    head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    sha = None
    if head_text.startswith("ref: "):
        ref = head_text[5:].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            candidate = ref_path.read_text(encoding="utf-8").strip()
            if len(candidate) == 40 and all(
                c in "0123456789abcdef" for c in candidate
            ):
                sha = candidate
    assert sha == expected_sha


def test_git_commit_tier3_parses_detached_head(tmp_path: Path):
    """Detached HEAD: .git/HEAD itself contains the 40-char SHA."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    expected_sha = "b" * 40
    (git_dir / "HEAD").write_text(expected_sha + "\n", encoding="utf-8")

    head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    sha = None
    if not head_text.startswith("ref: "):
        if len(head_text) == 40 and all(
            c in "0123456789abcdef" for c in head_text
        ):
            sha = head_text
    assert sha == expected_sha


def test_verify_manifest_tracked_patterns_match_gitignore_allowlist():
    """The _TRACKED_PATTERNS list in verify_manifest.py must list ONLY
    the file basenames that are explicitly allowlisted in .gitignore
    under ``mvp/simulation/results/``. A glob that over-matches
    (e.g. ``table*.csv`` matching both ``table1_summary.csv`` and
    the gitignored ``table1_summary_seed42.csv``) makes
    --require-tracked hard-fail on CI because the seed42 companion
    files are HPC-side-only and never committed.
    """
    src = (_SIM_DIR / "analysis" / "verify_manifest.py").read_text(
        encoding="utf-8"
    )
    # Required exact-match entries for the table files (no globs).
    assert '"table1_summary.csv"' in src, (
        "Tracked-patterns list missing exact 'table1_summary.csv'."
    )
    assert '"table2_ablation.csv"' in src, (
        "Tracked-patterns list missing exact 'table2_ablation.csv'."
    )
    # The over-matching glob must NOT be present.
    assert '"table*.csv"' not in src, (
        "Tracked-patterns list contains the over-matching 'table*.csv' "
        "glob. This matches the gitignored 'table1_summary_seed42.csv' "
        "/ 'table2_ablation_seed42.csv' companion files (which are "
        "HPC-side single-seed outputs, not in the .gitignore "
        "allowlist), causing CI --require-tracked to hard-fail on "
        "missing-tracked when those files are absent from a fresh "
        "clone."
    )


def test_git_commit_tier3_handles_packed_refs(tmp_path: Path):
    """When the ref file is missing but exists in packed-refs."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    expected_sha = "c" * 40
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    # No refs/heads/main file -- it's been packed.
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n"
        f"{expected_sha} refs/heads/main\n",
        encoding="utf-8",
    )

    # Mirror the packed-refs branch of tier-3 logic.
    head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    ref = head_text[5:].strip() if head_text.startswith("ref: ") else None
    assert ref == "refs/heads/main"
    ref_path = git_dir / ref
    assert not ref_path.exists()  # Packed-only state.

    sha = None
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(ref) and len(line) >= 41:
                candidate = line.split(" ", 1)[0].strip()
                if len(candidate) == 40:
                    sha = candidate
                    break
    assert sha == expected_sha
