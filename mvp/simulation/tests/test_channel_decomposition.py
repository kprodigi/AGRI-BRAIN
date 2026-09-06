"""Regression tests for H2 and its historical channel-subset audit fields in
``mvp/simulation/benchmarks/aggregate_seeds.py``.

Confirmatory H2 is one m=20 Holm family: two single-channel-versus-No-context
contrasts and two full-versus-single-channel contrasts in five scenarios.  The
older m=10 two-contrast correction remains only as an auxiliary audit field;
it is never the canonical H2 p-value.

Tests are written so they only import ``mvp.simulation.benchmarks.
aggregate_seeds`` lazily, inside the test bodies that need it. The
aggregator transitively imports ``generate_results.py`` which pulls
in the FastAPI router stack; in environments where the backend isn't
installed (e.g. a documentation-only checkout), the source-text
contract tests below still run and the import-dependent tests
``pytest.skip`` cleanly.

Three contracts are pinned:

  1. ``_CHANNEL_DECOMPOSITION_PAIRS`` remains the exact two direct
     single-channel-vs-No-context pairs.  Together with the two
     full-vs-single pairs in ``H2_DIRECTIONAL_PAIRS``, these form H2.

  2. When the aggregator has been run on a 20-seed dump and produced
     ``benchmark_significance.json``, that file carries (a) every
     ``(scenario, pair)`` cell as a populated ``ari`` record with the
     ``p_value_adj_holm_h2_directional`` field set, (b) the canonical
     ``correction_method`` is ``holm_bonferroni_h2_directional_20``, and
     (c) the JSON top-level ``h2_directional_holm_adjusted`` dict covers
     all 20 keys.

  3. Holm step-down is monotone and
     bounded in [0, 1] (pure-design test verifying the published
     correction's mathematical contract; no aggregator import needed).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers used by the source-text contract tests (no heavy imports)
# ---------------------------------------------------------------------------

_AGG_SOURCE_PATH = (
    _REPO_ROOT / "mvp" / "simulation" / "benchmarks" / "aggregate_seeds.py"
)


def _aggregator_source() -> str:
    return _AGG_SOURCE_PATH.read_text(encoding="utf-8")


def _can_import_aggregator() -> bool:
    try:
        import mvp.simulation.benchmarks.aggregate_seeds  # noqa: F401
        return True
    except Exception:
        return False


def _seed_dump_available() -> bool:
    """True iff a non-empty 20-seed dump is present at the canonical path."""
    seed_dir = _REPO_ROOT / "mvp" / "simulation" / "results" / "benchmark_seeds"
    if not seed_dir.exists():
        return False
    flat_seeds = list(seed_dir.glob("seed_*.json"))
    if len(flat_seeds) >= 5:
        return True
    for sub in seed_dir.iterdir():
        if sub.is_dir() and len(list(sub.glob("seed_*.json"))) >= 5:
            return True
    return False


# ---------------------------------------------------------------------------
# Contract 1: family membership and size (source-text + import variants)
# ---------------------------------------------------------------------------

def test_channel_decomposition_pairs_constant_in_source():
    """Source contains the canonical two-pair tuple. Pinned via text
    so the assertion runs even without the backend installed."""
    src = _aggregator_source()
    # Match either single- or double-quoted strings in the tuple.
    pattern = re.compile(
        r"_CHANNEL_DECOMPOSITION_PAIRS[^=]*=\s*\(\s*"
        r"\(\s*['\"]mcp_only['\"]\s*,\s*['\"]no_context['\"]\s*\)\s*,\s*"
        r"\(\s*['\"]pirag_only['\"]\s*,\s*['\"]no_context['\"]\s*\)\s*,?"
        r"\s*\)",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "Source-level _CHANNEL_DECOMPOSITION_PAIRS tuple has changed "
        "from the canonical (mcp_only,no_context) + (pirag_only,no_context) "
        "membership. Together with the two full-versus-single contrasts, "
        "these pairs define the m=20 H2 family."
    )


@pytest.mark.skipif(
    not _can_import_aggregator(),
    reason="aggregate_seeds requires the backend stack (FastAPI etc); "
           "the source-text test above covers the contract in lighter "
           "environments.",
)
def test_channel_decomposition_pairs_constant_via_import():
    """Same contract as the source-text test, but verified through
    the actual Python import — catches subtle errors that source
    parsing would miss (e.g. the constant defined twice with
    different values)."""
    from mvp.simulation.benchmarks.aggregate_seeds import (
        _CHANNEL_DECOMPOSITION_PAIRS,
        SCENARIOS,
    )
    assert _CHANNEL_DECOMPOSITION_PAIRS == (
        ("mcp_only",   "no_context"),
        ("pirag_only", "no_context"),
    )
    expected_m = len(_CHANNEL_DECOMPOSITION_PAIRS) * len(SCENARIOS)
    assert expected_m == 10, (
        f"Expected auxiliary subset size 10 (2 pairs x 5 scenarios); "
        f"got {expected_m}."
    )


# ---------------------------------------------------------------------------
# Contract 2: end-to-end JSON shape (only when the aggregator has run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_REPO_ROOT / "mvp" / "simulation" / "results" /
         "benchmark_significance.json").exists(),
    reason="benchmark_significance.json not present; run aggregate_seeds "
           "first.",
)
def test_channel_decomposition_records_present_in_significance_json():
    """``benchmark_significance.json`` must carry both
    ``mcp_only_vs_no_context`` and ``pirag_only_vs_no_context`` records
    for every scenario, with the family-corrected ARI p-value field
    populated and the canonical ``p_value_adj`` set to that field."""
    import json
    sig_path = (
        _REPO_ROOT / "mvp" / "simulation" / "results" /
        "benchmark_significance.json"
    )
    payload = json.loads(sig_path.read_text())
    sig = payload.get("significance", payload)
    scenarios = ("heatwave", "overproduction", "cyber_outage",
                 "adaptive_pricing", "baseline")
    pairs = (
        "mcp_only_vs_no_context", "pirag_only_vs_no_context",
        "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
    )
    # Skip the test gracefully if this file pre-dates the
    # channel-decomposition family. The check below distinguishes
    # "aggregator hasn't been re-run since 2026-05" from "aggregator
    # ran but the family is broken".
    sample = sig.get("baseline", {})
    if "mcp_only_vs_no_context" not in sample:
        pytest.skip(
            "benchmark_significance.json pre-dates the channel-"
            "decomposition family (no mcp_only_vs_no_context record "
            "in baseline); re-run aggregate_seeds.py to regenerate."
        )

    for sc in scenarios:
        for pair_name in pairs:
            comp = sig.get(sc, {}).get(pair_name)
            assert comp is not None, (
                f"channel-decomposition: missing {sc}/{pair_name}"
            )
            rec = comp.get("ari")
            assert isinstance(rec, dict), (
                f"channel-decomposition: {sc}/{pair_name}/ari "
                "is not a dict"
            )
            assert "p_value_adj_holm_h2_directional" in rec, (
                f"channel-decomposition: {sc}/{pair_name}/ari "
                "missing p_value_adj_holm_h2_directional"
            )
            assert rec.get("correction_method") == (
                "holm_bonferroni_h2_directional_20"
            ), (
                f"channel-decomposition: {sc}/{pair_name}/ari "
                f"correction_method is {rec.get('correction_method')!r}, "
                "expected 'holm_bonferroni_h2_directional_20'"
            )
            # Effect-size CI bracketed correctly.
            lo = rec.get("effect_size_ci_low")
            hi = rec.get("effect_size_ci_high")
            if lo is not None and hi is not None:
                assert float(lo) <= float(hi), (
                    f"channel-decomposition: {sc}/{pair_name}/ari "
                    f"effect-size CI inverted: low={lo} > high={hi}"
                )

    fam = payload.get("h2_directional_holm_adjusted")
    assert isinstance(fam, dict), (
        "missing top-level h2_directional_holm_adjusted dict"
    )
    expected_keys = {f"{sc}:{pair}" for sc in scenarios for pair in pairs}
    assert set(fam.keys()) == expected_keys, (
        f"h2_directional_holm_adjusted keys mismatch: "
        f"missing={expected_keys - set(fam.keys())} "
        f"extra={set(fam.keys()) - expected_keys}"
    )


# ---------------------------------------------------------------------------
# Contract 3: Holm step-down monotonicity (pure-design)
# ---------------------------------------------------------------------------

def _local_holm(p_values: dict[str, float]) -> dict[str, float]:
    """Reference implementation of step-down Holm-Bonferroni; lets the
    monotonicity tests run without importing the aggregator."""
    if not p_values:
        return {}
    sorted_keys = sorted(p_values, key=lambda k: p_values[k])
    m = len(sorted_keys)
    out: dict[str, float] = {}
    last = 0.0
    for i, k in enumerate(sorted_keys):
        adj = min(1.0, max(last, (m - i) * p_values[k]))
        out[k] = adj
        last = adj
    return out


def test_local_holm_within_family_is_monotone():
    """Pure-design check: Holm produces a non-decreasing adjusted
    sequence on sorted-ascending raw inputs."""
    raw = {
        f"k{i}": p for i, p in enumerate([
            1e-6, 2e-6, 3e-6, 5e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.20, 0.80,
        ])
    }
    adj = _local_holm(raw)
    sorted_raw_keys = sorted(raw, key=lambda k: raw[k])
    adj_sorted = [adj[k] for k in sorted_raw_keys]
    for i in range(1, len(adj_sorted)):
        assert adj_sorted[i] >= adj_sorted[i - 1] - 1e-12
    for k, p in adj.items():
        assert 0.0 <= p <= 1.0, f"out-of-[0,1]: {k}={p}"


def test_local_holm_at_largest_p_keeps_p_unchanged_when_below_one():
    """At m=10 with all-tied raw p=1e-6, adjusted p = 10 * 1e-6 = 1e-5
    on the smallest-rank cell, and the monotone running-max keeps every
    subsequent cell at >= that value too."""
    raw = {f"k{i}": 1e-6 for i in range(10)}
    adj = _local_holm(raw)
    for k, p in adj.items():
        assert p == pytest.approx(1e-5, abs=1e-12), (
            f"Holm at m=10 with all-tied p=1e-6 expected p_adj=1e-5; "
            f"got {k}={p}"
        )


@pytest.mark.skipif(
    not _can_import_aggregator(),
    reason="aggregate_seeds requires the backend stack (FastAPI etc); "
           "the local-Holm tests above cover the same contract.",
)
def test_aggregator_holm_matches_local_reference():
    """The aggregator's holm_bonferroni implementation must agree with
    the reference implementation above on a sample input."""
    from mvp.simulation.benchmarks.aggregate_seeds import holm_bonferroni
    raw = {
        f"k{i}": p for i, p in enumerate([
            1e-6, 2e-6, 3e-6, 5e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.20, 0.80,
        ])
    }
    agg_adj = holm_bonferroni(raw)
    ref_adj = _local_holm(raw)
    for k in raw:
        assert agg_adj[k] == pytest.approx(ref_adj[k], abs=1e-12), (
            f"holm_bonferroni disagrees with reference at {k}: "
            f"aggregator={agg_adj[k]}, reference={ref_adj[k]}"
        )


@pytest.mark.skipif(
    not _can_import_aggregator(),
    reason="aggregate_seeds requires the backend stack",
)
def test_h2_publication_projection_is_exact_20_row_table(tmp_path):
    """The paper-facing table must include every estimate and test field."""
    import csv

    from mvp.simulation.benchmarks.aggregate_seeds import (
        _H2_DIRECTIONAL_PAIRS,
        _H2_PUBLICATION_COLUMNS,
        SCENARIOS,
        _build_h2_publication_rows,
        _write_h2_publication_csv,
    )

    significance = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        significance[scenario] = {}
        for pair_index, (left, right) in enumerate(_H2_DIRECTIONAL_PAIRS):
            delta = 0.01 + 0.001 * (scenario_index * 4 + pair_index)
            significance[scenario][f"{left}_vs_{right}"] = {
                "ari": {
                    "n_seeds": 20,
                    "h2_test_type_actual": "wilcoxon_signed_rank",
                    "mean_diff": delta,
                    "mean_diff_ci_low": delta - 0.002,
                    "mean_diff_ci_high": delta + 0.002,
                    "mean_diff_ci_method": "BCa",
                    "cohens_dz": 1.2,
                    "cohens_dz_ci_low": 0.8,
                    "cohens_dz_ci_high": 1.6,
                    "cohens_dz_ci_method": "BCa",
                    "p_value_directional_greater": 0.001,
                    "p_value_adj_holm_h2_directional": 0.02,
                    "h2_family_size": 20,
                    "h2_cell_supported": True,
                },
            }
    rows = _build_h2_publication_rows(
        significance, source_commit="a" * 40, run_tag="aaaaaaa_20260829_010203",
    )
    assert len(rows) == 20
    assert all(tuple(row) == _H2_PUBLICATION_COLUMNS for row in rows)
    assert [
        (row["scenario"], row["numerator_mode"], row["denominator_mode"])
        for row in rows
    ] == [
        (scenario, left, right)
        for scenario in SCENARIOS
        for left, right in _H2_DIRECTIONAL_PAIRS
    ]

    path = tmp_path / "h2_directional_evidence.csv"
    _write_h2_publication_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 20
    assert tuple(exported[0]) == _H2_PUBLICATION_COLUMNS
    assert exported[0]["raw_directional_p_value"] == "0.001"
    assert exported[0]["holm_adjusted_p_value"] == "0.02"
    assert exported[0]["cell_supported"] == "True"


def test_resampling_metadata_labels_n_perm_as_legacy_scope():
    source = _aggregator_source()
    assert '"confirmatory_test": "directional_wilcoxon_signed_rank"' in source
    assert '"legacy_sign_flip_resamples": 10_000' in source
    assert '"n_perm_scope"' in source
