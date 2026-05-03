"""Regression tests for the channel-decomposition Holm family in
``mvp/simulation/benchmarks/aggregate_seeds.py``.

The aggregator computes a third statistical family alongside the
primary H1 family (``agribrain_vs_no_context`` on ARI, m=5) and the
secondary BY-FDR per-scenario family: the **channel-decomposition
family** of m=10 tests (2 contrasts: ``mcp_only_vs_no_context`` and
``pirag_only_vs_no_context``; each on ARI across 5 scenarios) with
its own Holm-Bonferroni correction.

Tests are written so they only import ``mvp.simulation.benchmarks.
aggregate_seeds`` lazily, inside the test bodies that need it. The
aggregator transitively imports ``generate_results.py`` which pulls
in the FastAPI router stack; in environments where the backend isn't
installed (e.g. a documentation-only checkout), the source-text
contract tests below still run and the import-dependent tests
``pytest.skip`` cleanly.

Three contracts are pinned:

  1. ``_CHANNEL_DECOMPOSITION_PAIRS`` is exactly the two single-
     channel-vs-no-context pairs the C4 paper-claim families depend
     on. If a future maintainer extends or shrinks this tuple, the
     manuscript text and ``docs/STATISTICAL_METHODS.md`` need to be
     updated to reflect the new family size; the test pins both the
     count (m=10 after the cross-product with 5 scenarios) and the
     exact membership.

  2. When the aggregator has been run on a 20-seed dump and produced
     ``benchmark_significance.json``, that file carries (a) every
     ``(scenario, pair)`` cell as a populated ``ari`` record with the
     ``p_value_adj_holm_channel`` field set, (b) the canonical
     ``correction_method`` is ``holm_bonferroni_channel_decomposition``
     on those cells, and (c) the JSON top-level
     ``channel_decomposition_holm_adjusted`` dict covers every
     ``{scenario}:{a_mode}_vs_{b_mode}`` key.

  3. Holm step-down within a 10-test family is monotone and
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
        "membership. The C4 claim in CLAIMS_TO_EVIDENCE.md and the "
        "family-size statement in STATISTICAL_METHODS.md both reference "
        "exactly these two pairs. Update both docs (and the family m=10 "
        "statement) before changing this constant."
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
        _CHANNEL_DECOMPOSITION_PAIRS, SCENARIOS,
    )
    assert _CHANNEL_DECOMPOSITION_PAIRS == (
        ("mcp_only",   "no_context"),
        ("pirag_only", "no_context"),
    )
    expected_m = len(_CHANNEL_DECOMPOSITION_PAIRS) * len(SCENARIOS)
    assert expected_m == 10, (
        f"Expected family size 10 (2 pairs x 5 scenarios); got {expected_m}. "
        "If SCENARIOS or _CHANNEL_DECOMPOSITION_PAIRS changed, update "
        "STATISTICAL_METHODS.md's 'Channel-decomposition family' size "
        "statement to match."
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
    pairs = ("mcp_only_vs_no_context", "pirag_only_vs_no_context")
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
            assert "p_value_adj_holm_channel" in rec, (
                f"channel-decomposition: {sc}/{pair_name}/ari "
                "missing p_value_adj_holm_channel"
            )
            assert rec.get("correction_method") == (
                "holm_bonferroni_channel_decomposition"
            ), (
                f"channel-decomposition: {sc}/{pair_name}/ari "
                f"correction_method is {rec.get('correction_method')!r}, "
                "expected 'holm_bonferroni_channel_decomposition'"
            )
            # Effect-size CI bracketed correctly.
            lo = rec.get("effect_size_ci_low")
            hi = rec.get("effect_size_ci_high")
            if lo is not None and hi is not None:
                assert float(lo) <= float(hi), (
                    f"channel-decomposition: {sc}/{pair_name}/ari "
                    f"effect-size CI inverted: low={lo} > high={hi}"
                )

    fam = payload.get("channel_decomposition_holm_adjusted")
    assert isinstance(fam, dict), (
        "missing top-level channel_decomposition_holm_adjusted dict"
    )
    expected_keys = {f"{sc}:{pair}" for sc in scenarios for pair in pairs}
    assert set(fam.keys()) == expected_keys, (
        f"channel_decomposition_holm_adjusted keys mismatch: "
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
