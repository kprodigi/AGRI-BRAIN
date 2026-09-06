#!/usr/bin/env python3
"""Aggregate multi-seed benchmark results into canonical benchmark files.

Reads ``results/benchmark_seeds/seed_*.json`` and writes

- ``results/benchmark_summary.json``    , per-(scenario, mode, metric) means,
  standard deviations, and 95 % bootstrap CIs.
- ``results/benchmark_significance.json``, paired Wilcoxon signed-rank
  p-values, effect
  sizes, and multiplicity-adjusted p-values using prespecified primary and
  secondary correction families:

  1. Holm-Bonferroni across the five scenario-level primary H1 tests
     (agribrain vs no_context, metric = ARI, one test per scenario). This
     matches the primary-family multiplicity control documented in
     docs/STATISTICAL_METHODS.md.
     Reported as ``p_value_adj_holm`` on the five primary entries and as the
     canonical ``p_value_adj`` on the same entries.
  2. Holm-Bonferroni across all twenty directional H2 tests: MCP-only and
     Retrieval-only versus No-external-context, and AGRI-BRAIN versus each
     single-channel arm, on ARI in five scenarios.  A separate five-cell
     interaction contrast is reported as an exploratory superadditivity
     diagnostic rather than being relabelled as observed "synergy".
  3. Benjamini-Yekutieli FDR within each scenario across all (baseline,
     metric) secondary comparisons. Reported as the canonical ``p_value_adj``
     on non-primary entries; Benjamini-Hochberg is retained as a supplementary
     diagnostic.

The input and output directories are mandatory. Writing the canonical results
tree additionally requires ``--publication`` and the locked publisher
environment; ordinary direct execution cannot overwrite publication evidence.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from ..analysis.experiment_accounting import (
        PRIMARY_PUBLICATION_MODES,
        build_episode_accounting,
    )
    from ..analysis.protocol_statistics import (
        H1_PRACTICAL_MARGIN,
        H2_DIRECTIONAL_PAIRS,
        h2_synergy_interaction,
    )
    from ..analysis.recovery_provenance import (
        recovery_context_from_environment,
    )
except ImportError:
    _REPO_IMPORT_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_IMPORT_ROOT))
    from mvp.simulation.analysis.experiment_accounting import (  # noqa: E402
        PRIMARY_PUBLICATION_MODES,
        build_episode_accounting,
    )
    from mvp.simulation.analysis.protocol_statistics import (  # noqa: E402
        H1_PRACTICAL_MARGIN,
        H2_DIRECTIONAL_PAIRS,
        h2_synergy_interaction,
    )
    from mvp.simulation.analysis.recovery_provenance import (  # noqa: E402
        recovery_context_from_environment,
    )

SEEDS = [42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
         606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515]

# Reproducibility contract for every resampling stream created below.  This is
# serialized with the results so an archive consumer has the exact seed
# derivation, generator identity, namespace, and observation order needed to
# reproduce the finite 10,000-draw Monte Carlo realization (not merely the
# asymptotic bootstrap procedure).
_RESAMPLING_IDENTITY_VERSION = 1
_RESAMPLING_SCOPES = (
    "bootstrap_ci",
    "bootstrap_diff_ci",
    "d_ci_pooled",
    "d_ci_dz",
    "wilcoxon_fallback",
    "paired_perm",
    "mannwhitney_fallback",
)
# Scenario and mode lists come from the simulator's canonical definitions so
# the aggregator follows exactly the locked eight primary and three secondary
# modes without maintaining a second list.
_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from generate_results import (
    _MULTI_EPISODE_MODES as _SIM_MULTI_EPISODE_MODES,
)
from generate_results import (
    MODES as _SIM_MODES,
)
from generate_results import (  # noqa: E402
    SCENARIOS as _SIM_SCENARIOS,
)
from src.models.footprint import (  # noqa: E402
    DEFAULT_ASSUMED_ACTIVE_POWER_W,
    DEFAULT_ENERGY_PER_PROXY_STEP_J,
    DEFAULT_WATER_PER_PROXY_STEP_L,
    DEFAULT_WATER_RATE_L_PER_SERVER_SECOND,
)

SCENARIOS = list(_SIM_SCENARIOS)
MODES = list(_SIM_MODES)
METRICS = ("ari", "waste", "rle", "slca", "carbon", "equity")
# Extra metrics exposed by run_single_seed.py when they are present in the
# per-seed dump. Aggregator does bootstrap CIs on these the same way as the
# core METRICS; missing values (e.g. context_honor_rate for static) are
# filtered out per-cell so aggregation does not crash.
EXTRA_METRICS = (
    # Exploratory ARI/emissions ratio is computed within each seed before
    # bootstrapping, preserving numerator-denominator covariance.  Green-AI
    # fields are descriptive, hardware-dependent activity estimates for the
    # declared decision-path timer; fixed-step proxies remain separately named.
    "carbon_efficiency_ari_per_kgco2e_proxy",
    "decision_path_compute_energy_estimate_j",
    "decision_path_compute_water_estimate_l",
    "decision_path_elapsed_seconds",
    "decision_step_count_energy_proxy_j",
    "decision_step_count_water_proxy_l",
    # Required columns for the legacy table1/table2 CSV schema and for
    # validate_results.py's DecisionLatencyMs / ConstraintViolationRate /
    # operating-envelope rate bounds checks. Keeping them here means the
    # CSV rewrite below populates the same columns the validator expects.
    "mean_decision_latency_ms",
    "constraint_violation_rate",
    "compliance_violation_rate",
    # §4.7 paper-evidence metrics.
    "operational_violation_rate", "regulatory_violation_rate",
    "operating_envelope_violation_rate",
    "context_active_fraction", "context_honor_rate",
    "context_active_steps", "context_honored_steps",
    # 2026-05 context-influence rate (fig 9 panel-c headline). Honor
    # rate is retained alongside as a supplementary-methods companion;
    # both rates are reported with the same bootstrap-CI machinery so
    # a reviewer can read either off the same benchmark_summary cell.
    "context_influence_rate", "context_influenced_steps",
    # Outcome-side violation disposition: policy-quality score on the
    # env-driven violation event set. See resilience.py
    # compute_violation_disposition. The aggregator runs the same
    # bootstrap CI machinery on these as for the headline metrics so the
    # CSV picks up DownstreamViolationRate / ContainedViolationRate
    # columns alongside ConstraintViolationRate.
    "downstream_violation_rate", "redistribute_violation_rate",
    "contained_violation_rate", "violation_event_count",
    # Publication execution-integrity evidence. Strict runs require every
    # failure/truncation count to be zero; aggregating the retained per-episode
    # values keeps that fact visible in benchmark_summary.json as well as in
    # the raw seed envelopes.
    "protocol_interaction_count",
    "protocol_jsonrpc_error_count", "protocol_tool_iserror_count",
    "protocol_real_tool_iserror_count", "protocol_error_count",
    "protocol_dropped_interaction_count", "dispatcher_tool_failure_count",
    "context_execution_error_count",
)

# Columns exposed in the stochastic CSV rewrites below. First element of
# each tuple is the source key in benchmark_summary.json; second is the
# human-facing display name (kept identical to the legacy single-seed CSV
# so the paper's Tables 7 and 9 and the validate_results.py row["..."]
# reads continue to work against the 20-seed CSV).
# Implementation note: 2026-04 deep-audit fix (commit 1d9caf0).
# Two coupled fixes were applied to make the per-mode constraint and
# compliance columns symmetric across every mode:
#
#  (a) ``constraint_violation_steps`` in generate_results.py is now
#      counted only on (temp_violation OR quality_violation) — both
#      ambient-driven, so the metric is symmetric across every mode by
#      construction. The new ``constraint_violation_rate_is_environmental``
#      tag in the per-episode summary makes this framing explicit.
#  (b) the underlying synthetic operating-envelope evaluator is now applied
#      uniformly on every step regardless of mode (previously gated on
#      _MCP_WASTE_MODES, which
#      pinned compliance_violation_steps to zero on static/hybrid_rl).
#      its operating-envelope violation rate is now
#      directly comparable across all benchmark modes.
#
# Schema-side: the public ``ConstraintViolationRate`` CSV column maps to the
# canonical ``constraint_violation_rate`` record.  The retained
# ``operational_violation_rate`` alias has the same seed-level values, but it
# must not be used as an independent bootstrap source: cell-keyed Monte Carlo
# streams can otherwise give two slightly different finite-resample intervals
# for the same estimand.  One canonical record keeps the JSON and CSV endpoints
# identical by construction. The
# ``OperatingEnvelopeViolationRate`` maps to the explicitly non-regulatory
# operating-envelope alias and is uniform across all modes.
_TABLE1_COLUMNS = (
    ("ari", "ARI"), ("rle", "RLE"), ("waste", "Waste"),
    ("slca", "SLCA"), ("carbon", "Carbon"), ("equity", "Equity"),
    ("constraint_violation_rate", "ConstraintViolationRate"),
    ("operating_envelope_violation_rate", "OperatingEnvelopeViolationRate"),
    # Outcome-side disposition on the common violation-event set.
    ("downstream_violation_rate", "DownstreamViolationRate"),
    ("contained_violation_rate", "ContainedViolationRate"),
)
_TABLE2_COLUMNS = (
    ("ari", "ARI"), ("rle", "RLE"), ("waste", "Waste"), ("slca", "SLCA"),
    ("carbon", "Carbon"), ("equity", "Equity"),
    ("constraint_violation_rate", "ConstraintViolationRate"),
    ("downstream_violation_rate", "DownstreamViolationRate"),
    ("contained_violation_rate", "ContainedViolationRate"),
)
_TABLE1_ROW_METHODS = (
    "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
    "mcp_only", "pirag_only", "agribrain",
)
BASELINES = (
    "mcp_only", "pirag_only", "no_context", "no_pinn", "no_slca",
    "hybrid_rl", "static",
)

# These direct single-channel-versus-No-external-context records are two of
# the four prespecified H2 contrasts.  A historical 10-test subset correction
# is retained only as an explicitly auxiliary compatibility field; it is not
# the H2 result.  Confirmatory H2 uses one Holm family over all four
# directional contrasts in all five scenarios (m=20).
_CHANNEL_DECOMPOSITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("mcp_only",   "no_context"),
    ("pirag_only", "no_context"),
)
"""Cross-baseline pairs that test each context channel directly
against the no-context floor (C4 paper claim)."""

# H2 is an explicitly directional, four-contrast design.  The first two
# contrasts establish whether each isolated external channel improves on the
# No-external-context floor.  The latter two establish whether the integrated
# system improves on each single-channel arm.  Treating only the first two as
# the multiplicity family left the integration part of the manuscript claim
# statistically unprotected.
_H2_DIRECTIONAL_PAIRS = H2_DIRECTIONAL_PAIRS
"""Compatibility alias for the prespecified 20-cell H2 family."""

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
seed_dir = _SCRIPT_DIR / "results" / "benchmark_seeds"


def _cell_seed(scope: str, cell_key: tuple) -> int:
    """Deterministic but cell-keyed RNG seed.

    Implementation note: 2025-04 cell-correlation fix +
    2026-04 cross-process reproducibility fix.

    Previous revisions used a constant seed (42, 24, 123) for every
    bootstrap and permutation call, which made adjacent (scenario, mode,
    metric) cells share the same resample sequence and therefore have
    correlated bootstrap noise. The 2025-04 fix derived a 32-bit seed
    from ``hash((scope, *cell_key))`` so each cell got independent
    resampling. The 2026-04 fix replaced ``hash()`` with
    ``hashlib.blake2b()``: Python's built-in ``hash()`` is
    PYTHONHASHSEED-randomised by default for str / bytes / tuple
    inputs, so two HPC runs in different Python processes (or the
    same process with different PYTHONHASHSEED) produced different
    bootstrap samples for the same cell - silently breaking the
    "fully reproducible run-to-run" claim this docstring made. The
    blake2b digest is purely deterministic and gives the same 32-bit
    seed across processes / OSes / Python versions.
    """
    import hashlib
    payload = "::".join((scope,) + tuple(str(p) for p in cell_key))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


def _resampling_identity(seed_order: list[int]) -> dict:
    """Return the complete, JSON-safe resampling reproducibility contract."""
    return {
        "schema_version": _RESAMPLING_IDENTITY_VERSION,
        "generator": "numpy.random.Generator",
        "bit_generator": "PCG64",
        "seed_derivation": {
            "algorithm": "BLAKE2b",
            "digest_size_bytes": 4,
            "key_hex": "",
            "salt_hex": "",
            "personalization_hex": "",
            "payload_encoding": "UTF-8",
            "payload_template": "scope::cell_key[0]::cell_key[1]::...",
            "integer_conversion": "unsigned big-endian",
        },
        "scopes": list(_RESAMPLING_SCOPES),
        "summary_cell_key": ["scenario", "mode", "metric"],
        "comparison_cell_key": ["scenario", "comparison", "metric"],
        "observation_order": {
            "kind": "explicit_seed_order",
            "seeds": [int(seed) for seed in seed_order],
        },
        "example": {
            "scope": "bootstrap_ci",
            "cell_key": ["heatwave", "agribrain", "ari"],
            "derived_seed": _cell_seed(
                "bootstrap_ci", ("heatwave", "agribrain", "ari")
            ),
        },
    }


def bootstrap_ci(vals, n_boot=10_000, alpha=0.05, cell_key=("global",)):
    """BCa bootstrap CI for the mean with 10,000 resamples.

    Bias-corrected and accelerated (Efron, 1987). BCa adjusts the
    bootstrap quantiles for estimated bias and skew using the bootstrap
    distribution and a jackknife acceleration estimate. cell_key seeds the resampler so
    adjacent cells have independent Monte Carlo error.

    2026-05: zero-variance inputs are short-circuited BEFORE calling
    BCa. Such cells (deterministic-by-construction quantities like
    context_active_steps with a hardcoded schedule, or null-mean rates
    like context_honor_rate on static) cannot have a BCa CI -- z0 is
    mathematically undefined when every bootstrap replicate equals
    theta_hat. Pre-2026-05 these were silently routed through the
    percentile fallback and counted as "BCa fallback rate", which
    inflated the headline 8.3 % stat with cells where BCa was never
    going to apply. The honest rate is the rate of cases where BCa
    was attempted on variance > 0 input AND the machinery still
    couldn't recover z0 -- which is ~0.
    """
    arr = np.array(vals, dtype=float)
    if len(arr) < 2:
        return float(np.mean(arr)) if len(arr) else 0.0, float(np.mean(arr)) if len(arr) else 0.0

    theta_hat = float(np.mean(arr))
    # Zero-variance short-circuit: BCa is mathematically undefined.
    # Emit (theta_hat, theta_hat) and increment the deterministic
    # counter so the _meta block can report the cell as "ci_method =
    # deterministic" rather than as a BCa fallback.
    if float(np.std(arr, ddof=1)) == 0.0:
        _BCA_STATS["deterministic_cells"] += 1
        return theta_hat, theta_hat

    rng = np.random.default_rng(_cell_seed("bootstrap_ci", cell_key))
    boots = np.array([
        float(np.mean(rng.choice(arr, len(arr), replace=True)))
        for _ in range(n_boot)
    ])

    # Jackknife acceleration on the data
    n = len(arr)
    jacks = np.array([float(np.mean(np.delete(arr, i))) for i in range(n)])

    return _bca_ci_from_boots(boots, theta_hat, jacks, alpha)


def bootstrap_mean_diff_ci(a, b, n_boot=10_000, alpha=0.05, paired=True, cell_key=("global",)):
    """BCa bootstrap CI for mean(a) - mean(b) with 10,000 resamples.

    paired=True resamples a single index applied to both arms (correct
    when a and b come from a matched-seed paired design). paired=False
    independently resamples each arm (correct when the two arms have
    independent seeds). BCa correction is applied (Efron 1987).
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return 0.0, 0.0

    # Zero-variance short-circuit. For paired diffs the relevant
    # quantity is the variance of (x - y); for unpaired the variances
    # of x and y separately. Either way, if all the seed-level
    # estimates that go into the bootstrap are constant, BCa is
    # mathematically undefined. Same reasoning as bootstrap_ci above.
    if paired and x.shape == y.shape:
        diff = x - y
        if float(np.std(diff, ddof=1)) == 0.0:
            _BCA_STATS["deterministic_cells"] += 1
            theta_hat = float(np.mean(diff))
            return theta_hat, theta_hat
    else:
        if (float(np.std(x, ddof=1)) == 0.0
                and float(np.std(y, ddof=1)) == 0.0):
            _BCA_STATS["deterministic_cells"] += 1
            theta_hat = float(np.mean(x) - np.mean(y))
            return theta_hat, theta_hat

    rng = np.random.default_rng(_cell_seed("bootstrap_diff_ci", cell_key))
    boots = []
    if paired and x.shape == y.shape:
        idx = np.arange(len(x))
        for _ in range(n_boot):
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            boots.append(float(np.mean(x[sample_idx] - y[sample_idx])))
        theta_hat = float(np.mean(x - y))
        jacks = np.array([
            float(np.mean(np.delete(x, i) - np.delete(y, i)))
            for i in range(len(x))
        ])
    else:
        idx_a = np.arange(len(x))
        idx_b = np.arange(len(y))
        for _ in range(n_boot):
            mean_a = float(np.mean(x[rng.choice(idx_a, size=len(idx_a), replace=True)]))
            mean_b = float(np.mean(y[rng.choice(idx_b, size=len(idx_b), replace=True)]))
            boots.append(mean_a - mean_b)
        theta_hat = float(np.mean(x) - np.mean(y))
        jacks = np.empty(len(x) + len(y))
        for i in range(len(x)):
            jacks[i] = float(np.mean(np.delete(x, i)) - np.mean(y))
        for j in range(len(y)):
            jacks[len(x) + j] = float(np.mean(x) - np.mean(np.delete(y, j)))

    return _bca_ci_from_boots(np.asarray(boots, dtype=float), theta_hat, jacks, alpha)


# Module-level fallback counters. These are reset at the start of each
# aggregator run by aggregate_main(), incremented by _bca_ci_from_boots
# when the percentile fallback fires, and emitted into
# benchmark_summary._meta.bca_fallback_stats so reviewers (and the
# methods section) can quote the exact percentage of cells that fell
# back from BCa to plain percentile. With n=20, finite-sample instability is
# possible for highly discrete or skewed cells; the fallback is silent without
# these counters, which is exactly the
# silent-fallback pattern the post-2026-04 audit flagged.
# 2026-05 honest-counter restructure. Pre-2026-05 the counter set
# conflated two semantically-different events:
#   1. "BCa is mathematically undefined for this cell" -- the input
#      array has zero across-seed variance (deterministic-by-construction
#      cells like context_active_steps=72 every seed, or null-mean cells
#      like context_honor_rate on static). 218 of the 218 reported
#      "fallbacks" on the d33b8de run were actually this.
#   2. "BCa attempted on real-variance data but the bias-correction
#      machinery couldn't recover z0" -- a true statistical fallback
#      that should be rare and is the only thing the methods section
#      should report as a "BCa fallback rate".
# The 8.3 % "fallback rate" on the d33b8de run was 100 % case-1
# events, none case-2. Splitting the counter:
#   bca_calls           = cells where BCa was actually attempted
#                          (input variance > 0)
#   bca_fallbacks       = cells where BCa failed despite variance > 0
#                          (the "true" fallback rate, target ~0)
#   deterministic_cells = cells skipped because variance is 0
#                          (mathematically can't have a BCa CI; emits
#                          ci_method="deterministic" instead of "bca")
_BCA_STATS = {
    "bca_calls": 0,
    "bca_fallbacks": 0,
    "fallback_scipy_unavailable": 0,
    "deterministic_cells": 0,
}

# Records any declared Wilcoxon cell that had to use the sign-flip fallback.
# The set is emitted in the output metadata and each affected comparison so a
# dependency/runtime problem can never silently change the inferential test.
_WILCOXON_FALLBACK_CELLS: set[tuple] = set()


def _bca_ci_from_boots(boots: np.ndarray, theta_hat: float,
                        jacks: np.ndarray, alpha: float = 0.05):
    """Compute BCa percentiles from a precomputed bootstrap distribution.

    Falls back to the plain percentile method when the BCa correction
    cannot be estimated. Two failure modes:
      - ``fallback_p0_degenerate`` (legacy aggregate label): the empirical
        bias or jackknife acceleration term is degenerate.
      - ``fallback_scipy_unavailable``: scipy.special.ndtri is missing
        (rare; primary fix is to install scipy via the pyproject.toml
        dependency).

    Each fallback increments a module-level counter that is emitted
    into the aggregator's _meta block so reviewers can see what
    fraction of cells used percentile fallback. Silent fallback was
    the post-2026-04 audit flag.
    """
    from math import erf, sqrt
    _BCA_STATS["bca_calls"] += 1
    if len(boots) < 2:
        return float(theta_hat), float(theta_hat)
    # Mid-rank treatment of ties matches the usual empirical percentile used
    # for BCa bias correction: P(T* < T) + 0.5 P(T* = T).  Using strict '<'
    # alone can spuriously drive z0 to -inf for discrete simulation endpoints
    # with many bootstrap ties.
    p0 = float(
        (np.count_nonzero(boots < theta_hat)
         + 0.5 * np.count_nonzero(boots == theta_hat)) / len(boots)
    )
    if p0 <= 0.0 or p0 >= 1.0:
        # 2026-05: this is the "true" BCa fallback (target ~0). The
        # callers (bootstrap_ci / bootstrap_mean_diff_ci) already
        # short-circuit on zero-variance input, so by the time we get
        # here the bootstrap distribution should have non-trivial
        # spread. Hitting p0 in {0, 1} despite that means the data has
        # an extreme distribution shape (e.g. heavy one-sided point
        # mass with a thin continuous tail) where BCa's bias-correction
        # heuristic still degenerates. Falling back to plain percentile
        # is the standard defensive move.
        _BCA_STATS["bca_fallbacks"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))
    try:
        from scipy.special import ndtri  # type: ignore
        z0 = float(ndtri(p0))
        z_lo = float(ndtri(alpha / 2))
        z_hi = float(ndtri(1.0 - alpha / 2))
    except Exception:
        _BCA_STATS["fallback_scipy_unavailable"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))

    m = float(np.mean(jacks))
    num = float(np.sum((m - jacks) ** 3))
    jack_ss = float(np.sum((m - jacks) ** 2))
    if jack_ss <= 0.0 or not np.isfinite(jack_ss):
        _BCA_STATS["bca_fallbacks"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))
    a_acc = num / (6.0 * jack_ss ** 1.5)

    def _adj(z_q: float) -> float | None:
        denominator = 1.0 - a_acc * (z0 + z_q)
        # Preserve the sign of the BCa denominator.  The former max(...,
        # 1e-12) silently changed the formula whenever it was negative.
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            return None
        x = z0 + (z0 + z_q) / denominator
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    p_lo, p_hi = _adj(z_lo), _adj(z_hi)
    if (
        p_lo is None or p_hi is None
        or not np.isfinite(p_lo) or not np.isfinite(p_hi)
        or p_lo > p_hi
    ):
        _BCA_STATS["bca_fallbacks"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))
    p_lo = max(min(p_lo, 1.0 - 1e-9), 1e-9)
    p_hi = max(min(p_hi, 1.0 - 1e-9), 1e-9)
    return float(np.quantile(boots, p_lo)), float(np.quantile(boots, p_hi))


def _bca_fallback_stats_snapshot() -> dict:
    """Return the current bootstrap-CI stats as a JSON-friendly dict.

    Three counters, three different stories the methods section may
    want to report (2026-05 honest restructure):

      bca_calls           -- cells where BCa was attempted (input had
                              non-zero across-seed variance).
      bca_fallbacks       -- cells where BCa was attempted but
                              degenerated to plain percentile despite
                              variance > 0. This is the "true" BCa
                              fallback rate; target ~0.
      fallback_scipy_unavailable
                          -- scipy.special.ndtri import failed; should
                              be 0 in the production env.
      deterministic_cells -- cells short-circuited because their input
                              array had zero variance (BCa is
                              mathematically undefined). These cells
                              emit a deterministic ci_method marker
                              and a [mean, mean] CI; they were always
                              going to be deterministic, not a
                              statistical "fallback".

    Headline rate is now ``bca_fallback_rate = bca_fallbacks /
    max(bca_calls, 1)`` -- this is the only honest number the methods
    section should quote. Pre-2026-05 the headline rate conflated
    bca_fallbacks + deterministic_cells into one stat (the d33b8de
    8.3 % was 100 % deterministic, 0 % true BCa fallback).
    """
    bca_calls = _BCA_STATS["bca_calls"]
    bca_fallbacks = _BCA_STATS["bca_fallbacks"]
    scipy_unavail = _BCA_STATS["fallback_scipy_unavailable"]
    deterministic = _BCA_STATS["deterministic_cells"]
    total_fallback = bca_fallbacks + scipy_unavail
    snapshot = {
        "bca_calls": bca_calls,
        "bca_fallbacks": bca_fallbacks,
        "fallback_scipy_unavailable": scipy_unavail,
        "deterministic_cells": deterministic,
        # The "true" BCa fallback rate (denominator excludes
        # deterministic cells where BCa was never going to apply).
        "bca_fallback_rate": (
            float(total_fallback / bca_calls) if bca_calls else 0.0
        ),
        # Back-compat aliases so the previous _meta consumers
        # (validation scripts, the export_paper_evidence.py meta
        # propagation, methods footnotes) don't crash on key absence.
        # The "fallback_rate" here matches the pre-2026-05 semantics
        # (fraction of all bootstrap_ci calls where the percentile
        # path was taken, INCLUDING deterministic-by-construction
        # cells) so downstream readers get the same number. New
        # consumers should prefer "bca_fallback_rate".
        "calls": bca_calls + deterministic,
        "fallback_p0_degenerate": bca_fallbacks + deterministic,
        "fallback_total": total_fallback + deterministic,
        "fallback_rate": (
            float((total_fallback + deterministic) / (bca_calls + deterministic))
            if (bca_calls + deterministic) else 0.0
        ),
    }
    return snapshot


def _reset_bca_fallback_stats() -> None:
    """Reset bootstrap-CI counters to zero (called at aggregator start)."""
    _BCA_STATS["bca_calls"] = 0
    _BCA_STATS["bca_fallbacks"] = 0
    _BCA_STATS["fallback_scipy_unavailable"] = 0
    _BCA_STATS["deterministic_cells"] = 0


def _ci_method_since(before: dict, low, high) -> str:
    """Classify the CI path used by the immediately preceding CI call.

    The aggregate metadata counts fallback paths globally, but every published
    cell also needs its own honest method label.  Callers take ``dict(_BCA_STATS)``
    immediately before invoking a bootstrap helper and pass the returned bounds
    here immediately afterwards.
    """
    if low is None or high is None:
        return "undefined"
    if _BCA_STATS["deterministic_cells"] > before["deterministic_cells"]:
        return "deterministic"
    if (
        _BCA_STATS["bca_fallbacks"] > before["bca_fallbacks"]
        or _BCA_STATS["fallback_scipy_unavailable"]
        > before["fallback_scipy_unavailable"]
    ):
        return "percentile_fallback"
    if _BCA_STATS["bca_calls"] > before["bca_calls"]:
        return "BCa"
    return "point_interval"


def wilcoxon_signed_rank_pvalue(
    a, b, cell_key=("global",), *, alternative: str = "two-sided",
):
    """Wilcoxon signed-rank p-value via SciPy with labelled fallback.

    The Wilcoxon signed-rank test assumes a symmetric distribution of paired
    differences and tests its location relative to zero. It is used because it
    matches the declared analysis plan; it is not distribution-free with
    respect to asymmetry. When SciPy is unavailable, we fall back to a
    sign-flip permutation and label that fallback clearly in the method
    metadata.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unsupported Wilcoxon alternative: {alternative!r}")
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) == 0:
        return 1.0
    d = x - y
    nz = d[d != 0]
    if len(nz) < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        # Zero differences are removed above.  Confirmatory H1 and H2 use the
        # prespecified directional ``greater`` alternative; descriptive grids
        # retain two-sided tests.
        res = wilcoxon(
            nz, zero_method="wilcox", alternative=alternative, method="auto",
        )
        return float(res.pvalue)
    except Exception:
        # Fallback to sign-flip permutation and record the exact cell so the
        # output metadata and per-comparison record expose the method change.
        _WILCOXON_FALLBACK_CELLS.add(tuple(cell_key))
        rng = np.random.default_rng(
            _cell_seed("wilcoxon_fallback", (*tuple(cell_key), alternative))
        )
        observed_raw = float(np.mean(d))
        observed = abs(observed_raw) if alternative == "two-sided" else observed_raw
        ge = 0
        n_perm = 10_000
        for _ in range(n_perm):
            signs = rng.choice([-1.0, 1.0], size=len(d))
            permuted = float(np.mean(d * signs))
            extreme = (
                abs(permuted) >= observed
                if alternative == "two-sided"
                else permuted >= observed
                if alternative == "greater"
                else permuted <= observed
            )
            if extreme:
                ge += 1
        return float((ge + 1) / (n_perm + 1))


def paired_permutation_pvalue(a, b, n_perm=10_000, cell_key=("global",)):
    """Paired sign-flip permutation p-value (legacy alias).

    Kept for backward compatibility. New code should call
    `wilcoxon_signed_rank_pvalue` for paired comparisons because the
    sign-flip null requires symmetry about zero, which is not
    guaranteed under our multiplicative noise model. See Implementation note
    on `wilcoxon_signed_rank_pvalue`.
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) == 0:
        return 1.0
    d = x - y
    observed = abs(float(np.mean(d)))
    rng = np.random.default_rng(_cell_seed("paired_perm", cell_key))
    ge = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        if abs(float(np.mean(d * signs))) >= observed:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def mann_whitney_pvalue(a, b, cell_key=("global",)):
    """Two-sided Mann-Whitney U p-value (unpaired non-parametric).

    Falls back to an unpaired-mean-difference permutation test when the
    scipy call fails. The previous implementation returned a silent
    p=1.0 on any exception, which silently nullified the headline
    AgriBrain-vs-Static and AgriBrain-vs-Hybrid-RL significance claims
    on HPC clusters whose scipy was older than the ``alternative=
    "two-sided"`` keyword (or on edge cases where mannwhitneyu raised
    on perfect rank separation). The fallback gives the correct p
    estimate (typically ~1/n_perm for huge effects) consistent with
    ``paired_permutation_pvalue`` returned in
    ``p_value_legacy_signflip``.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return 1.0
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        pass
    # Fallback: unpaired permutation on the difference of means. Pool
    # both arms, repeatedly shuffle the partition into two equal-size
    # halves, and count the fraction with |mean diff| >= observed.
    rng = np.random.default_rng(_cell_seed("mannwhitney_fallback", cell_key))
    observed = abs(float(np.mean(x) - np.mean(y)))
    pooled = np.concatenate([x, y])
    n_a = x.size
    n_perm = 10_000
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(float(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])))
        if diff >= observed:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def cohens_dz(a, b):
    """Paired Cohen's d_z = mean(a-b) / std(a-b).

    Appropriate for repeated-measures / matched designs where (a, b) are
    paired observations. Standardised by the within-pair standard
    deviation, which is small when the two arms share environmental
    variance — large d_z values reflect both effect size AND the
    precision of the paired design. It is the primary standardized effect for
    the confirmatory matched-seed comparisons.
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) < 2:
        return None
    d = x - y
    sd = np.std(d, ddof=1)
    mean_d = float(np.mean(d))
    if sd > 0:
        return float(mean_d / sd)
    # A constant nonzero paired difference has no finite standardized effect;
    # returning zero would incorrectly encode "no effect". A constant zero
    # difference is reported as zero by explicit convention.
    return 0.0 if mean_d == 0.0 else None


def cohens_d_pooled(a, b):
    """Unpaired (pooled) Cohen's d = (mean(a) - mean(b)) / s_pooled.

    s_pooled = sqrt(((n_a-1)*var(a) + (n_b-1)*var(b)) / (n_a+n_b-2)).

    This statistic is retained as a secondary descriptive standardization. It
    must not be interpreted as deployment variability because the seed panel is
    synthetic and paired. The matched-design primary effect is ``cohens_dz``.
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    n_a, n_b = len(x), len(y)
    var_a = np.var(x, ddof=1)
    var_b = np.var(y, ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def hedges_g(a, b, paired: bool = False):
    """Hedges' g — small-sample-corrected Cohen's d.

    g = J(df) * d, where J(df) = 1 - 3/(4*df - 1). For 20 pairs the
    correction is approximately 0.960 when applied to d_z (df=19); for two
    independent groups of 20 it is approximately 0.980 (df=38).
    """
    if paired:
        d = cohens_dz(a, b)
        df = max(len(a) - 1, 1)
    else:
        d = cohens_d_pooled(a, b)
        df = max(len(a) + len(b) - 2, 1)
    if d is None:
        return None
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(j * d)


def bootstrap_effect_size_ci(a, b, n_boot: int = 10_000, alpha: float = 0.05,
                              paired: bool = True, cell_key=("global",),
                              statistic: str = "pooled"):
    """95 % BCa bootstrap CI on the requested Cohen's d statistic.

    ``statistic="pooled"`` returns a CI on ``cohens_d_pooled``.
    ``statistic="dz"`` returns a CI on the matched-design primary effect,
    ``cohens_dz``.

    BCa correction (Efron 1987) handles the bias and skew that the
    plain percentile method misses on n=20 with non-normal residuals.
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(_cell_seed(f"d_ci_{statistic}", cell_key))

    def _stat(xa, xb):
        if statistic == "dz":
            return cohens_dz(xa, xb)
        return cohens_d_pooled(xa, xb)

    theta_hat = _stat(x, y)
    if theta_hat is None or not np.isfinite(theta_hat):
        return None, None
    boots = []
    if paired and x.shape == y.shape:
        idx = np.arange(len(x))
        for _ in range(n_boot):
            sel = rng.choice(idx, size=len(idx), replace=True)
            value = _stat(x[sel], y[sel])
            if value is not None and np.isfinite(value):
                boots.append(value)
    else:
        idx_a = np.arange(len(x))
        idx_b = np.arange(len(y))
        for _ in range(n_boot):
            sa = rng.choice(idx_a, size=len(idx_a), replace=True)
            sb = rng.choice(idx_b, size=len(idx_b), replace=True)
            value = _stat(x[sa], y[sb])
            if value is not None and np.isfinite(value):
                boots.append(value)

    if len(boots) < 2:
        return None, None
    return _bca_quantiles(np.asarray(boots, dtype=float), theta_hat,
                           paired_xy=(x, y) if (paired and x.shape == y.shape) else None,
                           unpaired_xy=(x, y) if not (paired and x.shape == y.shape) else None,
                           statistic_fn=_stat, alpha=alpha)


def _bca_quantiles(boots: np.ndarray, theta_hat: float,
                   paired_xy=None, unpaired_xy=None,
                   statistic_fn=None, alpha: float = 0.05):
    """Return BCa-corrected lower/upper percentiles for a bootstrap sample.

    Falls back to the plain percentile method when the acceleration or
    bias-correction terms cannot be estimated (rare; happens when all
    bootstrap replicates equal theta_hat exactly). Increments the
    module-level ``_BCA_STATS`` counters on each fallback path so the
    aggregator's _meta block surfaces a non-zero ``fallback_rate``
    when the effect-size BCa step degenerates - mirrors the
    instrumentation in ``_bca_ci_from_boots``. Earlier, only the
    mean-CI BCa path incremented the counters and the effect-size CI
    fallbacks were silent, which under-reported the published
    fallback_rate value.
    """
    from math import erf, sqrt
    _BCA_STATS["bca_calls"] += 1
    n_boot = len(boots)
    if n_boot < 2:
        return float(theta_hat), float(theta_hat)

    # Bias correction z0
    p0 = float(
        (np.count_nonzero(boots < theta_hat)
         + 0.5 * np.count_nonzero(boots == theta_hat)) / len(boots)
    )
    if p0 <= 0.0 or p0 >= 1.0:
        # All bootstrap values on one side of theta_hat -> percentile fallback.
        # 2026-05: callers that wrap this function should already have
        # short-circuited on zero-variance input, so reaching this branch
        # means the bootstrap distribution had non-trivial spread but
        # BCa's z0 still degenerated -- the "true" BCa fallback,
        # tracked under bca_fallbacks. Target ~0.
        _BCA_STATS["bca_fallbacks"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))

    def _phi_inv(p: float) -> float:
        # Beasley-Springer-Moro inverse normal CDF (good enough for n=20)
        from scipy.special import ndtri  # type: ignore
        return float(ndtri(p))

    try:
        z0 = _phi_inv(p0)
    except Exception:
        _BCA_STATS["fallback_scipy_unavailable"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))

    # Acceleration via jackknife on the original observations
    a_acc = 0.0
    if paired_xy is not None and statistic_fn is not None:
        x, y = paired_xy
        n = len(x)
        jacks = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            value = statistic_fn(x[mask], y[mask])
            if value is None or not np.isfinite(value):
                _BCA_STATS["bca_fallbacks"] += 1
                return (float(np.quantile(boots, alpha / 2)),
                        float(np.quantile(boots, 1 - alpha / 2)))
            jacks[i] = value
        m = jacks.mean()
        num = np.sum((m - jacks) ** 3)
        jack_ss = float(np.sum((m - jacks) ** 2))
        if jack_ss <= 0.0 or not np.isfinite(jack_ss):
            _BCA_STATS["bca_fallbacks"] += 1
            return (float(np.quantile(boots, alpha / 2)),
                    float(np.quantile(boots, 1 - alpha / 2)))
        a_acc = float(num / (6.0 * jack_ss ** 1.5))
    elif unpaired_xy is not None and statistic_fn is not None:
        x, y = unpaired_xy
        nx, ny = len(x), len(y)
        jacks = np.empty(nx + ny)
        for i in range(nx):
            mask = np.ones(nx, dtype=bool); mask[i] = False
            value = statistic_fn(x[mask], y)
            if value is None or not np.isfinite(value):
                _BCA_STATS["bca_fallbacks"] += 1
                return (float(np.quantile(boots, alpha / 2)),
                        float(np.quantile(boots, 1 - alpha / 2)))
            jacks[i] = value
        for j in range(ny):
            mask = np.ones(ny, dtype=bool); mask[j] = False
            value = statistic_fn(x, y[mask])
            if value is None or not np.isfinite(value):
                _BCA_STATS["bca_fallbacks"] += 1
                return (float(np.quantile(boots, alpha / 2)),
                        float(np.quantile(boots, 1 - alpha / 2)))
            jacks[nx + j] = value
        m = jacks.mean()
        num = np.sum((m - jacks) ** 3)
        jack_ss = float(np.sum((m - jacks) ** 2))
        if jack_ss <= 0.0 or not np.isfinite(jack_ss):
            _BCA_STATS["bca_fallbacks"] += 1
            return (float(np.quantile(boots, alpha / 2)),
                    float(np.quantile(boots, 1 - alpha / 2)))
        a_acc = float(num / (6.0 * jack_ss ** 1.5))

    z_lo = _phi_inv(alpha / 2)
    z_hi = _phi_inv(1.0 - alpha / 2)

    def _adj(z_q: float) -> float | None:
        # Standard normal CDF via erf.  A singular acceleration correction is
        # an explicit percentile fallback, not an opportunity to alter the
        # denominator's sign.
        denominator = 1.0 - a_acc * (z0 + z_q)
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            return None
        x = z0 + (z0 + z_q) / denominator
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    p_lo, p_hi = _adj(z_lo), _adj(z_hi)
    if (
        p_lo is None or p_hi is None
        or not np.isfinite(p_lo) or not np.isfinite(p_hi)
        or p_lo > p_hi
    ):
        _BCA_STATS["bca_fallbacks"] += 1
        return (float(np.quantile(boots, alpha / 2)),
                float(np.quantile(boots, 1 - alpha / 2)))
    p_lo = max(min(p_lo, 1.0 - 1e-9), 1e-9)
    p_hi = max(min(p_hi, 1.0 - 1e-9), 1e-9)
    return float(np.quantile(boots, p_lo)), float(np.quantile(boots, p_hi))


def benjamini_yekutieli(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini-Yekutieli step-up FDR correction (valid under arbitrary dependence).

    Differs from BH-FDR by a factor c(m) = sum_{i=1..m} 1/i in the
    threshold formula. More conservative than BH but doesn't require
    PRDS — the right choice when the m hypotheses can have negative
    correlations (e.g., waste vs ARI metrics that share simulation
    traces).

    Implementation note: added 2025-04 in response to the dependence-violation
    concern. Within-scenario metrics are mechanically correlated with
    sign varying by metric pair; PRDS is not guaranteed.
    """
    keys = list(p_values.keys())
    m = len(keys)
    if m == 0:
        return {}
    c_m = sum(1.0 / i for i in range(1, m + 1))
    ordered = sorted(((k, float(p_values[k])) for k in keys), key=lambda kv: kv[1])
    adjusted = {}
    prev = 1.0
    for rank_rev, (k, p) in enumerate(reversed(ordered), start=1):
        i = m - rank_rev + 1
        q = min(prev, (p * m * c_m) / max(i, 1))
        adjusted[k] = float(min(max(q, 0.0), 1.0))
        prev = adjusted[k]
    return adjusted


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg step-up FDR correction.

    Controls the false discovery rate at alpha. Preserves input keys.
    Returns each key's BH-adjusted p-value. Order-independent in the output.

    Implementation note: post-2026-04 propagation-bug fix. The earlier
    body used ``prev = q`` after the clip-to-[0, 1] step, which left
    ``prev`` carrying the unclipped pre-clip ``q`` from the previous
    iteration when ``p * m / i > 1``. BY-FDR (above) correctly used
    ``prev = adjusted[k]`` so the propagated bound is the clipped
    value. Asymmetry between the two FDR routines was confusing for
    a maintainer reading the code; switched BH to the same
    ``prev = adjusted[k]`` idiom so both functions track the same
    monotonic-in-rank step-up envelope.
    """
    keys = list(p_values.keys())
    m = len(keys)
    if m == 0:
        return {}
    ordered = sorted(((k, float(p_values[k])) for k in keys), key=lambda kv: kv[1])
    adjusted = {}
    prev = 1.0
    for rank_rev, (k, p) in enumerate(reversed(ordered), start=1):
        i = m - rank_rev + 1
        q = min(prev, (p * m) / max(i, 1))
        adjusted[k] = float(min(max(q, 0.0), 1.0))
        prev = adjusted[k]
    return adjusted


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down FWER correction.

    Controls the family-wise error rate. Stricter than BH-FDR. Preserves
    input keys. Matches paper Section 3.13's declared multiplicity control
    for the primary H1 family (the five scenario-level agribrain vs
    no_context comparisons on ARI).
    """
    keys = list(p_values.keys())
    m = len(keys)
    if m == 0:
        return {}
    ordered = sorted(((k, float(p_values[k])) for k in keys), key=lambda kv: kv[1])
    adjusted = {}
    running = 0.0
    for rank_idx, (k, p) in enumerate(ordered):
        # Holm step-down: p_(i) * (m - i + 1), then monotone non-decreasing
        q = min(1.0, p * (m - rank_idx))
        running = max(running, q)
        adjusted[k] = float(running)
    return adjusted


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args(argv)
    input_seed_dir = args.seed_root.resolve()
    out_dir = args.output_dir.resolve()
    canonical_results = (_SCRIPT_DIR / "results").resolve()
    canonical_seed_dir = (canonical_results / "benchmark_seeds").resolve()
    if out_dir == canonical_results:
        if (
            not args.publication
            or os.environ.get("STRICT_VALIDATION") != "1"
            or os.environ.get("AGRIBRAIN_PUBLICATION_AGGREGATION") != "1"
            or input_seed_dir != canonical_seed_dir
        ):
            raise RuntimeError(
                "canonical benchmark aggregation is restricted to the locked "
                "HPC publisher and exact flat archived seed directory"
            )
    elif args.publication:
        raise RuntimeError("--publication requires the canonical results directory")
    if not input_seed_dir.is_dir():
        raise FileNotFoundError(f"benchmark seed root is missing: {input_seed_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _SCRIPT_DIR.parents[1]
    recovery_provenance = recovery_context_from_environment(
        results_dir=out_dir,
        repo_root=repo_root,
    )
    if recovery_provenance is not None and not args.publication:
        raise RuntimeError(
            "deterministic recovery provenance is valid only for the locked "
            "publication aggregation path"
        )

    # Reset BCa fallback counters so the per-run stats reflect only
    # this aggregator invocation, not residue from prior calls in the
    # same Python process (relevant in tests).
    _reset_bca_fallback_stats()
    _WILCOXON_FALLBACK_CELLS.clear()

    seed_csv = os.environ.get(
        "BENCHMARK_SEEDS",
        "42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515",
    ).strip()
    seeds = []
    for raw in seed_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            seeds.append(int(raw))
        except ValueError:
            continue
    if not seeds:
        seeds = SEEDS
    print(f"Configured seed count: {len(seeds)}")

    # Load seed results.
    #
    # Per-seed JSON envelope (post-2026-05, written by run_single_seed.py):
    #     {"seed": int,
    #      "scenarios": {sc: {mode: {metric: value}}},
    #      "traces":    {sc: {mode: {trace_field: [floats]}}}}
    #
    # Legacy flat format (pre-2026-05): scenarios at the top level
    # directly:
    #     {sc: {mode: {metric: value}}}
    #
    # Detect the envelope by the presence of a top-level "scenarios"
    # key whose value is a dict; unwrap if so. The aggregator's per-
    # cell access pattern (``all_data[s][sc][mode][met]``) is the
    # legacy flat shape, so we normalise on load. Without this fix
    # every metric is silently filtered out (the .get(sc, {}) check
    # returns empty), the BCa loop never runs (calls=0), and every
    # summary cell ends up {} -- the failure mode that surfaced on
    # HPC RUN_TAG 485c769_20260505_0349.
    all_data = {}
    for seed in seeds:
        f = input_seed_dir / f"seed_{seed}.json"
        if f.exists():
            def _reject_constant(value: str, f: Path = f):
                raise ValueError(f"non-finite JSON constant {value!r} in {f}")

            payload = json.loads(
                f.read_text(encoding="utf-8"), parse_constant=_reject_constant
            )
            if payload.get("_trace_failures"):
                raise RuntimeError(
                    f"Seed {seed} contains trace serialization failures: "
                    f"{payload['_trace_failures']}"
                )
            scenarios_block = payload.get("scenarios")
            if isinstance(scenarios_block, dict):
                all_data[seed] = scenarios_block
            else:
                all_data[seed] = payload
            print(f"Loaded seed {seed}")
        else:
            print(f"WARNING: {f} not found, skipping")

    if len(all_data) < 2:
        print(f"ERROR: Only {len(all_data)} seed(s) found, need at least 2")
        sys.exit(1)

    print(f"Aggregating {len(all_data)} seeds...")

    # Publication runs are complete balanced panels. Silently aggregating a
    # different seed subset in different cells would destroy both the matched
    # design and the declared sample size, so fail before computing anything.
    expected_seed_set = set(seeds)
    loaded_seed_set = set(all_data)
    if loaded_seed_set != expected_seed_set:
        missing = sorted(expected_seed_set - loaded_seed_set)
        extra = sorted(loaded_seed_set - expected_seed_set)
        raise RuntimeError(
            "Incomplete benchmark seed panel: "
            f"missing={missing}, unexpected={extra}."
        )
    required_metrics = set(METRICS)
    incomplete_cells = []
    for seed, seed_data in all_data.items():
        for sc in SCENARIOS:
            for mode in MODES:
                rec = seed_data.get(sc, {}).get(mode)
                if not isinstance(rec, dict):
                    incomplete_cells.append(f"seed={seed}/{sc}/{mode}:missing")
                    continue
                missing_metrics = sorted(required_metrics - set(rec))
                if missing_metrics:
                    incomplete_cells.append(
                        f"seed={seed}/{sc}/{mode}:missing={missing_metrics}"
                    )
    if incomplete_cells:
        preview = "; ".join(incomplete_cells[:12])
        suffix = "" if len(incomplete_cells) <= 12 else (
            f"; ... {len(incomplete_cells) - 12} more"
        )
        raise RuntimeError(
            "Incomplete benchmark scenario/mode/metric panel: "
            f"{preview}{suffix}"
        )

    # Build summary. Iterate over core METRICS plus any EXTRA_METRICS that
    # the per-seed JSON carries so operational / regulatory CVR and honor
    # rate also get bootstrap CIs instead of being dropped.
    summary = {}
    all_metrics = tuple(METRICS) + tuple(EXTRA_METRICS)
    for sc in SCENARIOS:
        summary[sc] = {}
        for mode in MODES:
            summary[sc][mode] = {}
            for met in all_metrics:
                vals = [
                    all_data[s][sc][mode][met]
                    for s in all_data
                    if mode in all_data[s].get(sc, {})
                    and met in all_data[s][sc][mode]
                    and all_data[s][sc][mode][met] is not None
                ]
                if not vals:
                    continue
                # Key the resampling stream to the actual summary cell.  Using
                # the default ``("global",)`` key here made every
                # scenario/mode/metric CI reuse the same bootstrap index
                # sequence, contrary to the declared cell-specific design.
                ci_before = dict(_BCA_STATS)
                lo, hi = bootstrap_ci(vals, cell_key=(sc, mode, met))
                summary[sc][mode][met] = {
                    "mean": float(np.mean(vals)),
                    "std": (
                        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    ),
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_method": _ci_method_since(ci_before, lo, hi),
                    "n_seeds": len(vals),
                }

    # Build significance with two-level multiplicity control.
    # Pass 1: collect raw p-values for every (scenario, baseline, metric) cell.
    significance: dict = {}
    per_scenario_pvals: dict[str, dict[str, float]] = {sc: {} for sc in SCENARIOS}
    primary_h1_pvals: dict[str, float] = {}
    pinn_ablation_pvals: dict[str, float] = {}
    h2_directional_pvals: dict[str, float] = {}
    h2_synergy_pvals: dict[str, float] = {}

    # Pairing scope.
    # ----------------------------------------------------------------
    # Every (mode, seed) within a scenario-seed-episode cell shares the same
    # initial conditions and mode-independent exogenous stream keys. Draws are
    # source- and counter-keyed, so conditional branches do not shift another
    # arm's later exogenous draws. Actions may still create different endogenous
    # trajectories. Seed is the matched inferential unit.
    #
    # The Wilcoxon signed-rank test pairs by seed and tests whether the
    # within-seed difference is centered away from zero. Pairing is justified
    # by matched scenario-seed cells and their common exogenous streams.
    # Cohen's d_z and the pooled standardized difference are both reported;
    # neither is tuned through added noise.
    _PAIRED_BASELINES = {
        "no_context", "no_pinn", "mcp_only", "pirag_only", "no_slca",
        "static", "hybrid_rl",
    }

    for sc in SCENARIOS:
        significance[sc] = {}
        for baseline in BASELINES:
            seeds_paired = sorted(
                s for s in all_data
                if "agribrain" in all_data[s].get(sc, {})
                and baseline in all_data[s].get(sc, {})
            )
            if not seeds_paired:
                continue
            is_paired = baseline in _PAIRED_BASELINES
            comp: dict = {"is_paired_design": is_paired,
                          "test_type": "wilcoxon_signed_rank" if is_paired
                                       else "mann_whitney_u",
                          "effect_size_primary": "cohens_dz" if is_paired
                                                 else "cohens_d_pooled"}
            for met in METRICS:
                a = [all_data[s][sc]["agribrain"][met] for s in seeds_paired]
                b = [all_data[s][sc][baseline][met] for s in seeds_paired]
                cell_key = (sc, baseline, met)
                # Test selection: paired Wilcoxon when seeds match;
                # unpaired Mann-Whitney when they don't. The legacy
                # paired_permutation result is also kept as a consistency
                # cross-check field.
                if is_paired:
                    p_value = wilcoxon_signed_rank_pvalue(a, b, cell_key=cell_key)
                else:
                    p_value = mann_whitney_pvalue(a, b, cell_key=cell_key)
                p_perm_legacy = paired_permutation_pvalue(a, b, cell_key=cell_key)
                dz = cohens_dz(a, b) if is_paired else float("nan")
                d_pooled = cohens_d_pooled(a, b)
                hg = hedges_g(a, b, paired=is_paired)
                diff_ci_before = dict(_BCA_STATS)
                lo_diff, hi_diff = bootstrap_mean_diff_ci(
                    a, b, paired=is_paired, cell_key=cell_key
                )
                diff_ci_method = _ci_method_since(
                    diff_ci_before, lo_diff, hi_diff,
                )
                pooled_ci_before = dict(_BCA_STATS)
                pooled_lo, pooled_hi = bootstrap_effect_size_ci(
                    a, b, paired=is_paired, cell_key=cell_key,
                    statistic="pooled",
                )
                pooled_ci_method = _ci_method_since(
                    pooled_ci_before, pooled_lo, pooled_hi,
                )
                if is_paired:
                    dz_ci_before = dict(_BCA_STATS)
                    dz_lo, dz_hi = bootstrap_effect_size_ci(
                        a, b, paired=True, cell_key=cell_key,
                        statistic="dz",
                    )
                    dz_ci_method = _ci_method_since(
                        dz_ci_before, dz_lo, dz_hi,
                    )
                    paired_diff = np.array(a, dtype=float) - np.array(b, dtype=float)
                    within_pair_sd = float(np.std(paired_diff, ddof=1)) if len(paired_diff) >= 2 else 0.0
                else:
                    dz_lo, dz_hi = float("nan"), float("nan")
                    dz_ci_method = "not_applicable"
                    within_pair_sd = float("nan")
                primary_d = dz if is_paired else d_pooled
                primary_lo = dz_lo if is_paired else pooled_lo
                primary_hi = dz_hi if is_paired else pooled_hi
                mean_diff = float(np.mean(a) - np.mean(b))
                p_directional = None
                directional_cell_key = None
                if met == "ari" and baseline in {
                    "no_context", "no_pinn", "mcp_only", "pirag_only",
                }:
                    directional_scope = (
                        "pinn_ablation_directional"
                        if baseline == "no_pinn" else "h2_directional"
                    )
                    directional_cell_key = (*cell_key, directional_scope)
                    p_directional = wilcoxon_signed_rank_pvalue(
                        a,
                        b,
                        cell_key=directional_cell_key,
                        alternative="greater",
                    )
                comp[met] = {
                    "test_type_actual": (
                        "sign_flip_permutation_fallback"
                        if tuple(cell_key) in _WILCOXON_FALLBACK_CELLS
                        else ("wilcoxon_signed_rank" if is_paired else "mann_whitney_u")
                    ),
                    "p_value": p_value,
                    "p_value_directional_greater": p_directional,
                    "directional_test_type_actual": (
                        "sign_flip_permutation_fallback"
                        if directional_cell_key is not None
                        and tuple(directional_cell_key)
                        in _WILCOXON_FALLBACK_CELLS
                        else "wilcoxon_signed_rank"
                        if directional_cell_key is not None
                        else None
                    ),
                    "p_value_legacy_signflip": p_perm_legacy,
                    # Legacy alias follows the design-appropriate primary effect.
                    "cohens_d": primary_d,
                    "cohens_dz": dz,
                    "cohens_dz_undefined_zero_variance": bool(
                        is_paired and dz is None
                    ),
                    "cohens_d_pooled": d_pooled,
                    "hedges_g": hg,
                    "effect_size_ci_low": primary_lo,
                    "effect_size_ci_high": primary_hi,
                    "effect_size_ci_method": (
                        dz_ci_method if is_paired else pooled_ci_method
                    ),
                    "cohens_dz_ci_low": dz_lo,
                    "cohens_dz_ci_high": dz_hi,
                    "cohens_dz_ci_method": dz_ci_method,
                    "cohens_d_pooled_ci_low": pooled_lo,
                    "cohens_d_pooled_ci_high": pooled_hi,
                    "cohens_d_pooled_ci_method": pooled_ci_method,
                    "within_pair_sd": within_pair_sd,
                    "design_tax_note": (
                        "d_z uses the seed-level paired-difference SD under "
                        "the matched scenario-seed design; pooled d is also reported."
                    ) if is_paired else "unpaired comparison",
                    "mean_diff": mean_diff,
                    "mean_diff_ci_low": lo_diff,
                    "mean_diff_ci_high": hi_diff,
                    "mean_diff_ci_method": diff_ci_method,
                    "n_seeds": len(seeds_paired),
                }
                per_scenario_pvals[sc][f"{baseline}:{met}"] = p_value
                if baseline == "no_context" and met == "ari":
                    primary_h1_pvals[sc] = float(p_directional)
                if baseline == "no_pinn" and met == "ari":
                    pinn_ablation_pvals[sc] = float(p_directional)
                if (
                    p_directional is not None
                    and baseline in {"mcp_only", "pirag_only"}
                ):
                    h2_directional_pvals[
                        f"{sc}:agribrain_vs_{baseline}"
                    ] = float(p_directional)
            significance[sc][f"agribrain_vs_{baseline}"] = comp

    # Pass 1.5: Channel-decomposition family.
    # Direct tests for the C4 claim that each context channel (MCP,
    # piR) contributes to quality improvements. The agribrain_vs_X
    # family alone leaves C4 inferable only by transitivity; this loop
    # adds the single-channel-vs-no_context contrasts on the same
    # paired-seed design.
    #
    # Both modes in each pair occupy the same scenario-seed-episode cell and
    # share source/counter-keyed exogenous draws. Their actions may create
    # different endogenous states. Cohen's d_z is the primary matched-design
    # effect size; pooled d is retained as a secondary standardization.
    channel_pvals: dict[str, float] = {}
    for sc in SCENARIOS:
        for a_mode, b_mode in _CHANNEL_DECOMPOSITION_PAIRS:
            seeds_paired = sorted(
                s for s in all_data
                if a_mode in all_data[s].get(sc, {})
                and b_mode in all_data[s].get(sc, {})
            )
            if not seeds_paired:
                continue
            comp: dict = {
                "is_paired_design": True,
                "test_type": "wilcoxon_signed_rank",
                "effect_size_primary": "cohens_dz",
                "_family": "channel_decomposition",
            }
            for met in METRICS:
                a = [all_data[s][sc][a_mode][met] for s in seeds_paired]
                b = [all_data[s][sc][b_mode][met] for s in seeds_paired]
                cell_key = (sc, f"{a_mode}_vs_{b_mode}", met)
                p_value = wilcoxon_signed_rank_pvalue(a, b, cell_key=cell_key)
                p_directional = (
                    wilcoxon_signed_rank_pvalue(
                        a,
                        b,
                        cell_key=(*cell_key, "h2_directional"),
                        alternative="greater",
                    )
                    if met == "ari" else None
                )
                p_perm_legacy = paired_permutation_pvalue(a, b, cell_key=cell_key)
                dz = cohens_dz(a, b)
                d_pooled = cohens_d_pooled(a, b)
                hg = hedges_g(a, b, paired=True)
                diff_ci_before = dict(_BCA_STATS)
                lo_diff, hi_diff = bootstrap_mean_diff_ci(
                    a, b, paired=True, cell_key=cell_key
                )
                diff_ci_method = _ci_method_since(
                    diff_ci_before, lo_diff, hi_diff,
                )
                pooled_ci_before = dict(_BCA_STATS)
                pooled_lo, pooled_hi = bootstrap_effect_size_ci(
                    a, b, paired=True, cell_key=cell_key,
                    statistic="pooled",
                )
                pooled_ci_method = _ci_method_since(
                    pooled_ci_before, pooled_lo, pooled_hi,
                )
                dz_ci_before = dict(_BCA_STATS)
                dz_lo, dz_hi = bootstrap_effect_size_ci(
                    a, b, paired=True, cell_key=cell_key,
                    statistic="dz",
                )
                dz_ci_method = _ci_method_since(
                    dz_ci_before, dz_lo, dz_hi,
                )
                paired_diff = np.array(a, dtype=float) - np.array(b, dtype=float)
                within_pair_sd = (
                    float(np.std(paired_diff, ddof=1))
                    if len(paired_diff) >= 2 else 0.0
                )
                mean_diff = float(np.mean(a) - np.mean(b))
                comp[met] = {
                    "test_type_actual": (
                        "sign_flip_permutation_fallback"
                        if tuple(cell_key) in _WILCOXON_FALLBACK_CELLS
                        else "wilcoxon_signed_rank"
                    ),
                    "p_value": p_value,
                    "p_value_directional_greater": p_directional,
                    "directional_test_type_actual": (
                        "sign_flip_permutation_fallback"
                        if p_directional is not None
                        and (*cell_key, "h2_directional")
                        in _WILCOXON_FALLBACK_CELLS
                        else "wilcoxon_signed_rank"
                        if p_directional is not None
                        else None
                    ),
                    "p_value_legacy_signflip": p_perm_legacy,
                    "cohens_d": dz,
                    "cohens_dz": dz,
                    "cohens_dz_undefined_zero_variance": bool(dz is None),
                    "cohens_d_pooled": d_pooled,
                    "hedges_g": hg,
                    "effect_size_ci_low": dz_lo,
                    "effect_size_ci_high": dz_hi,
                    "effect_size_ci_method": dz_ci_method,
                    "cohens_dz_ci_low": dz_lo,
                    "cohens_dz_ci_high": dz_hi,
                    "cohens_dz_ci_method": dz_ci_method,
                    "cohens_d_pooled_ci_low": pooled_lo,
                    "cohens_d_pooled_ci_high": pooled_hi,
                    "cohens_d_pooled_ci_method": pooled_ci_method,
                    "within_pair_sd": within_pair_sd,
                    "design_tax_note": (
                        "d_z uses the seed-level paired-difference SD under "
                        "the matched scenario-seed design; pooled d is also reported."
                    ),
                    "mean_diff": mean_diff,
                    "mean_diff_ci_low": lo_diff,
                    "mean_diff_ci_high": hi_diff,
                    "mean_diff_ci_method": diff_ci_method,
                    "n_seeds": len(seeds_paired),
                }
                # Per-scenario secondary FDR participation: the channel
                # contrasts are within-scenario secondary endpoints
                # exactly like the agribrain_vs_X cells, so they go
                # through the same per-scenario BY-FDR / BH-FDR
                # correction below. Keying convention prefixes with
                # the comparison name to avoid collision with the
                # ``baseline:metric`` keys that agribrain_vs_X uses.
                per_scenario_pvals[sc][f"{a_mode}_vs_{b_mode}:{met}"] = p_value
                # Channel-decomposition Holm family covers the ARI
                # endpoint only (matching the primary H1 family
                # convention of one metric per scenario per contrast).
                # Other metrics participate via per-scenario FDR.
                if met == "ari":
                    channel_pvals[f"{sc}:{a_mode}_vs_{b_mode}"] = p_value
                    h2_directional_pvals[
                        f"{sc}:{a_mode}_vs_{b_mode}"
                    ] = float(p_directional)
            significance[sc][f"{a_mode}_vs_{b_mode}"] = comp

    # Exploratory superadditivity diagnostic.  Joint dominance (Full exceeds
    # both single-channel arms) does not by itself establish a positive
    # interaction.  The seed-level interaction below is therefore reported
    # separately and its observed sign is never converted into a required
    # validator outcome.
    for sc in SCENARIOS:
        seeds_paired = sorted(
            s for s in all_data
            if all(
                mode in all_data[s].get(sc, {})
                for mode in ("agribrain", "mcp_only", "pirag_only", "no_context")
            )
        )
        if not seeds_paired:
            continue
        interactions = h2_synergy_interaction(
            [all_data[s][sc]["agribrain"]["ari"] for s in seeds_paired],
            [all_data[s][sc]["mcp_only"]["ari"] for s in seeds_paired],
            [all_data[s][sc]["pirag_only"]["ari"] for s in seeds_paired],
            [all_data[s][sc]["no_context"]["ari"] for s in seeds_paired],
        )
        zeros = np.zeros_like(interactions)
        cell_key = (sc, "h2_synergy_interaction", "ari")
        p_directional = wilcoxon_signed_rank_pvalue(
            interactions,
            zeros,
            cell_key=cell_key,
            alternative="greater",
        )
        ci_before = dict(_BCA_STATS)
        ci_low, ci_high = bootstrap_ci(
            interactions,
            cell_key=cell_key,
        )
        ci_method = _ci_method_since(ci_before, ci_low, ci_high)
        interaction_sd = float(np.std(interactions, ddof=1))
        interaction_mean = float(np.mean(interactions))
        interaction_dz = (
            interaction_mean / interaction_sd if interaction_sd > 0.0 else None
        )
        significance[sc]["h2_synergy_interaction"] = {
            "is_paired_design": True,
            "test_type": "wilcoxon_signed_rank_greater",
            "effect_size_primary": "cohens_dz",
            "exploratory": True,
            "interpretation": (
                "positive values indicate superadditivity: Full - MCP-only - "
                "Retrieval-only + No-external-context"
            ),
            "ari": {
                "p_value_directional_greater": float(p_directional),
                "mean_interaction": interaction_mean,
                "mean_interaction_ci_low": float(ci_low),
                "mean_interaction_ci_high": float(ci_high),
                "mean_interaction_ci_method": ci_method,
                "cohens_dz": interaction_dz,
                "within_pair_sd": interaction_sd,
                "n_seeds": len(seeds_paired),
                "positive_point_estimate": bool(interaction_mean > 0.0),
                "ci_excludes_zero_in_positive_direction": bool(ci_low > 0.0),
            },
        }
        h2_synergy_pvals[sc] = float(p_directional)

    # Pass 2a: Holm-Bonferroni across the primary H1 family (5 scenarios).
    # The primary family is fixed in docs/STATISTICAL_METHODS.md: one
    # contrast (agribrain vs no_context) on one metric (ARI) per
    # scenario, m=5.
    primary_h1_holm = holm_bonferroni(primary_h1_pvals)
    expected_pinn_ablation_keys = set(SCENARIOS)
    if set(pinn_ablation_pvals) != expected_pinn_ablation_keys:
        raise RuntimeError(
            "Incomplete PINN-ablation directional family: "
            f"missing={sorted(expected_pinn_ablation_keys - set(pinn_ablation_pvals))}, "
            f"unexpected={sorted(set(pinn_ablation_pvals) - expected_pinn_ablation_keys)}"
        )
    pinn_ablation_holm = holm_bonferroni(pinn_ablation_pvals)
    expected_h2_keys = {
        f"{sc}:{a_mode}_vs_{b_mode}"
        for sc in SCENARIOS
        for a_mode, b_mode in _H2_DIRECTIONAL_PAIRS
    }
    if set(h2_directional_pvals) != expected_h2_keys:
        raise RuntimeError(
            "Incomplete H2 directional family: "
            f"missing={sorted(expected_h2_keys - set(h2_directional_pvals))}, "
            f"unexpected={sorted(set(h2_directional_pvals) - expected_h2_keys)}"
        )
    h2_directional_holm = holm_bonferroni(h2_directional_pvals)
    h2_synergy_holm = holm_bonferroni(h2_synergy_pvals)

    # Auxiliary 1: Holm across an extended paired-baseline grid (3
    # endpoints x 7 paired baselines x 5 scenarios = 105 tests after
    # the post-2026-04 pairing-extension fix). Lets headline robustness
    # be judged under a moderately-wider family.
    extended_pvals: dict[str, float] = {}
    for sc in SCENARIOS:
        for baseline in _PAIRED_BASELINES:
            comp = significance.get(sc, {}).get(f"agribrain_vs_{baseline}")
            if comp is None:
                continue
            for met in ("ari", "rle", "slca"):
                rec = comp.get(met)
                if rec is None:
                    continue
                extended_pvals[f"{sc}:{baseline}:{met}"] = float(rec["p_value"])
    extended_holm = holm_bonferroni(extended_pvals) if extended_pvals else {}

    # Auxiliary 2: Holm across the FULL grid of every (scenario,
    # baseline, metric) cell that has a p-value. m = scenarios *
    # (seven AGRI-BRAIN baselines plus two direct channel contrasts) *
    # metrics (5 * 9 * 6 = 270). This is the
    # strictest end-to-end FWER control: any p_value_adj_holm_full
    # below alpha rejects the null at family-wise alpha across
    # everything reported in benchmark_significance.json. Reviewers
    # who want to read significance off the full table without
    # restricting to the primary H1 family can use this column.
    # Per docs/STATISTICAL_METHODS.md the canonical p_value_adj is
    # still BY-FDR within scenario for secondary endpoints (less
    # conservative under arbitrary dependence), with this full-grid
    # Holm column reported alongside as the FWER-strict alternative.
    full_grid_pvals: dict[str, float] = {}
    for sc in SCENARIOS:
        for baseline in BASELINES:
            comp = significance.get(sc, {}).get(f"agribrain_vs_{baseline}")
            if comp is None:
                continue
            for met in METRICS:
                rec = comp.get(met)
                if rec is None:
                    continue
                full_grid_pvals[f"{sc}:{baseline}:{met}"] = float(rec["p_value"])
        # Channel-decomposition contrasts also participate in the
        # full-grid Holm. Same shape as agribrain_vs_X cells, just
        # different comparison name.
        for a_mode, b_mode in _CHANNEL_DECOMPOSITION_PAIRS:
            comp = significance.get(sc, {}).get(f"{a_mode}_vs_{b_mode}")
            if comp is None:
                continue
            for met in METRICS:
                rec = comp.get(met)
                if rec is None:
                    continue
                full_grid_pvals[f"{sc}:{a_mode}_vs_{b_mode}:{met}"] = float(rec["p_value"])
    full_grid_holm = (holm_bonferroni(full_grid_pvals)
                       if full_grid_pvals else {})

    # Auxiliary 3: Channel-decomposition Holm-Bonferroni.
    # 2 contrasts (mcp_only_vs_no_context, pirag_only_vs_no_context) x
    # 5 scenarios = 10 tests on ARI. Closes the C4 paper-claim gap
    # (each context channel contributes to quality improvements) by
    # applying a family-honest multiple-comparison correction within
    # the channel-decomposition family alone — separate from the
    # primary H1 family (5 tests) and the extended/full grids. The
    # ``p_value_adj_holm_channel`` retains this historical subset correction
    # for audit. Pass 4 below overwrites generic ``p_value_adj`` with the
    # prespecified 20-test H2 correction on all four H2 contrasts.
    channel_holm = holm_bonferroni(channel_pvals) if channel_pvals else {}

    # Pass 2b: BH-FDR (PRDS-assuming) AND BY-FDR (arbitrary-dependence)
    # within each scenario across all (baseline, metric) pairs. Reporting
    # both surfaces the conservative bound (BY) when within-scenario
    # metric correlations have mixed signs.
    per_scenario_bh: dict[str, dict[str, float]] = {
        sc: benjamini_hochberg(per_scenario_pvals[sc]) for sc in SCENARIOS
    }
    per_scenario_by: dict[str, dict[str, float]] = {
        sc: benjamini_yekutieli(per_scenario_pvals[sc]) for sc in SCENARIOS
    }

    # Pass 3: write adjusted p-values back into each comparison record. Each
    # cell gets both fields (p_value_adj_bh and, where applicable,
    # p_value_adj_holm) plus a canonical p_value_adj and correction_method.
    for sc in SCENARIOS:
        bh_map = per_scenario_bh.get(sc, {})
        by_map = per_scenario_by.get(sc, {})
        for baseline in BASELINES:
            comp_key = f"agribrain_vs_{baseline}"
            comp = significance[sc].get(comp_key)
            if comp is None:
                continue
            for met in METRICS:
                rec = comp.get(met)
                if rec is None:
                    continue
                key = f"{baseline}:{met}"
                p_bh = float(bh_map.get(key, rec["p_value"]))
                p_by = float(by_map.get(key, rec["p_value"]))
                rec["p_value_adj_bh"] = p_bh
                rec["p_value_adj_by"] = p_by
                # Auxiliary extended-Holm field on every record where
                # the contrast is part of the extended grid (3 endpoints
                # x 7 paired baselines x 5 scenarios = 105 tests).
                ext_key = f"{sc}:{baseline}:{met}"
                if ext_key in extended_holm:
                    rec["p_value_adj_holm_extended"] = float(extended_holm[ext_key])
                # End-to-end Holm across the FULL grid (every scenario
                # x baseline x metric cell with a p-value). This is the
                # strictest FWER control: a cell can be read off as
                # significant at family-wise alpha across the entire
                # significance table without restricting to the primary
                # H1 family. Always populated; the canonical
                # p_value_adj for secondary endpoints continues to be
                # BY-FDR within scenario per STATISTICAL_METHODS.md
                # because it is less conservative under arbitrary
                # dependence and matches the published reporting
                # convention; full-grid Holm is the FWER-strict
                # alternative.
                if ext_key in full_grid_holm:
                    rec["p_value_adj_holm_full"] = float(full_grid_holm[ext_key])
                # M3/M4 descriptive-only flags. RLE on static is
                # structurally zero (static always picks cold_chain),
                # so the RLE contrast against static is descriptive
                # only — it measures the policy ceiling, not a
                # comparable RLE. regulatory_violation_rate USED to be
                # structurally zero for non-MCP baselines (static,
                # hybrid_rl, no_slca) because they didn't
                # invoke the compliance tool, but the post-2026-04
                # deep-audit fix routes check_compliance uniformly on
                # every step regardless of mode (commit 1d9caf0), so
                # compliance_violation_rate / regulatory_violation_rate
                # are now directly comparable across every mode and the
                # descriptive_only flag for non-MCP baselines is
                # retired. Only the static-RLE flag remains because
                # that one is genuinely structural (static never
                # selects a recovery action; the metric scores the
                # *action*, not the environment).
                if met == "rle" and baseline == "static":
                    rec["descriptive_only"] = True
                    rec["descriptive_only_reason"] = (
                        "static RLE is structurally 0 (always cold_chain); "
                        "the contrast measures policy ceiling, not RLE"
                    )
                if baseline == "no_context" and met == "ari":
                    p_holm = float(primary_h1_holm.get(sc, rec["p_value"]))
                    rec["p_value_adj_holm"] = p_holm
                    rec["p_value_adj"] = p_holm
                    rec["correction_method"] = "holm_bonferroni_across_scenarios"
                    rec["h1_test"] = "paired_wilcoxon_signed_rank_greater"
                    rec["h1_raw_p_value_directional_greater"] = float(
                        rec["p_value_directional_greater"]
                    )
                    rec["canonical_raw_p_value_field"] = (
                        "h1_raw_p_value_directional_greater"
                    )
                    rec["h1_family_size"] = len(SCENARIOS)
                    rec["h1_positive_effect_supported"] = bool(
                        p_holm < 0.05 and rec["mean_diff"] > 0.0
                    )
                    rec["h1_practical_margin"] = H1_PRACTICAL_MARGIN
                    rec["h1_practical_margin_supported"] = bool(
                        rec["mean_diff_ci_method"] == "BCa"
                        and rec["mean_diff_ci_low"] > H1_PRACTICAL_MARGIN
                    )
                    rec["h1_practical_claim_rule"] = (
                        "optional claim; the 95% paired BCa mean-difference CI "
                        "lower bound must exceed 0.005 ARI; a labelled CI "
                        "fallback cannot support this prespecified claim"
                    )
                elif baseline == "no_pinn" and met == "ari":
                    p_holm = float(pinn_ablation_holm.get(sc, rec["p_value"]))
                    rec["pinn_ablation_test"] = (
                        "paired_wilcoxon_signed_rank_greater"
                    )
                    rec["pinn_ablation_raw_p_value_directional_greater"] = float(
                        rec["p_value_directional_greater"]
                    )
                    rec["p_value_adj_holm_pinn_ablation"] = p_holm
                    rec["p_value_adj"] = p_holm
                    rec["correction_method"] = (
                        "holm_bonferroni_across_pinn_ablation_scenarios"
                    )
                    rec["pinn_ablation_family_size"] = len(SCENARIOS)
                    rec["pinn_ablation_positive_effect_supported"] = bool(
                        p_holm < 0.05 and rec["mean_diff"] > 0.0
                    )
                else:
                    # Canonical p_value_adj on secondary endpoints uses
                    # the more conservative BY-FDR (valid under arbitrary
                    # dependence). BH retained as a less-conservative
                    # comparator under the PRDS assumption.
                    rec["p_value_adj"] = p_by
                    rec["correction_method"] = "by_fdr_within_scenario"

        # Pass 3b: write adjusted p-values into the channel-decomposition
        # records (mcp_only_vs_no_context, pirag_only_vs_no_context).
        # Each cell carries:
        #   p_value_adj_bh / p_value_adj_by  — per-scenario FDR (matches
        #                                      the agribrain_vs_X cells'
        #                                      treatment of secondary
        #                                      endpoints; the same
        #                                      per_scenario_pvals[sc]
        #                                      dictionary fed BH/BY)
        #   p_value_adj_holm_full            — full-grid Holm (FWER-strict
        #                                      across the entire significance
        #                                      table)
        #   p_value_adj_holm_channel         — Holm within the
        #                                      channel-decomposition family
        #                                      of 10 tests (ARI only)
        #   p_value_adj                      — temporary subset value on ARI;
        #                                      Pass 4 replaces it with the
        #                                      canonical m=20 H2 value
        #   correction_method                — names which correction was
        #                                      applied
        for a_mode, b_mode in _CHANNEL_DECOMPOSITION_PAIRS:
            comp_key = f"{a_mode}_vs_{b_mode}"
            comp = significance[sc].get(comp_key)
            if comp is None:
                continue
            for met in METRICS:
                rec = comp.get(met)
                if rec is None:
                    continue
                key = f"{a_mode}_vs_{b_mode}:{met}"
                p_bh = float(bh_map.get(key, rec["p_value"]))
                p_by = float(by_map.get(key, rec["p_value"]))
                rec["p_value_adj_bh"] = p_bh
                rec["p_value_adj_by"] = p_by
                full_key = f"{sc}:{a_mode}_vs_{b_mode}:{met}"
                if full_key in full_grid_holm:
                    rec["p_value_adj_holm_full"] = float(full_grid_holm[full_key])
                if met == "ari":
                    ch_key = f"{sc}:{a_mode}_vs_{b_mode}"
                    p_holm_channel = float(
                        channel_holm.get(ch_key, rec["p_value"])
                    )
                    rec["p_value_adj_holm_channel"] = p_holm_channel
                    rec["p_value_adj"] = p_holm_channel
                    rec["correction_method"] = (
                        "holm_bonferroni_channel_decomposition"
                    )
                else:
                    rec["p_value_adj"] = p_by
                    rec["correction_method"] = "by_fdr_within_scenario"

    # Pass 4: apply the prespecified 20-test directional H2 family.  Keep the
    # existing two-sided/BY fields for backward compatibility, but make the H2
    # inferential result explicit rather than asking readers to infer it from
    # unrelated columns.
    h2_cell_support: dict[str, bool] = {}
    for sc in SCENARIOS:
        for a_mode, b_mode in _H2_DIRECTIONAL_PAIRS:
            comp_key = f"{a_mode}_vs_{b_mode}"
            rec = significance[sc][comp_key]["ari"]
            family_key = f"{sc}:{comp_key}"
            p_adj = float(h2_directional_holm[family_key])
            supported = bool(
                rec["mean_diff"] > 0.0
                and rec["p_value_directional_greater"] < 0.05
                and p_adj < 0.05
            )
            rec["p_value_adj_holm_h2_directional"] = p_adj
            # Make generic table/export consumers use the declared H2 family,
            # not the historical 10-test subset or a per-scenario FDR field.
            rec["p_value_adj"] = p_adj
            rec["correction_method"] = (
                "holm_bonferroni_h2_directional_20"
            )
            rec["canonical_raw_p_value_field"] = (
                "p_value_directional_greater"
            )
            rec["h2_family_size"] = len(expected_h2_keys)
            rec["h2_direction"] = f"{a_mode} > {b_mode}"
            rec["h2_cell_supported"] = supported
            rec["h2_correction_method"] = (
                "holm_bonferroni_across_20_directional_ari_contrasts"
            )
            rec["h2_test_type_actual"] = (
                rec["directional_test_type_actual"]
            )
            h2_cell_support[family_key] = supported

        synergy = significance[sc].get("h2_synergy_interaction", {}).get("ari")
        if synergy is not None:
            synergy_adj = float(h2_synergy_holm[sc])
            synergy["p_value_adj_holm_exploratory"] = synergy_adj
            synergy["exploratory_superadditivity_supported"] = bool(
                synergy["mean_interaction"] > 0.0
                and synergy["p_value_directional_greater"] < 0.05
                and synergy_adj < 0.05
            )

    h2_supported_all_cells = bool(h2_cell_support) and all(
        h2_cell_support.values()
    )
    h1_cell_support = {
        sc: bool(
            float(significance[sc]["agribrain_vs_no_context"]["ari"]["mean_diff"])
            > 0.0
            and float(primary_h1_holm[sc]) < 0.05
        )
        for sc in SCENARIOS
    }
    h1_supported_all_cells = bool(h1_cell_support) and all(
        h1_cell_support.values()
    )

    # Save
    bca_stats = _bca_fallback_stats_snapshot()
    if os.environ.get("STRICT_VALIDATION", "").strip() == "1":
        if _WILCOXON_FALLBACK_CELLS:
            raise RuntimeError(
                "STRICT_VALIDATION forbids a change from the declared "
                "Wilcoxon test to sign-flip fallback; affected cells: "
                f"{sorted(_WILCOXON_FALLBACK_CELLS)!r}"
            )
        if bca_stats["fallback_scipy_unavailable"]:
            raise RuntimeError(
                "STRICT_VALIDATION requires SciPy for BCa intervals; "
                "one or more cells used a percentile fallback because SciPy "
                "was unavailable"
            )
    # Provenance pin (post-2026-05): record the seed count and the
    # source-code commit alongside the bootstrap parameters. A
    # reviewer reading benchmark_summary.json should see at a glance
    # which version of the simulator produced which numbers, without
    # cross-referencing artifact_manifest.json.
    #
    # Resolution order (each tier falls through to the next on failure):
    #   1. AGRIBRAIN_GIT_COMMIT env var (HPC pipelines export this via
    #      sbatch --export so the stamp survives slurm contexts where
    #      git is not in PATH).
    #   2. ``git rev-parse HEAD`` subprocess (local-dev path; requires
    #      the ``git`` binary on PATH).
    #   3. Direct read of ``.git/HEAD`` (slurm-compute-node path; works
    #      even when ``git`` is not installed on the worker, as long
    #      as the repo's ``.git`` directory is on the shared filesystem
    #      that the worker can see). Handles both detached-HEAD and
    #      ref-based forms.
    #   4. None (last resort; verify_manifest.py --strict-commit on
    #      the artifact manifest still gates the artifact set, so a
    #      None here is informational rather than a hard failure).
    #
    # The post-HPC RUN_TAG 485c769_20260505_0349 incident: a manual
    # ``sbatch hpc/hpc_aggregate.sh`` resubmission bypassed the env
    # export from ``hpc_run.sh`` and the slurm worker's PATH didn't
    # include ``git`` -- tiers 1 and 2 both returned None, ``_meta.git_commit``
    # ended up null. Tier 3 added so this fallback chain reaches a
    # real SHA on every realistic HPC + local invocation path.
    import os as _os_meta
    import subprocess as _subprocess_meta
    _git_commit_meta: str | None = (
        _os_meta.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip() or None
    )
    _git_root_meta_path = repo_root
    if _git_commit_meta is None:
        try:
            _git_commit_meta = _subprocess_meta.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_git_root_meta_path),
                stderr=_subprocess_meta.PIPE,
            ).decode("utf-8").strip() or None
        except Exception:
            _git_commit_meta = None
    if _git_commit_meta is None:
        # Tier 3: read .git/HEAD directly. Handles slurm workers
        # without git on PATH.
        try:
            _head_path = _git_root_meta_path / ".git" / "HEAD"
            _head_text = _head_path.read_text(encoding="utf-8").strip()
            if _head_text.startswith("ref: "):
                _ref = _head_text[5:].strip()  # e.g. "refs/heads/main"
                _ref_path = _git_root_meta_path / ".git" / _ref
                if _ref_path.exists():
                    _sha = _ref_path.read_text(encoding="utf-8").strip()
                    if len(_sha) == 40 and all(
                        c in "0123456789abcdef" for c in _sha
                    ):
                        _git_commit_meta = _sha
                else:
                    # Packed refs fallback: ref might be in
                    # ``.git/packed-refs`` instead of an unpacked file.
                    _packed = _git_root_meta_path / ".git" / "packed-refs"
                    if _packed.exists():
                        for _line in _packed.read_text(encoding="utf-8").splitlines():
                            if _line.endswith(_ref) and len(_line) >= 41:
                                _candidate = _line.split(" ", 1)[0].strip()
                                if len(_candidate) == 40:
                                    _git_commit_meta = _candidate
                                    break
            elif len(_head_text) == 40 and all(
                c in "0123456789abcdef" for c in _head_text
            ):
                # Detached HEAD: HEAD itself contains the SHA.
                _git_commit_meta = _head_text
        except Exception:
            _git_commit_meta = None

    # Fresh publication evidence remains single-provenance.  The only
    # exception is an independently authorized deterministic recovery over
    # byte-bound preserved simulation outputs.  In that narrow path the two
    # identities are kept separate instead of relabelling the simulation as a
    # run of the repaired publication code.
    if recovery_provenance is None:
        _analysis_code_commit = _git_commit_meta
        _dual_provenance = False
        _recovery_authorization = None
    else:
        _git_commit_meta = str(
            recovery_provenance["simulation_source_commit"]
        )
        _analysis_code_commit = str(
            recovery_provenance["publication_code_commit"]
        )
        _dual_provenance = True
        _recovery_authorization = recovery_provenance[
            "recovery_authorization"
        ]

    _resampling_meta = _resampling_identity(list(all_data))
    episode_budget_by_mode = {
        mode: int(_SIM_MULTI_EPISODE_MODES.get(mode, 1)) for mode in MODES
    }
    episode_accounting = build_episode_accounting(
        scenarios=SCENARIOS,
        configured_modes=MODES,
        episode_budget_by_mode=episode_budget_by_mode,
        n_seeds=len(all_data),
        primary_modes=PRIMARY_PUBLICATION_MODES,
    )
    payload_summary = {
        "_meta": {
            "n_boot": 10_000,
            "n_perm": 10_000,
            "n_perm_scope": (
                "legacy two-sided sign-flip audit values and emergency "
                "fallbacks only; confirmatory H1/H2 use directional "
                "Wilcoxon signed-rank tests"
            ),
            "legacy_sign_flip_resamples": 10_000,
            "confirmatory_test": "directional_wilcoxon_signed_rank",
            "bootstrap_alpha": 0.05,
            "std_ddof": 1,
            "seeds_loaded": sorted(all_data),
            "n_seeds": len(all_data),
            "git_commit": _git_commit_meta,
            "source_commit": _git_commit_meta,
            "simulation_source_commit": _git_commit_meta,
            "analysis_code_commit": _analysis_code_commit,
            "dual_provenance": _dual_provenance,
            "recovery_authorization": _recovery_authorization,
            "run_tag": _os_meta.environ.get("RUN_TAG", "").strip() or None,
            "resampling_rng": _resampling_meta,
            "episode_accounting": episode_accounting,
            "canonical_table_sources": {
                "table1_summary.csv": "summary",
                "table2_ablation.csv": "summary",
                "ConstraintViolationRate": "constraint_violation_rate",
            },
            "derived_metric_contracts": {
                "carbon_efficiency_ari_per_kgco2e_proxy": {
                    "equation": "episode_mean_ari/episode_carbon_kgco2e_proxy",
                    "unit": "ARI per kg CO2-e modeled transport indicator",
                    "scale_factor": 1.0,
                    "status": "exploratory ratio; not a canonical outcome",
                    "uncertainty": "BCa bootstrap of within-seed ratios",
                },
                "green_ai_decision_path": {
                    "energy_equation": (
                        "decision_path_elapsed_seconds*assumed_active_power_W"
                    ),
                    "water_equation": (
                        "decision_path_elapsed_seconds*"
                        "water_rate_L_per_server_second"
                    ),
                    "assumed_active_power_W": DEFAULT_ASSUMED_ACTIVE_POWER_W,
                    "water_rate_L_per_server_second": (
                        DEFAULT_WATER_RATE_L_PER_SERVER_SECOND
                    ),
                    "energy_per_step_proxy_J": (
                        DEFAULT_ENERGY_PER_PROXY_STEP_J
                    ),
                    "water_per_step_proxy_L": (
                        DEFAULT_WATER_PER_PROXY_STEP_L
                    ),
                    "measurement_scope": (
                        "coordinator.step action-selection wall time only; "
                        "not whole-job resource use"
                    ),
                    "status": (
                        "descriptive activity-based estimates and separately "
                        "labelled fixed-step proxies; not hardware telemetry"
                    ),
                },
            },
            "descriptive_only_metrics": [
                "mean_decision_latency_ms",
                "decision_path_compute_energy_estimate_j",
                "decision_path_compute_water_estimate_l",
                "decision_path_elapsed_seconds",
                "decision_step_count_energy_proxy_j",
                "decision_step_count_water_proxy_l",
            ],
            # BCa and deterministic-cell diagnostics. The canonical fallback
            # rate excludes zero-variance cells for which BCa is undefined by
            # construction; those cells receive point intervals and are
            # counted separately.
            "bca_fallback_stats": bca_stats,
        },
        "summary": summary,
    }
    if bca_stats["bca_fallback_rate"] > 0.10:
        print(
            f"WARNING: BCa percentile fallback fired on "
            f"{bca_stats['bca_fallbacks'] + bca_stats['fallback_scipy_unavailable']} "
            f"of {bca_stats['bca_calls']} attempted BCa calls "
            f"({100.0 * bca_stats['bca_fallback_rate']:.1f}%). Threshold for "
            "this warning is 10%; cells where the fallback fired may have "
            "wider-than-BCa CIs and should be flagged in the manuscript "
            "as percentile-fallback rather than BCa."
        )
    payload_significance = {
        "_meta": {
            "n_seeds": len(all_data),
            "git_commit": _git_commit_meta,
            "source_commit": _git_commit_meta,
            "simulation_source_commit": _git_commit_meta,
            "analysis_code_commit": _analysis_code_commit,
            "dual_provenance": _dual_provenance,
            "recovery_authorization": _recovery_authorization,
            "run_tag": _os_meta.environ.get("RUN_TAG", "").strip() or None,
            "resampling_rng": _resampling_meta,
            "episode_accounting": episode_accounting,
            "primary_h1_family": "agribrain_vs_no_context on ARI, 5 scenarios",
            "primary_h1_alternative": "agribrain greater than no_context",
            "primary_h1_canonical_p_value": (
                "h1_raw_p_value_directional_greater"
            ),
            "primary_h1_correction": "holm_bonferroni",
            "primary_h1_practical_margin_optional": H1_PRACTICAL_MARGIN,
            "primary_h1_practical_claim_rule": (
                "95% paired BCa mean-difference CI lower bound exceeds 0.005 "
                "ARI; a percentile fallback cannot support this claim"
            ),
            "pinn_ablation_family": (
                "agribrain_vs_no_pinn on ARI, 5 scenarios"
            ),
            "pinn_ablation_alternative": "agribrain greater than no_pinn",
            "pinn_ablation_correction": "holm_bonferroni",
            "pinn_ablation_scope": (
                "separate prespecified paired mechanistic-residual ablation; "
                "not part of H1 or H2"
            ),
            "h2_directional_family": (
                "{mcp_only,pirag_only}_vs_no_context and "
                "agribrain_vs_{mcp_only,pirag_only} on ARI, 5 scenarios "
                "(4 contrasts x 5 = 20 tests)"
            ),
            "h2_directional_alternative": "first mode greater than second mode",
            "h2_directional_correction": "holm_bonferroni",
            "h2_directional_canonical_field": (
                "p_value_adj_holm_h2_directional"
            ),
            "h2_global_support_rule": (
                "all 20 directional cells have positive mean differences and "
                "Holm-adjusted p < 0.05"
            ),
            "h2_synergy_definition": (
                "agribrain - mcp_only - pirag_only + no_context"
            ),
            "h2_synergy_status": "exploratory five-scenario family",
            "secondary_correction": "by_fdr",
            "secondary_family_scope": "per-scenario, all (comparison, metric) pairs",
            # Historical two-contrast subset retained as an auxiliary audit
            # field only. It must never be presented as the confirmatory H2
            # family, which is the 20-test family above.
            "channel_decomposition_family": (
                "{mcp_only,pirag_only}_vs_no_context on ARI, 5 scenarios "
                "(auxiliary 2 contrasts x 5 = 10 tests; not H2)"
            ),
            "channel_decomposition_correction": "holm_bonferroni",
            "channel_decomposition_auxiliary_field": "p_value_adj_holm_channel",
            "channel_decomposition_status": (
                "historical auxiliary subset; not confirmatory H2"
            ),
            "n_perm": 10_000,
            "n_perm_scope": (
                "legacy two-sided sign-flip audit values and emergency "
                "fallbacks only; not the canonical H1/H2 test"
            ),
            "legacy_sign_flip_resamples": 10_000,
            "confirmatory_test": "directional_wilcoxon_signed_rank",
            "paired": True,
            "wilcoxon_fallback_count": len(_WILCOXON_FALLBACK_CELLS),
            "wilcoxon_fallback_cells": [
                list(cell) for cell in sorted(_WILCOXON_FALLBACK_CELLS)
            ],
        },
        "primary_h1_holm_adjusted": primary_h1_holm,
        "primary_h1_supported_by_cell": h1_cell_support,
        "primary_h1_supported_all_cells": h1_supported_all_cells,
        "pinn_ablation_holm_adjusted": pinn_ablation_holm,
        "pinn_ablation_supported_by_cell": {
            sc: bool(
                significance[sc]["agribrain_vs_no_pinn"]["ari"].get(
                    "pinn_ablation_positive_effect_supported", False
                )
            )
            for sc in SCENARIOS
        },
        "pinn_ablation_supported_all_cells": all(
            bool(
                significance[sc]["agribrain_vs_no_pinn"]["ari"].get(
                    "pinn_ablation_positive_effect_supported", False
                )
            )
            for sc in SCENARIOS
        ),
        "h2_directional_holm_adjusted": h2_directional_holm,
        "h2_directional_supported_by_cell": h2_cell_support,
        "h2_directional_supported_all_cells": h2_supported_all_cells,
        "h2_synergy_holm_adjusted_exploratory": h2_synergy_holm,
        # Historical auxiliary subset, not the confirmatory H2 result.
        "channel_decomposition_holm_adjusted": channel_holm,
        "significance": significance,
    }
    h2_evidence_rows = _build_h2_publication_rows(
        significance,
        source_commit=_git_commit_meta,
        run_tag=_os_meta.environ.get("RUN_TAG", "").strip() or None,
    )
    payload_significance["h2_directional_evidence"] = h2_evidence_rows
    if recovery_provenance is not None:
        # Re-read and revalidate the immutable authorization immediately before
        # committing outputs so a long aggregation cannot hide a concurrent
        # replacement of either receipt or the preserved-input manifest.
        final_recovery_provenance = recovery_context_from_environment(
            results_dir=out_dir,
            repo_root=repo_root,
        )
        if final_recovery_provenance != recovery_provenance:
            raise RuntimeError(
                "publication recovery provenance changed during aggregation"
            )
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(payload_summary, indent=2, allow_nan=False)
    )
    (out_dir / "benchmark_significance.json").write_text(
        json.dumps(payload_significance, indent=2, allow_nan=False)
    )
    _write_h2_publication_csv(
        out_dir / "h2_directional_evidence.csv", h2_evidence_rows,
    )
    print("Saved benchmark_summary.json")
    print("Saved benchmark_significance.json")
    print("Saved h2_directional_evidence.csv (20 prespecified rows)")

    # Print key results
    print()
    for sc in SCENARIOS:
        a = summary[sc]["agribrain"]["ari"]
        print(f"  {sc}: ARI mean={a['mean']:.4f} CI=[{a['ci_low']:.4f}, {a['ci_high']:.4f}] std={a['std']:.6f}")

    print()
    print("Primary directional H1 family (Holm-Bonferroni across 5 scenarios):")
    for sc in SCENARIOS:
        p_raw = primary_h1_pvals.get(sc)
        p_adj = primary_h1_holm.get(sc)
        if p_raw is None or p_adj is None:
            continue
        print(f"  {sc} agribrain_vs_no_context ARI: p={p_raw:.4f} p_holm={p_adj:.4f}")

    print()
    print("Secondary (per-scenario BY-FDR) selected comparisons, ARI:")
    print(f"    {'Scenario':<22} {'Comparison':<28} {'p_adj':>7} {'d_z':>7} {'d_pooled':>9}")
    for sc in SCENARIOS:
        for comp_name in ("agribrain_vs_no_context", "agribrain_vs_hybrid_rl"):
            rec = significance[sc].get(comp_name, {}).get("ari")
            if rec is None:
                continue
            dz_text = (
                f"{rec['cohens_dz']:+7.3f}"
                if rec.get("cohens_dz") is not None else "    n/a"
            )
            print(f"    {sc:<22} {comp_name:<28} {rec['p_value_adj']:>7.4f} "
                  f"{dz_text} {rec.get('cohens_d_pooled', 0.0):>+9.3f}")

    print()
    print("Auxiliary single-channel subset (Holm across 2 x 5 = 10 ARI tests; not H2):")
    print(f"    {'Scenario':<22} {'Comparison':<32} {'p_adj_holm':>11} {'d_pooled':>9}")
    for sc in SCENARIOS:
        for a_mode, b_mode in _CHANNEL_DECOMPOSITION_PAIRS:
            comp_name = f"{a_mode}_vs_{b_mode}"
            rec = significance[sc].get(comp_name, {}).get("ari")
            if rec is None:
                continue
            print(
                f"    {sc:<22} {comp_name:<32} "
                f"{rec.get('p_value_adj_holm_channel', float('nan')):>11.4f} "
                f"{rec.get('cohens_d_pooled', 0.0):>+9.3f}"
            )

    print()
    print("H2 directional family (Holm across 4 x 5 = 20 ARI tests):")
    for sc in SCENARIOS:
        for a_mode, b_mode in _H2_DIRECTIONAL_PAIRS:
            comp_name = f"{a_mode}_vs_{b_mode}"
            rec = significance[sc][comp_name]["ari"]
            print(
                f"    {sc:<22} {comp_name:<32} "
                f"delta={rec['mean_diff']:+.4f} "
                f"p_holm={rec['p_value_adj_holm_h2_directional']:.4f} "
                f"supported={rec['h2_cell_supported']}"
            )
        synergy = significance[sc]["h2_synergy_interaction"]["ari"]
        print(
            f"    {sc:<22} {'exploratory superadditivity':<32} "
            f"S={synergy['mean_interaction']:+.4f} "
            f"p_holm={synergy['p_value_adj_holm_exploratory']:.4f} "
            f"supported={synergy['exploratory_superadditivity_supported']}"
        )

    # ------------------------------------------------------------------
    # Rewrite the publication CSVs with 20-seed statistics. Any existing
    # unsuffixed files are overwritten because their run provenance may be
    # unknown; per-seed evidence remains in benchmark_seeds/seed_*.json.
    # The canonical tables carry 20-seed bootstrap means and 95% CIs. The
    # column names (ARI, RLE, Waste, SLCA, Carbon, Equity) are preserved
    # with their values replaced by 20-seed means, and new _ci_low /
    # _ci_high columns are appended per metric.
    # ------------------------------------------------------------------
    _rewrite_stochastic_csvs(out_dir, summary)
    # The prespecified diagnostic secondary-ablation family is deliberately
    # exported separately from the confirmatory significance grid.  Generate
    # it from the same exact seed envelopes during every core aggregation so
    # publication validation can replay the JSON and CSV byte-for-byte.
    try:
        from ..analysis.export_secondary_ablations import analyse, write_outputs
    except ImportError:
        from mvp.simulation.analysis.export_secondary_ablations import (  # noqa: E402
            analyse,
            write_outputs,
        )
    secondary_payload = analyse(input_seed_dir, tuple(seeds))
    write_outputs(
        secondary_payload,
        out_dir / "secondary_ablation_analysis.json",
        out_dir / "secondary_ablation_analysis.csv",
    )
    print("Saved secondary_ablation_analysis.json")
    print("Saved secondary_ablation_analysis.csv")


def _fmt(x: float, precision: int = 4) -> str:
    """Format a float for CSV output; used so DataFrame-free builds of
    table1/table2 still produce the same number of decimals the legacy
    single-seed files used."""
    if x is None:
        return ""
    return f"{x:.{precision}f}"


_H2_PUBLICATION_COLUMNS = (
    "source_commit", "run_tag", "scenario", "comparison",
    "numerator_mode", "denominator_mode", "direction", "endpoint",
    "n_seeds", "paired_design", "test", "alternative",
    "mean_difference", "mean_difference_ci_low",
    "mean_difference_ci_high", "mean_difference_ci_method",
    "cohens_dz", "cohens_dz_ci_low", "cohens_dz_ci_high",
    "cohens_dz_ci_method", "raw_directional_p_value",
    "holm_adjusted_p_value", "holm_family_size", "alpha",
    "positive_mean", "cell_supported",
)


def _build_h2_publication_rows(
    significance: dict,
    *,
    source_commit: str | None,
    run_tag: str | None,
) -> list[dict]:
    """Project the exact 20-cell H2 family into a paper-ready table.

    This is deliberately a projection of the canonical significance records,
    not a second statistical implementation.  The independent publication
    validator replays Holm and cross-checks every exported value against the
    JSON records and raw seed evidence.
    """
    rows: list[dict] = []
    for scenario in SCENARIOS:
        for numerator, denominator in _H2_DIRECTIONAL_PAIRS:
            comparison = f"{numerator}_vs_{denominator}"
            record = significance[scenario][comparison]["ari"]
            row = {
                "source_commit": source_commit,
                "run_tag": run_tag,
                "scenario": scenario,
                "comparison": comparison,
                "numerator_mode": numerator,
                "denominator_mode": denominator,
                "direction": f"{numerator} > {denominator}",
                "endpoint": "ari",
                "n_seeds": int(record["n_seeds"]),
                "paired_design": True,
                "test": record["h2_test_type_actual"],
                "alternative": "greater",
                "mean_difference": float(record["mean_diff"]),
                "mean_difference_ci_low": float(record["mean_diff_ci_low"]),
                "mean_difference_ci_high": float(record["mean_diff_ci_high"]),
                "mean_difference_ci_method": record["mean_diff_ci_method"],
                "cohens_dz": record["cohens_dz"],
                "cohens_dz_ci_low": record["cohens_dz_ci_low"],
                "cohens_dz_ci_high": record["cohens_dz_ci_high"],
                "cohens_dz_ci_method": record["cohens_dz_ci_method"],
                "raw_directional_p_value": float(
                    record["p_value_directional_greater"]
                ),
                "holm_adjusted_p_value": float(
                    record["p_value_adj_holm_h2_directional"]
                ),
                "holm_family_size": int(record["h2_family_size"]),
                "alpha": 0.05,
                "positive_mean": bool(record["mean_diff"] > 0.0),
                "cell_supported": bool(record["h2_cell_supported"]),
            }
            rows.append(row)
    if len(rows) != len(SCENARIOS) * len(_H2_DIRECTIONAL_PAIRS):
        raise RuntimeError("H2 publication table is not the exact 20-cell family")
    return rows


def _write_h2_publication_csv(path: Path, rows: list[dict]) -> None:
    """Persist the complete H2 evidence table with a fixed column schema."""
    if len(rows) != 20 or any(set(row) != set(_H2_PUBLICATION_COLUMNS) for row in rows):
        raise RuntimeError("refusing to write an incomplete H2 evidence table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_H2_PUBLICATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_stochastic_csvs(out_dir, summary):
    """Rewrite table1_summary.csv and table2_ablation.csv as 20-seed means
    + 95% CIs from ``summary``. Existing files are replaced rather than
    relabelled as a particular seed without evidence of their provenance.

    Column layout: same display names as the legacy single-seed CSVs
    (ARI, Waste, ...), followed by ``ARI_ci_low``, ``ARI_ci_high``, etc.
    Downstream readers that only index by ``row["ARI"]`` continue to work
    and now get the 20-seed mean; readers that want CIs pick up the new
    columns.
    """
    import csv
    # table1_summary.csv: primary publication modes across scenarios.
    t1_path = out_dir / "table1_summary.csv"
    header = ["Scenario", "Method"]
    for _key, disp in _TABLE1_COLUMNS:
        header.extend([disp, f"{disp}_ci_low", f"{disp}_ci_high"])
    header.extend(["n_seeds"])
    rows = []
    for sc in SCENARIOS:
        for mode in _TABLE1_ROW_METHODS:
            bucket = summary.get(sc, {}).get(mode, {})
            if not bucket:
                continue
            row = [sc, mode]
            n_seeds_row = 0
            for key, _ in _TABLE1_COLUMNS:
                rec = bucket.get(key)
                if rec is None:
                    row.extend(["", "", ""])
                    continue
                precision = 0 if key == "carbon" else 4
                row.append(_fmt(rec["mean"], precision))
                row.append(_fmt(rec["ci_low"], precision))
                row.append(_fmt(rec["ci_high"], precision))
                n_seeds_row = max(n_seeds_row, int(rec.get("n_seeds", 0)))
            row.append(str(n_seeds_row))
            rows.append(row)
    with open(t1_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved 20-seed {t1_path.name} ({len(rows)} rows)")

    # table2_ablation.csv: compact architectural ablation across scenarios.
    t2_path = out_dir / "table2_ablation.csv"
    header = ["Scenario", "Variant"]
    for _key, disp in _TABLE2_COLUMNS:
        header.extend([disp, f"{disp}_ci_low", f"{disp}_ci_high"])
    header.extend(["n_seeds"])
    rows = []
    table2_modes = (
        "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
        "agribrain",
    )
    for sc in SCENARIOS:
        for mode in table2_modes:
            bucket = summary.get(sc, {}).get(mode, {})
            if not bucket:
                continue
            row = [sc, mode]
            n_seeds_row = 0
            for key, _ in _TABLE2_COLUMNS:
                rec = bucket.get(key)
                if rec is None:
                    row.extend(["", "", ""])
                    continue
                row.append(_fmt(rec["mean"], 4))
                row.append(_fmt(rec["ci_low"], 4))
                row.append(_fmt(rec["ci_high"], 4))
                n_seeds_row = max(n_seeds_row, int(rec.get("n_seeds", 0)))
            row.append(str(n_seeds_row))
            rows.append(row)
    with open(t2_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved 20-seed {t2_path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
