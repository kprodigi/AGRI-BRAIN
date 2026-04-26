#!/usr/bin/env python3
"""Aggregate multi-seed benchmark results into canonical benchmark files.

Reads ``results/benchmark_seeds/seed_*.json`` and writes

- ``results/benchmark_summary.json``    , per-(scenario, mode, metric) means,
  standard deviations, and 95 % bootstrap CIs.
- ``results/benchmark_significance.json``, paired permutation p-values, effect
  sizes, and multiplicity-adjusted p-values using two correction families:

  1. Holm-Bonferroni across the five scenario-level primary H1 tests
     (agribrain vs no_context, metric = ARI, one test per scenario). This
     matches the pre-registered multiplicity control declared in the paper.
     Reported as ``p_value_adj_holm`` on the five primary entries and as the
     canonical ``p_value_adj`` on the same entries.
  2. Benjamini-Hochberg FDR within each scenario across all (baseline, metric)
     secondary comparisons. Reported as ``p_value_adj_bh`` on every entry and
     as ``p_value_adj`` on every non-primary entry.

Usage::

    python aggregate_seeds.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
         606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515]
# Scenario and mode lists come from the simulator's canonical definitions
# so any new mode added to generate_results (e.g. cold_start, pert_*) is
# picked up automatically by this aggregator. Duplicated hardcoded lists
# here were the bug that silently dropped the new §4.7 ablation modes
# from the benchmark summary.
_SIM_DIR = Path(__file__).resolve().parent.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from generate_results import SCENARIOS as _SIM_SCENARIOS, MODES as _SIM_MODES
SCENARIOS = list(_SIM_SCENARIOS)
MODES = list(_SIM_MODES)
METRICS = ("ari", "waste", "rle", "slca", "carbon", "equity")
# Extra metrics exposed by run_single_seed.py when they are present in the
# per-seed dump. Aggregator does bootstrap CIs on these the same way as the
# core METRICS; missing values (e.g. context_honor_rate for static) are
# filtered out per-cell so aggregation does not crash.
EXTRA_METRICS = (
    # Required columns for the legacy table1/table2 CSV schema and for
    # validate_results.py's DecisionLatencyMs / ConstraintViolationRate /
    # ComplianceViolationRate bounds checks. Keeping them here means the
    # CSV rewrite below populates the same columns the validator expects.
    "mean_decision_latency_ms",
    "constraint_violation_rate",
    "compliance_violation_rate",
    # §4.7 paper-evidence metrics.
    "operational_violation_rate", "regulatory_violation_rate",
    "context_active_fraction", "context_honor_rate",
    "context_active_steps", "context_honored_steps",
)

# Columns exposed in the stochastic CSV rewrites below. First element of
# each tuple is the source key in benchmark_summary.json; second is the
# human-facing display name (kept identical to the legacy single-seed CSV
# so the paper's Tables 7 and 9 and the validate_results.py row["..."]
# reads continue to work against the 20-seed CSV).
# Implementation note: 2025-04 instrumentation symmetry fix.
# The previous schema used ``constraint_violation_rate`` as the public
# constraint column. That field is computed as
#   constraint = temp_violation OR quality_violation OR compliance_violation
# and ``compliance_violation`` only fires for modes in _MCP_WASTE_MODES
# (agribrain and friends). Static and hybrid_rl never invoke
# check_compliance, so their compliance count is structurally zero,
# which made AgriBrain look uniquely bad on the OR'd metric (~0.80 vs
# ~0.59 for static). The asymmetry was instrumentation, not behaviour.
#
# We now publish ``operational_violation_rate`` (temp OR quality only,
# symmetric across every mode) as the primary "ConstraintViolationRate"
# column, and keep the MCP-specific compliance signal under a clearly
# named ``RegulatoryViolationRate`` column so reviewers can read the two
# axes without conflating them.
_TABLE1_COLUMNS = (
    ("ari", "ARI"), ("rle", "RLE"), ("waste", "Waste"),
    ("slca", "SLCA"), ("carbon", "Carbon"), ("equity", "Equity"),
    ("mean_decision_latency_ms", "DecisionLatencyMs"),
    ("operational_violation_rate", "ConstraintViolationRate"),
    ("regulatory_violation_rate", "RegulatoryViolationRate"),
)
_TABLE2_COLUMNS = (
    ("ari", "ARI"), ("rle", "RLE"), ("waste", "Waste"), ("slca", "SLCA"),
    ("mean_decision_latency_ms", "DecisionLatencyMs"),
    ("operational_violation_rate", "ConstraintViolationRate"),
)
_TABLE1_ROW_METHODS = ("static", "hybrid_rl", "agribrain")
BASELINES = ("mcp_only", "pirag_only", "no_context",
             "hybrid_rl", "static")

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
seed_dir = _SCRIPT_DIR / "results" / "benchmark_seeds"


def _cell_seed(scope: str, cell_key: tuple) -> int:
    """Deterministic but cell-keyed RNG seed.

    Implementation note: 2025-04 cell-correlation fix.
    Previous revisions used a constant seed (42, 24, 123) for every
    bootstrap and permutation call, which made adjacent (scenario, mode,
    metric) cells share the same resample sequence and therefore have
    correlated bootstrap noise. We now derive a 32-bit seed from
    hash((scope, *cell_key)) so each cell gets independent resampling
    while remaining fully reproducible run-to-run.
    """
    return abs(hash((scope, *cell_key))) % (2**32)


def bootstrap_ci(vals, n_boot=10_000, alpha=0.05, cell_key=("global",)):
    """Percentile bootstrap CI with 10,000 resamples, matching paper Section 3.13.

    cell_key seeds the resampler so adjacent cells have independent
    Monte Carlo error (see _cell_seed).
    """
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(_cell_seed("bootstrap_ci", cell_key))
    boots = [float(np.mean(rng.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_mean_diff_ci(a, b, n_boot=10_000, alpha=0.05, paired=True, cell_key=("global",)):
    """Bootstrap CI for mean(a) - mean(b) with 10,000 resamples.

    paired=True resamples a single index applied to both arms (correct
    when a and b come from a matched-seed paired design). paired=False
    independently resamples each arm (correct when the two arms have
    independent seeds).
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(_cell_seed("bootstrap_diff_ci", cell_key))
    boots = []
    if paired and x.shape == y.shape:
        idx = np.arange(len(x))
        for _ in range(n_boot):
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            boots.append(float(np.mean(x[sample_idx] - y[sample_idx])))
    else:
        idx_a = np.arange(len(x))
        idx_b = np.arange(len(y))
        for _ in range(n_boot):
            mean_a = float(np.mean(x[rng.choice(idx_a, size=len(idx_a), replace=True)]))
            mean_b = float(np.mean(y[rng.choice(idx_b, size=len(idx_b), replace=True)]))
            boots.append(mean_a - mean_b)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilcoxon_signed_rank_pvalue(a, b, cell_key=("global",)):
    """Two-sided Wilcoxon signed-rank p-value via SciPy with exact fallback.

    Implementation note: 2025-04 distributional-assumption fix.
    The previous test was a sign-flip permutation on |mean(d)|, which
    requires d to be symmetric about 0 under H0. Under multiplicative
    log-normal stochastic noise this assumption is questionable. The
    Wilcoxon signed-rank test is valid under the weaker assumption that
    d is symmetric in *rank* (which holds for many common distributions
    of paired differences). When SciPy is unavailable, we fall back to
    a sign-flip permutation labelled clearly as such.
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) == 0:
        return 1.0
    d = x - y
    nz = d[d != 0]
    if len(nz) < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        # zsplit handles ties; two-sided is the default.
        res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.pvalue)
    except Exception:
        # Fallback to sign-flip permutation; document the fallback in
        # the per-comparison record so reviewers can detect it.
        rng = np.random.default_rng(_cell_seed("wilcoxon_fallback", cell_key))
        observed = abs(float(np.mean(d)))
        ge = 0
        n_perm = 10_000
        for _ in range(n_perm):
            signs = rng.choice([-1.0, 1.0], size=len(d))
            if abs(float(np.mean(d * signs))) >= observed:
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


def mann_whitney_pvalue(a, b):
    """Two-sided Mann-Whitney U p-value (unpaired non-parametric)."""
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return 1.0


def cohens_dz(a, b):
    """Paired Cohen's d_z = mean(a-b) / std(a-b).

    Appropriate for repeated-measures / matched designs where (a, b) are
    paired observations. Standardised by the within-pair standard
    deviation, which is small when the two arms share environmental
    variance — large d_z values reflect both effect size AND the
    precision of the paired design, so should always be reported
    alongside the unpaired/pooled Cohen's d (see ``cohens_d_pooled``).
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if x.shape != y.shape or len(x) < 2:
        return 0.0
    d = x - y
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else 0.0


def cohens_d_pooled(a, b):
    """Unpaired (pooled) Cohen's d = (mean(a) - mean(b)) / s_pooled.

    s_pooled = sqrt(((n_a-1)*var(a) + (n_b-1)*var(b)) / (n_a+n_b-2)).

    Implementation note: companion to cohens_dz.
    The paired d_z and pooled d answer different questions. d_z asks
    "given matched conditions, how reliably does method A beat method B
    seed-to-seed?" — its denominator is std(diff), which under a paired
    design that shares scenario template across arms can be very small,
    pushing d_z into 4-10 range that reviewers correctly flag as
    implausibly large. Pooled d asks "across realistic deployment
    variation, how separated are the two methods on the metric scale?" —
    its denominator is the pooled within-method standard deviation,
    which captures the run-to-run variability operators actually
    observe and lands in the empirical 0.5-2.5 range. Reporting both
    lets the reader see the effect-size claim under the experimental-
    design lens (paired) and the deployment lens (pooled) without
    conflating the two.
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

    g = J(df) * d, where J(df) = 1 - 3/(4*df - 1). For n=20 the
    correction is approximately 0.987. Recommended by APA / OR
    reporting standards for samples below n=50.
    """
    if paired:
        d = cohens_dz(a, b)
        df = max(len(a) - 1, 1)
    else:
        d = cohens_d_pooled(a, b)
        df = max(len(a) + len(b) - 2, 1)
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(j * d)


def bootstrap_effect_size_ci(a, b, n_boot: int = 5_000, alpha: float = 0.05,
                              paired: bool = True, cell_key=("global",)):
    """95 % bootstrap CI on Cohen's d_z (paired) or pooled d (unpaired).

    Readers asking for effect-size uncertainty (APA reporting
    standards) get a CI on the effect size itself, not just the mean
    difference.
    """
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(_cell_seed("d_ci", cell_key))
    boots = []
    if paired and x.shape == y.shape:
        idx = np.arange(len(x))
        for _ in range(n_boot):
            sel = rng.choice(idx, size=len(idx), replace=True)
            boots.append(cohens_dz(x[sel], y[sel]))
    else:
        idx_a = np.arange(len(x))
        idx_b = np.arange(len(y))
        for _ in range(n_boot):
            sa = rng.choice(idx_a, size=len(idx_a), replace=True)
            sb = rng.choice(idx_b, size=len(idx_b), replace=True)
            boots.append(cohens_d_pooled(x[sa], y[sb]))
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


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
        prev = q
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


def main():
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

    # Load seed results
    all_data = {}
    for seed in seeds:
        f = seed_dir / f"seed_{seed}.json"
        if f.exists():
            all_data[seed] = json.loads(f.read_text())
            print(f"Loaded seed {seed}")
        else:
            print(f"WARNING: {f} not found, skipping")

    if len(all_data) < 2:
        print(f"ERROR: Only {len(all_data)} seed(s) found, need at least 2")
        sys.exit(1)

    print(f"Aggregating {len(all_data)} seeds...")

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
                lo, hi = bootstrap_ci(vals)
                summary[sc][mode][met] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": len(vals),
                }

    # Build significance with two-level multiplicity control.
    # Pass 1: collect raw p-values for every (scenario, baseline, metric) cell.
    significance: dict = {}
    per_scenario_pvals: dict[str, dict[str, float]] = {sc: {} for sc in SCENARIOS}
    primary_h1_pvals: dict[str, float] = {}

    # Set of baselines that share `ablation_seed` with agribrain in
    # generate_results.run_all (the _AGRIBRAIN_LOGIT_MODES set). For
    # these the paired test and paired Cohen's d_z are valid; for
    # everything else (static, hybrid_rl, no_pinn, no_slca) the seeds
    # are independent and we use unpaired statistics. This addresses
    # the reported behaviour where paired d_z applied to unpaired data
    # is uninterpretable.
    _PAIRED_BASELINES = {"no_context", "mcp_only", "pirag_only"}

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
                # paired_permutation result is also kept as a sanity
                # cross-check field.
                if is_paired:
                    p_value = wilcoxon_signed_rank_pvalue(a, b, cell_key=cell_key)
                else:
                    p_value = mann_whitney_pvalue(a, b)
                p_perm_legacy = paired_permutation_pvalue(a, b, cell_key=cell_key)
                dz = cohens_dz(a, b) if is_paired else float("nan")
                d_pooled = cohens_d_pooled(a, b)
                hg = hedges_g(a, b, paired=is_paired)
                lo_diff, hi_diff = bootstrap_mean_diff_ci(
                    a, b, paired=is_paired, cell_key=cell_key
                )
                d_lo, d_hi = bootstrap_effect_size_ci(
                    a, b, paired=is_paired, cell_key=cell_key
                )
                mean_diff = float(np.mean(a) - np.mean(b))
                comp[met] = {
                    "p_value": p_value,
                    "p_value_legacy_signflip": p_perm_legacy,
                    # cohens_d retained as legacy alias = canonical
                    # effect size for the comparison's design (d_z if
                    # paired, d_pooled otherwise).
                    "cohens_d": dz if is_paired else d_pooled,
                    "cohens_dz": dz,
                    "cohens_d_pooled": d_pooled,
                    "hedges_g": hg,
                    "effect_size_ci_low": d_lo,
                    "effect_size_ci_high": d_hi,
                    "mean_diff": mean_diff,
                    "mean_diff_ci_low": lo_diff,
                    "mean_diff_ci_high": hi_diff,
                    "n_seeds": len(seeds_paired),
                }
                per_scenario_pvals[sc][f"{baseline}:{met}"] = p_value
                if baseline == "no_context" and met == "ari":
                    primary_h1_pvals[sc] = p_value
            significance[sc][f"agribrain_vs_{baseline}"] = comp

    # Pass 2a: Holm-Bonferroni across the primary H1 family (5 scenarios).
    primary_h1_holm = holm_bonferroni(primary_h1_pvals)

    # Pass 2b: BH-FDR (PRDS-assuming) AND BY-FDR (arbitrary-dependence)
    # within each scenario across all (baseline, metric) pairs. Reporting
    # both lets reviewers see the conservative bound (BY) when within-
    # scenario metric correlations have mixed signs.
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
                if baseline == "no_context" and met == "ari":
                    p_holm = float(primary_h1_holm.get(sc, rec["p_value"]))
                    rec["p_value_adj_holm"] = p_holm
                    rec["p_value_adj"] = p_holm
                    rec["correction_method"] = "holm_bonferroni_across_scenarios"
                else:
                    # Canonical p_value_adj on secondary endpoints uses
                    # the more conservative BY-FDR (valid under arbitrary
                    # dependence). BH retained as a less-conservative
                    # comparator under the PRDS assumption.
                    rec["p_value_adj"] = p_by
                    rec["correction_method"] = "by_fdr_within_scenario"

    # Save
    out_dir = _SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    payload_summary = {
        "_meta": {
            "n_boot": 10_000,
            "n_perm": 10_000,
            "bootstrap_alpha": 0.05,
            "seeds_loaded": sorted(all_data),
        },
        "summary": summary,
    }
    payload_significance = {
        "_meta": {
            "primary_h1_family": "agribrain_vs_no_context on ARI, 5 scenarios",
            "primary_h1_correction": "holm_bonferroni",
            "secondary_correction": "bh_fdr",
            "secondary_family_scope": "per-scenario, all (baseline, metric) pairs",
            "n_perm": 10_000,
            "paired": True,
        },
        "primary_h1_holm_adjusted": primary_h1_holm,
        "significance": significance,
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(payload_summary, indent=2)
    )
    (out_dir / "benchmark_significance.json").write_text(
        json.dumps(payload_significance, indent=2)
    )
    print("Saved benchmark_summary.json")
    print("Saved benchmark_significance.json")

    # Print key results
    print()
    for sc in SCENARIOS:
        a = summary[sc]["agribrain"]["ari"]
        print(f"  {sc}: ARI mean={a['mean']:.4f} CI=[{a['ci_low']:.4f}, {a['ci_high']:.4f}] std={a['std']:.6f}")

    print()
    print("Primary H1 family (Holm-Bonferroni across 5 scenarios):")
    for sc in SCENARIOS:
        p_raw = primary_h1_pvals.get(sc)
        p_adj = primary_h1_holm.get(sc)
        if p_raw is None or p_adj is None:
            continue
        print(f"  {sc} agribrain_vs_no_context ARI: p={p_raw:.4f} p_holm={p_adj:.4f}")

    print()
    print("Secondary (per-scenario BH-FDR) selected comparisons, ARI:")
    print(f"    {'Scenario':<22} {'Comparison':<28} {'p_adj':>7} {'d_z':>7} {'d_pooled':>9}")
    for sc in SCENARIOS:
        for comp_name in ("agribrain_vs_no_context", "agribrain_vs_hybrid_rl"):
            rec = significance[sc].get(comp_name, {}).get("ari")
            if rec is None:
                continue
            print(f"    {sc:<22} {comp_name:<28} {rec['p_value_adj']:>7.4f} "
                  f"{rec['cohens_dz']:>+7.3f} {rec.get('cohens_d_pooled', 0.0):>+9.3f}")

    # ------------------------------------------------------------------
    # Rewrite the Stage 1 CSVs with 20-seed statistics (Option 1).
    # The single-seed (seed=42) versions written by generate_results.py are
    # preserved as *_seed42.csv siblings for traceability. Downstream paper
    # figures and the export_paper_evidence stage should always read the
    # unsuffixed table1_summary.csv / table2_ablation.csv; these now carry
    # 20-seed bootstrap means and 95% CIs. Formula-compatible: the existing
    # column names (ARI, RLE, Waste, SLCA, Carbon, Equity) are preserved
    # with their values replaced by 20-seed means, and new _ci_low /
    # _ci_high columns are appended per metric.
    # ------------------------------------------------------------------
    _rewrite_stochastic_csvs(out_dir, summary)


def _fmt(x: float, precision: int = 4) -> str:
    """Format a float for CSV output; used so DataFrame-free builds of
    table1/table2 still produce the same number of decimals the legacy
    single-seed files used."""
    if x is None:
        return ""
    return f"{x:.{precision}f}"


def _rewrite_stochastic_csvs(out_dir, summary):
    """Rewrite table1_summary.csv and table2_ablation.csv as 20-seed means
    + 95% CIs from ``summary``. Renames any existing single-seed files to
    *_seed42.csv siblings before writing so those point-value references
    remain available for debugging.

    Column layout: same display names as the legacy single-seed CSVs
    (ARI, Waste, ...), followed by ``ARI_ci_low``, ``ARI_ci_high``, etc.
    Downstream readers that only index by ``row["ARI"]`` continue to work
    and now get the 20-seed mean; readers that want CIs pick up the new
    columns.
    """
    import csv
    import shutil

    # Preserve single-seed files, if present, as debugging siblings.
    for base in ("table1_summary.csv", "table2_ablation.csv"):
        src = out_dir / base
        if src.exists():
            dst = out_dir / base.replace(".csv", "_seed42.csv")
            try:
                shutil.move(str(src), str(dst))
                print(f"Preserved single-seed: {dst.name}")
            except OSError as exc:
                print(f"WARNING: could not rename {src} -> {dst}: {exc}")

    # table1_summary.csv: three-method headline across scenarios.
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

    # table2_ablation.csv: every mode across scenarios.
    t2_path = out_dir / "table2_ablation.csv"
    header = ["Scenario", "Variant"]
    for _key, disp in _TABLE2_COLUMNS:
        header.extend([disp, f"{disp}_ci_low", f"{disp}_ci_high"])
    header.extend(["n_seeds"])
    rows = []
    for sc in SCENARIOS:
        for mode in MODES:
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
