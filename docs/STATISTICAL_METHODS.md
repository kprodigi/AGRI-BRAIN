# Statistical Methods (Benchmark Inference)

## Scope

This note defines the inferential statistics used for multi-seed benchmark claims in AGRI-BRAIN.

## Hypotheses

For each scenario, baseline, and metric:

- Null hypothesis: seed-level mean difference between `agribrain` and baseline is zero.
- Alternative hypothesis: seed-level mean difference is non-zero.

The pre-registered primary hypothesis H1 is specifically `agribrain` vs `no_context`
on ARI, tested once per scenario. The five resulting p-values form the primary
family. All other `(baseline, metric)` comparisons are secondary.

## Paired vs unpaired comparisons

The simulator's seed plumbing assigns the same `ablation_seed` to all
context-aware modes (`agribrain`, `no_context`, `mcp_only`, `pirag_only`,
and the §4.7 ablation variants) so they share the per-step environmental
realisation. Comparisons against these baselines are therefore **paired**.

Comparisons against `static`, `hybrid_rl`, `no_pinn`, `no_slca` use
**independent** mode seeds for the comparator arm, so paired statistics
do not apply. The aggregator selects the test type per comparison based
on this design, recording the choice in the `is_paired_design` and
`test_type` fields of every comparison record.

## Tests and Effect Size

- **Paired comparisons (vs `no_context`, `mcp_only`, `pirag_only`, ablation variants)**:
  - Wilcoxon signed-rank test (SciPy `wilcoxon` with `zsplit` tie handling).
  - Paired effect size `cohens_dz`.
  - 10,000-resample percentile bootstrap CI for both the mean difference
    and the effect size.
- **Unpaired comparisons (vs `static`, `hybrid_rl`, `no_pinn`, `no_slca`)**:
  - Mann-Whitney U test.
  - Pooled effect size `cohens_d_pooled`.
  - 10,000-resample independent-arm percentile bootstrap CI for both
    the mean difference and the effect size.
- **Effect-size CIs are reported in every record** (`effect_size_ci_low`,
  `effect_size_ci_high`).
- **Hedges' g** small-sample correction (`hedges_g`) is reported alongside
  Cohen's d for transparency. With n=20 the correction is approximately
  0.987.
- **Legacy sign-flip permutation** (`p_value_legacy_signflip`) is also
  recorded for paired comparisons so reviewers can compare the two test
  bases. The Wilcoxon p-value is the canonical `p_value`.

Bootstrap CIs in this implementation use the percentile method with
10,000 resamples and per-cell deterministic seeds derived from
`hash((scope, scenario, mode, metric))` so adjacent cells have
independent Monte Carlo error.

## Multiple Testing Control

Two-level multiplicity control:

1. **Primary H1 family** (5 tests, one per scenario, `agribrain` vs
   `no_context` on ARI): **Holm-Bonferroni** step-down correction.
   Controls the family-wise error rate. The canonical `p_value_adj` on
   the five primary records uses this correction, and the same value
   also appears as `p_value_adj_holm`.
2. **Secondary family** (within each scenario: all baselines × all
   metrics): **Benjamini-Yekutieli (BY) FDR** correction applied per
   scenario, valid under arbitrary dependence (which the within-
   scenario metric correlations may violate, since waste vs ARI are
   mechanically negatively correlated). Canonical `p_value_adj` on
   non-primary records uses BY. **Benjamini-Hochberg (BH)** is also
   reported (`p_value_adj_bh`) for transparency, though it requires
   PRDS which we do not assume.

Every record additionally carries a `correction_method` field naming
the canonical adjustment used.

## Alpha and Interpretation

- Nominal alpha: `0.05`.
- Claims should be based on adjusted p-values and practical effect size
  jointly, not p-values alone.
- Primary decision rule for strong claims:
  - statistical: `p_value_adj < 0.05`
  - practical (paired): `|cohens_dz| >= 0.20` (small-or-greater paired effect)
  - practical (unpaired): `|cohens_d_pooled| >= 0.20`
  - directional consistency: sign of `mean_diff` matches claimed direction
- For metrics where lower is better (Waste, Carbon), a negative
  `mean_diff` supports AGRI-BRAIN superiority. For metrics where higher
  is better (ARI, RLE, SLCA, Equity), a positive `mean_diff` does.

## On the magnitude of `cohens_dz`

The paired Cohen's d_z values reported by the aggregator are sensitive
to the size of the within-pair standard deviation, which under our
matched-seed design captures only the residual variance after
partialling out the shared environmental realisation. We document this
explicitly because reviewers correctly observe that paired d_z values
above ~3 are uncommon in empirical operations-research literature; in
our setting d_z lands at ~1.5–3 because the simulator includes a
per-(mode, seed) policy-temperature draw that introduces mode-
differential noise (`STOCH_POLICY_TEMP_STD=0.25`). Reporting the
unpaired `cohens_d_pooled` alongside d_z lets the reader interpret the
effect size under both the experimental-design lens (paired) and the
deployment-variation lens (pooled).

## Seed Policy

- Publication benchmark default uses the 20-seed fixed list
  `42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505, 606, 707, 808, 909,
  1010, 1111, 1212, 1313, 1414, 1515` (see `mvp/simulation/reproduce_core.py`
  and `.env.example`).
- The list can be overridden via the `BENCHMARK_SEEDS` env var
  (comma-separated integers). Publication artefacts should always
  report the exact list used; the manifest `git_commit` plus the
  resolved seed list together pin the run.

## Baseline Fairness Protocol

- All compared methods use the same scenario generator, episode horizon, and observation stream.
- Paired comparisons share `ablation_seed` for environmental noise; unpaired comparisons receive independent mode seeds.
- Metrics and post-processing are identical across methods; only method logic differs.
- Canonical publication statistics are taken from `benchmarks/aggregate_seeds.py` outputs, not mixed alternate benchmark sources.

## Deterministic vs Stochastic Use

- Deterministic mode is for reproducibility gates and exact drift checks.
- Stochastic mode is required for uncertainty quantification and inferential claims.

## Validator policy

The post-aggregation validator (`validation/validate_results.py`) runs
in **report mode by default** and exits 0 even when pre-registered
range or ordering checks fire. It writes `validation_report.json` with
the full list of flagged items so reviewers can inspect them. Set
`STRICT_VALIDATION=1` to restore the old hard-gate behaviour. This
decouples the build from the hypothesis-confirmation gates that
earlier reviewers correctly identified as a confirmation-bias check.
