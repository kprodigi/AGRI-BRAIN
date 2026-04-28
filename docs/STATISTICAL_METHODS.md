# Statistical Methods (Benchmark Inference)

## Scope

This note defines the inferential statistics used for multi-seed benchmark claims in AGRI-BRAIN.

## Hypotheses

For each scenario, baseline, and metric:

- Null hypothesis: seed-level mean difference between `agribrain` and baseline is zero.
- Alternative hypothesis: seed-level mean difference is non-zero.

The primary hypothesis H1 is `agribrain` vs `no_context` on ARI,
tested once per scenario. The five resulting p-values form the
primary family. All other `(baseline, metric)` comparisons are
secondary. Until an external pre-registration record exists,
manuscript text should describe these as "the analysis specified in
`docs/STATISTICAL_METHODS.md@<commit>`" rather than "pre-registered".

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
in **strict mode by default** as of 2026-04 and exits non-zero when
the range / interval checks declared in the validator source fire. It
writes `validation_report.json` with the full list of flagged items so
reviewers can inspect them. Set `STRICT_VALIDATION=0` to restore the
previous report-only behaviour for local debugging — this is not the
canonical configuration.

The previous default (report-mode) was an explicit response to an
earlier reviewer's concern that range/ordering gates encoded the
manuscript's preferred ordering and risked confirmation bias. To
retain that protection, this version of the validator (i) ships
*ranges* not orderings (each metric is checked against an absolute
interval, not against a "agribrain > X" ordering), and (ii) gates
the build only on those interval checks. Ordering claims in the
manuscript are decided by the bootstrap CIs and adjusted p-values
reported by `aggregate_seeds.py`, never by the validator.

## References

The statistical methods used here follow established practice. Full
citations for each method named above:

- Wilcoxon, F. (1945). Individual comparisons by ranking methods.
  *Biometrics Bulletin*, 1(6), 80–83. — Wilcoxon signed-rank test.
- Mann, H.B. & Whitney, D.R. (1947). On a test of whether one of two
  random variables is stochastically larger than the other. *Annals of
  Mathematical Statistics*, 18(1), 50–60. — Mann-Whitney U test.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
  Sciences* (2nd ed.). Lawrence Erlbaum Associates, Hillsdale, NJ.
  ISBN 0-8058-0283-5. — Cohen's d_z (paired) and d_pooled (independent).
- Hedges, L.V. (1981). Distribution theory for Glass's estimator of
  effect size and related estimators. *Journal of Educational
  Statistics*, 6(2), 107–128. — Hedges' g small-sample correction.
- Holm, S. (1979). A simple sequentially rejective multiple test
  procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. —
  Holm-Bonferroni step-down correction (primary H1 family).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
  rate: A practical and powerful approach to multiple testing. *Journal
  of the Royal Statistical Society B*, 57(1), 289–300. — BH FDR
  (reported alongside BY).
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false
  discovery rate in multiple testing under dependency. *Annals of
  Statistics*, 29(4), 1165–1188. — BY FDR (canonical secondary-family
  correction; valid under arbitrary dependence).
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the
  Bootstrap*. Monographs on Statistics and Applied Probability, 57.
  Chapman & Hall, New York. ISBN 0-412-04231-2. — Percentile bootstrap
  CI used for both mean differences and effect sizes.
- Clopper, C.J. & Pearson, E.S. (1934). The use of confidence or
  fiducial limits illustrated in the case of the binomial. *Biometrika*,
  26(4), 404–413. — Exact binomial CI used for stress-suite pass rates.
- Lakens, D. (2013). Calculating and reporting effect sizes to
  facilitate cumulative science: a practical primer for t-tests and
  ANOVAs. *Frontiers in Psychology*, 4, 863. — Effect-size reporting
  conventions for paired designs.
