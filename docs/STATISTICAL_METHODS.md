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

Every (mode, scenario) seed shares the **same scenario trajectory**
(same `df_scenario`, `scenario_rng`, ambient temperature/RH/inventory
time-series) by construction in `generate_results.run_all`. What
differs across modes within a scenario is (a) the per-mode policy-
temperature draw (Source 8, σ=0.25 in log-space) and (b) for non-
`_AGRIBRAIN_LOGIT_MODES` modes, the ablation-seed integer used for
the policy-internal RNG.

Wilcoxon signed-rank and other paired tests pair on **shared scenario
trajectory**, not on shared ablation-seed integer. Within-pair
correlation comes from the common environmental realisation, which is
the dominant noise source. Treating any baseline as unpaired wastes
that within-seed correlation and produces conservatively-loose
p-values. The post-2026-04 audit therefore extended pairing to **all
five baselines** (`no_context`, `mcp_only`, `pirag_only`, `static`,
`hybrid_rl`); the aggregator's `_PAIRED_BASELINES` set carries this
union and the `is_paired_design` field is set to `True` on every
record.

The reported `cohens_dz` for `static`/`hybrid_rl` is therefore the
seed-list-aligned within-pair effect size. Reviewers who prefer the
design-independent statistic should read `cohens_d_pooled` (also
reported on every record), which standardises by the pooled within-
method standard deviation rather than the within-pair difference SD.

## Tests and Effect Size

- **Paired comparisons (all five baselines: `no_context`, `mcp_only`,
  `pirag_only`, `static`, `hybrid_rl`)**:
  - Wilcoxon signed-rank test (SciPy `wilcoxon` with `zsplit` tie handling).
  - Paired effect size `cohens_dz`.
  - Canonical effect size `cohens_d_pooled` reported alongside.
  - 10,000-resample BCa bootstrap CI (Efron 1987) for both the mean
    difference and both effect-size statistics. When the BCa
    correction is mathematically undefined (e.g. all bootstrap
    replicates equal the point estimate, or scipy.special.ndtri is
    unavailable) the routine falls back to the plain percentile
    method and increments a per-call counter so the aggregator's
    `_meta.bca_fallback_stats.fallback_rate` field surfaces the
    fraction of cells that fell back from BCa.
- **Unpaired path** (retained for `no_pinn`, `no_slca` which don't
  participate in the canonical paired family per `_PAIRED_BASELINES`):
  - Mann-Whitney U test.
  - Pooled effect size `cohens_d_pooled`.
  - Same BCa bootstrap CI structure (independent-arm resampling).
- **Effect-size CIs are reported in every record** (`effect_size_ci_low`,
  `effect_size_ci_high`, `effect_size_ci_method = "BCa"`).
- **Hedges' g** small-sample correction (`hedges_g`) is reported alongside
  Cohen's d for transparency. With n=20 the correction is approximately
  0.987.
- **Legacy sign-flip permutation** (`p_value_legacy_signflip`) is also
  recorded for paired comparisons so the two test bases can be
  compared. The Wilcoxon p-value is the canonical `p_value`.

Bootstrap CI seeds. Per-cell deterministic seeds are derived from
`blake2b((scope, scenario, mode, metric))` (a 32-bit BLAKE2b digest
of the canonical-string-joined cell key). The earlier
`hash((scope, *cell_key))` derivation was PYTHONHASHSEED-randomised
by default, so two HPC runs in different processes produced
different bootstrap samples for the same cell. The blake2b digest is
purely deterministic and gives the same 32-bit seed across processes,
operating systems, and Python versions.

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
explicitly because paired d_z values above ~3 are uncommon in
empirical operations-research literature; in our setting d_z lands at
~1.5–3 because the simulator includes a
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
they can be inspected. Set `STRICT_VALIDATION=0` to restore the
previous report-only behaviour for local debugging — this is not the
canonical configuration.

The previous default (report-mode) was an explicit response to an
earlier concern that range/ordering gates encoded the manuscript's
preferred ordering and risked confirmation bias. To
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
