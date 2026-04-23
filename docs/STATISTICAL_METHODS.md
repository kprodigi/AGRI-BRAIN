# Statistical Methods (Benchmark Inference)

## Scope

This note defines the inferential statistics used for multi-seed benchmark claims in AGRI-BRAIN.

## Hypotheses

For each scenario, baseline, and metric:

- Null hypothesis: paired seed-level mean difference between `agribrain` and baseline is zero.
- Alternative hypothesis: paired seed-level mean difference is non-zero.

The pre-registered primary hypothesis H1 is specifically `agribrain` vs `no_context`
on ARI, tested once per scenario. The five resulting p-values form the primary
family. All other `(baseline, metric)` comparisons are secondary.

## Tests and Effect Size

- Paired permutation test for p-values (`paired_permutation_pvalue`),
  **10,000 permutations** per test.
- Bias-corrected accelerated bootstrap 95 % confidence interval for the paired mean
  difference, **10,000 resamples**.
- Paired effect size `cohens_dz` computed on per-seed differences.
- Mean difference reported as `E[agribrain - baseline]`.

## Multiple Testing Control

Two-level multiplicity control:

1. **Primary H1 family** (5 tests, one per scenario, `agribrain` vs `no_context` on
   ARI): **Holm-Bonferroni** step-down correction. Matches the paper's
   pre-registered multiplicity control and controls the family-wise error rate.
   The canonical `p_value_adj` on the five primary records uses this correction,
   and the same value also appears as `p_value_adj_holm`.
2. **Secondary family** (within each scenario: all baselines × all metrics, 6×6 = 36
   tests): **Benjamini-Hochberg FDR** correction applied per scenario.
   Reported as `p_value_adj_bh` on every record and as the canonical `p_value_adj`
   on non-primary records.

Every record additionally carries a `correction_method` field with the method
name (`"holm_bonferroni_across_scenarios"` or `"bh_fdr_within_scenario"`).

## Alpha and Interpretation

- Nominal alpha: `0.05`.
- Claims should be based on adjusted p-values and practical effect size (`cohens_dz`) jointly, not p-values alone.
- Primary decision rule for strong claims:
  - statistical: `p_value_adj < 0.05`
  - practical: `|cohens_dz| >= 0.20` (small-or-greater paired effect)
  - directional consistency: sign of `mean_diff` matches claimed improvement direction
- For metrics where lower is better (e.g., Waste, Carbon), a negative `mean_diff` supports AGRI-BRAIN superiority.
- For metrics where higher is better (e.g., ARI, RLE, SLCA, Equity), a positive `mean_diff` supports AGRI-BRAIN superiority.

## Seed Policy

- Publication benchmark default uses 20 fixed seeds:
  `42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515`
- Seed list can be overridden via `BENCHMARK_SEEDS`, but publication artifacts should explicitly report the exact list used.

## Baseline Fairness Protocol

- All compared methods use the same scenario generator, episode horizon, and observation stream.
- Per-seed comparisons are paired: all methods are evaluated under aligned seed conditions for each scenario.
- Metrics and post-processing are identical across methods; only method logic differs.
- Canonical publication statistics are taken from `benchmarks/aggregate_seeds.py` outputs, not mixed alternate benchmark sources.

## Deterministic vs Stochastic Use

- Deterministic mode is for reproducibility gates and exact drift checks.
- Stochastic mode is required for uncertainty quantification and inferential claims.
