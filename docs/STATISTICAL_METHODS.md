# Statistical Methods (Benchmark Inference)

## Scope

This note defines the inferential statistics used for multi-seed benchmark claims in AGRI-BRAIN.

## Hypotheses

For each scenario, baseline, and metric:

- Null hypothesis: paired seed-level mean difference between `agribrain` and baseline is zero.
- Alternative hypothesis: paired seed-level mean difference is non-zero.

## Tests and Effect Size

- Paired permutation test for p-values (`paired_permutation_pvalue`).
- Bootstrap confidence interval (95%) for paired mean difference.
- Paired effect size `cohens_dz` computed on per-seed differences.
- Mean difference reported as `E[agribrain - baseline]`.

## Multiple Testing Control

- Benjamini-Hochberg FDR correction is applied to the family of p-values for each scenario-baseline set across metrics.
- Report both raw `p_value` and adjusted `p_value_adj`.

## Alpha and Interpretation

- Nominal alpha: `0.05`.
- Claims should be based on adjusted p-values and practical effect size (`cohens_dz`) jointly, not p-values alone.

## Seed Policy

- Publication benchmark default uses 20 fixed seeds:
  `42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515`
- Seed list can be overridden via `BENCHMARK_SEEDS`, but publication artifacts should explicitly report the exact list used.

## Deterministic vs Stochastic Use

- Deterministic mode is for reproducibility gates and exact drift checks.
- Stochastic mode is required for uncertainty quantification and inferential claims.
