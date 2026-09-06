# Statistical Methods for the Submission Benchmark

## Status and inferential unit

This file defines the analysis implemented by the matching source commit. It is
not an external preregistration. The independent inferential unit is the
simulation seed (`n = 20`), not a timestep, routing decision, training episode,
or agent message.

The canonical seeds are:

`42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515`.

All modes within a scenario-seed-episode cell share the same initial conditions
and mode-independent exogenous stream keys. Operational random draws are keyed
by source and counter, so a conditional branch in one arm cannot shift later
environmental draws in another arm. Policy actions can still create different
endogenous trajectories, so shared exogenous randomness does not imply
identical realized states. Learned comparators adapt on episodes 0--2 and
retain episode 3 as a frozen no-update evaluation. Learner state persists only
within that adaptation block and is
reset before the next scenario. Static is a one-pass, non-learning reference.
Every episode starts with a fresh in-memory decision ledger, so `chain_query`
history does not cross episodes, modes, scenarios, or seeds. Policy-temperature
effect-size targeting is disabled.

## Metrics

The primary endpoint is the declared Adaptive Resilience Index
(ARI):

`ARI = (1 - waste) * social_performance * (1 - modeled_spoilage_risk)`.

ARI is a simulation composite, not the Adjusted Rand Index and not an externally
validated resilience scale. Waste fraction per routing opportunity,
the modeled emissions indicator, severity-weighted RLE, social-performance proxy,
latency, and violation disposition are secondary endpoints. Absolute
values are conditional on the synthetic scenario parameters.

Each 15-minute step is one equal-weight standardized simulated dispatch
opportunity. Waste is the mean modeled loss fraction across those opportunities,
not a measured share of physical throughput. The emissions endpoint is a
routing-distance indicator under declared distance and
per-kilometre factors; it contains no shipment-payload or tonne-kilometre model
and is not a measured network footprint.

The context-influence rate is an internal mechanism diagnostic. A step is
eligible when `max(abs(context_modifier)) > 0.10`. The live action is compared
with a context-ablated policy call reconstructed from the random-number state
saved immediately before live action selection. On the stochastic policy
path, both calls consume the same categorical variate and the context modifier
is the only controlled difference. The live call consumes that variate before
the author-declared probability-gap rule is evaluated, so an override discards
the sampled action without skipping the live arm's draw. Explicit deterministic policy evaluations
consume no action draw.
Sampling away from a policy argmax therefore cannot count as influence by
itself. Each retained context-enabled ledger row records the paired
context-ablated action and probabilities (legacy ledger fields retain
`counterfactual_*` names for compatibility), the raw action-change flag, the
eligibility flag, and whether the change enters the numerator. Sensitivity is
reported at thresholds 0.05, 0.10, 0.15, and 0.20. Under
`STRICT_VALIDATION=1`, a failed paired replay aborts the run rather than being
scored as no change. The same row records raw fused retrieval strength and the
post-rerank ordering score as separate fields; only the former drives the
normalized fused-rank feature and author-declared RRF-floor gate. Neither
quantity is a calibrated probability or an independent retrieval-quality
assessment.

## Hypotheses

### H1: external-context contribution

For each of the five scenarios, compare `agribrain` with `no_context` on ARI.
The two arms use the same base policy, declared priors, learning budget, agents,
forecasts, peer-message mechanism, and stochastic initialization. `no_context` bypasses
both external context channels (MCP tool outputs and institutional retrieval).
H1
therefore tests the contribution of those external channels; it does not isolate
peer messaging or establish a causal effect in a real supply chain.

The five scenario-level tests form one family. H1 is supported only for cells
whose Holm-adjusted directional p-value is below 0.05, whose mean paired difference
has the claimed sign, and whose uncertainty/effect size is reported. A global
statement across all scenarios requires all five cells to satisfy the stated
criterion. A practical claim of at least 0.005 ARI additionally requires the
95% paired BCa confidence-interval lower bound to exceed 0.005.

### Frozen-residual ablation

For each scenario, compare `agribrain` with `no_pinn` on ARI. The paired
`no_pinn` arm disables the frozen residual only in the policy-observed
spoilage estimate. Both arms are scored against the same noise-free,
mode-invariant trajectory from the declared independent synthetic DGP;
policy, context, peer, social-proxy, learning-budget, and random-stream
capabilities are otherwise identical. The five directional Wilcoxon
signed-rank tests form a separate
Holm family and are not added to H1 or H2. Support is reported per cell and
globally only when all five adjusted tests have the declared positive sign.

### H2: joint channel contribution and conditional feature-group masking

Within each scenario, H2 tests four paired directional ARI contrasts:
MCP-only minus No-external-context, Retrieval-only minus No-external-context, full AGRI-BRAIN
minus MCP-only, and full AGRI-BRAIN minus Retrieval-only. The 20 tests form one
Holm-corrected family. A universal claim requires positive differences and
Holm-adjusted p < 0.05 in all 20 cells; otherwise conclusions remain
cell-specific. The complete table is persisted as
`h2_directional_evidence.csv`; each row carries the paired mean difference,
95% interval and method label, paired effect and interval, raw directional
p-value, 20-test Holm value, and cell-support flag. Superadditivity is a
separate exploratory interaction,
`Full - MCP - Retrieval + No-external-context > 0`, and is not inferred merely because
the full arm ranks first. Within full-system ledgers, conditional observed-state
masking algebraically retains the MCP-derived or retrieval-derived context
features and recomputes the recorded policy argmax. Seed-cluster bootstrap
intervals and seed-level sign-flip diagnostics are used; decisions are never
treated as independent replicates.

Because the context modifier is additive in logit space, this decision-level
analysis concerns nonlinear argmax sensitivity. It reuses the observed MCP
results, retrieval output, and guards, so it cannot estimate what would happen
if a communication, tool, or retrieval channel were disabled. Only the separate
experimental-arm contrasts support channel-contribution statements.

### H3: robustness equivalence

The declared stressors are sensor noise, 10% missing telemetry, four-step
telemetry delay, MCP fault injection, and their compounded configuration. For
each stressed condition, AGRI-BRAIN starts from the same declared priors,
adapts on episodes 0--2, and is evaluated without updates on episode 3. The
nominal episode-3 endpoint is reused from the primary benchmark rather than
executed again. Learner state never crosses stressors, scenarios, seeds, or
modes, and each episode starts with empty decision history. The MCP reliability
configuration remains at the canonical disabled value, so it is not confounded with fault
exposure. The fixed rule `int(hour) % 11 == 0` creates 28 scheduled fault
opportunities on a complete 288-step trace. The artifacts report scheduled
opportunities, observed trigger steps, and individual tool results replaced
separately; observed exposure can be lower when MCP is unavailable (including
during the declared cyber outage). At an observed trigger, all tool results
actually invoked on that step are replaced. For
each of five scenarios and each stressor, the paired seed-level difference is:

`delta_s = ARI_stressed,s - ARI_nominal,s`.

A one-sample two-one-sided test (TOST) evaluates whether the mean difference is
equivalent to zero within ±0.01 at alpha = 0.05. Equivalence is established when
the 90% confidence interval lies wholly inside (-0.01, 0.01), equivalently when
both one-sided tests reject. The one-percentage-point ARI margin is a
prespecified simulation tolerance, not an externally validated deployment
threshold. H3 is supported globally only if all 25
scenario-stressor cells establish equivalence and every cell has verified
nonzero treatment exposure. This global rule is an
intersection-union test and therefore does not require an additional
multiplicity adjustment. The fraction of individual seeds
within the margin and its Clopper-Pearson interval are descriptive, not the
formal H3 test.

## Estimation and tests

- Confirmatory H1/H2 and the separate frozen-residual-ablation paired
  directional Wilcoxon signed-rank tests use SciPy with
  `zero_method="wilcox"`; zero differences are removed by the test definition.
- The metadata field `n_perm=10000` is a retained compatibility alias only for
  legacy two-sided sign-flip audit values and an emergency labelled fallback;
  it does not describe the canonical H1/H2 test. Strict publication validation
  requires zero Wilcoxon fallback cells.
- Mean paired differences, Cohen's `d_z`, pooled-standardized Cohen's `d`, and
  Hedges' `g` are reported.
- Across-seed standard deviations use the sample convention (`ddof=1`).
- Mean and effect-size intervals use 10,000-resample BCa bootstrap with paired
  index resampling. Deterministic zero-variance cells receive degenerate
  `[estimate, estimate]` intervals labelled `deterministic`. If BCa cannot be
  evaluated despite nonzero input variance, the output explicitly records the
  percentile fallback.
- Cell-specific bootstrap seeds are derived deterministically with BLAKE2b from
  the analysis scope, scenario, comparison, and metric.
- The retained unpaired Mann-Whitney path is not used by the canonical primary
  comparison.

## Multiple testing

- H1: Holm-Bonferroni across five scenario-level ARI tests.
- Frozen-residual ablation: a separate Holm-Bonferroni family across five
  scenario-level AGRI-BRAIN-minus-`no_pinn` ARI tests.
- H2 joint-channel family: Holm-Bonferroni across 20 directional ARI tests.
- For the secondary diagnostic column, Benjamini-Yekutieli false-discovery-rate
  correction is applied within scenario across all reported comparison-metric
  cells; H1 and H2 retain their family-specific Holm values as their canonical
  adjusted p-values. Benjamini-Hochberg is reported only as a supplementary
  diagnostic.
- A full-grid Holm value is also emitted as a conservative diagnostic.

The validator never requires AGRI-BRAIN to outperform another arm. Direction,
effect magnitude, and adjusted significance are observed results, not build
conditions.

## Completeness and provenance gates

`aggregate_seeds.py` rejects an incomplete or unbalanced seed panel. A valid
publication run must contain exactly the declared seed set and all required
scenario-mode metrics. The publication job also verifies 55 retained
final-episode decision ledgers per seed (11 modes × 5 scenarios), the complete
H3 panel, finite/bounded metrics, `publication_environment.json`, the artifact
SHA-256 manifest, the leakage-free internal forecast-validation receipt, and a
concrete clean Git commit.

## Interpretation limits

- The design evaluates behavior under synthetic scenarios; it is not external
  or field validation.
- ARI, RLE, social performance, temporal social performance, waste, route
  exposure, and emissions are modeled constructs with declared parameters.
- Statistical uncertainty across seeds does not cover structural model error,
  parameter misspecification, or institutional-corpus validity.
- The frozen residual is trained and evaluated only against the declared
  independent synthetic DGP (36 trajectories, locked 24/6/6 split). It is not
  empirical or external predictive validation; the paired `no_pinn` arm is the
  mechanistic-only comparator.

## Primary references

- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics
  Bulletin*, 1(6), 80–83.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery
  rate in multiple testing under dependency. *Annals of Statistics*, 29(4),
  1165–1188.
- Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the
  American Statistical Association*, 82(397), 171–185.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
  (2nd ed.). Lawrence Erlbaum Associates.
- Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
  size and related estimators. *Journal of Educational Statistics*, 6(2),
  107–128.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure
  and the power approach for assessing equivalence of average bioavailability.
  *Journal of Pharmacokinetics and Biopharmaceutics*, 15, 657–680.
- Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial
  limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413.
