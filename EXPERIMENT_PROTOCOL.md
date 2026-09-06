# AGRI-BRAIN locked simulation protocol

This file is the human-readable specification for the next publication run.
The implementation, validators, statistical analysis, tables, figures, and
manuscript must all agree with this protocol.  Numerical claims are written
only after the run; the model is never tuned to recover numbers from an older
manuscript.

## Scope and interpretation

AGRI-BRAIN is evaluated as a synthetic, stochastic simulation benchmark.  Its
waste, modeled emissions, social-performance, resilience, and temporal
social-performance-stability quantities are model outputs under declared
assumptions.  The legacy internal key `equity` denotes only the last proxy; it
is not demographic equity.  None of these quantities is a field measurement,
causal effect, or externally validated deployment performance.

Every paired arm is scored against the same noise-free outcome trajectory from
the declared independent synthetic data-generating process (DGP).  This scored
trajectory is not a PINN prediction.  For the policy-observed spoilage state,
every arm uses the same mechanistic Arrhenius-lag baseline; all arms except
`no_pinn` then add one identical frozen physics-informed neural residual.  The
residual is trained offline before HPC against 36 trajectories from the
versioned DGP (24 train, 6 validation, and 6 untouched test trajectories).
Targets are generated from a separately
integrated augmented rate law containing declared packaging, handling, and
humidity-transient effects; they are not copied from the mechanistic baseline
or its numerical ODE residual.  The bounded correction is
`delta_C = 0.08 tanh(f_theta(x))`.  Its loss contains data MSE, ODE-residual,
initial-boundary, residual-size, and monotonicity terms with weights
`[1.00, 0.20, 1.00, 0.01, 0.20]`.  Three fixed initialization seeds are
selected only by validation RMSE, and the checkpoint, trajectory splits,
history, hyperparameters, metrics, and SHA-256 bindings are retained under
`mvp/simulation/pinn/artifacts/`.

The raw neural correction is not asserted to be physically admissible by
itself.  On the untouched synthetic test trajectories, the raw prediction has
115 monotonicity violations and 41 unit-interval violations.  The estimator
actually deployed by every residual-enabled simulation arm is defined exactly
as `clip(C_mech + delta_C, 0, 1)` followed by a cumulative minimum within each
trajectory; its test diagnostics have zero violations of both constraints.
Raw and deployed RMSEs are retained separately in the checkpoint manifest.

This exercise is synthetic internal validation, not empirical spinach
shelf-life validation.  The paper and release must not describe the targets as
observations, field measurements, or externally validated kinetic data.
`no_pinn`, reported as **Mechanistic-only (No-PINN)**, is capability-identical
to AGRI-BRAIN except that it omits the frozen residual from the
policy-observed spoilage estimate; its scored DGP outcome, exogenous streams,
and episode budget remain paired.
The base values are `k_ref = 0.0021 h^-1` and `Ea/R = 8000 K`.  Once per
episode, the stochastic layer multiplies them by `1 + Normal(0, 0.20)` and
`1 + Normal(0, 0.14)`, respectively, with floors of `1e-6 h^-1` and `100 K`.
These counter-keyed draws are shared across paired modes.  Thus the DGP
outcome and each policy estimator are deterministic conditional on the episode
parameters and environmental stream, but the publication experiment is
stochastic across its declared seeds.

## Forecast lock

Forecast families were compared by one-step rolling-origin evaluation on the
repository's 288-row synthetic spinach series.  The temporal split is the first
60% for development, the next 20% for selection, and the final 20% for a
single untouched test report.  The test segment never selects a model.

The validation-RMSE rule selects non-seasonal Holt-linear demand forecasting
(7.5693 versus 7.7047 for persistence and 16.2671 for the LSTM) and persistence
for the inventory-based supply proxy (536.6288 versus 634.9944 for
Holt-linear).  These are the confirmatory defaults; the LSTM and the rejected
alternatives remain diagnostic implementations only.  Test performance is
reported even when inconvenient: demand Holt-linear RMSE is 8.0705 versus
8.4496 for persistence but its MAE is slightly worse (4.7382 versus 4.6724),
while the validation-selected supply persistence forecast is worse than
Holt-linear on the test segment (RMSE 20.7389 versus 12.3748).  Supply interval
coverage is also poor.  These instabilities are limitations, not evidence of
external predictive validity.  The forecast exercise is internal validation on
synthetic benchmark data only.

## Confirmatory benchmark

The five scenarios are `heatwave`, `overproduction`, `cyber_outage`,
`adaptive_pricing`, and `baseline`.  The fixed 20-seed panel is recorded in
`mvp/simulation/experiment_protocol.json`.

The eight primary modes are:

1. `static`
2. `hybrid_rl`
3. `no_pinn` (reported as **Mechanistic-only (No-PINN)**)
4. `no_slca`
5. `no_context` (reported as **No-external-context**; peer messages remain)
6. `mcp_only`
7. `pirag_only` (reported as **Retrieval-only**)
8. `agribrain`

For every context-enabled mode, the primary role first obtains its own bounded
external-context modifier.  During the cooperative overlay interval
`12 <= hour < 30`, the ordinary composition is
`0.70 * primary + 0.30 * cooperative`.  If, and only if, the cooperative MCP operating-envelope result
is critical while the primary MCP result is not critical, the composition is
instead the cooperative modifier plus `[-0.20, +0.20, 0.00]` in
`[cold-chain, local-redistribute, recovery]` order.  The result is clipped to
`[-1,+1]`.  This author-declared synthetic adjustment is distinct from the
later probability-gap action rule and is not a legal or calibrated safety
rule.

Each learned arm executes three adaptation episodes and then one frozen
evaluation episode.  Learners are reset before every
scenario-by-mode-by-seed block and cannot update during the retained evaluation
episode.  Static executes only the matched evaluation stream.  All paired arms
share initial conditions and counter-keyed exogenous random streams; their
endogenous trajectories may diverge after different actions.

This yields 800 retained primary evaluation cells but 2,900 actual 72-hour
episode executions (835,200 simulated decision steps).  The 800 retained cells
must not be described as “800 stochastic simulation episodes.”

## Secondary one-factor ablations

The same four-episode learned-arm budget applies to:

- `agribrain_standard_rag`: no state-conditioned query expansion, no
  lexical/Arrhenius reranking, and no piRAG temporal multiplier; the optional
  hard physics-consistency gate is disabled in both confirmatory arms;
- `agribrain_no_peer`: peer-message generation and delivery disabled while the
  agent schedule and external channels remain unchanged;
- `agribrain_sign_unconstrained`: sign projection disabled while initialization,
  learning rate, shrinkage, and magnitude caps remain unchanged.

These add 300 retained cells and 1,200 actual episode executions.  A claim that
piRAG improves retrieval quality additionally requires a separately supplied,
independently judged query set; downstream synthetic ARI alone is insufficient.

## Hypotheses and inference

The seed is the inferential unit.  All intervals and tests operate on paired
seed-level differences, never on the 288 within-episode decisions as if they
were independent samples.

- **H1 — external-context value:** AGRI-BRAIN minus No-external-context in each
  scenario.  Holm correction covers five directional tests.  Report the paired
  mean, 95% BCa confidence interval, paired effect size, and adjusted p-value.
  A practical claim of at least 0.005 ARI requires the confidence-interval lower
  bound, not merely the point estimate, to exceed 0.005.
- **H2 — joint channel contribution:** per scenario test MCP-only minus
  No-external-context, Retrieval-only minus No-external-context, AGRI-BRAIN
  minus MCP-only, and AGRI-BRAIN minus Retrieval-only.  Holm correction covers
  all 20 directional tests.  Universal support requires all 20 to pass.
  Superadditivity, if discussed, is a separate interaction contrast:
  `Full - MCP - Retrieval + No-external-context > 0`.
- **H3 — sensing and tool-channel robustness:** for each of five scenarios and
  five stressors, compare stressed AGRI-BRAIN with its paired nominal arm by
  TOST at margin +/-0.01 ARI.  Equivalence requires the 90% confidence interval
  to lie strictly inside the margin.  The global claim is an intersection-union
  claim and requires all 25 cells plus verified nonzero fault exposure.

The five H3 stressors are sensor noise, missing telemetry, telemetry delay,
MCP-result fault injection, and compounded stress.  The primary nominal arm is
reused, so H3 adds 500 retained stressed cells and 2,000 episode executions.
The core benchmark, H3, and three secondary ablations therefore contain 1,600
unique retained cells, 6,100 actual episodes, and 1,756,800 simulated steps.

## Structural sensitivity

A separately labelled 100-point, seed-locked Latin-hypercube design varies 29
active declared model parameters.  It is a structural sensitivity analysis,
not probability-based uncertainty unless externally justified parameter
distributions are later supplied.  Each design point evaluates the eight
primary scenario panels and the five AGRI-BRAIN stress panels: 6,500 retained
cells and 24,500 episode executions (7,056,000 simulated steps) in total.

Report rank stability, sign stability for every H1/H2 contrast, H3-margin
stability, and PRCC/Spearman associations.  Inactive or removed quantities
(including mode-specific physical outcome multipliers) must not be included
merely because an older document named them.  The frozen neural residual is
not retrained or tuned at any sensitivity point.

## Reproducibility boundary

Any change to policy logits, context gating, learner gradients, peer messaging,
retrieval, outcome/reward equations, scenarios, stochastic streams, forecast
inputs, stress exposure, or ablation definitions requires a fresh simulation
run under a new source commit and run tag. Deterministic replay may be used to
verify that analyses and rendering reproduce unchanged, hash-verified raw
artifacts. A shippable release may use either one clean source identity for the
complete pipeline or the narrowly authorized dual-provenance publication-only
recovery in `docs/PUBLICATION_RECOVERY.md`. Recovery requires byte-preserved
completed simulation outputs, verified failed-publisher accounting, distinct
simulation and publication commit/tree identities, `simulation_rerun: false`,
and all recovery plus combined-submission validators. A change to simulation
semantics or raw outputs still requires the affected simulation pipeline to be
rerun; deterministic publication, validation, or manuscript corrections do
not authorize changing the preserved simulation evidence.
