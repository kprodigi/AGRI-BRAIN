# Focused weight-sensitivity protocol

This is a prospective sensitivity extension designed after the primary results, not a prespecified part of that run. Its purpose is to assess dependence on selected numerical weights. No positive conclusion is required for a task to pass.

## Design

Only Static, No-Context and AGRI-BRAIN are evaluated. The unmodified source is commit `83c91ce8592c1d40234e5a6bb2a8ce342412f383`; a separate restoring wrapper applies the perturbations. The original sign assumptions, route-outcome equations, scenario generation, retrieval configuration and learning algorithm remain unchanged.

Twenty seeds and five scenarios are fully crossed with 23 settings: one nominal, 18 one-factor settings (nine factors, each at 0.8 and 1.2 times its nominal value), and four joint settings. Joint profiles A and B use opposite alternating signs across the nine factors, at 10% and 20%. They test simultaneous changes without selecting settings based on outcomes. They are not an exhaustive search or a probability distribution of plausible weights.

| Factor | Nominal | Intervention | Source |
|---|---|---|---|
| Base-policy prior | Full 3 × 10 THETA matrix | Multiply nonzero and zero entries by factor before existing seed perturbation; signs/zeros preserved | `agribrain/backend/src/models/action_selection.py`, `THETA` |
| Tool-context prior | THETA_CONTEXT columns 0, 1, 4 | Scale columns before constructing and adapting the context learner | `agribrain/backend/pirag/context_to_logits.py` |
| Retrieval-context prior | THETA_CONTEXT columns 2, 3 | Scale columns before constructing and adapting the context learner | same |
| Carbon score weight | 0.30 | Multiply selected weight, then normalize all four score weights to sum to one | `agribrain/backend/src/models/policy.py`, `w_c` |
| Labour score weight | 0.20 | same | `w_l` |
| Community score weight | 0.25 | same | `w_r` |
| Price-information score weight | 0.25 | same | `w_p` |
| Waste reward penalty | 0.50 | 0.40 / 0.60 | `eta` |
| Risk reward penalty | 0.50 | 0.40 / 0.60 | `eta_rho` |

The factor on a score weight is applied BEFORE normalization. Report effective weights, not an inaccurate claim that the final normalized weight changed by exactly 20%. Multiplicative perturbations do not test activation of nominal zeros or sign reversal. Scaling a prior also changes the magnitude of relative learning bounds; this is an initial-prior sensitivity test, not a fixed-policy post hoc multiplier.

The 20% ranges are explicit author-selected local stress envelopes, not empirical calibration intervals. The joint 10% profiles provide a smaller perturbation check. Other choices—including volatility coefficients, social logit shaping priors, action-specific social values, waste-saving assumptions, and model parameters—remain fixed. Conclusions must be limited to these selected factors and ranges.

## Execution and pairing

Each setting × seed × scenario task freshly executes all three modes. Static has one evaluation; each adaptive mode has three adaptation episodes followed by one frozen evaluation. No nominal checkpoint is reused for adaptation under a changed setting. Score weights also affect rewards, so this study reruns the complete feedback process rather than reporting fixed-trajectory rescoring as behavioral sensitivity.

Shared factors apply identically to the three modes wherever they enter their calculation. Context factors apply only where context is enabled. All modes and all settings retain matched source-keyed environmental and policy random streams. Individual decision histories may diverge.

Totals: **2,300 tasks; 20,700 executed episodes; 13,800 adaptation episodes; 6,900 evaluation episodes; 1,987,200 evaluation routing choices.** The independent nominal repeat used for quality assurance adds nine episodes and is excluded from analysis. Pilot study cells are reused, not counted twice.

All adaptation and evaluation ledgers and complete episode archives are retained. Record the source manifest, harness hash, actual matrices/weights, full dependency receipt, freeze status, metrics and routes. Outputs must be outside the source package. Failed attempts remain available; only validated completed tasks enter collection. The collector refuses incomplete studies.

## Local and HPC checks

Local pilots are explicitly marked development-only when using the available Python 3.12 runtime. They check scientific outputs against retained primary evidence and against an independent repeat. They are never merged into the HPC study. HPC requires Python 3.11 and the exact dependency lock. Its own pilot checks nominal reproduction, an independent repeat and representative perturbations before releasing the bulk arrays.

Nominal adaptation/evaluation results for both adaptive modes are compared with all 100 primary seed-scenario references when those nominal cells run. Metrics, routing probabilities and ARI traces must agree within 1e-10; actions must match exactly. Environmental trajectories are compared numerically within 1e-10: cross-platform roundoff can change literal hashes even when numerical differences are approximately 1e-15. Exact hashes are still retained; within-task pairing remains exact. Repeated local/HPC tasks must agree within 1e-12 for scientific quantities. Wall-clock latency, timestamps and their derived records are excluded from deterministic equivalence.

## Analysis

Primary comparison: AGRI-BRAIN minus No-Context ARI. AGRI-BRAIN minus Static is secondary. Report each scenario and the overall equal-scenario average. For overall inference, average scenarios within each seed before summarizing the 20 seed values.

Compute 10,000-resample BCa 95% confidence intervals across seeds for paired mean gains and changes in gain relative to the nominal setting. Use the same seeds for those changes. Report per-mode ARI, other retained metrics, relative percentage gain, the number of negative seed-level gains and setting/scenario mean ranking reversals. Record route differences between modes and changes from each mode's nominal routes. Degenerate bootstrap distributions are flagged; a constant quantity receives a zero-width interval.

These intervals are pointwise, not simultaneous confidence guarantees across all settings. No new family of significance tests is planned. A +0.01 ARI line is a descriptive reference, not a new prespecified threshold. Report how many settings have lower interval bounds above zero and above 0.01 without interpreting these counts as a corrected global hypothesis test.

The figure shows paired gains under the perturbations with uncertainty. A second panel shows the change in the gain relative to nominal. Include scenario summaries and a parameter/results table as CSV. Publish unfavorable and null findings as well as favorable ones. Report the minimum observed gain and its setting/scenario; never conclude “weights have no effect” merely because rankings are unchanged.

Stable findings within these ranges would support local robustness of the comparison. They would not validate synthetic route-outcome assumptions, establish routing optimality, or demonstrate robustness to arbitrary weights.
