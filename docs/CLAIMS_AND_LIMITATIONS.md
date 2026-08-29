# Scientific claims and limitations

This page defines the public claim boundary for the methodology-aligned
AGRI-BRAIN source. It is a concise guide; the locked equations and inferential
rules remain authoritative in [EXPERIMENT_PROTOCOL.md](../EXPERIMENT_PROTOCOL.md),
[docs/METHODS_REPRO_APPENDIX.md](METHODS_REPRO_APPENDIX.md), and
[docs/STATISTICAL_METHODS.md](STATISTICAL_METHODS.md).

## What the implementation contributes

AGRI-BRAIN is a coordinator-mediated heterogeneous multi-agent framework for a
synthetic perishable-supply-chain routing benchmark. Four sequential
decision-owner roles exchange typed peer messages; a cooperative role acts as
an overlapping advisory layer during its declared time window.

Three information paths can affect the policy:

1. typed peer messages produce a separate clipped additive logit term;
2. structured MCP outputs contribute through MCP-designated context features;
3. institutional retrieval contributes through retrieval-designated features,
   with retrieval-strength and temporal-continuity gates.

The peer term is not fused into the five-dimensional external-context vector.
The retrieval gates do not attenuate MCP. The binary operating-regime flag has
an action-specific logit bias, so it changes relative softmax probabilities.
The policy-gradient implementation differentiates the same gated, clipped,
and cooperatively composed modifier used by the forward policy.

The spoilage component separates two quantities: a policy-side mechanistic
estimate, optionally corrected by a frozen neural residual, and an independent
noise-free synthetic data-generating process used to score paired outcomes.
The residual is trained on declared synthetic trajectories with a five-term
physics-regularized loss. This is synthetic validation, not empirical
shelf-life validation.

Each retained decision can record the applied context, logit components,
categorical random variate, action, outcome fields, and local Merkle evidence.
The trace supports calculation reconstruction from retained inputs; it is not
a causal explanation of every upstream model.

## Experimental scale

The primary design has 800 retained evaluation cells:

`5 scenarios x 8 modes x 20 paired seeds = 800 retained cells`.

It is not an 800-episode experiment. Adaptation episodes make the primary
execution total 2,900 episodes and 835,200 simulated decision steps. Including
H3 and the declared secondary ablations, the core design contains 1,600 unique
retained cells, 6,100 executed episodes, and 1,756,800 steps. The separate
structural design contains 6,500 retained cells, 24,500 executed episodes, and
7,056,000 steps.

Paired arms share source- and counter-keyed exogenous streams and initial
conditions. Their endogenous paths may diverge after different actions.

## Confirmatory hypotheses

- **H1:** five directional paired Wilcoxon comparisons of AGRI-BRAIN against
  No-external-context, one per scenario, with Holm correction across five.
  The practical margin additionally requires the lower 95% paired BCa bound
  to exceed 0.005. The No-PINN comparison is a separate five-test family.
- **H2:** four directional external-channel contrasts per scenario, forming
  one 20-test Holm family. Universal support requires all 20 adjusted tests to
  pass. Superadditivity is a separately reported interaction contrast, not a
  synonym for H2.
- **H3:** 25 paired TOST cells from five scenarios and five stressors, with
  equivalence margin +/-0.01, a 90% interval strictly inside the margin, both
  one-sided tests significant, and verified nonzero exposure. Global support
  requires all 25 cells.

Hypotheses are supported only when the complete prespecified rule passes.
Partial or failed cells must be reported as partial, inconclusive, or
unsupported.

## Outcome interpretation

- Waste is an author-declared synthetic loss fraction per routing opportunity.
- Carbon is a modeled routing-emissions indicator, not a lifecycle footprint
  or fleet measurement.
- The social-performance quantity is a simulation proxy, not demographic or
  distributional equity and not an empirical social life-cycle assessment.
- Carbon Efficiency is episode-mean ARI divided by episode-summed modeled
  emissions in kilograms of CO2e; no factor of 1,000 is applied.
- Green-AI values use measured coordinator action-selection wall time with
  declared 10 W and `1.8e-6 L/s` conversion factors. They exclude forecasting,
  outcome scoring, learning, I/O, idle allocation, and whole-job computation.
- The optional permissioned-ledger interface is disabled in the confirmatory
  treatment; reproducibility evidence uses local off-chain Merkle roots.

## Claims this repository does not establish

The constructed benchmark does not establish field effectiveness, causal
deployment effects, measured food-waste reduction, lifecycle carbon impact,
demographic equity, legal compliance, marketability, safety, or generalization
to other crops and supply chains. Those require external data and a separate
validation design.
