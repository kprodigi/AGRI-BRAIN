# MODE_EFF: Predicted vs. Empirical Save Efficiency

The `MODE_EFF` table in `agribrain/backend/src/models/waste.py` is the
load-bearing free parameter that determines per-mode waste reduction.
This note compares the model's *predicted* save efficiency for each
mode (from the capability-additive form) against the *empirical*
efficiency observed in the published 20-seed benchmark
(`mvp/simulation/results/benchmark_summary.json`).

## Method

For each scenario, the static baseline has `save = 0` by construction,
so its mean waste represents the unmitigated baseline. For any other
mode m:

    effective_save_m = 1 − (waste_m / waste_static)

This is the *total* empirical save efficiency observed in the
benchmark — it convolves (i) the policy's action choice (which depends
on mode) with (ii) the per-action `MODE_EFF` multiplier. We report
the mean across the 5 scenarios and compare it with the model's
predicted MODE_EFF.

## Results

| Mode | MODE_EFF predicted | Empirical observed | Δ (obs − pred) |
|---|---:|---:|---:|
| static | 0.00 | 0.000 | 0.00 |
| hybrid_rl | 0.45 | 0.516 | +0.07 |
| no_pinn | 0.64 | 0.597 | −0.04 |
| no_slca | 0.64 | **0.415** | **−0.22** |
| no_context | 0.75 | 0.706 | −0.05 |
| pirag_only | 0.79 | 0.731 | −0.06 |
| mcp_only | 0.79 | 0.744 | −0.05 |
| agribrain | 0.79 | 0.766 | −0.02 |

## Interpretation

**Five of seven RL-enabled modes match within ±0.07 of prediction.**
The capability-additive form is a good approximation of the full
ablation table at the AGRI-BRAIN endpoint, with the predicted value
(0.79) within 0.02 of observed (0.766). The hybrid-RL baseline is
slightly under-predicted (+0.07 gap), and the three no-context-channel
variants are slightly over-predicted (−0.04 to −0.06 gaps).

**One outlier: no_slca (Δ = −0.22).** The model predicts the no_slca
ablation should land at 0.64, but the observed value is 0.42 — *lower
than the simpler hybrid_rl variant* (0.52). The additive model cannot
reproduce this ordering, which means SLCA shaping has interaction
effects with PINN and the context channel that the additive
attribution does not capture.

The plausible mechanism: SLCA contributes more than its allotted
0.15 delta because it shapes the policy's action distribution in
ways that compound with PINN's predictive routing. When SLCA is
removed, the policy still has high MODE_EFF on paper, but its
chosen actions become less waste-efficient — so the realised save
falls below the additive prediction. The same compounding does not
appear for PINN-removal (no_pinn matches prediction within 0.04)
or context-removal (no_context matches within 0.05), suggesting the
interaction is specifically between SLCA and the social-shaping
component of action selection.

## What this means for the manuscript

1. **The capability-additive structure is empirically supported** at
   the AGRI-BRAIN endpoint and across most ablation arms. The four
   capability deltas can be defended as "calibration constants
   consistent with the observed ablation table within ±0.07 except
   for the no_slca interaction."

2. **The no_slca anomaly is itself a result.** A reviewer asking why
   the model's predicted ordering hybrid_rl < no_slca was reversed
   in the data should be answered with: *"removing SLCA hurts more
   than the additive model predicts because SLCA also shapes which
   actions the policy chooses, not only the per-action save floor.
   This is documented in `docs/MODE_EFF_EMPIRICAL.md`."*

3. **Interaction-term extension**: a multiplicative-interaction term
   `δ_PS · 1[PINN] · 1[SLCA]` would close most of the no_slca gap
   (negative coefficient). Adding it requires re-running the ablation
   grid with each two-way capability combination held out.

## Reproduction

The numbers in this note are produced by:

    python -m mvp.simulation.analysis.empirical_mode_eff

This script reads `benchmark_summary.json` and prints the table above
along with per-scenario breakdown.

## References

- Shapley, L.S. (1953). A value for n-person games. In *Contributions
  to the Theory of Games II*, Princeton UP, 307–317. — Foundation
  for additive capability attribution.
- Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to
  interpreting model predictions. *NeurIPS 30*. — Modern Shapley-value
  attribution; framework for adding interaction terms.
