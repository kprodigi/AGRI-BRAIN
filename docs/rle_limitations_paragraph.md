# RLE Limitations Paragraph (Draft)

Drop-in text for §Limitations or §Discussion of the AgriBrain manuscript,
explaining the choice of EU-hierarchy + severity-weighted Reverse
Logistics Efficiency (RLE) as the canonical metric and the operational
frictions that remain absent from the simulator.

---

## Short version (≈110 words)

We report Reverse Logistics Efficiency (RLE) as the EU 2008/98/EC
waste-hierarchy + severity-weighted form proposed by
Papargyropoulou et al. (2014). Each at-risk timestep (ρ > 0.10) is
weighted both by spoilage severity ρ and by the chosen action's tier
in the directive (``local_redistribute`` = 1.00, ``recovery`` = 0.40,
``cold_chain`` = 0.00). This avoids the saturation pathology of the
naïve ``recovered / at_risk`` binary form, which scores any always-
reroute policy at 1.0 regardless of which reroute action is chosen.
Earlier drafts of this codebase also exposed match-quality and
capacity-constrained variants; both have been retired in favour of
the single hierarchy-weighted form, whose action weights are the only
ones derived from a peer-reviewed regulatory hierarchy rather than
from author choices.

## Longer version (≈260 words, recommended for §Limitations)

We define Reverse Logistics Efficiency as the severity- and
hierarchy-weighted form

> RLE = Σ_t [ρ(t) · w(a_t) · 1[ρ(t) > θ]] /
>       Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

with at-risk threshold θ = 0.10 (the standard 10 %-quality-loss
inflection in postharvest physiology) and action weights
w(local_redistribute) = 1.00, w(recovery) = 0.40, w(cold_chain) = 0.00
reflecting the human-consumption-first ordering of the EU Waste
Framework Directive (2008/98/EC, Article 4) as operationalised in
Papargyropoulou et al. (2014) and reaffirmed in Garcia-Garcia et al.
(2017). The 1.00 / 0.40 weight ratio encodes the *ranking* prescribed
by the directive; sensitivity to the recovery weight in [0.20, 0.60]
is exercised in
``tests/test_metric_variants.py::test_rle_distinguishes_redistribute_from_recovery``.
Under this metric the score reaches 1.0 only when every at-risk
timestep is routed to ``local_redistribute``; a policy that uniformly
chooses ``recovery`` lands at exactly the 0.40 ratio.

We rejected three alternative RLE forms. The naïve binary
``recovered / at_risk`` saturates at 1.0 for any always-reroute policy
and cannot distinguish ``local_redistribute`` from ``recovery``. A
match-quality form scoring tier-vs-severity match per timestep
introduced three additional band-edge parameters (LR-full, LR-zero,
Recovery-base) that the literature does not pin down. A
capacity-constrained form scoring the realised rather than chosen
routing required modelling LR / Recovery intake capacities for which
we do not have reliable empirical anchors. The hierarchy-weighted
form is the only variant whose action weights are externally
defensible without further calibration.

The simulator does not model real-world rerouting frictions —
finite reroute capacity, scheduling lock-outs, the cost of executing
a reroute, or sensor-noise-induced misclassification of borderline
batches. Under any of these frictions a deployed system would observe
RLE < 1, with the gap growing in the magnitude of the friction. We
treat the headline RLE as the value achievable under perfect-
feasibility operational assumptions and leave a full Recovery Value
Index in the Govindan, Soleimani & Kannan (2015) and Steeneck & Sarin
(2018) family, with empirically-calibrated per-action recovery
factors, as future work — it would require cold-chain reverse-
logistics datasets we do not have.

## Suggested citations

- European Parliament and Council of the European Union (2008).
  Directive 2008/98/EC on waste (Waste Framework Directive),
  Article 4 — waste hierarchy.
- Papargyropoulou, E., Lozano, R., Steinberger, J. K., Wright, N. &
  Ujang, Z. (2014). The food waste hierarchy as a framework for the
  management of food surplus and food waste. *Journal of Cleaner
  Production*, 76, 106–115.
- Garcia-Garcia, G., Woolley, E., Rahimifard, S., Colwill, J., White,
  R. & Needham, L. (2017). A methodology for sustainable management
  of food waste. *Waste and Biomass Valorization*, 8, 2209–2227.
- Govindan, K., Soleimani, H. & Kannan, D. (2015). Reverse logistics
  and closed-loop supply chain: A comprehensive review to explore the
  future. *European Journal of Operational Research*, 240(3), 603–626.
- Steeneck, D. W. & Sarin, S. C. (2018). Pricing and production
  planning for reverse logistics: a review. *IIE Transactions*, 49(7),
  693–712.
- Hodges, R. J., Buzby, J. C. & Bennett, B. (2011). Postharvest
  losses and waste in developed and less developed countries:
  opportunities to improve resource use. *Journal of Agricultural
  Science*, 149(S1), 37–45.
