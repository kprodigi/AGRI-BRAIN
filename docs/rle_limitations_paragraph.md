# RLE Limitations Paragraph (Draft)

Drop-in text for §Limitations or §Discussion of the AgriBrain manuscript,
addressing the known saturation behaviour of the binary Reverse Logistics
Efficiency metric and introducing the severity- and hierarchy-weighted
variant ``compute_rle_weighted`` that ships alongside it as a robustness
check.

---

## Short version (≈110 words)

The headline Reverse Logistics Efficiency (RLE) metric is the standard
binary fraction of at-risk batches (ρ > 0.10) routed to either
local-redistribution or recovery rather than the cold chain. By
construction this metric saturates at 1.0 for any policy that always
intervenes on at-risk batches and cannot distinguish action quality
once the rerouting threshold is met. AgriBrain's RLE = 1 reflects
this saturation: the learned policy correctly reroutes every at-risk
step under the simulator's unconstrained-feasibility assumption.
Alongside the binary form we now report a severity- and
hierarchy-weighted variant (RLE_w) grounded in the EU 2008/98/EC food
waste hierarchy, which does not saturate and rewards selecting the
top-of-hierarchy ``local_redistribute`` action.

## Longer version (≈260 words, recommended for §Limitations)

The headline Reverse Logistics Efficiency metric follows the standard
binary form: the fraction of at-risk batches (spoilage risk ρ > 0.10)
that the policy routes to local-redistribution or recovery rather
than leaving them on the cold-chain path. This formulation has two
limitations that are visible in our experiments and worth flagging
explicitly.

First, the metric saturates at 1.0 for any policy that *always*
reroutes at-risk batches, regardless of *which* reroute action is
chosen. A policy that sends every at-risk batch to recovery
(compost / feed / biofuel) scores identically to a policy that
correctly chooses local-redistribution while batches are still
marketable, even though the two policies recover very different
amounts of economic and nutritional value. AgriBrain reaches RLE ≈
1.0 in our experiments because the learned policy reroutes every
at-risk step; this is the *correct* behaviour under our simulator's
assumption of unconstrained rerouting feasibility, but the binary
metric does not reward picking the *right* reroute.

Second, the simulator does not model real-world rerouting frictions —
finite reroute capacity, scheduling lock-outs, the cost of executing
a reroute, or sensor-noise-induced misclassification of borderline
batches. Under any of these frictions a deployed system would observe
RLE < 1, with the gap growing in the magnitude of the friction.

To address the first limitation we additionally report a
severity- and hierarchy-weighted variant (``compute_rle_weighted``):

> RLE_w = Σ_t [ρ(t) · w(a_t) · 1[ρ(t) > θ]] /
>         Σ_t [ρ(t) · w_max · 1[ρ(t) > θ]]

with action weights w(local_redistribute) = 1.00, w(recovery) = 0.40,
w(cold_chain) = 0.00 reflecting the human-consumption-first ordering
of the EU Waste Framework Directive (2008/98/EC, Article 4), as
operationalised by Papargyropoulou et al. (2014). RLE_w reaches 1.0
only when every at-risk timestep is sent to ``local_redistribute``;
a policy that uniformly chooses ``recovery`` lands at the documented
0.40 ratio. The 1.00 / 0.40 weight ratio encodes the *ranking*
prescribed by the directive; sensitivity to the recovery weight in
[0.20, 0.60] is exercised in
``tests/test_metric_variants.py::test_weighted_rle_distinguishes_redistribute_from_recovery``,
and the rank ordering of methods under the weighted metric agrees
with the ordering under the binary metric across this range.

The second limitation — operational frictions absent from the
simulator — is left as future work. A full Recovery Value Index in
the Govindan, Soleimani & Kannan (2015) and Steeneck & Sarin (2018)
family, with empirically-calibrated per-action recovery factors as a
function of ρ, would require cold-chain reverse-logistics datasets we
do not have. We treat the binary RLE in the headline tables as a
coarse upper bound under perfect-feasibility operational assumptions,
and the weighted RLE_w as a robustness companion that is sensitive to
action quality.

## Suggested citations

- Govindan, K., Soleimani, H. & Kannan, D. (2015). Reverse logistics
  and closed-loop supply chain: A comprehensive review to explore the
  future. *European Journal of Operational Research*, 240(3), 603–626.
- Steeneck, D. W. & Sarin, S. C. (2018). Pricing and production
  planning for reverse logistics: a review. *IIE Transactions*, 49(7),
  693–712.
- European Parliament and Council of the European Union (2008).
  Directive 2008/98/EC on waste (Waste Framework Directive),
  Article 4 — waste hierarchy.
- Hodges, R. J., Buzby, J. C. & Bennett, B. (2011). Postharvest
  losses and waste in developed and less developed countries:
  opportunities to improve resource use. *Journal of Agricultural
  Science*, 149(S1), 37–45.
