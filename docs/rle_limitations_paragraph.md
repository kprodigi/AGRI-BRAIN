# RLE Limitations Paragraph (Draft)

Drop-in text for §Limitations or §Discussion of the AgriBrain manuscript,
addressing the known saturation behaviour of the binary Reverse Logistics
Efficiency metric.

---

## Short version (≈90 words)

The Reverse Logistics Efficiency (RLE) metric used throughout this work is
the standard binary fraction of at-risk batches (ρ > 0.10) routed to
either local-redistribution or recovery rather than the cold chain. By
construction the metric saturates at 1.0 for any policy that always
intervenes on at-risk batches, and therefore cannot distinguish between
action choices once the rerouting threshold is met. AgriBrain's RLE = 1
in our experiments reflects this saturation: the policy correctly reroutes
every at-risk batch under the simulator's assumption of unconstrained
rerouting feasibility.

## Longer version (≈200 words, recommended for §Limitations)

The Reverse Logistics Efficiency metric we report follows the standard
binary form: the fraction of at-risk batches (spoilage risk ρ > 0.10)
that the policy routes to local-redistribution or recovery rather than
leaving them on the cold-chain path. This formulation has two
limitations that are visible in our experiments and worth flagging
explicitly.

First, the metric saturates at 1.0 for any policy that *always* reroutes
at-risk batches, regardless of *which* reroute action is chosen. A
policy that sends every at-risk batch to recovery (compost / feed /
biofuel) scores identically to a policy that correctly chooses
local-redistribution while batches are still marketable, even though the
two policies recover very different amounts of economic and nutritional
value. AgriBrain reaches RLE ≈ 1.0 in our experiments because the
learned policy reroutes every at-risk step; this is the *correct*
behaviour under our simulator's assumption of unconstrained rerouting
feasibility, but the metric itself does not reward picking the
*right* reroute.

Second, the simulator does not model real-world rerouting frictions —
finite reroute capacity, scheduling lock-outs, the cost of executing
a reroute, or sensor-noise-induced misclassification of borderline
batches. Under any of these frictions a deployed system would observe
RLE < 1, with the gap growing in the magnitude of the friction.

A value-weighted variant in the Recovery Value Index / Effective
Recovery Rate family (Govindan, Soleimani & Kannan, 2015; Steeneck &
Sarin, 2018), reflecting the cascading-use hierarchy of the EU Waste
Framework Directive (2008/98/EC, Article 4), would distinguish
local-redistribution from recovery and weight contributions by spoilage
severity, addressing the saturation but introducing free parameters
(per-action recovery factors as a function of ρ) that would themselves
require empirical calibration from cold-chain reverse-logistics studies.
We leave this calibration and the corresponding value-weighted RLE to
future work, and treat the binary metric in this paper as a coarse
upper bound on what the policy could achieve under perfect-feasibility
operational assumptions.

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
