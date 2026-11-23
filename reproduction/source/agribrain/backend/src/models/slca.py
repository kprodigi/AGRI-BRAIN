"""
4-component social-performance proxy for short- vs. long-chain comparison.

Honest framing
--------------
This module implements a stylised social-performance scoring rule that
sits in the same conceptual space as a Social Life-Cycle Assessment
(SLCA) but is *not* a UNEP/SETAC SLCA. UNEP/SETAC (2020) and the
Roundtable for Product Social Metrics (Goedkoop et al., 2018) require
indicator-level measurement against an audited inventory; we instead score
each routing action against four author-declared base values. The literature
motivates attention to social indicators but does not establish the numerical
route ordering used here.

The base values below encode an explicit *scenario assumption*
(``local_redistribute > recovery > cold_chain`` on every social pillar) and
should not be read as measurements or as values reported by the cited reviews.
Fresh-result reporting must include the prespecified sensitivity analysis;
``tests/test_metric_variants.py::test_slca_scores_remain_bounded_under_weight_swap``
checks software boundedness under one weight swap but is not empirical
sensitivity evidence. For
work that requires absolute social-performance levels, these priors must be
replaced with an inventory-backed assessment appropriate to the geography,
sector, stakeholders, and functional unit. The present scoring is labelled a
"sustainability/social-performance proxy" in the manuscript.

Components
----------
C  - Inverse modeled-emissions term: C = max(0, 1 - carbon_kg / carbon_cap)
     Normalised inverse modeled-emissions proxy evaluated for one standardized routing
     opportunity. Carbon_kg is computed by ``carbon.py`` from the selected
     action's route distance, an author-declared vehicle-kilometre emission
     factor, and a thermal multiplier. ``carbon_cap`` has the same per-routing-
     opportunity time basis; it is not an episode cap. No payload or tonne-
     kilometre term is modelled. The factor is a benchmark assumption, not a
     value attributed to a specific EPA table.

L  - Labour-practice prior : Author-declared route constants only:
     cold_chain=0.60, local_redistribute=0.82, recovery=0.70.
     No labour inventory or observed labour outcome is modeled.

R  - Community-network prior: Author-declared route constants only:
     cold_chain=0.55, local_redistribute=0.78, recovery=0.72.
     No community-level effect is estimated.

P  - Price-information prior: Author-declared route constants only:
     cold_chain=0.55, local_redistribute=0.78, recovery=0.68.
     No price-transparency audit or consumer outcome is modeled.

Composite:
    S = w_c*C + w_l*L + w_r*R + w_p*P
with default weights  w_c=0.30, w_l=0.20, w_r=0.25, w_p=0.25.
The weights are author-specified and close to equal. Carbon is modelled from
transport activity; none of the four pillars is a field measurement.

References
----------
    - UNEP (2020). Guidelines for Social Life Cycle Assessment of
      Products and Organizations. UNEP, Paris.
    - Goedkoop, M., Indrane, D., de Beer, I. (2018). Product Social
      Impact Assessment Handbook 2018. Roundtable for Product Social
      Metrics, Amersfoort.
    - Benoît-Norris, C., Vickery-Niederman, G., Valdivia, S., Franze,
      J., Traverso, M., Ciroth, A. & Mazijn, B. (2011). Introducing
      the UNEP/SETAC methodological sheets for subcategories of
      social LCA. International Journal of Life Cycle Assessment,
      16(7), 682–690.
    - Arcese, G., Lucchetti, M.C., Massa, I. & Valente, C. (2018).
      State of the art in S-LCA: integrating literature review and
      automatic text analysis. International Journal of Life Cycle
      Assessment, 23(3), 394–405.
    - Iofrida, N., Strano, A., Gulisano, G. & De Luca, A.I. (2018).
      Why social life cycle assessment is struggling in development?
      International Journal of Life Cycle Assessment, 23(2), 201–203.
    - Petti, L., Serreli, M. & Di Cesare, S. (2018). Systematic
      literature review in social life cycle assessment.
      International Journal of Life Cycle Assessment, 23(3), 422–431.
    - Eisfeldt, F. & Ciroth, A. (2017). PSILCA — A Product Social
      Impact Life Cycle Assessment database, Version 2. GreenDelta
      GmbH, Berlin.
    - Krüger, S., Eisfeldt, F. & Ciroth, A. (2024). PSILCA database
      for social life cycle assessment: worker hours vs. raw values
      approach. International Journal of Life Cycle Assessment,
      29(11), 2129–2144.
    - GreenDelta (2025). PSILCA v4.0 Product Social Impact Life
      Cycle Assessment Database — Manual. GreenDelta GmbH, Berlin.
    - World Resources Institute & World Business Council for
      Sustainable Development (2004). The Greenhouse Gas Protocol:
      A Corporate Accounting and Reporting Standard, Revised Edition.
    - U.S. Environmental Protection Agency (2023). Emission Factors
      for Greenhouse Gas Inventories. EPA Climate Leaders.
"""
from __future__ import annotations

from math import isfinite
from typing import Dict, Mapping, Optional

from .action_aliases import resolve_action as _resolve_action


DEFAULT_CARBON_CAP_KG_PER_ROUTING_OPPORTUNITY: float = 50.0
"""Author-declared carbon normalizer for one routing opportunity, not an episode."""


# Per-action base scores keyed by canonical action family.
# See module docstring for the declared synthetic definition of each value.
#
# These values are an explicit synthetic scenario design. UNEP/SETAC and the
# cited reviews motivate inventory-based social assessment but do not prescribe
# these numerical gaps. Deployment requires measured, inventory-backed inputs.
_ACTION_BASES: Dict[str, Dict[str, float]] = {
    "cold_chain":         {"L": 0.60, "R": 0.55, "P": 0.55},
    "local_redistribute": {"L": 0.82, "R": 0.78, "P": 0.78},
    "recovery":           {"L": 0.70, "R": 0.72, "P": 0.68},
}


def slca_score(
    carbon_kg: float,
    action: str = "cold_chain",
    *,
    w_c: float = 0.30,
    w_l: float = 0.20,
    w_r: float = 0.25,
    w_p: float = 0.25,
    carbon_cap: float = DEFAULT_CARBON_CAP_KG_PER_ROUTING_OPPORTUNITY,
    fairness: Optional[float] = None,
    resilience: Optional[float] = None,
    transparency: Optional[float] = None,
    action_bases: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, float]:
    """Compute the four-component sustainability/social-performance proxy.

    Parameters
    ----------
    carbon_kg : modelled kg CO2-eq for the action at one standardized routing
        opportunity. This is not cumulative episode emissions.
    action : routing decision string (resolved via alias table).
    w_c, w_l, w_r, w_p : component weights (should sum to 1).
    carbon_cap : per-routing-opportunity denominator for carbon normalisation
        (default 50 kg).
        Provides good dynamic range across action distances:
        cold_chain (120 km × 0.12 = 14.4 kg) → C ≈ 0.71,
        local_redistribute (45 km × 0.12 = 5.4 kg) → C ≈ 0.89,
        recovery (80 km × 0.12 = 9.6 kg) → C ≈ 0.81.
    fairness, resilience, transparency :
        Optional overrides for L, R, P (use per-action defaults when None).

    Returns
    -------
    dict with keys ``C``, ``L``, ``R``, ``P``, ``composite``, ``action_family``.
    """
    carbon_kg = float(carbon_kg)
    carbon_cap = float(carbon_cap)
    if not isfinite(carbon_kg) or carbon_kg < 0.0:
        raise ValueError("carbon_kg must be finite and non-negative")
    if not isfinite(carbon_cap) or carbon_cap <= 0.0:
        raise ValueError("carbon_cap must be finite and positive")

    family = _resolve_action(action)
    if action_bases is None:
        bases = _ACTION_BASES[family]
    else:
        bases = action_bases[family]

    # Author-defined carbon-normalization term.
    C = max(0.0, 1.0 - carbon_kg / carbon_cap)
    L = fairness if fairness is not None else bases["L"]
    R = resilience if resilience is not None else bases["R"]
    P = transparency if transparency is not None else bases["P"]

    # Author-defined weighted composite. The UNEP S-LCA guidelines do not
    # prescribe these pillars, weights, or route-specific base scores.
    composite = w_c * C + w_l * L + w_r * R + w_p * P
    composite = float(max(0.0, min(1.0, composite)))

    return {
        "C": round(C, 4),
        "L": round(L, 4),
        "R": round(R, 4),
        "P": round(P, 4),
        "composite": round(composite, 4),
        "action_family": family,
    }
