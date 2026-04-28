"""
Multi-objective reward function for the AGRI-BRAIN decision policy.

The reward balances social lifecycle performance against operational waste:

    R(t) = SLCA_composite(t) − η × waste(t)

where:
    SLCA_composite  = attenuated 4-component social LCA score (see slca.py)
    waste           = net waste fraction after intervention (see waste.py)
    η               = waste penalty coefficient (default 0.50)

Design rationale
----------------
The reward is a linear scalarisation of the two primary objectives.
Linear scalarisation is the simplest member of the multi-objective RL
toolkit (Roijers et al., 2013) and is appropriate when the Pareto
front in objective space is *known to be convex* and the relative
importance of the objectives is fixed at design time. It is *not*
appropriate when the Pareto front is non-convex, because policies
that achieve dominated points on a non-convex front cannot be
recovered by any choice of scalarisation weight (Vamplew et al.,
2011, §4). For AGRI-BRAIN the SLCA-vs-waste front is approximately
convex over the operating range we report (see Figure XX of the
manuscript and the per-η sweep in
tests/test_metric_variants.py::test_eta_sensitivity_ranking), so the
linear form is defensible *for our problem* — but the choice should
be re-examined if the action set or the SLCA scoring rule changes.

Choice of η = 0.50
------------------
η = 0.50 is the median of the swept range {0.10, 0.25, 0.50, 1.00,
2.00}. The AGRI-BRAIN method ranking is invariant across this range
(verified in test_eta_sensitivity_ranking), so the policy comparison
is robust to the exact choice. For deployments where the operator
has a calibrated cost-of-waste in dollars per kg, η should be set to
that cost expressed in the same units as the SLCA composite.

Reasons the linear form is retained:

1. Transparency: a linear reward makes it easy to interpret why the
   policy chose a particular action.
2. Tunable: the single parameter η controls the waste-vs-social
   trade-off. Higher η → more waste-averse policy.
3. Sufficient under convexity: as discussed above, on a convex front
   linear scalarisation can recover every Pareto-optimal point.
4. Captures the five primary dimensions: with SLCA already encoding
   carbon, labour, resilience, and transparency as a composite, the
   reward covers operational and social dimensions jointly.

A hypervolume-indicator based evaluation (Zitzler & Thiele, 1998;
Hayes et al., 2022) is the natural extension if a non-convex front
is observed in future scenarios. We treat that as future work.

The optional energy and water penalties (from footprint.py) are tracked
separately for Green AI reporting but do not influence routing decisions,
since their magnitude (~10⁻² J, ~10⁻⁶ L per step) is negligible
compared to the supply chain impacts.

Extended reward (used in backend router for reporting):
    R_ext = SLCA − α_E × energy_J − β_W × water_L − η × waste

References
----------
    - Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An
      Introduction (2nd ed.). MIT Press.
    - Roijers, D.M., Vamplew, P., Whiteson, S. & Dazeley, R. (2013).
      A survey of multi-objective sequential decision-making.
      Journal of Artificial Intelligence Research, 48, 67–113.
    - Vamplew, P., Yearwood, J., Dazeley, R. & Berry, A. (2011).
      Empirical evaluation methods for multiobjective reinforcement
      learning algorithms. Machine Learning, 84(1), 51–80.
    - Hayes, C.F., Rădulescu, R., Bargiacchi, E., Källström, J., et al.
      (2022). A practical guide to multi-objective reinforcement
      learning and planning. Autonomous Agents and Multi-Agent
      Systems, 36(1), 26.
    - Zitzler, E. & Thiele, L. (1998). Multiobjective optimization
      using evolutionary algorithms — A comparative case study. In
      Parallel Problem Solving from Nature V, LNCS 1498, 292–301.
"""
from __future__ import annotations


def compute_reward(
    slca_composite: float,
    waste: float,
    eta: float = 0.50,
) -> float:
    """Compute the single-step multi-objective reward.

    R(t) = SLCA_composite − η × waste

    Parameters
    ----------
    slca_composite : attenuated SLCA composite score for this timestep.
    waste : net waste fraction after intervention.
    eta : waste penalty coefficient (higher → more waste-averse).

    Returns
    -------
    Scalar reward value.
    """
    return slca_composite - eta * waste


def compute_reward_extended(
    slca_composite: float,
    waste: float,
    energy_J: float = 0.0,
    water_L: float = 0.0,
    eta: float = 0.50,
    alpha_E: float = 0.05,
    beta_W: float = 0.03,
) -> dict[str, float]:
    """Compute the extended reward with Green AI penalty decomposition.

    R_ext = SLCA − α_E × energy − β_W × water − η × waste

    Parameters
    ----------
    slca_composite : attenuated SLCA composite score.
    waste : net waste fraction.
    energy_J : energy consumed by inference (Joules).
    water_L : water consumed by inference (Litres).
    eta : waste penalty coefficient.
    alpha_E : energy penalty coefficient.
    beta_W : water penalty coefficient.

    Returns
    -------
    dict with ``total``, ``slca``, ``energy_penalty``, ``water_penalty``,
    ``waste_penalty`` keys.
    """
    energy_penalty = alpha_E * energy_J
    water_penalty = beta_W * water_L
    waste_penalty = eta * waste
    total = slca_composite - energy_penalty - water_penalty - waste_penalty

    return {
        "slca": round(slca_composite, 4),
        "energy_penalty": round(energy_penalty, 6),
        "water_penalty": round(water_penalty, 8),
        "waste_penalty": round(waste_penalty, 4),
        "total": round(total, 4),
    }
