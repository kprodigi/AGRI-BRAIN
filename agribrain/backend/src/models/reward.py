"""
Multi-objective reward function for the AGRI-BRAIN decision policy.

The reward balances social lifecycle performance against operational
waste *and* spoilage risk:

    R(t) = SLCA_composite(t) − η_w × waste(t) − η_ρ × ρ(t)

where:
    SLCA_composite  = attenuated 4-component social LCA score (see slca.py)
    waste           = net waste fraction after intervention (see waste.py)
    ρ               = spoilage risk in [0, 1] (see spoilage.py / resilience.py)
    η_w             = waste penalty coefficient (default 0.50)
    η_ρ             = spoilage-risk penalty coefficient (default 0.50)

Design rationale
----------------
The reward is a linear scalarisation of the three primary objectives.
Linear scalarisation is the simplest member of the multi-objective RL
toolkit (Roijers et al., 2013) and is appropriate when the Pareto
front in objective space is *known to be convex* and the relative
importance of the objectives is fixed at design time. It is *not*
appropriate when the Pareto front is non-convex, because policies
that achieve dominated points on a non-convex front cannot be
recovered by any choice of scalarisation weight (Vamplew et al.,
2011, §4). For AGRI-BRAIN the (SLCA, waste, ρ) front is approximately
convex over the operating range we report, so the linear form is
defensible *for our problem* — but the choice should be re-examined
if the action set or the SLCA scoring rule changes.

Relationship to ARI
-------------------
The paper's headline metric is the multiplicative

    ARI(t) = (1 − waste(t)) · SLCA(t) · (1 − ρ(t))
           = SLCA − SLCA·waste − SLCA·ρ + SLCA·waste·ρ

The linear reward above approximates ARI to first order over the
operating range — the missing element is the cross term
``SLCA·waste·ρ``, which in our regime is small (≤ 0.04 in absolute
units when SLCA ≈ 0.7, waste ≤ 0.10, ρ ≤ 0.5). The linear
formulation retains the convex-scalarisation theory while giving the
policy direct, separate gradients on each of the three objectives the
paper grades it on. We do *not* claim the reward equals ARI; we claim
the linear scalarisation closely tracks per-step ARI in our regime,
which is the strongest defensible position under the Roijers et al.
framework.

Choice of η_w = η_ρ = 0.50
--------------------------
η_w = 0.50 is the median of the swept range {0.10, 0.25, 0.50, 1.00,
2.00}. The AGRI-BRAIN method ranking is invariant across this range
(verified in test_eta_sensitivity_ranking).

η_ρ defaults to the same 0.50 so the per-step gradient signal on ρ
is comparable in magnitude to the gradient on waste at the operating
ρ scale (waste ≈ 0.02–0.10, ρ ≈ 0.0–0.5; with η_w = η_ρ = 0.50 the
two penalty terms have similar order of magnitude during heatwave).
A separate η_ρ sensitivity test is exercised in
test_eta_rho_sensitivity_ranking and the method ranking is invariant
across {0.10, 0.25, 0.50, 1.00, 2.00}.

For deployments where the operator has a calibrated cost-of-spoilage
in dollars per kg, η_ρ should be set to that cost expressed in the
same units as the SLCA composite, in parallel with η_w.

Reasons the linear form is retained:

1. Transparency: a linear reward makes it easy to interpret why the
   policy chose a particular action.
2. Tunable: two parameters (η_w, η_ρ) control the waste-vs-spoilage-
   vs-social trade-offs independently.
3. Sufficient under convexity: as discussed above, on a convex front
   linear scalarisation can recover every Pareto-optimal point.
4. Aligned with the headline metric: the new ρ term closes the
   previously-missing connection to ARI.
5. Captures the five primary dimensions: with SLCA already encoding
   carbon, labour, resilience, and transparency as a composite, the
   reward covers operational, social, and freshness dimensions
   jointly.

A hypervolume-indicator based evaluation (Zitzler & Thiele, 1998;
Hayes et al., 2022) is the natural extension if a non-convex front
is observed in future scenarios. We treat that as future work.

Backward compatibility
----------------------
The ``rho`` parameter defaults to 0.0, which makes :func:`compute_reward`
behave identically to the previous SLCA-minus-waste form when callers
have not been updated. This lets existing tests and call sites that
do not yet pass ρ continue to produce the old reward values until they
are migrated.

The optional energy and water penalties (from footprint.py) are tracked
separately for Green AI reporting but do not influence routing decisions,
since their magnitude (~10⁻² J, ~10⁻⁶ L per step) is negligible
compared to the supply chain impacts.

Extended reward (used in backend router for reporting):
    R_ext = SLCA − α_E × energy − β_W × water − η_w × waste − η_ρ × ρ

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
    rho: float = 0.0,
    eta: float = 0.50,
    eta_rho: float = 0.50,
) -> float:
    """Compute the single-step multi-objective reward.

    R(t) = SLCA_composite − η_w × waste − η_ρ × ρ

    Parameters
    ----------
    slca_composite : attenuated SLCA composite score for this timestep.
    waste : net waste fraction after intervention.
    rho : spoilage risk for this timestep, in [0, 1]. Defaults to 0.0
          so callers that have not yet been migrated to the
          ρ-penalised form produce the previous reward values.
    eta : waste penalty coefficient (higher → more waste-averse).
    eta_rho : spoilage-risk penalty coefficient (higher → more
              spoilage-averse).

    Returns
    -------
    Scalar reward value.
    """
    return slca_composite - eta * waste - eta_rho * rho


def compute_reward_extended(
    slca_composite: float,
    waste: float,
    rho: float = 0.0,
    energy_J: float = 0.0,
    water_L: float = 0.0,
    eta: float = 0.50,
    eta_rho: float = 0.50,
    alpha_E: float = 0.05,
    beta_W: float = 0.03,
) -> dict[str, float]:
    """Compute the extended reward with Green AI penalty decomposition.

    R_ext = SLCA − α_E × energy − β_W × water − η_w × waste − η_ρ × ρ

    Parameters
    ----------
    slca_composite : attenuated SLCA composite score.
    waste : net waste fraction.
    rho : spoilage risk for this timestep, in [0, 1]. Defaults to 0.0
          for backward compatibility with un-migrated callers.
    energy_J : energy consumed by inference (Joules).
    water_L : water consumed by inference (Litres).
    eta : waste penalty coefficient.
    eta_rho : spoilage-risk penalty coefficient.
    alpha_E : energy penalty coefficient.
    beta_W : water penalty coefficient.

    Returns
    -------
    dict with ``total``, ``slca``, ``energy_penalty``, ``water_penalty``,
    ``waste_penalty``, ``rho_penalty`` keys.
    """
    energy_penalty = alpha_E * energy_J
    water_penalty = beta_W * water_L
    waste_penalty = eta * waste
    rho_penalty = eta_rho * rho
    total = (
        slca_composite
        - energy_penalty
        - water_penalty
        - waste_penalty
        - rho_penalty
    )

    return {
        "slca": round(slca_composite, 4),
        "energy_penalty": round(energy_penalty, 6),
        "water_penalty": round(water_penalty, 8),
        "waste_penalty": round(waste_penalty, 4),
        "rho_penalty": round(rho_penalty, 4),
        "total": round(total, 4),
    }
