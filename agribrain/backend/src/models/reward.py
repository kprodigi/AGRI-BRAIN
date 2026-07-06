"""
Multi-objective reward function for the AGRI-BRAIN decision policy.

The reward balances social lifecycle performance against operational
waste *and* route-conditioned spoilage risk:

    R(t) = SLCA_composite(t) − η_w × waste(t) − η_ρ × ρ_eff(t)

where:
    SLCA_composite  = attenuated 4-component social LCA score (see slca.py)
    waste           = net waste fraction after intervention (see waste.py)
    ρ_eff           = temperature-conditional route-effective spoilage
                      risk in [0, 1]
                       = route_rho_factor(chosen_action, T_amb) × ρ_env
                      The route factor is the per-step ambient-exposure
                      fraction the chosen action's transit envelope
                      carries (resilience.route_rho_factor):
                        - cold_chain at T_amb < 30 degC: 0.15 (nominal
                          cold-chain integrity, Mercier 2017)
                        - cold_chain at 30-35 degC:      0.40 (stressed)
                        - cold_chain at T_amb > 35 degC: 1.00 (overwhelmed)
                        - local_redistribute (any T):    0.45 (short
                          dwell, partial cooling - Ndraha 2018)
                        - recovery (any T):              0.00 (leaves
                          retail-bound pool entirely)
                      The policy gradient on the rho channel is
                      therefore not just action-conditioned but also
                      *condition-conditioned*: under nominal ambient,
                      cold chain is strictly preferred for retail-pool
                      quality; only when the cold chain becomes
                      stressed or overwhelmed does the rho gradient
                      tip toward local_redistribute or recovery. This
                      matches realistic supply-chain physics: cold
                      chain literally exists to lower retail rho, and
                      the AGRI-BRAIN value proposition is composite-
                      ARI optimisation (carbon, labour, resilience,
                      price) rather than raw rho minimisation.
    η_w             = waste penalty coefficient (default 0.50)
    η_ρ             = spoilage-risk penalty coefficient (default 0.50)

When called without an action (or with route_factor=None), the reward
falls back to the previous form using the unfactored env_rho. This
keeps backward compatibility with un-migrated call sites.

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
is observed in future scenarios.

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
    route_factor: float | None = None,
) -> float:
    """Compute the single-step multi-objective reward.

    R(t) = SLCA_composite − η_w × waste − η_ρ × ρ_eff

    where ρ_eff = route_factor × rho when route_factor is provided, else
    ρ_eff = rho (legacy un-migrated callers).

    Parameters
    ----------
    slca_composite : attenuated SLCA composite score for this timestep.
    waste : net waste fraction after intervention.
    rho : environmental spoilage risk for this timestep, in [0, 1].
          Defaults to 0.0 so legacy callers produce the previous
          reward values.
    eta : waste penalty coefficient (higher → more waste-averse).
    eta_rho : spoilage-risk penalty coefficient (higher → more
              spoilage-averse).
    route_factor : optional thermal-exposure factor for the action
        chosen this step. The caller computes this via
        ``resilience.route_rho_factor(action, ambient_temp_c)``: a
        temperature-conditional value that returns 0.15 / 0.40 /
        1.00 for cold_chain at nominal / stressed / overwhelmed
        ambient, 0.45 (constant) for local_redistribute, and 0.00
        for recovery. When supplied, the rho penalty is route- and
        condition-conditioned. When omitted, falls back to the
        unfactored env_rho for backward compatibility.

    Returns
    -------
    Scalar reward value.
    """
    rho_eff = rho if route_factor is None else float(route_factor) * rho
    return slca_composite - eta * waste - eta_rho * rho_eff


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
    route_factor: float | None = None,
) -> dict[str, float]:
    """Compute the extended reward with Green AI penalty decomposition.

    R_ext = SLCA − α_E × energy − β_W × water − η_w × waste − η_ρ × ρ_eff

    where ρ_eff is route-conditioned when ``route_factor`` is supplied
    (see :func:`compute_reward`).

    Parameters
    ----------
    slca_composite : attenuated SLCA composite score.
    waste : net waste fraction.
    rho : environmental spoilage risk for this timestep, in [0, 1].
          Defaults to 0.0 for backward compatibility with un-migrated
          callers.
    energy_J : energy consumed by inference (Joules).
    water_L : water consumed by inference (Litres).
    eta : waste penalty coefficient.
    eta_rho : spoilage-risk penalty coefficient.
    alpha_E : energy penalty coefficient.
    beta_W : water penalty coefficient.
    route_factor : optional value from
        ``resilience.route_rho_factor(action, ambient_temp_c)``
        (temperature-conditional). When supplied,
        ρ_eff = route_factor * rho.

    Returns
    -------
    dict with ``total``, ``slca``, ``energy_penalty``, ``water_penalty``,
    ``waste_penalty``, ``rho_penalty``, ``route_factor`` keys.
    """
    rho_eff = rho if route_factor is None else float(route_factor) * rho
    energy_penalty = alpha_E * energy_J
    water_penalty = beta_W * water_L
    waste_penalty = eta * waste
    rho_penalty = eta_rho * rho_eff
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
        "route_factor": float(route_factor) if route_factor is not None else 1.0,
        "total": round(total, 4),
    }
