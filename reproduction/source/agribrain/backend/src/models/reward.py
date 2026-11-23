"""Transparent synthetic reward used by the AGRI-BRAIN policy learner.

The implemented single-step reward is

    R(t) = S_proxy(t) - eta_w * waste(t) - eta_rho * rho_env(t),

where ``S_proxy`` is the author-defined sustainability/social-performance
proxy and ``rho_env`` is the common latent mechanistic environmental risk.
The default coefficients (eta_w = eta_rho = 0.50) and proxy values are
simulation choices; they are not field-calibrated costs. A legacy optional
``route_factor`` remains available for explicitly exploratory callers, but the
confirmatory benchmark does not supply it.

This linear scalarisation is intentionally inspectable and tunable. It is not
claimed to equal the multiplicative Adaptive Resilience Index, to recover an
unknown Pareto frontier, or to represent monetary welfare. The ``rho`` default
and legacy ``slca_composite`` argument name are retained for API compatibility.

Optional energy and water terms are available in ``compute_reward_extended``
for decomposition/reporting. They do not establish a life-cycle inventory.

References
----------
    - Roijers, D.M., Vamplew, P., Whiteson, S. & Dazeley, R. (2013).
      A survey of multi-objective sequential decision-making.
      Journal of Artificial Intelligence Research, 48, 67–113.
    - Hayes, C.F., Rădulescu, R., Bargiacchi, E., Källström, J., et al.
      (2022). A practical guide to multi-objective reinforcement
      learning and planning. Autonomous Agents and Multi-Agent
      Systems, 36(1), 26.
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

    R(t) = S_proxy − η_w × waste − η_ρ × ρ_env

    The confirmatory path uses ``rho`` directly. If an explicitly exploratory
    caller supplies ``route_factor``, the last term instead uses
    ``route_factor × rho`` for backward compatibility.

    Parameters
    ----------
    slca_composite : sustainability/social-performance proxy for this timestep
        (legacy parameter name).
    waste : net waste fraction after intervention.
    rho : environmental spoilage risk for this timestep, in [0, 1].
          Defaults to 0.0 so legacy callers produce the previous
          reward values.
    eta : waste penalty coefficient (higher → more waste-averse).
    eta_rho : spoilage-risk penalty coefficient (higher → more
              spoilage-averse).
    route_factor : optional *exploratory* thermal-exposure factor for the action
        chosen this step. The caller computes this via
        ``resilience.route_rho_factor(action, ambient_temp_c)``: a
        temperature-conditional value that returns 0.15 / 0.40 /
        1.00 for cold_chain at nominal / stressed / overwhelmed
        ambient, 0.20 / 0.45 / 0.65 / 0.85 for local_redistribute
        across its cool / nominal / stressed / hot bands, and 0.00
        for recovery. When supplied, the rho penalty is route- and
        condition-conditioned. This option is excluded from the confirmatory
        publication path. When omitted, the common environmental rho is used.

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

    R_ext = S_proxy − α_E × energy − β_W × water − η_w × waste − η_ρ × ρ_eff

    The confirmatory form uses the common environmental rho. The optional
    route-conditioned branch is retained only for exploratory compatibility
    (see :func:`compute_reward`).

    Parameters
    ----------
    slca_composite : sustainability/social-performance proxy (legacy name).
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
