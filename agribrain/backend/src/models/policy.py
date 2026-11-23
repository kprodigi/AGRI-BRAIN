"""
Extended Policy configuration model.

Backward-compatible with the original fields (max_temp_c, min_shelf_reroute,
min_shelf_expedite, carbon_per_km, km_farm_to_dc, km_dc_to_retail,
km_expedited, msrp) while adding all paper-derived parameters.

Defaults define a synthetic case study unless explicitly supplied by a user.
They are not field-calibrated estimates or regulatory thresholds.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .slca import DEFAULT_CARBON_CAP_KG_PER_ROUTING_OPPORTUNITY


class Policy(BaseModel):
    # ---- original fields (backward compatible) ----
    max_temp_c: float = Field(
        8.0,
        description="Declared synthetic cold-chain warning threshold (°C); "
        "not a legal limit or product-validation result.",
    )
    min_shelf_reroute: float = Field(
        0.70,
        description="Declared synthetic remaining-quality trigger for reroute consideration.",
    )
    min_shelf_expedite: float = Field(
        0.50,
        description="Declared synthetic remaining-quality trigger for expedited handling.",
    )
    carbon_per_km: float = Field(
        0.12,
        description="Declared synthetic kg CO2-eq per vehicle-km; replace with a "
        "fleet- and region-specific inventory factor before deployment.",
    )
    km_farm_to_dc: float = Field(
        280.0, description="Farm-to-distribution-center distance (km). Legacy field."
    )
    km_dc_to_retail: float = Field(
        50.0, description="Distribution-center-to-retail distance (km). Legacy field."
    )
    km_expedited: float = Field(
        160.0, description="Expedited route distance (km). Legacy field."
    )
    msrp: float = Field(1.50, description="Manufacturer suggested retail price (USD/unit).")

    # ---- sustainability/social-proxy weights ----
    # These are author-declared synthetic policy weights, not values estimated
    # from stakeholders or prescribed by an S-LCA standard.
    w_c: float = Field(0.30, description="Proxy weight: carbon reduction. Range: 0.20-0.40.")
    w_l: float = Field(0.20, description="Proxy weight: labour fairness. Range: 0.10-0.30.")
    w_r: float = Field(0.25, description="Proxy weight: community resilience. Range: 0.15-0.35.")
    w_p: float = Field(0.25, description="Proxy weight: price transparency. Range: 0.15-0.35.")

    # ---- waste penalty ----
    eta: float = Field(
        0.50,
        description="Waste penalty coefficient in the reward function. "
        "Controls the trade-off between social-proxy improvement and waste reduction. "
        "Range: 0.3-1.0. Higher values make the policy more waste-averse.",
    )

    # ---- spoilage-risk penalty ----
    eta_rho: float = Field(
        0.50,
        description="Spoilage-risk penalty coefficient in the reward "
        "function. Penalises high rho directly (rho enters the reward "
        "as -eta_rho * rho), closing the previous gap where the reward "
        "had no rho term while ARI = (1-waste)*SLCA*(1-rho) does. "
        "Range: 0.3-1.0. Higher values make the policy more "
        "spoilage-averse.",
    )

    # ---- energy / water penalty coefficients ----
    alpha_E: float = Field(0.05, description="Energy penalty coefficient for Green AI tracking.")
    beta_W: float = Field(0.03, description="Water penalty coefficient for Green AI tracking.")

    # ---- Bollinger volatility parameters ----
    boll_window: int = Field(
        20,
        description="Bollinger rolling window size (number of 15-min steps = 5 hours). "
        "Chosen to capture intra-day demand patterns. Range: 10-30.",
    )
    boll_k: float = Field(
        2.0,
        description="Bollinger z-score threshold for anomaly detection. "
        "2.0 corresponds to ~95% confidence interval. Range: 1.5-3.0.",
    )

    # ---- volatility tilt parameters ----
    # When volatility is detected (tau=1), these shift the softmax logits.
    # Positive gamma_coldchain tilts toward cold-chain continuation during
    # volatility in this synthetic policy.
    #
    # The signs and magnitudes are declared synthetic policy priors. They are
    # not coefficients estimated from the real-options literature.
    gamma_coldchain: float = Field(
        0.25,
        description="Volatility tilt toward cold-chain (positive = prefer CC under volatility). "
        "Kept small to avoid over-conservative routing during demand noise, which "
        "would degrade social-proxy scores and ARI. Range: 0.1-0.8.",
    )
    gamma_local: float = Field(
        0.05,
        description="Volatility tilt for local redistribution (near-neutral: local markets "
        "are relatively adaptive to demand changes). Range: -0.5 to 0.5.",
    )
    gamma_recovery: float = Field(
        -0.25,
        description="Volatility tilt for recovery (discouraged under volatility "
        "since recovery capacity may be strained). Range: -1.5 to 0.0.",
    )

    # ---- Mechanistic spoilage-risk parameters (Arrhenius form) ----
    # The Arrhenius model k(T) = k_ref * exp[Ea_R * (1/T_ref - 1/T_K)] is
    # the standard in food science for temperature-dependent quality loss
    # (Labuza & Riboh, 1982; Giannakourou & Taoukis, 2003).
    k_ref: float = Field(
        0.0021,
        description="Declared synthetic reference rate at T_ref_K (h^-1); not "
        "fitted to observed spinach quality labels.",
    )
    Ea_R: float = Field(
        8000.0,
        description="Arrhenius activation energy / gas constant (K). "
        "Ea_R = Ea/R where Ea is in J/mol and R = 8.314 J/(mol·K). "
        "8000 K corresponds to Ea ≈ 66.5 kJ/mol and is a declared benchmark "
        "parameter rather than a fitted spinach-specific value.",
    )
    T_ref_K: float = Field(
        277.15,
        description="Declared reference temperature in Kelvin (= 4.0°C).",
    )
    beta_humidity: float = Field(
        0.25,
        description="Declared synthetic humidity-coupling coefficient for the "
        "modelled-risk equation.",
    )
    lag_lambda: float = Field(
        12.0,
        description="Declared synthetic lag-shape parameter (hours); not fitted "
        "to spinach observations.",
    )

    # ---- route distances ----
    # Declared synthetic network distances; replace with a study-region network.
    km_coldchain: float = Field(
        120.0,
        description="Declared synthetic cold-chain route distance (km).",
    )
    km_local: float = Field(
        45.0,
        description="Declared synthetic local-redistribution distance (km).",
    )
    km_recovery: float = Field(
        80.0,
        description="Declared synthetic recovery-route distance (km).",
    )

    # ---- SLCA carbon normalization ----
    # Implementation note: provenance.
    # The 50 kg cap is a per-standardized-routing-opportunity normalization
    # constant for the SLCA carbon component; it is NOT an emissions claim or
    # an episode cap. It was chosen so a cold-chain opportunity (~14.4 kg
    # CO2eq) maps to C ~= 0.71, a local-redistribution opportunity (~5.4 kg)
    # maps to C ~= 0.89, and a recovery opportunity (~9.6 kg) maps to C ~=
    # 0.81 — i.e., so the C component
    # produces a ~20-percentage-point spread across actions, comparable
    # to the L/R/P components. The "why 50?" question is answered by:
    # the absolute value is a sensitivity-free monotone rescaling of
    # carbon_kg; the figures report rankings, and any cap in [40, 70]
    # preserves both the rank order and the qualitative gap between
    # actions. Verified via spot-check at 30 and 80.
    carbon_cap: float = Field(
        DEFAULT_CARBON_CAP_KG_PER_ROUTING_OPPORTUNITY,
        gt=0.0,
        description="Carbon normalization cap (kg CO2-eq per standardized "
        "routing opportunity) for the SLCA carbon component; not an episode "
        "cap. C = max(0, 1 - carbon_kg/carbon_cap). "
        "Default 50 provides good dynamic range across actions. "
        "Range: 20-100.",
    )

    # ---- compatibility-safe research feature flags ----
    enable_mcp_qos_routing: bool = Field(
        False,
        description="Enable QoS-aware MCP routing; false keeps legacy static dispatch.",
    )
    enable_mcp_reliability: bool = Field(
        False,
        description="Enable MCP retry/fallback/circuit-breaker reliability layer.",
    )
    enable_pirag_counterfactual_eval: bool = Field(
        False,
        description=(
            "Enable the alternative-query retrieval diagnostic; the legacy "
            "field name does not denote removal of a retrieved document or a "
            "causal counterfactual."
        ),
    )
    enable_physics_consistency_gate: bool = Field(
        False,
        description="Enable additional passage-level physics consistency gating.",
    )
    enable_heterogeneous_profiles: bool = Field(
        False,
        description="Enable heterogeneous role profiles for multi-model experiments.",
    )
    enable_temporal_retrieval_weighting: bool = Field(
        True,
        description="Enable temporal recency weighting in context retrieval.",
    )
    enable_dynamic_knowledge_feedback: bool = Field(
        False,
        description=(
            "Enable periodic decision-history ingestion back into the KB. "
            "Disabled by default as of 2026-04 because the re-ingested "
            "documents are autogenerated summary statistics of the agent's "
            "own past actions (see pirag/dynamic_knowledge.py); leaving "
            "this on creates a self-amplification loop where retrieval "
            "biases toward whatever the system has been doing. Enable "
            "explicitly via the env var DYNAMIC_KB_FEEDBACK=true for "
            "ablation studies that report with/without the loop."
        ),
    )
    enable_failure_injection: bool = Field(
        False,
        description="Enable simulation-time MCP fault injection for robustness studies.",
    )
    enable_research_metrics: bool = Field(
        False,
        description="Enable additional research metrics/artifact outputs.",
    )

    # ---- destination capacity and acceptance criteria ----
    # Declared synthetic design values; see models/destination_capacity.py.
    # With ``destination_capacity_enabled=False`` every dispatch is accepted
    # in full, which is the outcome model without destination constraints.
    destination_capacity_enabled: bool = Field(
        False,
        description="Enforce destination capacity and acceptance criteria.",
    )
    capacity_kappa_lr: float = Field(
        0.5, ge=0.0,
        description="Redistribution absorption per decision interval at "
        "reference demand, in reference dispatches.",
    )
    capacity_kappa_rec: float = Field(
        0.25, ge=0.0,
        description="Recovery absorption per decision interval, in reference "
        "dispatches.",
    )
    capacity_bmax_lr: float = Field(
        8.0, ge=0.0,
        description="Maximum redistribution capacity, in reference dispatches.",
    )
    capacity_bmax_rec: float = Field(
        4.0, ge=0.0,
        description="Maximum recovery capacity, in reference dispatches.",
    )
    capacity_demand_ref: float = Field(
        15.0, gt=0.0,
        description="Reference demand (units per decision interval) for "
        "redistribution absorption.",
    )
    capacity_inventory_ref: float = Field(
        12_000.0, gt=0.0,
        description="Reference inventory (units) defining one reference dispatch.",
    )
    acceptance_rho_at: float = Field(
        0.10, ge=0.0, le=1.0,
        description="At-risk threshold: recovery accepts product above it.",
    )
    acceptance_rho_cut: float = Field(
        0.50, ge=0.0, le=1.0,
        description="Marketability cutoff: redistribution accepts product at "
        "or below it.",
    )
    acceptance_strict_redistribution: bool = Field(
        False,
        description="Sensitivity variant: redistribution also requires "
        "at-risk product.",
    )
    refusal_penalty_scale: float = Field(
        1.0, ge=0.0,
        description="Scale of the diverted-leg penalty applied to the "
        "refused share of a dispatch.",
    )
    refused_share_lost: bool = Field(
        False,
        description="If true, none of the refused share of a dispatch is "
        "sold: its waste fraction is 1.",
    )
    capacity_report_age_steps: int = Field(
        0, ge=0,
        description="Age, in decision intervals, of the capacity reports that "
        "reach the decision side (0 = current).",
    )
    published_rho_at: float | None = Field(
        None, ge=0.0, le=1.0,
        description="At-risk threshold stated in the acceptance-policy "
        "document; None publishes the threshold the destinations apply.",
    )
    published_rho_cut: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Marketability cutoff stated in the acceptance-policy "
        "document; None publishes the cutoff the destinations apply.",
    )
    query_scenario_terms: bool = Field(
        True,
        description="Append scenario search terms to the primary retrieval "
        "query, as in the submitted design. The revised design never uses "
        "them; False removes them from the submitted design as well.",
    )
    destination_acceptance_enabled: bool = Field(
        True,
        description="Destinations apply their acceptance criteria. False "
        "makes every destination accept any product, which with unlimited "
        "capacity is the outcome model without destination constraints.",
    )
    reward_objective: str = Field(
        "linear",
        pattern="^(linear|step_performance)$",
        description="Learning signal: the linear scalarization of the "
        "submitted design, or the step value of the performance index.",
    )
    revised_context_design: bool = Field(
        False,
        description="Use the revised context design: constraint and "
        "eligibility context features, capacity tool, retrieved acceptance "
        "policy, feasibility guard, no scenario terms in queries. False keeps "
        "the submitted design.",
    )
    missing_confirmation_closed: bool = Field(
        True,
        description="When a capacity report or the acceptance criteria were "
        "requested but could not be obtained: do not divert (True, "
        "fail-closed), or leave the route available and let the destination "
        "enforce its limits (False, fail-open).",
    )
    reservation_share: float = Field(
        1.0, gt=0.0, le=1.0,
        description="Share of the dispatch for which the reservation guard "
        "requires reported room.",
    )
    acceptance_document_present: bool = Field(
        True,
        description="Whether the acceptance document is in the corpus.",
    )
    acceptance_document_id: str | None = Field(
        None,
        description="Identifier under which the acceptance document is "
        "stored; None uses the default identifier. The criteria are read "
        "from the content of the retrieved passage, not from its identifier.",
    )
