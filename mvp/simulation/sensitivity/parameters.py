"""Declared parameter registry for the structural sensitivity study.

The bounds below are deterministic stress envelopes.  They are intentionally
not labelled as confidence intervals, probability supports, priors, or
population distributions: this synthetic benchmark has no calibration data
that would justify any such interpretation.  ``basis`` records whether a
range was already declared by the policy model or is an author-specified
proportional envelope used only to test structural stability.

Only quantities with an auditable path into the live simulator are included.
Retired neural-residual parameters, neutral mode-specific outcome multipliers,
and paper-only compliance multipliers are explicitly excluded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


ValueKind = Literal["float", "integer"]


@dataclass(frozen=True)
class SourceReference:
    """One static link in a parameter's production-code influence path."""

    path: str
    token: str
    purpose: str


@dataclass(frozen=True)
class StructuralParameter:
    """One independently stratified LHS coordinate."""

    key: str
    default: float
    lower: float
    upper: float
    unit: str
    kind: ValueKind
    group: str
    target: str
    basis: str
    rationale: str
    source_references: tuple[SourceReference, ...]

    def transform(self, unit_coordinate: float) -> float | int:
        """Map one coordinate in ``[0, 1)`` to the declared factor range."""

        u = float(unit_coordinate)
        if not 0.0 <= u < 1.0:
            raise ValueError(f"unit coordinate for {self.key!r} must be in [0, 1)")
        value = self.lower + u * (self.upper - self.lower)
        if self.kind == "integer":
            # Inclusive integer endpoints.  The LHS remains exact in the
            # stored unit coordinate; ties after quantisation are expected and
            # are handled as tied ranks by the sensitivity analysis.
            span = int(round(self.upper)) - int(round(self.lower)) + 1
            return min(int(round(self.upper)), int(round(self.lower)) + int(u * span))
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_POLICY = "agribrain/backend/src/models/policy.py"
_GEN = "mvp/simulation/generate_results.py"
_WASTE = "agribrain/backend/src/models/waste.py"
_CARBON = "agribrain/backend/src/models/carbon.py"
_SLCA = "agribrain/backend/src/models/slca.py"
_ACTION = "agribrain/backend/src/models/action_selection.py"
_CONTEXT = "agribrain/backend/pirag/context_to_logits.py"
_COORD = "agribrain/backend/src/agents/coordinator.py"
_STOCH = "mvp/simulation/stochastic.py"


def _refs(*items: tuple[str, str, str]) -> tuple[SourceReference, ...]:
    return tuple(SourceReference(*item) for item in items)


PARAMETERS: tuple[StructuralParameter, ...] = (
    # Mechanistic Arrhenius-lag model.  k_ref and Ea/R use the already-declared
    # 20% and 14% model-error amplitudes as symmetric structural bounds, not as
    # distributional coverage statements.  The other two use a broad 0.5-1.5x
    # envelope because no empirical calibration interval exists.
    StructuralParameter(
        "spoilage_k_ref", 0.0021, 0.00168, 0.00252, "h^-1", "float",
        "mechanistic_spoilage", "policy.k_ref", "existing declared ±20% model-error amplitude",
        "Tests the uncalibrated dry-condition reference decay rate while remaining positive.",
        _refs((_POLICY, "k_ref: float", "declared policy field"),
              (_GEN, "policy.k_ref", "episode-level Arrhenius parameter")),
    ),
    StructuralParameter(
        "spoilage_ea_over_r", 8000.0, 6880.0, 9120.0, "K", "float",
        "mechanistic_spoilage", "policy.Ea_R", "existing declared ±14% model-error amplitude",
        "Tests temperature sensitivity of the common Arrhenius trajectory without a neural residual.",
        _refs((_POLICY, "Ea_R: float", "declared policy field"),
              (_GEN, "policy.Ea_R", "episode-level Arrhenius parameter")),
    ),
    StructuralParameter(
        "humidity_coupling", 0.25, 0.125, 0.375, "dimensionless", "float",
        "mechanistic_spoilage", "policy.beta_humidity", "author structural envelope (0.5-1.5x)",
        "Tests the declared water-activity coupling while preserving a non-negative acceleration.",
        _refs((_POLICY, "beta_humidity: float", "declared policy field"),
              (_GEN, "policy.beta_humidity", "Arrhenius humidity argument")),
    ),
    StructuralParameter(
        "lag_lambda_hours", 12.0, 6.0, 18.0, "hours", "float",
        "mechanistic_spoilage", "policy.lag_lambda", "author structural envelope (0.5-1.5x)",
        "Tests the declared rational lag timescale without implying Baranyi-model calibration.",
        _refs((_POLICY, "lag_lambda: float", "declared policy field"),
              (_GEN, "policy.lag_lambda", "spoilage integration argument")),
    ),

    # Waste mapping and action-saving assumptions.  These are common physical
    # equations for all modes; no mode label appears in the varied factors.
    StructuralParameter(
        "waste_exposure_scale", 10.2976, 7.7232, 12.8720, "effective hours", "float",
        "waste", "waste.W_SCALE", "author structural envelope (±25%)",
        "Perturbs the uncalibrated rate-to-batch-loss conversion around its two synthetic anchors.",
        _refs((_WASTE, "W_SCALE: float", "declared waste constant"),
              (_WASTE, "k_array * float(w_scale)", "production waste equation"),
              (_GEN, "compute_waste_rate(", "simulation call site")),
    ),
    StructuralParameter(
        "waste_compression_exponent", 0.7339, 0.55, 0.90, "dimensionless", "float",
        "waste", "waste.W_ALPHA", "author sub-linear structural envelope",
        "Keeps the waste mapping sub-linear while testing the shape fitted only to synthetic anchors.",
        _refs((_WASTE, "W_ALPHA: float", "declared waste constant"),
              (_WASTE, ") ** float(w_alpha)", "production waste equation")),
    ),
    StructuralParameter(
        "surplus_waste_factor", 0.25, 0.125, 0.375, "dimensionless", "float",
        "waste", "waste.SURPLUS_WASTE_FACTOR", "author structural envelope (0.5-1.5x)",
        "Tests how strongly handling congestion converts inventory surplus into loss.",
        _refs((_WASTE, "SURPLUS_WASTE_FACTOR: float", "declared waste constant"),
              (_WASTE, "float(surplus_waste_factor) * float(surplus_ratio)", "production waste equation")),
    ),
    StructuralParameter(
        "surplus_save_penalty", 0.10, 0.05, 0.15, "dimensionless", "float",
        "waste", "waste.SURPLUS_SAVE_PENALTY", "author structural envelope (0.5-1.5x)",
        "Tests reciprocal intervention-capacity degradation under surplus.",
        _refs((_WASTE, "SURPLUS_SAVE_PENALTY: float", "declared waste constant"),
              (_WASTE, "surplus_save_penalty * surplus_ratio", "production save equation")),
    ),
    StructuralParameter(
        "local_redistribute_save_floor", 0.45, 0.35, 0.55, "fraction", "float",
        "waste", "waste.SAVE_FLOOR.local_redistribute", "author bounded action-effect envelope",
        "Tests the unmeasured waste-prevention assumption for local redistribution.",
        _refs((_WASTE, '"local_redistribute": 0.45', "declared action floor"),
              (_WASTE, "SAVE_FLOOR.get", "production save lookup")),
    ),
    StructuralParameter(
        "recovery_save_floor", 0.25, 0.15, 0.35, "fraction", "float",
        "waste", "waste.SAVE_FLOOR.recovery", "author bounded action-effect envelope",
        "Tests the unmeasured waste-prevention assumption for recovery routing.",
        _refs((_WASTE, '"recovery": 0.25', "declared action floor"),
              (_WASTE, "SAVE_FLOOR.get", "production save lookup")),
    ),

    # Route and emissions assumptions.  Distances and factors are physical
    # inputs shared by every mode; retired MODE_CARBON_EFF values are excluded.
    StructuralParameter(
        "km_coldchain", 120.0, 90.0, 150.0, "km", "float", "route_carbon",
        "policy.km_coldchain", "author route envelope (±25%)",
        "Tests sensitivity to the declared synthetic central cold-chain route length.",
        _refs((_POLICY, "km_coldchain: float", "declared route field"),
              (_GEN, "ACTION_KM_KEYS[action]", "selected route lookup")),
    ),
    StructuralParameter(
        "km_local", 45.0, 33.75, 56.25, "km", "float", "route_carbon",
        "policy.km_local", "author route envelope (±25%)",
        "Tests sensitivity to the declared synthetic local-redistribution distance.",
        _refs((_POLICY, "km_local: float", "declared route field"),
              (_GEN, "ACTION_KM_KEYS[action]", "selected route lookup")),
    ),
    StructuralParameter(
        "km_recovery", 80.0, 60.0, 100.0, "km", "float", "route_carbon",
        "policy.km_recovery", "author route envelope (±25%)",
        "Tests sensitivity to the declared synthetic recovery-route distance.",
        _refs((_POLICY, "km_recovery: float", "declared route field"),
              (_GEN, "ACTION_KM_KEYS[action]", "selected route lookup")),
    ),
    StructuralParameter(
        "transport_carbon_factor", 0.12, 0.09, 0.15, "kg CO2-eq/km", "float",
        "route_carbon", "policy.carbon_per_km", "author inventory-factor envelope (±25%)",
        "Tests an explicitly uncalibrated fleet emission factor without mode-specific multipliers.",
        _refs((_POLICY, "carbon_per_km: float", "declared emissions field"),
              (_GEN, "policy.carbon_per_km", "production carbon call")),
    ),
    StructuralParameter(
        "refrigeration_cop_penalty", 0.40, 0.20, 0.60, "fraction", "float",
        "route_carbon", "carbon.REFRIG_COP_PENALTY", "author bounded thermal envelope",
        "Tests the declared full-stress refrigeration penalty while keeping emissions non-negative.",
        _refs((_CARBON, "REFRIG_COP_PENALTY: float", "declared carbon constant"),
              (_CARBON, 'values["cop_penalty"] * values["thermal_stress"]', "production carbon equation")),
    ),

    # SLCA weights use three independent coordinates and derive w_p as the
    # exact remainder.  These deliberately narrower ranges keep all four
    # effective weights inside their Policy-declared ranges for every LHS row.
    StructuralParameter(
        "slca_weight_carbon", 0.30, 0.275, 0.325, "weight", "float", "slca",
        "policy.w_c", "bounded simplex-preserving envelope",
        "Varies the carbon pillar while the runner derives w_p so the four weights sum exactly to one.",
        _refs((_POLICY, "w_c: float", "declared pillar weight"),
              (_GEN, "w_c=policy.w_c", "production SLCA call")),
    ),
    StructuralParameter(
        "slca_weight_labour", 0.20, 0.175, 0.225, "weight", "float", "slca",
        "policy.w_l", "bounded simplex-preserving envelope",
        "Varies the labour pillar without treating stakeholder weights as measured data.",
        _refs((_POLICY, "w_l: float", "declared pillar weight"),
              (_GEN, "w_l=policy.w_l", "production SLCA call")),
    ),
    StructuralParameter(
        "slca_weight_resilience", 0.25, 0.225, 0.275, "weight", "float", "slca",
        "policy.w_r", "bounded simplex-preserving envelope",
        "Varies the community-resilience pillar; price transparency is the exact remainder.",
        _refs((_POLICY, "w_r: float", "declared pillar weight"),
              (_GEN, "w_r=policy.w_r", "production SLCA call")),
    ),
    StructuralParameter(
        "slca_carbon_cap", 50.0, 40.0, 70.0,
        "kg CO2-eq per routing opportunity", "float", "slca",
        "policy.carbon_cap", "Policy-documented structural range",
        "Tests the author-specified denominator used to normalise the carbon pillar; it is not an emissions limit.",
        _refs((_POLICY, "carbon_cap: float", "declared carbon-normalisation field"),
              (_GEN, "carbon_cap=policy.carbon_cap", "production SLCA call")),
    ),
    StructuralParameter(
        "social_score_contrast_scale", 1.0, 0.50, 1.50, "multiplier", "float",
        "slca", "slca._ACTION_BASES contrast", "author contrast envelope (0.5-1.5x)",
        "Contracts or expands action-specific L/R/P score gaps around each pillar mean while preserving their order.",
        _refs((_SLCA, "_ACTION_BASES", "declared action-specific social scores"),
              (_SLCA, 'bases = _ACTION_BASES[family]', "production SLCA lookup")),
    ),

    # Active policy/reward assumptions whose ranges are already declared in
    # Policy descriptions.
    StructuralParameter(
        "waste_reward_penalty", 0.50, 0.30, 1.00, "coefficient", "float",
        "policy", "policy.eta", "Policy-declared range",
        "Tests the learned policy's trade-off between the social proxy and waste.",
        _refs((_POLICY, "eta: float", "declared reward field"),
              (_GEN, "eta=policy.eta", "production reward call")),
    ),
    StructuralParameter(
        "risk_reward_penalty", 0.50, 0.30, 1.00, "coefficient", "float",
        "policy", "policy.eta_rho", "Policy-declared range",
        "Tests the learned policy's direct common-risk penalty.",
        _refs((_POLICY, "eta_rho: float", "declared reward field"),
              (_GEN, "eta_rho=policy.eta_rho", "production reward call")),
    ),
    StructuralParameter(
        "bollinger_window_steps", 20.0, 10.0, 30.0, "15-minute steps", "integer",
        "policy", "policy.boll_window", "Policy-declared range",
        "Tests the demand-volatility memory length; integer ties are retained honestly.",
        _refs((_POLICY, "boll_window: int", "declared volatility field"),
              (_GEN, 'getattr(policy, "boll_window"', "production rolling window")),
    ),
    StructuralParameter(
        "bollinger_z_threshold", 2.0, 1.5, 3.0, "standard deviations", "float",
        "policy", "policy.boll_k", "Policy-declared range",
        "Tests the declared volatility-trigger threshold without interpreting it as calibrated coverage.",
        _refs((_POLICY, "boll_k: float", "declared volatility field"),
              (_GEN, "float(policy.boll_k)", "production anomaly rule")),
    ),
    StructuralParameter(
        "volatility_tilt_coldchain", 0.25, 0.10, 0.80, "logit", "float",
        "policy", "policy.gamma_coldchain", "Policy-declared range",
        "Tests the declared positive cold-chain tilt under detected volatility.",
        _refs((_POLICY, "gamma_coldchain: float", "declared tilt field"),
              (_ACTION, "policy.gamma_coldchain", "production policy logits")),
    ),
    StructuralParameter(
        "volatility_tilt_local", 0.05, -0.50, 0.50, "logit", "float",
        "policy", "policy.gamma_local", "Policy-declared range",
        "Tests the declared near-neutral local-routing tilt.",
        _refs((_POLICY, "gamma_local: float", "declared tilt field"),
              (_ACTION, "policy.gamma_local", "production policy logits")),
    ),
    StructuralParameter(
        "volatility_tilt_recovery", -0.25, -1.50, 0.0, "logit", "float",
        "policy", "policy.gamma_recovery", "Policy-declared range",
        "Tests the non-positive recovery tilt while retaining the documented sign constraint.",
        _refs((_POLICY, "gamma_recovery: float", "declared tilt field"),
              (_ACTION, "policy.gamma_recovery", "production policy logits")),
    ),

    StructuralParameter(
        "context_prior_scale", 1.0, 0.50, 1.50, "multiplier", "float",
        "context", "pirag.THETA_CONTEXT", "author prior-magnitude envelope (0.5-1.5x)",
        "Tests context-weight initialization magnitude while preserving every declared sign and zero pattern.",
        _refs((_CONTEXT, "THETA_CONTEXT: np.ndarray", "declared context prior"),
              (_COORD, "THETA_CONTEXT", "production coordinator consumption")),
    ),
    StructuralParameter(
        "stochastic_noise_scale", 1.0, 0.50, 1.50, "multiplier", "float",
        "stochastic", "all nonzero canonical stochastic amplitudes", "author global amplitude envelope (0.5-1.5x)",
        "Jointly scales the ten nonzero declared stochastic amplitudes, keeping their relative calibration fixed.",
        _refs((_STOCH, "_CANONICAL_STOCH_DEFAULTS", "single source of stochastic defaults"),
              (_GEN, "make_stochastic_layer(", "production stochastic construction")),
    ),
)


# Explicit exclusions make it impossible for an old manuscript list to be
# mistaken for the implemented design. Some entries are retired concepts;
# others remain compatibility fields that are neutral in the publication path.
EXCLUDED_PARAMETERS: dict[str, str] = {
    "frozen PINN training hyperparameters": (
        "The synthetic PINN is trained and frozen before the confirmatory run; "
        "its training choices are not retuned inside structural sensitivity."
    ),
    "MODE_EFF and mode-specific effective-spoilage multipliers": (
        "The live outcome equation is mode-neutral; compatibility exports are zero."
    ),
    "MODE_CARBON_EFF and mode-specific carbon multipliers": (
        "The live transport equation uses eff_factor=1.0 for every mode."
    ),
    "compliance multipliers": (
        "Context can alter actions but cannot change a fixed action's physical outcome."
    ),
    "policy-temperature heterogeneity": (
        "The locked confirmatory protocol declares this optional source disabled; "
        "the sensitivity design does not activate a zero-default mechanism."
    ),
}


def registry_as_dict() -> dict[str, Any]:
    """Return a JSON-serializable declaration of factors and exclusions."""

    return {
        "analysis_label": "structural sensitivity",
        "probability_interpretation": False,
        "range_semantics": (
            "deterministic bounded stress envelopes for a space-filling design; "
            "not probability distributions or confidence intervals"
        ),
        "parameters": [parameter.to_dict() for parameter in PARAMETERS],
        "excluded_parameters": dict(EXCLUDED_PARAMETERS),
    }


def validate_parameter_registry(repo_root: Path | str) -> dict[str, Any]:
    """Validate uniqueness, bounds, and every declared static code link.

    This is a fail-closed source audit, not a claim of empirical calibration.
    Dynamic production-function probes live in :mod:`.overrides` and are run by
    the focused test suite and the CLI ``validate`` command.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")
    keys = [parameter.key for parameter in PARAMETERS]
    if len(keys) != len(set(keys)):
        raise ValueError("structural parameter keys must be unique")
    targets = [parameter.target for parameter in PARAMETERS]
    if len(targets) != len(set(targets)):
        raise ValueError("structural parameter targets must be unique")

    checked_refs = 0
    for parameter in PARAMETERS:
        if not parameter.lower < parameter.default < parameter.upper:
            raise ValueError(
                f"{parameter.key}: expected lower < default < upper, got "
                f"{parameter.lower}, {parameter.default}, {parameter.upper}"
            )
        if not parameter.source_references:
            raise ValueError(f"{parameter.key}: no production source references")
        for reference in parameter.source_references:
            path = root / reference.path
            if not path.is_file():
                raise ValueError(f"{parameter.key}: missing source file {path}")
            source = path.read_text(encoding="utf-8")
            if reference.token not in source:
                raise ValueError(
                    f"{parameter.key}: token {reference.token!r} not found in "
                    f"{reference.path} ({reference.purpose})"
                )
            checked_refs += 1
    return {
        "status": "pass",
        "n_parameters": len(PARAMETERS),
        "n_source_references_checked": checked_refs,
        "probability_interpretation": False,
    }


def parameter_by_key(key: str) -> StructuralParameter:
    for parameter in PARAMETERS:
        if parameter.key == key:
            return parameter
    raise KeyError(key)


def validate_parameter_values(values: dict[str, Any]) -> dict[str, float | int]:
    """Validate one complete design row and return canonical typed values."""

    expected = {parameter.key for parameter in PARAMETERS}
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"parameter keys do not match registry: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    out: dict[str, float | int] = {}
    for parameter in PARAMETERS:
        value = values[parameter.key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{parameter.key}: value must be numeric")
        numeric = float(value)
        if not parameter.lower <= numeric <= parameter.upper:
            raise ValueError(
                f"{parameter.key}: {numeric} outside [{parameter.lower}, {parameter.upper}]"
            )
        if parameter.kind == "integer":
            if not numeric.is_integer():
                raise ValueError(f"{parameter.key}: value must be an integer")
            out[parameter.key] = int(numeric)
        else:
            out[parameter.key] = numeric
    # Exact compositional constraint used by the runner.
    w_p = 1.0 - (
        float(out["slca_weight_carbon"])
        + float(out["slca_weight_labour"])
        + float(out["slca_weight_resilience"])
    )
    if not 0.15 <= w_p <= 0.35:
        raise ValueError(f"derived SLCA price-transparency weight {w_p} is invalid")
    return out


def derived_values(values: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic dependent values that are not LHS dimensions."""

    checked = validate_parameter_values(values)
    w_p = 1.0 - (
        float(checked["slca_weight_carbon"])
        + float(checked["slca_weight_labour"])
        + float(checked["slca_weight_resilience"])
    )
    return {
        "slca_weight_price_transparency": float(w_p),
        "slca_weight_sum": 1.0,
        "stochastic_scale_excludes_zero_default_policy_temperature": True,
    }
