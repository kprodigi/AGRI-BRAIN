"""Executable contract for every paper-facing per-step outcome.

The decision ledger records the state and action used by the publication
pipeline.  This module serialises every effective equation parameter beside
those records and independently reconstructs carbon, the raw and attenuated
social-performance proxy, waste, reward, and ARI.  Hash-valid records are not
accepted as scientifically valid unless this reconstruction also succeeds.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import carbon as carbon_model
from . import slca as slca_model
from . import waste as waste_model
from .action_selection import (
    ACTION_KM_KEYS,
    ACTIONS,
    SLCA_SURPLUS_ATTEN,
    SLCA_THERMAL_ATTEN,
    THERMAL_DELTA_MAX,
    THERMAL_T0,
    compute_slca_attenuation,
    compute_thermal_stress,
)
from .carbon import compute_transport_carbon
from .pinn_residual import (
    MAX_RESIDUAL,
    build_residual_feature_row,
    load_frozen_checkpoint,
    predict_residual,
)
from .resilience import compute_ari
from .reward import compute_reward
from .slca import slca_score
from .spoilage import advance_spoilage_risk_midpoint, arrhenius_k
from .synthetic_spoilage_dgp import (
    DEFAULT_PACKAGING_INDEX,
    HANDLING_SHOCK_LOG_RATE_COEFFICIENT,
    PACKAGING_CENTER,
    PACKAGING_LOG_RATE_COEFFICIENT,
    RH_TRANSIENT_LOG_RATE_COEFFICIENT,
    synthetic_dgp_provenance,
)
from .waste import compute_save_factor, compute_waste_rate


OUTCOME_EQUATION_CONTRACT_VERSION = 2
_CONTRACT_TYPE = "agribrain_publication_per_step_outcomes"
_STOCHASTIC_DRAW_ALGORITHM = "agribrain-stochastic-v1/source-counter-keyed"

_OVERRIDE_KEYS = {
    "inventory_baseline",
    "waste_exposure_scale",
    "waste_compression_exponent",
    "waste_cap_fraction",
    "surplus_waste_factor",
    "surplus_save_penalty",
    "action_save_fraction",
    "refrigeration_cop_penalty",
    "physical_efficiency_factor",
    "thermal_reference_c",
    "thermal_range_c",
    "slca_thermal_attenuation",
    "slca_surplus_attenuation",
    "slca_action_bases",
}


def _float(value: Any, *, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} must be a finite number")
    return result


def _action_scalar_map(value: Any, *, where: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(ACTIONS):
        raise ValueError(f"{where} must contain exactly {list(ACTIONS)!r}")
    return {action: _float(value[action], where=f"{where}/{action}") for action in ACTIONS}


def _slca_bases(value: Any, *, where: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(ACTIONS):
        raise ValueError(f"{where} must contain exactly {list(ACTIONS)!r}")
    result: dict[str, dict[str, float]] = {}
    for action in ACTIONS:
        row = value[action]
        if not isinstance(row, Mapping) or set(row) != {"L", "R", "P"}:
            raise ValueError(f"{where}/{action} must contain exactly L, R, and P")
        result[action] = {
            key: _float(row[key], where=f"{where}/{action}/{key}")
            for key in ("L", "R", "P")
        }
    return result


def build_outcome_equation_contract(
    policy: Any,
    *,
    effective_k_ref: float,
    effective_ea_r: float,
    stochastic_layer: Any,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the exact parameters used by one episode's outcome equations."""

    overrides = dict(parameter_overrides or {})
    unexpected = set(overrides) - _OVERRIDE_KEYS
    if unexpected:
        raise ValueError(f"unknown outcome-equation overrides: {sorted(unexpected)}")

    def selected(name: str, default: Any) -> Any:
        return overrides[name] if name in overrides else default

    action_save_fraction = _action_scalar_map(
        selected("action_save_fraction", waste_model.SAVE_FLOOR),
        where="action_save_fraction",
    )
    action_bases = _slca_bases(
        selected("slca_action_bases", slca_model._ACTION_BASES),
        where="slca_action_bases",
    )
    contract = {
        "schema_version": OUTCOME_EQUATION_CONTRACT_VERSION,
        "contract_type": _CONTRACT_TYPE,
        "state_semantics": {
            "temperature": "temp_outcome_environmental",
            "humidity": "rh_outcome_environmental",
            "inventory": "inventory_outcome_environmental",
            "spoilage_risk": "rho_outcome_environmental",
            "transport_multiplier": "transport_multiplier_outcome_environmental",
            "action": "action",
            "mode": "mode",
        },
        "stochastic_effective_parameter_provenance": {
            "enabled": bool(stochastic_layer.enabled),
            "draw_algorithm": _STOCHASTIC_DRAW_ALGORITHM,
            "k_ref_fraction_std": float(stochastic_layer.k_ref_frac_std),
            "ea_r_fraction_std": float(stochastic_layer.ea_r_frac_std),
        },
        "arrhenius": {
            "base_k_ref": float(policy.k_ref),
            "base_ea_over_r": float(policy.Ea_R),
            "effective_k_ref": float(effective_k_ref),
            "effective_ea_over_r": float(effective_ea_r),
            "reference_temperature_k": float(policy.T_ref_K),
            "humidity_coupling": float(policy.beta_humidity),
            "rational_lag_hours": float(policy.lag_lambda),
        },
        "waste": {
            "inventory_baseline": float(selected(
                "inventory_baseline", waste_model.INV_BASELINE,
            )),
            "exposure_scale": float(selected(
                "waste_exposure_scale", waste_model.W_SCALE,
            )),
            "compression_exponent": float(selected(
                "waste_compression_exponent", waste_model.W_ALPHA,
            )),
            "cap_fraction": float(selected(
                "waste_cap_fraction", waste_model.WASTE_CAP,
            )),
            "surplus_waste_factor": float(selected(
                "surplus_waste_factor", waste_model.SURPLUS_WASTE_FACTOR,
            )),
            "surplus_save_penalty": float(selected(
                "surplus_save_penalty", waste_model.SURPLUS_SAVE_PENALTY,
            )),
            "action_save_fraction": action_save_fraction,
        },
        "carbon": {
            "route_km_by_action": {
                action: float(getattr(policy, ACTION_KM_KEYS[action]))
                for action in ACTIONS
            },
            "carbon_per_km": float(policy.carbon_per_km),
            "refrigeration_cop_penalty": float(selected(
                "refrigeration_cop_penalty", carbon_model.REFRIG_COP_PENALTY,
            )),
            "physical_efficiency_factor": float(selected(
                "physical_efficiency_factor", 1.0,
            )),
            "thermal_reference_c": float(selected(
                "thermal_reference_c", THERMAL_T0,
            )),
            "thermal_range_c": float(selected(
                "thermal_range_c", THERMAL_DELTA_MAX,
            )),
        },
        "slca": {
            "weights": {
                "C": float(policy.w_c),
                "L": float(policy.w_l),
                "R": float(policy.w_r),
                "P": float(policy.w_p),
            },
            "carbon_cap": float(policy.carbon_cap),
            "action_bases": action_bases,
            "thermal_attenuation": float(selected(
                "slca_thermal_attenuation", SLCA_THERMAL_ATTEN,
            )),
            "surplus_attenuation": float(selected(
                "slca_surplus_attenuation", SLCA_SURPLUS_ATTEN,
            )),
        },
        "reward": {
            "waste_penalty": float(policy.eta),
            "risk_penalty": float(policy.eta_rho),
            "slca_ablation_mode": "no_slca",
            "slca_value_in_ablation": 0.0,
        },
        "ari": {
            "equation": "(1-waste)*slca_attenuated*(1-rho_outcome_environmental)",
        },
    }
    validate_outcome_equation_contract(contract)
    return contract


def _require_exact_keys(value: Any, expected: set[str], *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{where} schema mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return value


def validate_outcome_equation_contract(
    contract: Any,
    *,
    where: str = "outcome_equation_contract",
    expected_contract: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on a missing, malformed, or unexpected parameter contract."""

    top = _require_exact_keys(contract, {
        "schema_version", "contract_type", "state_semantics",
        "stochastic_effective_parameter_provenance", "arrhenius", "waste",
        "carbon", "slca", "reward", "ari",
    }, where=where)
    if top["schema_version"] != OUTCOME_EQUATION_CONTRACT_VERSION:
        raise ValueError(f"{where} has an unsupported schema version")
    if top["contract_type"] != _CONTRACT_TYPE:
        raise ValueError(f"{where} has the wrong contract type")
    semantics = _require_exact_keys(top["state_semantics"], {
        "temperature", "humidity", "inventory", "spoilage_risk",
        "transport_multiplier", "action", "mode",
    }, where=f"{where}/state_semantics")
    expected_semantics = {
        "temperature": "temp_outcome_environmental",
        "humidity": "rh_outcome_environmental",
        "inventory": "inventory_outcome_environmental",
        "spoilage_risk": "rho_outcome_environmental",
        "transport_multiplier": "transport_multiplier_outcome_environmental",
        "action": "action",
        "mode": "mode",
    }
    if dict(semantics) != expected_semantics:
        raise ValueError(f"{where} changes the locked state semantics")

    stochastic = _require_exact_keys(
        top["stochastic_effective_parameter_provenance"],
        {"enabled", "draw_algorithm", "k_ref_fraction_std", "ea_r_fraction_std"},
        where=f"{where}/stochastic_effective_parameter_provenance",
    )
    if not isinstance(stochastic["enabled"], bool):
        raise ValueError(f"{where} stochastic enabled flag is not boolean")
    if stochastic["draw_algorithm"] != _STOCHASTIC_DRAW_ALGORITHM:
        raise ValueError(f"{where} changes the stochastic draw algorithm")
    for key in ("k_ref_fraction_std", "ea_r_fraction_std"):
        if _float(stochastic[key], where=f"{where}/{key}") < 0.0:
            raise ValueError(f"{where}/{key} must be non-negative")

    arrhenius = _require_exact_keys(top["arrhenius"], {
        "base_k_ref", "base_ea_over_r", "effective_k_ref", "effective_ea_over_r",
        "reference_temperature_k", "humidity_coupling", "rational_lag_hours",
    }, where=f"{where}/arrhenius")
    for key in arrhenius:
        value = _float(arrhenius[key], where=f"{where}/arrhenius/{key}")
        if key not in {"humidity_coupling", "rational_lag_hours"} and value <= 0.0:
            raise ValueError(f"{where}/arrhenius/{key} must be positive")
    if _float(arrhenius["humidity_coupling"], where=f"{where}/arrhenius/humidity_coupling") < 0:
        raise ValueError(f"{where}/arrhenius/humidity_coupling must be non-negative")
    if _float(arrhenius["rational_lag_hours"], where=f"{where}/arrhenius/rational_lag_hours") < 0:
        raise ValueError(f"{where}/arrhenius/rational_lag_hours must be non-negative")

    waste = _require_exact_keys(top["waste"], {
        "inventory_baseline", "exposure_scale", "compression_exponent",
        "cap_fraction", "surplus_waste_factor", "surplus_save_penalty",
        "action_save_fraction",
    }, where=f"{where}/waste")
    for key in (
        "inventory_baseline", "exposure_scale", "compression_exponent", "cap_fraction",
    ):
        if _float(waste[key], where=f"{where}/waste/{key}") <= 0.0:
            raise ValueError(f"{where}/waste/{key} must be positive")
    for key in ("surplus_waste_factor", "surplus_save_penalty"):
        if _float(waste[key], where=f"{where}/waste/{key}") < 0.0:
            raise ValueError(f"{where}/waste/{key} must be non-negative")
    save = _action_scalar_map(
        waste["action_save_fraction"], where=f"{where}/waste/action_save_fraction",
    )
    if any(value < 0.0 or value > 1.0 for value in save.values()):
        raise ValueError(f"{where} action save fractions leave [0,1]")

    carbon = _require_exact_keys(top["carbon"], {
        "route_km_by_action", "carbon_per_km", "refrigeration_cop_penalty",
        "physical_efficiency_factor", "thermal_reference_c", "thermal_range_c",
    }, where=f"{where}/carbon")
    routes = _action_scalar_map(
        carbon["route_km_by_action"], where=f"{where}/carbon/route_km_by_action",
    )
    if any(value < 0.0 for value in routes.values()):
        raise ValueError(f"{where} route distances must be non-negative")
    for key in ("carbon_per_km", "physical_efficiency_factor", "thermal_range_c"):
        if _float(carbon[key], where=f"{where}/carbon/{key}") <= 0.0:
            raise ValueError(f"{where}/carbon/{key} must be positive")
    if _float(
        carbon["refrigeration_cop_penalty"],
        where=f"{where}/carbon/refrigeration_cop_penalty",
    ) < 0.0:
        raise ValueError(f"{where} COP penalty must be non-negative")
    _float(carbon["thermal_reference_c"], where=f"{where}/carbon/thermal_reference_c")

    slca = _require_exact_keys(top["slca"], {
        "weights", "carbon_cap", "action_bases", "thermal_attenuation",
        "surplus_attenuation",
    }, where=f"{where}/slca")
    weights = _require_exact_keys(
        slca["weights"], {"C", "L", "R", "P"}, where=f"{where}/slca/weights",
    )
    numeric_weights = {
        key: _float(value, where=f"{where}/slca/weights/{key}")
        for key, value in weights.items()
    }
    if any(value < 0.0 for value in numeric_weights.values()) or not math.isclose(
        math.fsum(numeric_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(f"{where} SLCA weights must be non-negative and sum to one")
    if _float(slca["carbon_cap"], where=f"{where}/slca/carbon_cap") <= 0.0:
        raise ValueError(f"{where} SLCA carbon cap must be positive")
    bases = _slca_bases(slca["action_bases"], where=f"{where}/slca/action_bases")
    if any(value < 0.0 or value > 1.0 for row in bases.values() for value in row.values()):
        raise ValueError(f"{where} SLCA action bases leave [0,1]")
    for key in ("thermal_attenuation", "surplus_attenuation"):
        if _float(slca[key], where=f"{where}/slca/{key}") < 0.0:
            raise ValueError(f"{where}/slca/{key} must be non-negative")

    reward = _require_exact_keys(top["reward"], {
        "waste_penalty", "risk_penalty", "slca_ablation_mode",
        "slca_value_in_ablation",
    }, where=f"{where}/reward")
    for key in ("waste_penalty", "risk_penalty"):
        if _float(reward[key], where=f"{where}/reward/{key}") < 0.0:
            raise ValueError(f"{where}/reward/{key} must be non-negative")
    if reward["slca_ablation_mode"] != "no_slca" or _float(
        reward["slca_value_in_ablation"],
        where=f"{where}/reward/slca_value_in_ablation",
    ) != 0.0:
        raise ValueError(f"{where} changes the locked no-SLCA reward ablation")
    ari = _require_exact_keys(top["ari"], {"equation"}, where=f"{where}/ari")
    if ari["equation"] != "(1-waste)*slca_attenuated*(1-rho_outcome_environmental)":
        raise ValueError(f"{where} changes the locked ARI equation")

    if expected_contract is not None:
        validate_outcome_equation_contract(expected_contract, where=f"{where}/expected")
        _compare_contract_values(contract, expected_contract, where=where)


def _compare_contract_values(observed: Any, expected: Any, *, where: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{where} differs from the expected parameter contract")
        for key in expected:
            _compare_contract_values(observed[key], expected[key], where=f"{where}/{key}")
        return
    if isinstance(expected, float):
        actual = _float(observed, where=where)
        if not math.isclose(actual, expected, rel_tol=1e-15, abs_tol=1e-15):
            raise ValueError(f"{where}={actual!r}, expected {expected!r}")
        return
    if observed != expected:
        raise ValueError(f"{where}={observed!r}, expected {expected!r}")


def _validated_spoilage_checkpoint(
    provenance: Mapping[str, Any] | None,
    *,
    where: str,
    allow_mechanistic_only: bool,
) -> tuple[Any | None, str]:
    """Validate one declared spoilage estimator and return its checkpoint."""

    if not isinstance(provenance, Mapping):
        raise ValueError(f"{where} must be an object")
    estimator_kind = str(provenance.get("kind", ""))
    if estimator_kind == "mechanistic_plus_frozen_synthetic_pinn_residual":
        if provenance.get("training_target_origin") != "independent_synthetic_dgp":
            raise ValueError(
                f"{where} target origin is not independent synthetic DGP"
            )
        if not math.isclose(
            _float(
                provenance.get("residual_bound_abs"),
                where=f"{where}/residual_bound_abs",
            ),
            MAX_RESIDUAL,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"{where} residual bound mismatch")
        if provenance.get("deployment_transform") != (
            "clip_quality_to_unit_interval_then_cumulative_minimum"
        ):
            raise ValueError(f"{where} deployment transform mismatch")
        if provenance.get("synthetic_only") is not True:
            raise ValueError(f"{where} must label the PINN synthetic-only")
        if provenance.get("external_validation") is not False:
            raise ValueError(f"{where} cannot claim external validation")
        checkpoint = load_frozen_checkpoint()
        if provenance.get("checkpoint_sha256") != checkpoint.checkpoint_sha256:
            raise ValueError(f"{where} checkpoint SHA-256 mismatch")
        if provenance.get("training_dataset_sha256") != checkpoint.dataset_sha256:
            raise ValueError(f"{where} training-data SHA-256 mismatch")
        return checkpoint, estimator_kind

    if estimator_kind == "mechanistic_only_no_pinn" and allow_mechanistic_only:
        if provenance.get("external_validation") is not False:
            raise ValueError(f"{where} cannot claim external validation")
        if provenance.get("checkpoint_sha256") is not None or (
            provenance.get("training_dataset_sha256") is not None
        ):
            raise ValueError(f"{where} no-PINN arm references a checkpoint")
        for key in (
            "training_target_origin", "residual_bound_abs",
            "deployment_transform",
        ):
            if provenance.get(key) is not None:
                raise ValueError(f"{where} no-PINN arm sets {key}")
        return None, estimator_kind

    raise ValueError(f"{where} has unknown kind {estimator_kind!r}")


def _validate_latent_spoilage_model(
    provenance: Mapping[str, Any] | None,
    *,
    parameters: Mapping[str, float],
    where: str,
) -> None:
    """Bind the scored trajectory to the exact common noise-free DGP."""

    expected = synthetic_dgp_provenance(
        k_ref=parameters["k_ref"],
        Ea_R=parameters["Ea_R"],
        T_ref_K=parameters["T_ref_K"],
        beta=parameters["beta"],
        lag_lambda=parameters["lag_lambda"],
        packaging_index=DEFAULT_PACKAGING_INDEX,
    )
    if provenance != expected:
        raise ValueError(
            f"{where} does not identify the locked independent synthetic DGP"
        )


def validate_recorded_spoilage_trajectories(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    spoilage_estimator: Mapping[str, Any] | None = None,
    latent_spoilage_model: Mapping[str, Any] | None = None,
    where: str = "decision_ledger",
    contract_validated: bool = False,
) -> None:
    """Reconstruct the common latent and policy-observed spoilage trajectories.

    Every arm is scored against one independent, noise-free synthetic DGP.
    The mode-specific ``spoilage_estimator`` controls only the policy-observed
    trajectory: residual-enabled arms apply the frozen residual checkpoint,
    while ``no_pinn`` exposes the uncorrected mechanistic estimate.  The
    policy-side residual is never fed recursively into its mechanistic state.
    """
    if not contract_validated:
        validate_outcome_equation_contract(contract, where=f"{where}/contract")
    if not records:
        raise ValueError(f"{where} has no records")
    arrhenius = contract["arrhenius"]
    parameters = {
        "k_ref": float(arrhenius["effective_k_ref"]),
        "Ea_R": float(arrhenius["effective_ea_over_r"]),
        "T_ref_K": float(arrhenius["reference_temperature_k"]),
        "beta": float(arrhenius["humidity_coupling"]),
        "lag_lambda": float(arrhenius["rational_lag_hours"]),
    }
    _validate_latent_spoilage_model(
        latent_spoilage_model,
        parameters=parameters,
        where=f"{where}/latent_spoilage_model",
    )
    policy_checkpoint, policy_kind = _validated_spoilage_checkpoint(
        spoilage_estimator,
        where=f"{where}/spoilage_estimator",
        allow_mechanistic_only=True,
    )
    latent_quality = 1.0
    previous_rh_transient = 0.0
    for index, current in enumerate(records):
        current_hour = _float(
            current.get("hour"), where=f"{where}:{index}/hour",
        )
        current_temp = _float(
            current.get("temp_outcome_environmental"),
            where=f"{where}:{index}/temp_outcome_environmental",
        )
        current_rh = _float(
            current.get("rh_outcome_environmental"),
            where=f"{where}:{index}/rh_outcome_environmental",
        )
        current_shock = _float(
            current.get("shock_g"), where=f"{where}:{index}/shock_g",
        )
        if index > 0:
            previous = records[index - 1]
            previous_hour = _float(
                previous.get("hour"), where=f"{where}:{index - 1}/hour",
            )
            delta_t = current_hour - previous_hour
            if delta_t <= 0.0:
                raise ValueError(
                    f"{where}:{index}/hour is not strictly increasing"
                )
            previous_temp = _float(
                previous.get("temp_outcome_environmental"),
                where=f"{where}:{index - 1}/temp_outcome_environmental",
            )
            previous_rh = _float(
                previous.get("rh_outcome_environmental"),
                where=f"{where}:{index - 1}/rh_outcome_environmental",
            )
            previous_shock = _float(
                previous.get("shock_g"),
                where=f"{where}:{index - 1}/shock_g",
            )
            current_rh_transient = abs(current_rh - previous_rh) / delta_t
            mid_time = 0.5 * (current_hour + previous_hour)
            base_rate = float(arrhenius_k(
                0.5 * (current_temp + previous_temp),
                k_ref=parameters["k_ref"],
                Ea_R=parameters["Ea_R"],
                T_ref_K=parameters["T_ref_K"],
                rh_frac=0.005 * (current_rh + previous_rh),
                beta=parameters["beta"],
            ))
            lag_lambda = parameters["lag_lambda"]
            alpha = (
                mid_time / (mid_time + lag_lambda)
                if lag_lambda > 0.0 else 1.0
            )
            log_multiplier = (
                PACKAGING_LOG_RATE_COEFFICIENT
                * (DEFAULT_PACKAGING_INDEX - PACKAGING_CENTER)
                + HANDLING_SHOCK_LOG_RATE_COEFFICIENT
                * 0.5 * (current_shock + previous_shock)
                + RH_TRANSIENT_LOG_RATE_COEFFICIENT
                * 0.5 * (current_rh_transient + previous_rh_transient)
            )
            latent_quality *= math.exp(
                -base_rate * alpha * math.exp(log_multiplier) * delta_t
            )
            previous_rh_transient = current_rh_transient
        observed_rho = _float(
            current.get("rho_outcome_environmental"),
            where=f"{where}:{index}/rho_outcome_environmental",
        )
        expected_rho = 1.0 - latent_quality
        if not math.isclose(
            observed_rho, expected_rho, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(
                f"{where}:{index}/rho_outcome_environmental violates the "
                "locked independent synthetic DGP trajectory"
            )

    mechanistic_rho = 0.0
    deployed_quality = 1.0
    for index, current in enumerate(records):
        current_hour = _float(
            current.get("hour"), where=f"{where}:{index}/hour",
        )
        current_temp = _float(
            current.get("temp_policy_observed"),
            where=f"{where}:{index}/temp_policy_observed",
        )
        current_rh = _float(
            current.get("rh_policy_observed"),
            where=f"{where}:{index}/rh_policy_observed",
        )
        if index > 0:
            previous = records[index - 1]
            previous_hour = _float(
                previous.get("hour"),
                where=f"{where}:{index - 1}/hour",
            )
            previous_temp = _float(
                previous.get("temp_policy_observed"),
                where=f"{where}:{index - 1}/temp_policy_observed",
            )
            previous_rh = _float(
                previous.get("rh_policy_observed"),
                where=f"{where}:{index - 1}/rh_policy_observed",
            )
            mechanistic_rho = float(advance_spoilage_risk_midpoint(
                mechanistic_rho,
                previous_temp_C=previous_temp,
                current_temp_C=current_temp,
                previous_rh_pct=previous_rh,
                current_rh_pct=current_rh,
                previous_hour=previous_hour,
                current_hour=current_hour,
                **parameters,
            ))
        if policy_checkpoint is None:
            expected_rho = mechanistic_rho
        else:
            if index == 0:
                rh_transient = 0.0
            else:
                step_h = current_hour - previous_hour
                rh_transient = (
                    abs(current_rh - previous_rh) / step_h
                    if step_h > 0.0 else 0.0
                )
            residual_features = build_residual_feature_row(
                time_h=current_hour,
                temp_c=current_temp,
                rh_pct=current_rh,
                shock_g=_float(
                    current.get("shock_g"),
                    where=f"{where}:{index}/shock_g",
                ),
                rh_transient_per_h=rh_transient,
                k_ref=parameters["k_ref"],
                ea_over_r=parameters["Ea_R"],
            )
            delta_quality = float(predict_residual(
                residual_features, policy_checkpoint,
            )[0])
            deployed_quality = min(
                deployed_quality,
                max(0.0, min(1.0, 1.0 - mechanistic_rho + delta_quality)),
            )
            expected_rho = 1.0 - deployed_quality
        observed_rho = _float(
            current.get("rho_policy_observed"),
            where=f"{where}:{index}/rho_policy_observed",
        )
        if not math.isclose(
            observed_rho, expected_rho, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(
                f"{where}:{index}/rho_policy_observed violates the locked "
                f"{policy_kind} trajectory"
            )


def reconstruct_step_outcomes(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    where: str = "record",
    contract_validated: bool = False,
) -> dict[str, Any]:
    """Recompute all paper-facing outcomes from one ledger state/action row."""

    if not contract_validated:
        validate_outcome_equation_contract(contract, where=f"{where}/contract")
    action = record.get("action")
    if action not in ACTIONS:
        raise ValueError(f"{where}/action is not canonical")
    mode = record.get("mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"{where}/mode is missing")
    temperature = _float(
        record.get("temp_outcome_environmental"),
        where=f"{where}/temp_outcome_environmental",
    )
    humidity = _float(
        record.get("rh_outcome_environmental"),
        where=f"{where}/rh_outcome_environmental",
    )
    inventory = _float(
        record.get("inventory_outcome_environmental"),
        where=f"{where}/inventory_outcome_environmental",
    )
    rho = _float(
        record.get("rho_outcome_environmental"),
        where=f"{where}/rho_outcome_environmental",
    )
    transport_multiplier = _float(
        record.get("transport_multiplier_outcome_environmental"),
        where=f"{where}/transport_multiplier_outcome_environmental",
    )
    if humidity < 0.0 or humidity > 100.0 or inventory < 0.0 or not 0.0 <= rho <= 1.0:
        raise ValueError(f"{where} environmental outcome state is outside its domain")
    if transport_multiplier < 0.0:
        raise ValueError(f"{where} transport multiplier is negative")

    waste_parameters = contract["waste"]
    carbon_parameters = contract["carbon"]
    slca_parameters = contract["slca"]
    arrhenius_parameters = contract["arrhenius"]
    reward_parameters = contract["reward"]

    inventory_baseline = float(waste_parameters["inventory_baseline"])
    surplus_ratio = max(0.0, inventory / inventory_baseline - 1.0)
    thermal_stress = compute_thermal_stress(
        temperature,
        thermal_t0=float(carbon_parameters["thermal_reference_c"]),
        thermal_delta_max=float(carbon_parameters["thermal_range_c"]),
    )
    km = (
        float(carbon_parameters["route_km_by_action"][action])
        * transport_multiplier
    )
    carbon = float(compute_transport_carbon(
        km,
        float(carbon_parameters["carbon_per_km"]),
        thermal_stress,
        cop_penalty=float(carbon_parameters["refrigeration_cop_penalty"]),
        eff_factor=float(carbon_parameters["physical_efficiency_factor"]),
    ))
    weights = slca_parameters["weights"]
    slca_components = slca_score(
        carbon,
        action,
        w_c=float(weights["C"]),
        w_l=float(weights["L"]),
        w_r=float(weights["R"]),
        w_p=float(weights["P"]),
        carbon_cap=float(slca_parameters["carbon_cap"]),
        action_bases=slca_parameters["action_bases"],
    )
    slca_quality = float(compute_slca_attenuation(
        thermal_stress,
        surplus_ratio,
        thermal_atten=float(slca_parameters["thermal_attenuation"]),
        surplus_atten=float(slca_parameters["surplus_attenuation"]),
    ))
    slca_attenuated = float(slca_components["composite"]) * slca_quality

    k_inst = float(arrhenius_k(
        temperature,
        float(arrhenius_parameters["effective_k_ref"]),
        float(arrhenius_parameters["effective_ea_over_r"]),
        float(arrhenius_parameters["reference_temperature_k"]),
        humidity / 100.0,
        float(arrhenius_parameters["humidity_coupling"]),
    ))
    waste_raw = float(compute_waste_rate(
        k_inst,
        surplus_ratio,
        w_scale=float(waste_parameters["exposure_scale"]),
        w_alpha=float(waste_parameters["compression_exponent"]),
        surplus_waste_factor=float(waste_parameters["surplus_waste_factor"]),
        waste_cap=float(waste_parameters["cap_fraction"]),
    ))
    save_factor = float(compute_save_factor(
        action,
        mode,
        surplus_ratio,
        surplus_save_penalty=float(waste_parameters["surplus_save_penalty"]),
        save_floor=waste_parameters["action_save_fraction"],
    ))
    waste = waste_raw * (1.0 - save_factor)
    reward_slca = (
        float(reward_parameters["slca_value_in_ablation"])
        if mode == reward_parameters["slca_ablation_mode"]
        else slca_attenuated
    )
    reward = float(compute_reward(
        reward_slca,
        waste,
        rho,
        eta=float(reward_parameters["waste_penalty"]),
        eta_rho=float(reward_parameters["risk_penalty"]),
    ))
    ari = float(compute_ari(waste, slca_attenuated, rho))
    return {
        "surplus_ratio": surplus_ratio,
        "thermal_stress": thermal_stress,
        "k_inst": k_inst,
        "waste_raw": waste_raw,
        "save_factor": save_factor,
        "waste": waste,
        "carbon_kg": carbon,
        "slca_raw": float(slca_components["composite"]),
        "slca_quality": slca_quality,
        "slca": slca_attenuated,
        "reward": reward,
        "ari": ari,
        "slca_component_trace": {
            **slca_components,
            "slca_quality": round(slca_quality, 4),
            "composite_attenuated": round(slca_attenuated, 4),
        },
    }


def validate_recorded_step_outcomes(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    where: str = "record",
    contract_validated: bool = False,
) -> dict[str, Any]:
    """Reject a row whose recorded outcomes do not follow the exact equations."""

    reconstructed = reconstruct_step_outcomes(
        record,
        contract,
        where=where,
        contract_validated=contract_validated,
    )
    for field in ("waste", "carbon_kg", "slca", "reward", "ari"):
        observed = _float(record.get(field), where=f"{where}/{field}")
        expected = float(reconstructed[field])
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{where}/{field} violates the outcome equation: "
                f"recorded={observed!r}, reconstructed={expected!r}"
            )
    return reconstructed
