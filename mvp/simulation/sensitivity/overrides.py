"""Fail-closed application and dynamic probes for structural parameters.

The sensitivity runner executes one task per process.  This module still uses
a restoring context manager so a failed task cannot leak factor settings into
another in-process diagnostic.  No production default is changed on disk.
"""
from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .parameters import PARAMETERS, derived_values, validate_parameter_values


_POLICY_KEY_TO_FIELD: dict[str, str] = {
    "spoilage_k_ref": "k_ref",
    "spoilage_ea_over_r": "Ea_R",
    "humidity_coupling": "beta_humidity",
    "lag_lambda_hours": "lag_lambda",
    "km_coldchain": "km_coldchain",
    "km_local": "km_local",
    "km_recovery": "km_recovery",
    "transport_carbon_factor": "carbon_per_km",
    "slca_weight_carbon": "w_c",
    "slca_weight_labour": "w_l",
    "slca_weight_resilience": "w_r",
    "slca_carbon_cap": "carbon_cap",
    "waste_reward_penalty": "eta",
    "risk_reward_penalty": "eta_rho",
    "bollinger_window_steps": "boll_window",
    "bollinger_z_threshold": "boll_k",
    "volatility_tilt_coldchain": "gamma_coldchain",
    "volatility_tilt_local": "gamma_local",
    "volatility_tilt_recovery": "gamma_recovery",
}


def default_parameter_values() -> dict[str, float | int]:
    return {
        parameter.key: (
            int(parameter.default)
            if parameter.kind == "integer"
            else float(parameter.default)
        )
        for parameter in PARAMETERS
    }


def _ensure_import_paths(repo_root: Path | str) -> Path:
    root = Path(repo_root).resolve()
    for path in (root, root / "agribrain" / "backend", root / "mvp" / "simulation"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


def policy_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    checked = validate_parameter_values(values)
    kwargs = {
        field: checked[key] for key, field in _POLICY_KEY_TO_FIELD.items()
    }
    kwargs["w_p"] = derived_values(checked)["slca_weight_price_transparency"]
    if abs(sum(float(kwargs[name]) for name in ("w_c", "w_l", "w_r", "w_p")) - 1.0) > 1e-12:
        raise ValueError("derived SLCA weights do not sum to one")
    return kwargs


def configure_policy_feature_flags(policy: Any, *, failure_injection: bool = False) -> Any:
    """Apply the same optional-feature environment posture as ``run_all``."""

    policy.enable_failure_injection = bool(failure_injection)
    policy.enable_mcp_reliability = (
        os.environ.get("MCP_RELIABILITY", "false").lower() == "true"
    )
    policy.enable_mcp_qos_routing = (
        os.environ.get("MCP_QOS_ROUTING", "false").lower() == "true"
    )
    policy.enable_pirag_counterfactual_eval = (
        os.environ.get("PIRAG_COUNTERFACTUAL", "false").lower() == "true"
    )
    policy.enable_physics_consistency_gate = (
        os.environ.get("PHYSICS_CONSISTENCY_GATE", "false").lower() == "true"
    )
    policy.enable_heterogeneous_profiles = (
        os.environ.get("HETEROGENEOUS_PROFILES", "false").lower() == "true"
    )
    policy.enable_research_metrics = (
        os.environ.get("RESEARCH_METRICS", "false").lower() == "true"
    )
    return policy


def _scaled_action_bases(
    original: dict[str, dict[str, float]], scale: float,
) -> dict[str, dict[str, float]]:
    out = copy.deepcopy(original)
    for pillar in ("L", "R", "P"):
        mean = float(np.mean([row[pillar] for row in original.values()]))
        for action, row in original.items():
            value = mean + float(scale) * (float(row[pillar]) - mean)
            out[action][pillar] = float(np.clip(value, 0.0, 1.0))
        before_order = sorted(original, key=lambda action: original[action][pillar])
        after_order = sorted(out, key=lambda action: out[action][pillar])
        if before_order != after_order:
            raise ValueError(f"social-score contrast scaling reversed {pillar} ordering")
    return out


@contextmanager
def applied_structural_parameters(
    values: dict[str, Any], repo_root: Path | str,
) -> Iterator[dict[str, Any]]:
    """Apply one full LHS row to live production call sites, then restore.

    The yielded mapping contains a fresh ``policy_factory`` used by both the
    primary and H3 runners.  Module constants with Python default-argument
    binding (waste/COP) are applied by wrapping the exact functions imported by
    ``generate_results``; merely reassigning an otherwise dormant constant
    would fail the active-parameter requirement.
    """

    checked = validate_parameter_values(values)
    root = _ensure_import_paths(repo_root)
    del root  # import-path validation is the only use here

    from mvp.simulation import generate_results as gr
    from mvp.simulation import stochastic
    from src.models import carbon, slca, waste
    from src.models.policy import Policy as PolicyClass
    from pirag import context_to_logits

    kwargs = policy_kwargs(checked)

    def factory() -> Any:
        return PolicyClass(**kwargs)

    saved_env = {
        key: os.environ.get(key)
        for key in (*stochastic.canonical_defaults().keys(), "DETERMINISTIC_MODE")
    }
    saved = {
        "gr_policy": gr.Policy,
        "gr_compute_waste_rate": gr.compute_waste_rate,
        "gr_compute_save_factor": gr.compute_save_factor,
        "gr_compute_transport_carbon": gr.compute_transport_carbon,
        "gr_build_outcome_equation_contract": gr.build_outcome_equation_contract,
        "save_floor": copy.deepcopy(waste.SAVE_FLOOR),
        "action_bases": copy.deepcopy(slca._ACTION_BASES),
        "theta_context": context_to_logits.THETA_CONTEXT.copy(),
    }
    try:
        # Force the stochastic path on; structural sensitivity is not a
        # deterministic-mode benchmark.  Scale only nonzero declared defaults,
        # leaving policy-temperature heterogeneity disabled at exactly zero.
        os.environ["DETERMINISTIC_MODE"] = "false"
        stochastic_scale = float(checked["stochastic_noise_scale"])
        for key, raw_default in stochastic.canonical_defaults().items():
            default = float(raw_default)
            os.environ[key] = str(default if default == 0.0 else default * stochastic_scale)

        waste.SAVE_FLOOR.clear()
        waste.SAVE_FLOOR.update(saved["save_floor"])
        waste.SAVE_FLOOR["local_redistribute"] = float(
            checked["local_redistribute_save_floor"]
        )
        waste.SAVE_FLOOR["recovery"] = float(checked["recovery_save_floor"])

        slca._ACTION_BASES.clear()
        slca._ACTION_BASES.update(_scaled_action_bases(
            saved["action_bases"], float(checked["social_score_contrast_scale"]),
        ))
        context_to_logits.THETA_CONTEXT = (
            saved["theta_context"] * float(checked["context_prior_scale"])
        )

        def compute_waste_rate_structural(
            k_inst: Any, surplus_ratio: float = 0.0, **_ignored: Any,
        ) -> Any:
            return waste.compute_waste_rate(
                k_inst,
                surplus_ratio,
                w_scale=float(checked["waste_exposure_scale"]),
                w_alpha=float(checked["waste_compression_exponent"]),
                surplus_waste_factor=float(checked["surplus_waste_factor"]),
            )

        def compute_save_factor_structural(
            action: str,
            mode: str,
            surplus_ratio: float = 0.0,
            compliance_data: dict | None = None,
            **_ignored: Any,
        ) -> float:
            return waste.compute_save_factor(
                action,
                mode,
                surplus_ratio,
                surplus_save_penalty=float(checked["surplus_save_penalty"]),
                compliance_data=compliance_data,
            )

        def compute_transport_carbon_structural(
            km: float,
            carbon_per_km: float,
            thermal_stress: float = 0.0,
            eff_factor: float = 1.0,
            **_ignored: Any,
        ) -> float:
            return carbon.compute_transport_carbon(
                km,
                carbon_per_km,
                thermal_stress,
                cop_penalty=float(checked["refrigeration_cop_penalty"]),
                eff_factor=eff_factor,
            )

        def build_outcome_equation_contract_structural(
            policy: Any,
            *,
            effective_k_ref: float,
            effective_ea_r: float,
            stochastic_layer: Any,
        ) -> dict[str, Any]:
            return saved["gr_build_outcome_equation_contract"](
                policy,
                effective_k_ref=effective_k_ref,
                effective_ea_r=effective_ea_r,
                stochastic_layer=stochastic_layer,
                parameter_overrides={
                    "waste_exposure_scale": float(
                        checked["waste_exposure_scale"]
                    ),
                    "waste_compression_exponent": float(
                        checked["waste_compression_exponent"]
                    ),
                    "surplus_waste_factor": float(
                        checked["surplus_waste_factor"]
                    ),
                    "surplus_save_penalty": float(
                        checked["surplus_save_penalty"]
                    ),
                    "action_save_fraction": dict(waste.SAVE_FLOOR),
                    "refrigeration_cop_penalty": float(
                        checked["refrigeration_cop_penalty"]
                    ),
                    "slca_action_bases": copy.deepcopy(slca._ACTION_BASES),
                },
            )

        gr.Policy = factory
        gr.compute_waste_rate = compute_waste_rate_structural
        gr.compute_save_factor = compute_save_factor_structural
        gr.compute_transport_carbon = compute_transport_carbon_structural
        gr.build_outcome_equation_contract = (
            build_outcome_equation_contract_structural
        )

        yield {
            "policy_factory": factory,
            "policy_kwargs": dict(kwargs),
            "derived_parameters": derived_values(checked),
            "stochastic_environment": {
                key: os.environ[key] for key in stochastic.canonical_defaults()
            },
            "generate_results_module": gr,
        }
    finally:
        gr.Policy = saved["gr_policy"]
        gr.compute_waste_rate = saved["gr_compute_waste_rate"]
        gr.compute_save_factor = saved["gr_compute_save_factor"]
        gr.compute_transport_carbon = saved["gr_compute_transport_carbon"]
        gr.build_outcome_equation_contract = saved[
            "gr_build_outcome_equation_contract"
        ]
        waste.SAVE_FLOOR.clear()
        waste.SAVE_FLOOR.update(saved["save_floor"])
        slca._ACTION_BASES.clear()
        slca._ACTION_BASES.update(saved["action_bases"])
        context_to_logits.THETA_CONTEXT = saved["theta_context"]
        for key, previous in saved_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def expected_structural_outcome_equation_contract(
    values: dict[str, Any],
    repo_root: Path | str,
    *,
    benchmark_seed: int,
    scenario: str,
    episode_index: int = 3,
) -> dict[str, Any]:
    """Independently derive the exact outcome contract for one LHS point."""

    with applied_structural_parameters(values, repo_root) as applied:
        gr = applied["generate_results_module"]
        from mvp.simulation import stochastic

        environment_seed = gr._stream_seed(
            benchmark_seed, scenario, episode_index, "environment",
        )
        layer = stochastic.make_stochastic_layer(
            np.random.default_rng(environment_seed),
            stream_seed=environment_seed,
        )
        policy = applied["policy_factory"]()
        effective_k_ref = layer.perturb_k_ref(policy.k_ref, counter=0)
        effective_ea_r = layer.perturb_ea_r(policy.Ea_R, counter=0)
        return gr.build_outcome_equation_contract(
            policy,
            effective_k_ref=effective_k_ref,
            effective_ea_r=effective_ea_r,
            stochastic_layer=layer,
        )


def _observable_for_parameter(
    key: str, values: dict[str, Any], repo_root: Path | str,
) -> np.ndarray:
    """Evaluate a small production-function observable for one factor row."""

    _ensure_import_paths(repo_root)
    from mvp.simulation import stochastic
    from src.models.action_selection import select_action
    from src.models.carbon import compute_transport_carbon
    from src.models.reward import compute_reward
    from src.models.slca import slca_score
    from src.models.spoilage import advance_spoilage_risk_midpoint
    from src.models.waste import compute_waste_rate
    from pirag.context_to_logits import compute_context_modifier

    checked = validate_parameter_values(values)
    kwargs = policy_kwargs(checked)
    from src.models.policy import Policy
    policy = Policy(**kwargs)

    if key in {
        "spoilage_k_ref", "spoilage_ea_over_r", "humidity_coupling",
        "lag_lambda_hours",
    }:
        return np.asarray([advance_spoilage_risk_midpoint(
            0.05,
            previous_temp_C=18.0,
            current_temp_C=24.0,
            previous_rh_pct=75.0,
            current_rh_pct=92.0,
            previous_hour=12.0,
            current_hour=24.0,
            k_ref=policy.k_ref,
            Ea_R=policy.Ea_R,
            T_ref_K=policy.T_ref_K,
            beta=policy.beta_humidity,
            lag_lambda=policy.lag_lambda,
        )])

    if key in {
        "waste_exposure_scale", "waste_compression_exponent",
        "surplus_waste_factor",
    }:
        return np.asarray([compute_waste_rate(
            0.004,
            0.25,
            w_scale=float(checked["waste_exposure_scale"]),
            w_alpha=float(checked["waste_compression_exponent"]),
            surplus_waste_factor=float(checked["surplus_waste_factor"]),
        )])

    if key in {
        "surplus_save_penalty", "local_redistribute_save_floor",
        "recovery_save_floor",
    }:
        # Use the same restoring override used by real tasks so dictionary
        # floors and bound default arguments are both exercised.
        with applied_structural_parameters(checked, repo_root) as applied:
            gr = applied["generate_results_module"]
            action = (
                "recovery" if key == "recovery_save_floor"
                else "local_redistribute"
            )
            return np.asarray([gr.compute_save_factor(
                action, "agribrain", surplus_ratio=0.8,
            )])

    if key in {
        "km_coldchain", "km_local", "km_recovery",
        "transport_carbon_factor", "refrigeration_cop_penalty",
    }:
        route = {
            "km_coldchain": policy.km_coldchain,
            "km_local": policy.km_local,
            "km_recovery": policy.km_recovery,
        }.get(key, policy.km_coldchain)
        return np.asarray([compute_transport_carbon(
            route,
            policy.carbon_per_km,
            thermal_stress=0.8,
            cop_penalty=float(checked["refrigeration_cop_penalty"]),
        )])

    if key in {
        "slca_weight_carbon", "slca_weight_labour",
        "slca_weight_resilience", "slca_carbon_cap",
        "social_score_contrast_scale",
    }:
        with applied_structural_parameters(checked, repo_root):
            score = slca_score(
                8.0,
                "recovery",
                w_c=policy.w_c,
                w_l=policy.w_l,
                w_r=policy.w_r,
                w_p=policy.w_p,
                carbon_cap=policy.carbon_cap,
            )
        return np.asarray([float(score["composite"])])

    if key in {"waste_reward_penalty", "risk_reward_penalty"}:
        return np.asarray([compute_reward(
            0.72, 0.11, eta=policy.eta, eta_rho=policy.eta_rho, rho=0.35,
        )])

    if key in {"bollinger_window_steps", "bollinger_z_threshold"}:
        # This is the exact rolling-z expression used by generate_results.
        import pandas as pd
        series = pd.Series(
            [10.0] * 8 + [11.0, 9.0, 10.5, 9.5] * 6 + [17.0, 4.0, 16.0, 5.0]
        )
        window = int(policy.boll_window)
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().fillna(0.0)
        z = np.where(std > 1e-12, (series - mean) / std, 0.0)
        return np.asarray([
            float(np.count_nonzero(np.abs(z) > float(policy.boll_k))),
            float(np.sum(np.abs(z))),
        ])

    if key in {
        "volatility_tilt_coldchain", "volatility_tilt_local",
        "volatility_tilt_recovery",
    }:
        diagnostics: dict[str, Any] = {}
        _, probabilities = select_action(
            "agribrain", 0.35, 12_000.0, 450.0, 12.0, 1.0,
            policy, np.random.default_rng(17), deterministic=True,
            context_modifier=np.zeros(3), out=diagnostics,
        )
        return np.asarray(probabilities, dtype=float)

    if key == "context_prior_scale":
        with applied_structural_parameters(checked, repo_root):
            modifier = compute_context_modifier(
                {
                    "check_compliance": {
                        "compliant": False,
                        "violations": [{"severity": "critical"}],
                    },
                    "spoilage_forecast": {"urgency": "high"},
                },
                {},
                None,
            )
        return np.asarray(modifier, dtype=float)

    if key == "stochastic_noise_scale":
        with applied_structural_parameters(checked, repo_root):
            layer = stochastic.make_stochastic_layer(
                np.random.default_rng(123), stream_seed=456,
            )
            return np.asarray([
                layer.temp_std_c, layer.rh_std, layer.demand_frac_std,
                layer.inventory_frac_std, layer.transport_km_frac_std,
                layer.k_ref_frac_std, layer.ea_r_frac_std,
                layer.onset_jitter_hours, layer.theta_noise_std,
                layer.delay_prob,
            ], dtype=float)
    raise KeyError(key)


def validate_dynamic_influence(repo_root: Path | str) -> dict[str, Any]:
    """Prove every registered coordinate changes a production observable.

    This diagnostic is deliberately small and deterministic; it does not run a
    simulation episode and it makes no claim about effect size or ranking.
    """

    baseline = default_parameter_values()
    records: list[dict[str, Any]] = []
    for parameter in PARAMETERS:
        low = dict(baseline)
        high = dict(baseline)
        low[parameter.key] = (
            int(parameter.lower) if parameter.kind == "integer" else parameter.lower
        )
        high[parameter.key] = (
            int(parameter.upper) if parameter.kind == "integer" else parameter.upper
        )
        low_observable = _observable_for_parameter(parameter.key, low, repo_root)
        high_observable = _observable_for_parameter(parameter.key, high, repo_root)
        changed = not np.allclose(
            low_observable, high_observable, rtol=1e-10, atol=1e-12,
        )
        if not changed:
            raise ValueError(
                f"registered parameter {parameter.key!r} did not change its "
                "production-function influence probe"
            )
        records.append({
            "parameter": parameter.key,
            "status": "pass",
            "low_observable": low_observable.tolist(),
            "high_observable": high_observable.tolist(),
        })
    return {
        "status": "pass",
        "n_parameters": len(records),
        "records": records,
        "scope": (
            "deterministic production-function influence probes; not outcome "
            "sensitivity estimates"
        ),
    }
