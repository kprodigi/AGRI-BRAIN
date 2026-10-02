"""Independent noise-free synthetic spoilage data-generating process.

This module implements the declared latent DGP used to construct the frozen
PINN training targets in ``mvp/simulation/pinn``.  It is deliberately separate
from both the mechanistic estimator and the frozen residual estimator: policy
arms may observe different estimates while being scored against one common
synthetic outcome trajectory.

The DGP is an author-declared simulation model, not an empirical spinach
quality model.  It does not add observation noise.  The PINN training-data
generator adds its separately declared observation noise only after this
noise-free latent trajectory has been integrated.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .spoilage import arrhenius_k


SYNTHETIC_DGP_SCHEMA_VERSION = 1
SYNTHETIC_DGP_KIND = "independent_synthetic_dgp_v1"

DEFAULT_PACKAGING_INDEX = 0.50
PACKAGING_CENTER = 0.50
PACKAGING_LOG_RATE_COEFFICIENT = 0.44
HANDLING_SHOCK_LOG_RATE_COEFFICIENT = 0.80
RH_TRANSIENT_LOG_RATE_COEFFICIENT = 0.0040


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _trajectory_packaging_index(
    frame: pd.DataFrame,
    packaging_index: float | None,
) -> float:
    """Resolve the trajectory-level packaging assumption without averaging."""

    if packaging_index is not None:
        resolved = _finite_float(packaging_index, name="packaging_index")
    elif "packaging_index" not in frame.columns:
        resolved = DEFAULT_PACKAGING_INDEX
    else:
        values = frame["packaging_index"].to_numpy(dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("packaging_index column must be finite and non-empty")
        resolved = float(values[0])
        if not np.all(values == resolved):
            raise ValueError(
                "packaging_index is a trajectory-level DGP assumption and "
                "must be constant within a frame"
            )
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("packaging_index must lie in [0, 1]")
    return resolved


def synthetic_dgp_provenance(
    *,
    k_ref: float = 0.0021,
    Ea_R: float = 8000.0,
    T_ref_K: float = 277.15,
    beta: float = 0.25,
    lag_lambda: float = 12.0,
    packaging_index: float = DEFAULT_PACKAGING_INDEX,
) -> dict[str, Any]:
    """Return the exact JSON-native provenance contract for one DGP path."""

    k_ref_value = _finite_float(k_ref, name="k_ref")
    ea_r_value = _finite_float(Ea_R, name="Ea_R")
    t_ref_value = _finite_float(T_ref_K, name="T_ref_K")
    beta_value = _finite_float(beta, name="beta")
    lag_value = _finite_float(lag_lambda, name="lag_lambda")
    packaging_value = _finite_float(
        packaging_index, name="packaging_index",
    )
    if k_ref_value <= 0.0 or ea_r_value <= 0.0 or t_ref_value <= 0.0:
        raise ValueError("DGP Arrhenius parameters must be positive")
    if beta_value < 0.0 or lag_value < 0.0:
        raise ValueError("DGP humidity coupling and lag must be non-negative")
    if not 0.0 <= packaging_value <= 1.0:
        raise ValueError("packaging_index must lie in [0, 1]")

    return {
        "schema_version": SYNTHETIC_DGP_SCHEMA_VERSION,
        "kind": SYNTHETIC_DGP_KIND,
        "role": "common_mode_invariant_noise_free_outcome_reference",
        "target_origin": "independent_synthetic_dgp",
        "synthetic_only": True,
        "external_validation": False,
        "empirical_claims_permitted": False,
        "noise_free": True,
        "state_variable": "remaining_quality_fraction",
        "initial_quality_fraction": 1.0,
        "integration": "midpoint_exponential_state_update",
        "state_equation": (
            "C_i=C_(i-1)*exp(-k_base(T_mid,RH_mid)*alpha(t_mid)*"
            "exp(u_i)*delta_t_i)"
        ),
        "lag_equation": "alpha(t)=t/(t+lag_lambda)",
        "log_rate_multiplier_equation": (
            "u=0.44*(packaging_index-0.50)+0.80*handling_shock_G_mid+"
            "0.0040*abs_dRH_dt_mid"
        ),
        "coefficients": {
            "packaging_center": PACKAGING_CENTER,
            "packaging_log_rate": PACKAGING_LOG_RATE_COEFFICIENT,
            "handling_shock_log_rate_per_g": (
                HANDLING_SHOCK_LOG_RATE_COEFFICIENT
            ),
            "rh_transient_log_rate_per_pct_per_hour": (
                RH_TRANSIENT_LOG_RATE_COEFFICIENT
            ),
        },
        "parameters": {
            "k_ref_per_h": k_ref_value,
            "ea_over_r_kelvin": ea_r_value,
            "reference_temperature_kelvin": t_ref_value,
            "humidity_coupling": beta_value,
            "lag_lambda_hours": lag_value,
            "packaging_index": packaging_value,
        },
    }


def compute_spoilage_independent_synthetic_dgp(
    frame: pd.DataFrame,
    *,
    k_ref: float = 0.0021,
    Ea_R: float = 8000.0,
    T_ref_K: float = 277.15,
    beta: float = 0.25,
    lag_lambda: float = 12.0,
    packaging_index: float | None = None,
) -> pd.DataFrame:
    """Integrate the declared independent noise-free synthetic DGP.

    The operation order intentionally matches
    ``mvp/simulation/pinn/generate_synthetic_spoilage_data.py`` exactly on its
    15-minute trajectories.  Variable positive timesteps are supported for
    reuse by the simulator, but no resampling, observation noise, clipping, or
    estimator prediction occurs here.

    Returns a copy of ``frame`` whose canonical ``shelf_left`` and
    ``spoilage_risk`` columns contain the common latent reference.  The
    explicit ``latent_quality_fraction``, ``latent_spoilage_risk``,
    ``rh_transient_per_h``, and ``packaging_index`` columns retain all DGP
    inputs and outputs needed for independent reconstruction.
    """

    required = {"timestamp", "tempC", "RH", "shockG"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"synthetic spoilage DGP input is missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("synthetic spoilage DGP requires at least one row")

    provenance = synthetic_dgp_provenance(
        k_ref=k_ref,
        Ea_R=Ea_R,
        T_ref_K=T_ref_K,
        beta=beta,
        lag_lambda=lag_lambda,
        packaging_index=_trajectory_packaging_index(frame, packaging_index),
    )
    parameters = provenance["parameters"]
    packaging_value = float(parameters["packaging_index"])

    timestamps = pd.to_datetime(frame["timestamp"])
    time_h = (
        (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(
            dtype=np.float64,
        )
        / 3600.0
    )
    temp = frame["tempC"].to_numpy(dtype=np.float64)
    rh = frame["RH"].to_numpy(dtype=np.float64)
    shock = frame["shockG"].to_numpy(dtype=np.float64)
    if not all(np.isfinite(values).all() for values in (time_h, temp, rh, shock)):
        raise ValueError("synthetic spoilage DGP inputs must all be finite")
    if len(time_h) > 1 and np.any(np.diff(time_h) <= 0.0):
        raise ValueError("synthetic spoilage DGP timestamps must be strictly increasing")

    latent_quality = np.ones(len(frame), dtype=np.float64)
    rh_transient = np.zeros(len(frame), dtype=np.float64)
    if len(frame) > 1:
        rh_transient[1:] = np.abs(np.diff(rh)) / np.diff(time_h)

    for index in range(1, len(frame)):
        delta_t = float(time_h[index] - time_h[index - 1])
        mid_time = 0.5 * (time_h[index] + time_h[index - 1])
        mid_temp = 0.5 * (temp[index] + temp[index - 1])
        # Keep the generator's literal multiplication order for exact replay.
        mid_rh = 0.005 * (rh[index] + rh[index - 1])
        base_rate = float(arrhenius_k(
            mid_temp,
            k_ref=float(parameters["k_ref_per_h"]),
            Ea_R=float(parameters["ea_over_r_kelvin"]),
            T_ref_K=float(parameters["reference_temperature_kelvin"]),
            rh_frac=mid_rh,
            beta=float(parameters["humidity_coupling"]),
        ))
        lag_value = float(parameters["lag_lambda_hours"])
        alpha = (
            mid_time / (mid_time + lag_value)
            if lag_value > 0.0 else 1.0
        )
        handling = 0.5 * (shock[index] + shock[index - 1])
        transient = 0.5 * (
            rh_transient[index] + rh_transient[index - 1]
        )
        log_multiplier = (
            PACKAGING_LOG_RATE_COEFFICIENT
            * (packaging_value - PACKAGING_CENTER)
            + HANDLING_SHOCK_LOG_RATE_COEFFICIENT * handling
            + RH_TRANSIENT_LOG_RATE_COEFFICIENT * transient
        )
        latent_rate = base_rate * alpha * float(np.exp(log_multiplier))
        latent_quality[index] = latent_quality[index - 1] * np.exp(
            -latent_rate * delta_t
        )

    result = frame.copy()
    result["packaging_index"] = packaging_value
    result["rh_transient_per_h"] = rh_transient
    result["latent_quality_fraction"] = latent_quality
    result["latent_spoilage_risk"] = 1.0 - latent_quality
    result["shelf_left"] = latent_quality
    result["spoilage_risk"] = 1.0 - latent_quality
    result.attrs["synthetic_spoilage_dgp"] = provenance
    return result


__all__ = [
    "SYNTHETIC_DGP_SCHEMA_VERSION",
    "SYNTHETIC_DGP_KIND",
    "DEFAULT_PACKAGING_INDEX",
    "PACKAGING_CENTER",
    "PACKAGING_LOG_RATE_COEFFICIENT",
    "HANDLING_SHOCK_LOG_RATE_COEFFICIENT",
    "RH_TRANSIENT_LOG_RATE_COEFFICIENT",
    "synthetic_dgp_provenance",
    "compute_spoilage_independent_synthetic_dgp",
]
