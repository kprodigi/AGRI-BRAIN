"""Independent reconstruction of every retained policy decision surface."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.agents.roles import ROLE_BIASES, stage_for_hour

from .action_selection import (
    GOVERNANCE_CC_PROB_CEILING,
    GOVERNANCE_LOCAL_ADVANTAGE_MIN,
    RHO_RECOVERY_KNEE,
    RHO_RECOVERY_KNEE_GAIN,
    RHO_RECOVERY_KNEE_LR_PENALTY,
    SLCA_BONUS,
    SLCA_RHO_BONUS,
    build_feature_vector,
    regime_logit_term,
)
from .mode_capabilities import capabilities_for


def _number(value: Any, *, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} is boolean, not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} is not finite")
    return result


def _vector(value: Any, length: int, *, where: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{where} is not a vector")
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{where} is not a finite {length}-vector")
    return array


def _matrix(value: Any, rows: int, columns: int, *, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (rows, columns) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{where} is not a finite {rows}x{columns} matrix"
        )
    return array


def _assert_close(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    where: str,
    tolerance: float = 1e-10,
) -> None:
    if observed.shape != expected.shape or np.max(
        np.abs(observed - expected), initial=0.0,
    ) > tolerance:
        raise ValueError(f"{where} violates the locked policy equation")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def validate_policy_record(
    record: Mapping[str, Any],
    *,
    policy: Any,
    policy_theta: np.ndarray,
    where: str = "record",
) -> dict[str, Any]:
    """Recompute phi, role/peer bias, logits, override, and probabilities."""
    mode = str(record.get("mode") or "")
    caps = capabilities_for(mode)
    rho = _number(record.get("rho_policy_observed"), where=f"{where}/rho")
    hour = _number(record.get("hour"), where=f"{where}/hour")
    phi = build_feature_vector(
        rho,
        _number(
            record.get("inventory_policy_observed"),
            where=f"{where}/inventory_policy_observed",
        ),
        _number(
            record.get("demand_forecast_policy_observed"),
            where=f"{where}/demand_forecast_policy_observed",
        ),
        _number(
            record.get("temp_policy_observed"),
            where=f"{where}/temp_policy_observed",
        ),
        supply_hat=_number(
            record.get("supply_forecast_policy_observed"),
            where=f"{where}/supply_forecast_policy_observed",
        ),
        supply_std=_number(
            record.get("supply_forecast_std_policy_observed"),
            where=f"{where}/supply_forecast_std_policy_observed",
        ),
        demand_std=_number(
            record.get("demand_forecast_std_policy_observed"),
            where=f"{where}/demand_forecast_std_policy_observed",
        ),
        price_signal=_number(
            record.get("price_signal"), where=f"{where}/price_signal",
        ),
    )
    stored_phi = _vector(record.get("phi"), 10, where=f"{where}/phi")
    _assert_close(stored_phi, phi, where=f"{where}/phi", tolerance=1e-12)

    role = str(record.get("role") or "")
    expected_role = stage_for_hour(hour)
    if role != expected_role:
        raise ValueError(f"{where}/role does not match the lifecycle stage")
    peer_bias = _vector(
        record.get("peer_message_bias"), 3,
        where=f"{where}/peer_message_bias",
    )
    expected_combined_bias = ROLE_BIASES[role].astype(float).copy()
    if 12.0 <= hour < 30.0:
        expected_combined_bias += ROLE_BIASES["cooperative"]
    if caps.peer_messages:
        expected_combined_bias += peer_bias
    elif np.any(np.abs(peer_bias) > 1e-15):
        raise ValueError(f"{where} no-peer arm has nonzero peer bias")
    combined_bias = _vector(
        record.get("combined_role_bias"), 3,
        where=f"{where}/combined_role_bias",
    )
    _assert_close(
        combined_bias, expected_combined_bias,
        where=f"{where}/combined_role_bias", tolerance=1e-12,
    )

    theta_delta = _matrix(
        record.get("effective_theta_delta"), 3, 10,
        where=f"{where}/effective_theta_delta",
    )
    slca_bonus_delta = _vector(
        record.get("effective_slca_bonus_delta"), 3,
        where=f"{where}/effective_slca_bonus_delta",
    )
    slca_rho_delta = _vector(
        record.get("effective_slca_rho_delta"), 3,
        where=f"{where}/effective_slca_rho_delta",
    )
    _vector(
        record.get("effective_no_slca_offset_delta"), 3,
        where=f"{where}/effective_no_slca_offset_delta",
    )

    if mode == "static":
        if (
            record.get("base_logits") is not None
            or record.get("post_context_logits_pre_override") is not None
            or record.get("policy_temperature") is not None
            or np.any(theta_delta)
            or np.any(slca_bonus_delta)
            or np.any(slca_rho_delta)
        ):
            raise ValueError(f"{where} static arm exposes an adaptive policy")
        return {
            "phi": phi,
            "preoverride_probs": np.array([1.0, 0.0, 0.0]),
            "governance_override": False,
        }

    theta = _matrix(policy_theta, 3, 10, where=f"{where}/policy_theta")
    tau = _number(
        record.get("bollinger_regime_flag"),
        where=f"{where}/bollinger_regime_flag",
    )
    try:
        regime_term = regime_logit_term(policy, tau)
    except ValueError as exc:
        raise ValueError(
            f"{where}/bollinger_regime_flag violates the binary regime contract"
        ) from exc
    stored_regime_term = _vector(
        record.get("regime_logit_bias"), 3,
        where=f"{where}/regime_logit_bias",
    )
    _assert_close(
        stored_regime_term, regime_term,
        where=f"{where}/regime_logit_bias", tolerance=1e-12,
    )
    effective_slca_bonus = SLCA_BONUS + slca_bonus_delta
    effective_slca_rho = SLCA_RHO_BONUS + slca_rho_delta
    if mode in {"hybrid_rl", "no_slca"}:
        logits = theta @ phi + regime_term
    else:
        logits = (
            theta @ phi + regime_term
            + effective_slca_bonus + effective_slca_rho * rho
        )
    if mode != "hybrid_rl" and rho > RHO_RECOVERY_KNEE:
        excess = (rho - RHO_RECOVERY_KNEE) / (1.0 - RHO_RECOVERY_KNEE)
        logits[2] += RHO_RECOVERY_KNEE_GAIN * excess
        logits[1] -= RHO_RECOVERY_KNEE_LR_PENALTY * excess
    logits = logits + theta_delta @ phi + expected_combined_bias
    stored_base = _vector(
        record.get("base_logits"), 3, where=f"{where}/base_logits",
    )
    _assert_close(stored_base, logits, where=f"{where}/base_logits")

    stored_shaping = _vector(
        record.get("slca_shaping"), 3, where=f"{where}/slca_shaping",
    )
    expected_shaping = effective_slca_bonus + effective_slca_rho * rho
    _assert_close(
        stored_shaping, expected_shaping, where=f"{where}/slca_shaping",
    )
    context_modifier = record.get("context_modifier")
    if caps.context_kind is None:
        if context_modifier is not None:
            raise ValueError(f"{where} no-context policy has a modifier")
        post_logits = logits.copy()
        expected_override = False
    else:
        modifier = _vector(
            context_modifier, 3, where=f"{where}/context_modifier",
        )
        amp = _number(record.get("slca_amp"), where=f"{where}/slca_amp")
        amplification = amp * min(abs(float(modifier[1])), 1.0)
        boost = np.zeros(3) if mode == "no_slca" else (
            expected_shaping * amplification
        )
        post_logits = logits + modifier + boost
        expected_override = False  # assigned after probability reconstruction
    temperature = _number(
        record.get("policy_temperature"),
        where=f"{where}/policy_temperature",
    )
    if temperature <= 0.0:
        raise ValueError(f"{where}/policy_temperature is not positive")
    post_logits = post_logits / temperature
    stored_post = _vector(
        record.get("post_context_logits_pre_override"), 3,
        where=f"{where}/post_context_logits_pre_override",
    )
    _assert_close(
        stored_post, post_logits,
        where=f"{where}/post_context_logits_pre_override",
    )
    preoverride_probs = _softmax(post_logits)
    stored_preoverride = _vector(
        record.get("policy_probs_pre_override"), 3,
        where=f"{where}/policy_probs_pre_override",
    )
    _assert_close(
        stored_preoverride, preoverride_probs,
        where=f"{where}/policy_probs_pre_override",
    )
    if caps.context_kind is not None:
        expected_override = bool(
            preoverride_probs[0] < GOVERNANCE_CC_PROB_CEILING
            and preoverride_probs[1] - preoverride_probs[0]
            > GOVERNANCE_LOCAL_ADVANTAGE_MIN
        )
    if record.get("governance_override") is not expected_override:
        raise ValueError(f"{where}/governance_override is inconsistent")
    returned_probs = (
        np.array([0.0, 1.0, 0.0])
        if expected_override else preoverride_probs
    )
    stored_returned = _vector(
        record.get("probs"), 3, where=f"{where}/probs",
    )
    _assert_close(stored_returned, returned_probs, where=f"{where}/probs")
    return {
        "phi": phi,
        "base_logits": logits,
        "post_context_logits_pre_override": post_logits,
        "preoverride_probs": preoverride_probs,
        "returned_probs": returned_probs,
        "governance_override": expected_override,
    }
