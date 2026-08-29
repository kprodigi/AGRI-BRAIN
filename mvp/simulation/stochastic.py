#!/usr/bin/env python3
"""Dual-mode stochastic perturbation layer for simulation.

Seven operational uncertainty sources plus an optional policy-temperature
sensitivity parameter:

  1. Sensor noise -- temperature sigma 2.5 degC, humidity sigma 7.0 %
  2. Observed-demand variability -- multiplicative CV 25 %, applied to the
     exogenous demand-observation series before forecasting and regime logic
  3. Inventory/yield uncertainty -- multiplicative CV 22 %
  4. Transport distance jitter -- route CV 22 %
  5. Spoilage model error -- k_ref CV 20 %, Ea_R CV 14 %
  6. Scenario onset jitter -- +/- 6 hour uniform shift
  7. Policy weight perturbation -- THETA noise sigma 0.15
  8. Optional policy temperature sensitivity -- disabled by default

Plus one orthogonal channel (not counted as a "source" per the paper
narrative): telemetry lag probability 0.10 (intermittent dropouts).

DETERMINISTIC_MODE=false (default) enables seeded, bounded perturbations.
DETERMINISTIC_MODE=true disables all perturbations for strict reproducibility.

Publication runs use source/counter-keyed draws. Per-step noise therefore
depends on the environmental stream id, source name, and timestep rather than
on mutable call order inside a policy arm.

Single-source-of-truth contract: :func:`canonical_defaults` returns the
canonical env-var -> default-value mapping that callers (and the
HOW_TO_RUN doc-drift test) must consult; do not duplicate these
literals elsewhere.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np


def _is_deterministic() -> bool:
    """Read DETERMINISTIC_MODE at call time, not import time."""
    return os.environ.get("DETERMINISTIC_MODE", "false").lower() == "true"


# Property-like module attribute for backward-compatible imports.
class _DetFlag:
    def __bool__(self): return _is_deterministic()
    def __repr__(self): return str(_is_deterministic())
    def __eq__(self, other): return _is_deterministic() == other

DETERMINISTIC_MODE = _DetFlag()


@dataclass(frozen=True)
class StochasticLayer:
    rng: np.random.Generator
    enabled: bool
    # --- Source 1: Sensor noise ---
    temp_std_c: float
    rh_std: float
    # --- Source 2: Demand variability ---
    demand_frac_std: float
    # --- Source 3: Inventory/yield uncertainty ---
    inventory_frac_std: float
    # --- Source 4: Transport distance jitter ---
    transport_km_frac_std: float
    # --- Source 5: Spoilage model error (per-episode) ---
    k_ref_frac_std: float
    ea_r_frac_std: float
    # --- Source 6: Scenario onset jitter ---
    onset_jitter_hours: float
    # --- Source 7: Policy weight perturbation ---
    theta_noise_std: float
    # --- Source 8: Policy temperature heterogeneity (per-mode-per-seed) ---
    # Disabled in the confirmatory benchmark. Non-zero values are reserved
    # for an explicitly labelled sensitivity analysis.
    policy_temp_std: float
    # --- Telemetry lag (kept from original) ---
    delay_prob: float
    # Stable root for source/counter-keyed common-random-number draws. When
    # omitted, methods retain the legacy sequential ``rng`` behaviour.
    stream_seed: int | None = None

    def _rng_for(self, source: str, counter: int | None) -> np.random.Generator:
        """Return a draw stream that is independent of call order.

        Publication callers provide ``stream_seed`` plus a semantic counter
        (timestep for per-step sources, zero for episode-level sources). A
        branch in one policy arm therefore cannot shift later environmental
        draws in another arm. Legacy callers without either value continue to
        consume ``self.rng`` sequentially.
        """
        if self.stream_seed is None or counter is None:
            return self.rng
        key = (
            f"agribrain-stochastic-v1|{int(self.stream_seed)}|"
            f"{source}|{int(counter)}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return np.random.default_rng(seed)

    # ---- Source 1: Sensor noise ----

    def perturb_temperature(
        self, temp_c: float, *, counter: int | None = None,
    ) -> float:
        if not self.enabled or self.temp_std_c <= 0.0:
            return float(temp_c)
        draw_rng = self._rng_for("temperature", counter)
        return float(np.clip(
            temp_c + draw_rng.normal(0.0, self.temp_std_c), -5.0, 55.0,
        ))

    def perturb_humidity(
        self, rh: float, *, counter: int | None = None,
    ) -> float:
        if not self.enabled or self.rh_std <= 0.0:
            return float(rh)
        draw_rng = self._rng_for("humidity", counter)
        return float(np.clip(
            rh + draw_rng.normal(0.0, self.rh_std), 0.0, 100.0,
        ))

    # ---- Source 2: Demand variability ----

    def perturb_demand(
        self, demand: float, *, counter: int | None = None,
    ) -> float:
        """Perturb one observed exogenous demand value, not its forecast."""
        if not self.enabled or self.demand_frac_std <= 0.0:
            return float(demand)
        draw_rng = self._rng_for("demand", counter)
        mult = 1.0 + float(draw_rng.normal(0.0, self.demand_frac_std))
        return float(max(0.0, demand * mult))

    # ---- Source 3: Inventory/yield uncertainty ----

    def perturb_inventory(
        self, inv: float, *, counter: int | None = None,
    ) -> float:
        if not self.enabled or self.inventory_frac_std <= 0.0:
            return float(inv)
        draw_rng = self._rng_for("inventory", counter)
        mult = 1.0 + float(draw_rng.normal(0.0, self.inventory_frac_std))
        return float(max(0.0, inv * mult))

    # ---- Source 4: Transport distance jitter ----

    def perturb_transport_km(
        self, km: float, *, counter: int | None = None,
    ) -> float:
        """Jitter transport distance (detours, traffic, loading delays)."""
        return float(max(
            0.0,
            float(km) * self.perturb_transport_multiplier(counter=counter),
        ))

    def perturb_transport_multiplier(
        self, *, counter: int | None = None,
    ) -> float:
        """Draw one action-independent transport multiplier.

        Callers draw this once per routing opportunity and only then multiply
        by the selected route's declared distance.  That ordering preserves a
        common environmental transport realization across policy arms.
        """
        if not self.enabled or self.transport_km_frac_std <= 0.0:
            return 1.0
        draw_rng = self._rng_for("transport", counter)
        return float(max(0.0, 1.0 + draw_rng.normal(
            0.0, self.transport_km_frac_std,
        )))

    # ---- Source 5: Spoilage model error (call ONCE per episode) ----

    def perturb_k_ref(
        self, k_ref: float, *, counter: int | None = None,
    ) -> float:
        """Batch-to-batch variation in produce decay rate."""
        if not self.enabled or self.k_ref_frac_std <= 0.0:
            return float(k_ref)
        draw_rng = self._rng_for("k_ref", counter)
        mult = 1.0 + float(draw_rng.normal(0.0, self.k_ref_frac_std))
        return float(max(1e-6, k_ref * mult))

    def perturb_ea_r(
        self, ea_r: float, *, counter: int | None = None,
    ) -> float:
        """Batch-to-batch variation in activation energy."""
        if not self.enabled or self.ea_r_frac_std <= 0.0:
            return float(ea_r)
        draw_rng = self._rng_for("ea_r", counter)
        mult = 1.0 + float(draw_rng.normal(0.0, self.ea_r_frac_std))
        return float(max(100.0, ea_r * mult))

    # ---- Source 6: Scenario onset jitter ----

    def jitter_onset_hour(
        self, base_hour: float, *, counter: int | None = None,
    ) -> float:
        """Shift scenario onset by ±onset_jitter_hours (uniform)."""
        if not self.enabled or self.onset_jitter_hours <= 0.0:
            return float(base_hour)
        draw_rng = self._rng_for("scenario_onset", counter)
        shift = float(draw_rng.uniform(
            -self.onset_jitter_hours, self.onset_jitter_hours,
        ))
        return float(base_hour + shift)

    # ---- Source 7: Policy weight perturbation (call ONCE per seed) ----

    def perturb_theta(
        self, theta: np.ndarray, *, counter: int | None = None,
    ) -> np.ndarray:
        """Add small Gaussian noise to policy weight matrix."""
        if not self.enabled or self.theta_noise_std <= 0.0:
            return theta.copy()
        draw_rng = self._rng_for("policy_theta", counter)
        noise = draw_rng.normal(0.0, self.theta_noise_std, size=theta.shape)
        return theta + noise

    # ---- Source 8: Policy-temperature heterogeneity (per-mode-per-seed) ----

    def policy_temperature(
        self, base: float = 1.0, *, counter: int | None = None,
    ) -> float:
        """Return a per-call softmax temperature draw.

        T = base * exp(N(0, policy_temp_std))

        When enabled, models deployment-to-deployment calibration
        heterogeneity. It must not be chosen to target a desired inferential
        statistic.
        """
        if not self.enabled or self.policy_temp_std <= 0.0:
            return float(base)
        draw_rng = self._rng_for("policy_temperature", counter)
        return float(base * np.exp(draw_rng.normal(0.0, self.policy_temp_std)))

    # ---- Telemetry lag ----

    def should_delay(self, *, counter: int | None = None) -> bool:
        """Return True with probability delay_prob (telemetry lag event)."""
        if not self.enabled or self.delay_prob <= 0.0:
            return False
        draw_rng = self._rng_for("telemetry_delay", counter)
        return float(draw_rng.random()) < self.delay_prob


_DISABLED = StochasticLayer(
    rng=np.random.default_rng(0),
    enabled=False,
    temp_std_c=0.0,
    rh_std=0.0,
    demand_frac_std=0.0,
    inventory_frac_std=0.0,
    transport_km_frac_std=0.0,
    k_ref_frac_std=0.0,
    ea_r_frac_std=0.0,
    onset_jitter_hours=0.0,
    theta_noise_std=0.0,
    policy_temp_std=0.0,
    delay_prob=0.0,
)


#: Canonical env-var -> default-value mapping. Single source of truth
#: for the documented stochastic layer defaults; the HOW_TO_RUN doc
#: drift test (agribrain/backend/tests/test_doc_stoch_defaults.py) reads this dict and
#: asserts the documented env-var table matches it.
#:
#: Keys are env-var names. Values are documented defaults as strings
#: (the form a reader would type into a shell). The order matches the
#: seven operational sources, optional policy-temperature sensitivity, and
#: one orthogonal lag channel in the module docstring.
_CANONICAL_STOCH_DEFAULTS: dict[str, str] = {
    "STOCH_TEMP_STD_C":         "2.5",
    "STOCH_RH_STD":             "7.0",
    "STOCH_DEMAND_FRAC_STD":    "0.25",
    "STOCH_INVENTORY_FRAC_STD": "0.22",
    "STOCH_TRANSPORT_KM_STD":   "0.22",
    "STOCH_K_REF_STD":          "0.20",
    "STOCH_EA_R_STD":           "0.14",
    "STOCH_ONSET_JITTER_H":     "6.0",
    "STOCH_THETA_NOISE_STD":    "0.15",
    "STOCH_POLICY_TEMP_STD":    "0.0",
    "STOCH_DELAY_PROB":         "0.10",
}


def canonical_defaults() -> dict[str, str]:
    """Return a copy of the canonical env-var -> default-value mapping.

    Tests, docs, and example .env files must read from this function
    rather than re-declaring the literals. Returning a copy prevents
    callers from accidentally mutating the source-of-truth dict.
    """
    return dict(_CANONICAL_STOCH_DEFAULTS)


def make_stochastic_layer(
    rng: np.random.Generator, *, stream_seed: int | None = None,
) -> StochasticLayer:
    """Build the stochastic perturbation layer.

    Values are declared benchmark assumptions and are exposed as environment
    variables for sensitivity analysis. The confirmatory run uses these values
    without outcome-dependent retuning.
    """
    if _is_deterministic():
        return _DISABLED
    # Read every env-knob through the canonical defaults dict so the
    # default literals live in exactly one place. Calibration rationale
    # for each value lives in the module docstring + the source-of-truth
    # mapping above; this constructor is now mechanical.
    d = _CANONICAL_STOCH_DEFAULTS
    def _f(key: str) -> float:
        return float(os.environ.get(key, d[key]))
    return StochasticLayer(
        rng=rng,
        enabled=True,
        temp_std_c=_f("STOCH_TEMP_STD_C"),
        rh_std=_f("STOCH_RH_STD"),
        demand_frac_std=_f("STOCH_DEMAND_FRAC_STD"),
        inventory_frac_std=_f("STOCH_INVENTORY_FRAC_STD"),
        transport_km_frac_std=_f("STOCH_TRANSPORT_KM_STD"),
        k_ref_frac_std=_f("STOCH_K_REF_STD"),
        ea_r_frac_std=_f("STOCH_EA_R_STD"),
        onset_jitter_hours=_f("STOCH_ONSET_JITTER_H"),
        theta_noise_std=_f("STOCH_THETA_NOISE_STD"),
        policy_temp_std=_f("STOCH_POLICY_TEMP_STD"),
        delay_prob=_f("STOCH_DELAY_PROB"),
        stream_seed=stream_seed,
    )
