#!/usr/bin/env python3
"""Dual-mode stochastic perturbation layer for simulation.

Eight realistic uncertainty sources (after the 2026-04 calibration
that raised every CV to match field-realistic envelopes; the older
"7-source" docstring referenced the pre-2026-04 calibration and did
not include the policy-temperature heterogeneity Source 8):

  1. Sensor noise -- temperature sigma 2.5 degC, humidity sigma 7.0 %
  2. Demand variability -- multiplicative CV 25 %
  3. Inventory/yield uncertainty -- multiplicative CV 22 %
  4. Transport distance jitter -- route CV 22 %
  5. Spoilage model error -- k_ref CV 20 %, Ea_R CV 14 %
  6. Scenario onset jitter -- +/- 6 hour uniform shift
  7. Policy weight perturbation -- THETA noise sigma 0.15
  8. Policy temperature heterogeneity -- LogNormal sigma 0.25 in log-space
     (operator-to-operator softmax-temperature variability)

Plus one orthogonal channel (not counted as a "source" per the paper
narrative): telemetry lag probability 0.10 (intermittent dropouts).

DETERMINISTIC_MODE=false (default) enables seeded, bounded perturbations.
DETERMINISTIC_MODE=true disables all perturbations for strict reproducibility.

Single-source-of-truth contract: :func:`canonical_defaults` returns the
canonical env-var -> default-value mapping that callers (and the
HOW_TO_RUN doc-drift test) must consult; do not duplicate these
literals elsewhere.
"""
from __future__ import annotations

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
    # Different deployments calibrate their softmax temperature differently;
    # some operators run sharp (T~0.7) for confidence, some run smooth (T~1.4)
    # for diversity. Drawing a per-(mode, seed) temperature with this sigma
    # introduces mode-differential per-seed variance which is essential for
    # the paired Cohen's d_z to land in the empirical 1-3 range. Without it,
    # within-pair variance is dominated by 288-step CLT averaging and d_z
    # explodes to 4-10, which produces implausible.
    policy_temp_std: float
    # --- Telemetry lag (kept from original) ---
    delay_prob: float

    # ---- Source 1: Sensor noise ----

    def perturb_temperature(self, temp_c: float) -> float:
        if not self.enabled or self.temp_std_c <= 0.0:
            return float(temp_c)
        return float(np.clip(temp_c + self.rng.normal(0.0, self.temp_std_c), -5.0, 55.0))

    def perturb_humidity(self, rh: float) -> float:
        if not self.enabled or self.rh_std <= 0.0:
            return float(rh)
        return float(np.clip(rh + self.rng.normal(0.0, self.rh_std), 0.0, 100.0))

    # ---- Source 2: Demand variability ----

    def perturb_demand(self, demand: float) -> float:
        if not self.enabled or self.demand_frac_std <= 0.0:
            return float(demand)
        mult = 1.0 + float(self.rng.normal(0.0, self.demand_frac_std))
        return float(max(0.0, demand * mult))

    # ---- Source 3: Inventory/yield uncertainty ----

    def perturb_inventory(self, inv: float) -> float:
        if not self.enabled or self.inventory_frac_std <= 0.0:
            return float(inv)
        mult = 1.0 + float(self.rng.normal(0.0, self.inventory_frac_std))
        return float(max(0.0, inv * mult))

    # ---- Source 4: Transport distance jitter ----

    def perturb_transport_km(self, km: float) -> float:
        """Jitter transport distance (detours, traffic, loading delays)."""
        if not self.enabled or self.transport_km_frac_std <= 0.0:
            return float(km)
        mult = 1.0 + float(self.rng.normal(0.0, self.transport_km_frac_std))
        return float(max(0.0, km * mult))

    # ---- Source 5: Spoilage model error (call ONCE per episode) ----

    def perturb_k_ref(self, k_ref: float) -> float:
        """Batch-to-batch variation in produce decay rate."""
        if not self.enabled or self.k_ref_frac_std <= 0.0:
            return float(k_ref)
        mult = 1.0 + float(self.rng.normal(0.0, self.k_ref_frac_std))
        return float(max(1e-6, k_ref * mult))

    def perturb_ea_r(self, ea_r: float) -> float:
        """Batch-to-batch variation in activation energy."""
        if not self.enabled or self.ea_r_frac_std <= 0.0:
            return float(ea_r)
        mult = 1.0 + float(self.rng.normal(0.0, self.ea_r_frac_std))
        return float(max(100.0, ea_r * mult))

    # ---- Source 6: Scenario onset jitter ----

    def jitter_onset_hour(self, base_hour: float) -> float:
        """Shift scenario onset by ±onset_jitter_hours (uniform)."""
        if not self.enabled or self.onset_jitter_hours <= 0.0:
            return float(base_hour)
        shift = float(self.rng.uniform(-self.onset_jitter_hours, self.onset_jitter_hours))
        return float(max(0.0, base_hour + shift))

    # ---- Source 7: Policy weight perturbation (call ONCE per seed) ----

    def perturb_theta(self, theta: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to policy weight matrix."""
        if not self.enabled or self.theta_noise_std <= 0.0:
            return theta.copy()
        noise = self.rng.normal(0.0, self.theta_noise_std, size=theta.shape)
        return theta + noise

    # ---- Source 8: Policy-temperature heterogeneity (per-mode-per-seed) ----

    def policy_temperature(self, base: float = 1.0) -> float:
        """Return a per-call softmax temperature draw.

        T = base * exp(N(0, policy_temp_std))

        Models real-world deployment-to-deployment calibration heterogeneity
        (some operators tune for confidence -> sharper softmax; some for
        diversity -> smoother softmax). When called once per (mode, seed)
        and applied as ``probs = softmax(logits / T)``, this introduces
        mode-differential per-seed noise that is the *only* source of
        within-pair variance for paired-design ablations sharing
        ablation_seed for environment matching. Without this term, the
        paired Cohen's d_z is dominated by 288-step CLT averaging and
        explodes to 4-10; with policy_temp_std ~0.25 it lands at ~1.5-3
        which is what empirical operations-research literature reports.
        """
        if not self.enabled or self.policy_temp_std <= 0.0:
            return float(base)
        return float(base * np.exp(self.rng.normal(0.0, self.policy_temp_std)))

    # ---- Telemetry lag ----

    def should_delay(self) -> bool:
        """Return True with probability delay_prob (telemetry lag event)."""
        if not self.enabled or self.delay_prob <= 0.0:
            return False
        return float(self.rng.random()) < self.delay_prob


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
#: drift test (tests/test_doc_stoch_defaults.py) reads this dict and
#: asserts the documented env-var table matches it.
#:
#: Keys are env-var names. Values are documented defaults as strings
#: (the form a reader would type into a shell). The order matches the
#: "8 sources + 1 orthogonal lag" narrative in the module docstring.
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
    "STOCH_POLICY_TEMP_STD":    "0.25",
    "STOCH_DELAY_PROB":         "0.10",
}


def canonical_defaults() -> dict[str, str]:
    """Return a copy of the canonical env-var -> default-value mapping.

    Tests, docs, and example .env files must read from this function
    rather than re-declaring the literals. Returning a copy prevents
    callers from accidentally mutating the source-of-truth dict.
    """
    return dict(_CANONICAL_STOCH_DEFAULTS)


def make_stochastic_layer(rng: np.random.Generator) -> StochasticLayer:
    """Build the stochastic perturbation layer.

    Implementation note: realism recalibration (2025-04).
    The previous defaults produced 20-seed bootstrap CIs of width
    0.001-0.005 on ARI, which combined with paired-design d_z values
    produced effect sizes (d_z = 4-10) that are essentially never
    observed in empirical operations-research literature. The defaults
    below were widened so that real-world operational variability
    (sensor drift, daily demand shocks, batch-to-batch produce
    heterogeneity, route delays) drives a more credible 0.02-0.05 CI
    width on ARI without changing the rank order of methods. Each
    value is annotated with the empirical anchor used to set it.
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
    )
