"""Optional exploratory operational-friction knobs.

This module is not called by the locked confirmatory simulator or by the
29-factor structural-sensitivity design. It provides three disabled-by-default
mechanisms—an author-declared token-bucket capacity, Gaussian observation
noise, and an action lockout—for separate exploratory software experiments.
The default magnitudes are illustrative hyperparameters, not field estimates,
literature calibrations, deployment constraints, or evidence for an upper-bound
performance claim. Any use requires a newly declared analysis and fresh
simulation episodes; these mechanisms cannot be applied after a run.

Frictions
---------

1. **Reroute capacity knob** — a token-bucket limiter with an
   illustrative default of 0.6 reroutes per simulated hour.

2. **Noise on observed ρ** — the exploratory policy sees
   ρ_obs = ρ_true + ε with ε ~ N(0, σ²); σ=0.03 is illustrative.

3. **Action lockout knob** — an action is held for ``lockout_steps``;
   the illustrative default is 4 steps (60 simulated minutes at the
   benchmark's 15-minute resolution).

The references below motivate broad modeling categories only. They do not
establish these defaults or validate this module for deployment.

References
----------
- Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*. Wiley.
  ISBN 0-471-49110-1. — Token-bucket and queueing fundamentals.
- Akkerman, R., Farahani, P. & Grunow, M. (2010). Quality, safety and
  sustainability in food distribution: a review of quantitative
  operations management approaches and challenges. *OR Spectrum*,
  32(4), 863–904.
- Govindan, K., Soleimani, H. & Kannan, D. (2015). Reverse logistics
  and closed-loop supply chain: A comprehensive review to explore the
  future. *European Journal of Operational Research*, 240(3), 603–626.
- Steeneck, D.W. & Sarin, S.C. (2018). Pricing and production planning
  for reverse logistics: a review. *International Journal of
  Production Research*, 49(13), 4023–4037.
- Mercier, S., Villeneuve, S., Mondor, M. & Uysal, I. (2017).
  Time-temperature management along the food cold chain: a review of
  recent developments. *Comprehensive Reviews in Food Science and
  Food Safety*, 16(4), 647–667.
- Jedermann, R., Nicometo, M., Uysal, I. & Lang, W. (2014).
  Reducing food losses by intelligent food logistics. *Philosophical
  Transactions of the Royal Society A*, 372(2017), 20130302.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .action_aliases import resolve_action as _resolve_action


@dataclass
class FrictionConfig:
    """Opt-in operational-friction configuration.

    All frictions are off when the corresponding ``enable_*`` flag is
    False (default). The locked confirmatory path does not instantiate this
    module. Values below are illustrative author-declared settings only.

    Attributes
    ----------
    enable_capacity_limit : bool
        Apply token-bucket reroute capacity constraint.
    capacity_per_hour : float
        Illustrative maximum reroute decisions per simulated hour (0.6).
    bucket_capacity : float
        Token-bucket peak burst size. Default 2.0 (two reroutes can
        be issued back-to-back before the rate limit kicks in).
    enable_sensor_noise : bool
        Apply Gaussian noise to observed ρ before the policy sees it.
    sigma_rho : float
        Illustrative standard deviation of ρ observation error (0.03).
    enable_lockout : bool
        Lock the action choice for ``lockout_steps`` after any reroute.
    lockout_steps : int
        Number of steps to hold the locked-in action. The illustrative
        default is 4 (60 simulated minutes at 15-minute resolution).
    rng_seed : int
        Seed for reproducible noise streams.
    """

    enable_capacity_limit: bool = False
    capacity_per_hour: float = 0.6
    bucket_capacity: float = 2.0

    enable_sensor_noise: bool = False
    sigma_rho: float = 0.03

    enable_lockout: bool = False
    lockout_steps: int = 4

    rng_seed: int = 0

    # Internal state (not user-facing)
    _bucket_tokens: float = field(default=0.0, init=False, repr=False)
    _last_step_hour: Optional[float] = field(default=None, init=False, repr=False)
    _lock_remaining: int = field(default=0, init=False, repr=False)
    _locked_action: Optional[str] = field(default=None, init=False, repr=False)
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._bucket_tokens = float(self.bucket_capacity)
        self._rng = np.random.default_rng(self.rng_seed)


class FrictionGate:
    """Applies opt-in operational frictions before the policy commits.

    Typical use (in the simulator's per-step loop):

        gate = FrictionGate(FrictionConfig(enable_capacity_limit=True))
        rho_obs = gate.observe_rho(rho_true)
        action_committed = gate.commit(action_proposed, rho_true, hour=t)

    With all frictions disabled (default config), ``observe_rho`` is
    the identity and ``commit`` always returns ``action_proposed``.
    """

    def __init__(self, config: Optional[FrictionConfig] = None) -> None:
        self.config = config if config is not None else FrictionConfig()

    # ------------------------------------------------------------------
    # Sensor noise
    # ------------------------------------------------------------------

    def observe_rho(self, rho_true: float) -> float:
        """Return ρ as the policy observes it.

        With sensor noise enabled, returns ρ_true + ε where
        ε ~ N(0, σ²), clipped to [0, 1]. Without noise, returns
        ρ_true unchanged.
        """
        if not self.config.enable_sensor_noise:
            return float(rho_true)
        eps = float(self.config._rng.normal(0.0, self.config.sigma_rho))
        return float(np.clip(rho_true + eps, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Capacity limit + scheduling lockout
    # ------------------------------------------------------------------

    def commit(
        self,
        proposed_action: str,
        rho_true: float,
        hour: float,
    ) -> str:
        """Commit a routing action subject to capacity and lockout.

        Parameters
        ----------
        proposed_action : the action the policy proposes.
        rho_true : true spoilage risk (used only to decide whether
            the proposed action is a reroute).
        hour : simulated time in hours (used to refill the token
            bucket).

        Returns
        -------
        Committed action. May differ from ``proposed_action`` when
        the capacity bucket is empty (proposed reroute downgrades to
        cold_chain) or a lock-out is in effect (committed action is
        the locked-in choice).
        """
        canonical = _resolve_action(proposed_action)

        # Lock-out: once a reroute is committed, the same action is
        # held for ``lockout_steps`` subsequent steps regardless of
        # what the policy proposes. Mirrors dispatch-window lock-in
        # in real reefer scheduling.
        if self.config.enable_lockout and self.config._lock_remaining > 0:
            self.config._lock_remaining -= 1
            return self.config._locked_action or canonical

        # Capacity: refill the token bucket based on elapsed hours
        # since the last step, then deduct one token if the proposed
        # action is a reroute.
        if self.config.enable_capacity_limit:
            if self.config._last_step_hour is not None:
                dt_h = max(0.0, hour - self.config._last_step_hour)
                self.config._bucket_tokens = min(
                    self.config.bucket_capacity,
                    self.config._bucket_tokens + dt_h * self.config.capacity_per_hour,
                )
            self.config._last_step_hour = hour

            if canonical in ("local_redistribute", "recovery"):
                if self.config._bucket_tokens >= 1.0:
                    self.config._bucket_tokens -= 1.0
                else:
                    # Capacity exhausted: downgrade to cold_chain
                    canonical = "cold_chain"

        # If a reroute committed and lockout is enabled, arm the lock.
        if (
            self.config.enable_lockout
            and canonical in ("local_redistribute", "recovery")
        ):
            self.config._lock_remaining = self.config.lockout_steps
            self.config._locked_action = canonical

        return canonical
