"""Per-batch FIFO inventory model with route-conditioned rho accumulation.

The aggregate-state simulator in ``mvp/simulation/generate_results.py``
tracks a single ``rho`` and a single ``inv`` per timestep. That makes the
"rho on retail-bound inventory" metric impossible to derive
mechanically - it has to be inferred from the action-probability mix.
``compute_effective_rho`` in :mod:`resilience` does that inference, but
a reviewer can fairly call it "an accounting transformation, not a
physical mechanism."

This module adds an explicit batch-level FIFO model that runs alongside
the aggregate state. Each batch carries its own ``current_rho`` updated
per step under one of four statuses:

  - ``dc``: at the distribution centre, accumulating rho at the DC
    coupling factor (refrigerated but not as tight as transit cold
    chain; per Mercier et al. (2017), DC temperature integrity is
    typically 0.15-0.30 of ambient deviation).
  - ``transit_cold_chain``: in transit on the long refrigerated route;
    accumulating rho at the cold-chain factor (1.00 of env_rho since
    env_rho is itself computed from the cab-level / ambient temperature
    trace).
  - ``transit_local_redistribute``: in transit on the short last-mile
    route; accumulating at 0.40 of env_rho.
  - ``transit_recovery``: in transit to a compost / feed / bioenergy
    facility; the batch leaves the retail-bound pool the moment this
    status is assigned, so its further rho accumulation is irrelevant
    to the retail-quality metric.

When a batch's transit time elapses, it transitions:

  - cold_chain or local_redistribute -> retail pool (with current rho)
  - recovery -> removed from system

The retail pool itself is FIFO: batches are sold off at a configured
rate, and the metric reported is the quantity-weighted mean rho across
batches currently in the retail pool.

This gives a physically grounded answer to "what fraction of produce
reaching the consumer is at-risk?", which is the question Figure 2
panel (b) actually wants to display.

References
----------
Aung, M.M., & Chang, Y.S. (2014). Temperature management for the
  quality assurance of a perishable food supply chain. Food Control,
  40, 198-207.
Mercier, S., Villeneuve, S., Mondor, M., & Uysal, I. (2017). Time-
  Temperature Management Along the Food Cold Chain. Comprehensive
  Reviews in Food Science and Food Safety, 16(4), 647-667.
Ndraha, N., et al. (2018). Time-temperature abuse in the food cold
  chain. Food Control, 89, 12-21.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .resilience import (
    DC_RHO_FACTOR,
    RHO_FOOD_SAFETY_CUTOFF,
    route_rho_factor,
)


# Default transit times (hours) per route, derived from the route
# distances in policy.py at a 50 km/h refrigerated-truck average speed.
# Range covers typical highway / urban / rural mixes per James & James
# (2010) Table 3.
TRANSIT_HOURS: dict[str, float] = {
    "cold_chain":         2.4,   # 120 km / 50 km/h
    "local_redistribute": 0.9,   # 45 km / 50 km/h
    "recovery":           1.6,   # 80 km / 50 km/h (irrelevant for retail metric)
}


# Default retail-pool sale rate: fraction of retail inventory cleared
# per hour. 0.04/h = ~50% half-life of 17h, matching typical fresh
# produce display-to-sale duration for leafy greens (Mercier 2017).
RETAIL_SALE_RATE_PER_HOUR: float = 0.04


@dataclass
class Batch:
    """A single batch of produce flowing through the system."""
    batch_id: int
    arrival_hour: float
    quantity: float
    current_rho: float = 0.0
    status: str = "dc"             # dc | transit_<route> | retail | sold | recovered
    transit_remaining_h: float = 0.0
    routed_to: Optional[str] = None  # the action that routed this batch
    routed_at_hour: Optional[float] = None


@dataclass
class BatchInventory:
    """FIFO batch-level inventory with route-conditioned rho tracking.

    The capacity-constraint infrastructure (LR / Recovery rolling
    admission windows, fallback ordering, lr_rejected_units counters,
    realized_route emission) was retired in 2026-04 along with the
    capacity-constrained RLE variant; the canonical RLE the paper
    reports is the EU-hierarchy + severity-weighted form
    (compute_rle in resilience.py) which scores the chosen action
    directly without needing a realised-action distinction.

    Parameters
    ----------
    initial_dc_quantity : starting batches' total quantity in the DC.
        Spread over ``initial_n_batches`` to seed the FIFO queue.
    initial_n_batches : number of batches to seed at t=0. 8 batches at
        baseline_inv (12 000 units) gives ~1500 units / batch which is
        consistent with typical DC palette sizes.
    fresh_arrival_rate : new batches per hour. 0.5/h means a fresh
        batch arrives every 2h on average. With dt=0.25h this is one
        new batch every 8 steps.
    fresh_batch_quantity : quantity of each fresh batch (units).
    sale_rate_per_hour : retail-pool clearing rate.
    dc_rho_factor : per-step env_rho coupling for DC-stored batches.
    """
    initial_dc_quantity: float = 12000.0
    initial_n_batches: int = 8
    fresh_arrival_rate_per_hour: float = 0.5
    fresh_batch_quantity: float = 1500.0
    sale_rate_per_hour: float = RETAIL_SALE_RATE_PER_HOUR
    dc_rho_factor: float = DC_RHO_FACTOR
    rng: Optional[np.random.Generator] = None

    batches: list[Batch] = field(default_factory=list)
    next_batch_id: int = 0
    next_arrival_hour: float = 0.0

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(0)
        per_batch = self.initial_dc_quantity / max(self.initial_n_batches, 1)
        for i in range(self.initial_n_batches):
            self._spawn_batch(arrival_hour=-i * 0.25, quantity=per_batch)
        # First fresh-batch arrival scheduled at t=0 + interval.
        self.next_arrival_hour = 1.0 / max(self.fresh_arrival_rate_per_hour, 1e-6)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _spawn_batch(self, arrival_hour: float, quantity: float) -> Batch:
        b = Batch(
            batch_id=self.next_batch_id,
            arrival_hour=float(arrival_hour),
            quantity=float(quantity),
        )
        self.next_batch_id += 1
        self.batches.append(b)
        return b

    def _oldest_dc_batch(self) -> Optional[Batch]:
        """FIFO: return the oldest DC batch (lowest arrival_hour)."""
        dc_batches = [b for b in self.batches if b.status == "dc"]
        if not dc_batches:
            return None
        return min(dc_batches, key=lambda b: b.arrival_hour)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def step(
        self,
        hour: float,
        d_env_rho: float,
        action_idx: int,
        ambient_temp_c: float = 4.0,
        arrival_rate_multiplier: float = 1.0,
        sale_rate_multiplier: float = 1.0,
        action_names: tuple[str, ...] = ("cold_chain", "local_redistribute", "recovery"),
        dt_hours: float = 0.25,
    ) -> dict[str, float]:
        """Advance the batch inventory one simulation step.

        Parameters
        ----------
        hour : current simulation hour (matches the hours array in the
            results JSON).
        d_env_rho : environmental rho increment for this step (the
            change in PINN env_rho since the previous step). Clamped
            to non-negative because Arrhenius spoilage is irreversible.
        action_idx : the action chosen by the policy this step. The
            oldest DC batch is routed to this action.
        ambient_temp_c : observed ambient temperature in degC at this
            step. Used by the temperature-conditional cold-chain factor
            (see resilience.route_rho_factor): batches in transit on
            the cold-chain route experience 0.15 / 0.40 / 1.00 of
            env_rho depending on whether ambient is below 30, in
            30-35, or above 35 degC. Defaults to 4 degC (cold storage)
            for legacy callers.
        arrival_rate_multiplier : per-step multiplier on the fresh
            batch arrival rate. Default 1.0 preserves baseline
            behaviour. The simulator drives this from the observed
            inventory level vs INV_BASELINE so that overproduction
            scenarios (where aggregate inventory climbs to ~5x
            baseline) actually push more batches through BatchInventory,
            forcing tier-capacity to bind under surge.
        sale_rate_multiplier : per-step multiplier on the retail
            sale rate. Default 1.0 preserves baseline. The simulator
            drives this from the observed demand level vs
            BASELINE_DEMAND so demand drops in overproduction reduce
            retail throughput in BatchInventory.
        action_names : ordered action names matching ACTIONS in
            action_selection.py.
        dt_hours : simulation timestep in hours (0.25 for 15-min ticks).

        Returns
        -------
        dict with the post-step retail-pool statistics:
            - ``effective_rho``: quantity-weighted mean rho on retail
            - ``retail_quantity``: total units in retail pool
            - ``dc_quantity``: total units still at DC
            - ``in_transit_quantity``: total units in transit
            - ``recovered_quantity``: total units sent to recovery
            - ``chosen_route``: the route the oldest DC batch was
              actually routed on this step. Equals the policy's
              chosen action unless the food-safety override fired
              (``current_rho`` > RHO_FOOD_SAFETY_CUTOFF), in which
              case it is forced to ``"recovery"``.
            - ``food_safety_override``: True iff the override fired
              this step.
        """
        d_env = max(float(d_env_rho), 0.0)

        # 1) Age all batches: each accumulates rho at its status's
        #    factor. The cold-chain factor is temperature-conditional:
        #    a batch on a CC truck during heatwave (T > 30) accumulates
        #    rho faster than at nominal temps, modelling refrigeration
        #    stress / failure per Mercier (2017) and Ndraha (2018).
        for b in self.batches:
            if b.status == "dc":
                b.current_rho = min(1.0, b.current_rho + self.dc_rho_factor * d_env)
            elif b.status.startswith("transit_"):
                route = b.status.replace("transit_", "")
                factor = route_rho_factor(route, float(ambient_temp_c))
                b.current_rho = min(1.0, b.current_rho + factor * d_env)
                b.transit_remaining_h = max(0.0, b.transit_remaining_h - dt_hours)
                if b.transit_remaining_h <= 1e-9:
                    if route == "recovery":
                        b.status = "recovered"
                    else:
                        b.status = "retail"
            elif b.status == "retail":
                # Retail-pool produce continues to age at DC factor
                # (display refrigeration is comparable to DC).
                b.current_rho = min(1.0, b.current_rho + self.dc_rho_factor * d_env)

        # 2) Route the oldest DC batch to the chosen action, subject to
        #    a food-safety hard cutoff: if the oldest DC batch has
        #    already accumulated rho above RHO_FOOD_SAFETY_CUTOFF
        #    (default 0.65, the rho threshold above which Garcia-Garcia
        #    (2017) records >=80% of food-bank network rejections at
        #    intake), the chosen action is overridden to ``recovery``
        #    regardless of policy. Real food-safety regulations behave
        #    this way: produce visibly past marketability cannot be
        #    sold even if the operator's optimisation routine would
        #    prefer otherwise. The override applies to *every* mode
        #    including Static, so it does not by itself differentiate
        #    AgriBrain from Static — the differentiation comes from
        #    the policy-side Recovery knee in action_selection.py
        #    (RHO_RECOVERY_KNEE = 0.30, soft logit boost) which
        #    triages earlier than the override fires.
        oldest = self._oldest_dc_batch()
        chosen_route = action_names[int(action_idx)]
        food_safety_override = False
        if oldest is not None and oldest.current_rho > RHO_FOOD_SAFETY_CUTOFF:
            chosen_route = "recovery"
            food_safety_override = True
        if oldest is not None:
            oldest.status = f"transit_{chosen_route}"
            oldest.transit_remaining_h = float(TRANSIT_HOURS[chosen_route])
            oldest.routed_to = chosen_route
            oldest.routed_at_hour = float(hour)

        # 4) Sell off retail-pool inventory at the configured rate,
        #    scaled by the demand multiplier (overproduction scenarios
        #    have demand below baseline so the retail pool clears
        #    slower; this is what keeps stock building up there too).
        effective_sale_rate = self.sale_rate_per_hour * max(
            float(sale_rate_multiplier), 0.0
        )
        sale_fraction = min(1.0, effective_sale_rate * dt_hours)
        retail = sorted(
            (b for b in self.batches if b.status == "retail"),
            key=lambda b: b.routed_at_hour or b.arrival_hour,
        )
        total_retail = sum(b.quantity for b in retail)
        to_sell = total_retail * sale_fraction
        for b in retail:
            if to_sell <= 1e-9:
                break
            sold = min(b.quantity, to_sell)
            b.quantity -= sold
            to_sell -= sold
            if b.quantity <= 1e-9:
                b.status = "sold"

        # 5) Spawn fresh arrivals at the surge-adjusted rate.
        #    Overproduction scenarios push the inventory level (and so
        #    the multiplier) above 1.0, increasing the per-hour
        #    arrival rate proportionally - so capacity actually binds
        #    when the underlying scenario surge would force it, not
        #    only when policy concentrates on LR.
        effective_arrival_rate = self.fresh_arrival_rate_per_hour * max(
            float(arrival_rate_multiplier), 0.0
        )
        if effective_arrival_rate > 0.0:
            interval = 1.0 / effective_arrival_rate
            while self.next_arrival_hour <= hour + dt_hours:
                self._spawn_batch(
                    arrival_hour=self.next_arrival_hour,
                    quantity=self.fresh_batch_quantity,
                )
                self.next_arrival_hour += interval

        # 6) Compute summary statistics for this step.
        retail_qty = 0.0
        retail_rho_qty = 0.0
        dc_qty = 0.0
        in_transit_qty = 0.0
        recovered_qty = 0.0
        for b in self.batches:
            if b.status == "retail":
                retail_qty += b.quantity
                retail_rho_qty += b.current_rho * b.quantity
            elif b.status == "dc":
                dc_qty += b.quantity
            elif b.status.startswith("transit_"):
                in_transit_qty += b.quantity
            elif b.status == "recovered":
                recovered_qty += b.quantity

        eff_rho = retail_rho_qty / retail_qty if retail_qty > 1e-9 else 0.0

        # 7) Garbage-collect fully sold/recovered batches to keep memory
        #    bounded. We retain a small recovered tally elsewhere.
        self.batches = [b for b in self.batches
                        if b.status not in ("sold",)]

        return {
            "effective_rho": float(eff_rho),
            "retail_quantity": float(retail_qty),
            "dc_quantity": float(dc_qty),
            "in_transit_quantity": float(in_transit_qty),
            "recovered_quantity": float(recovered_qty),
            "chosen_route": chosen_route,
            "food_safety_override": bool(food_safety_override),
        }
