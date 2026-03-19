"""Concrete supply chain agent roles with stage-specific biases.

Each agent corresponds to a lifecycle stage of the produce supply chain.
Role biases are small relative to THETA-phi and mode bonuses in
``action_selection.py``, nudging decisions toward the mandate of each
stage without overriding the global policy.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Observation, SupplyChainAgent
from .message import InterAgentMessage, MessageType


# ---------------------------------------------------------------------------
# Stage boundaries (hours since harvest)
# ---------------------------------------------------------------------------
_STAGE_BOUNDARIES = [
    ("farm",        0.0,  18.0),
    ("processor",  18.0,  36.0),
    ("distributor", 36.0, 54.0),
    ("recovery",   54.0,  float("inf")),
]
"""Primary stage boundaries mapping lifecycle hours to the four supply chain
stages. The cooperative agent does not appear here; it operates as an
always-active overlay during hours 12-30 (see ``AgentCoordinator.step``).
"""


def stage_for_hour(hour: float) -> str:
    """Return the primary supply chain stage for a given hour.

    The cooperative agent is not returned here — it participates as an
    overlay during hours 12-30, handled by the AgentCoordinator.
    """
    for role, lo, hi in _STAGE_BOUNDARIES:
        if lo <= hour < hi:
            return role
    return "recovery"


# ---------------------------------------------------------------------------
# FarmAgent
# ---------------------------------------------------------------------------
class FarmAgent(SupplyChainAgent):
    """Preserves freshness; sends SPOILAGE_ALERT when rho > 0.25."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="farm_agent",
            role="farm",
            role_bias=np.array([+0.12, -0.05, -0.07]),
        )

    def observe(self, env_state: Dict[str, Any], hour: float) -> Observation:
        return Observation(
            rho=env_state["rho"],
            inv=env_state["inv"],
            temp=env_state["temp"],
            rh=env_state["rh"],
            y_hat=env_state["y_hat"],
            tau=env_state["tau"],
            hour=hour,
            surplus_ratio=env_state.get("surplus_ratio", 0.0),
            raw=env_state,
            messages=self.flush_inbox(),
        )

    def update(self, action: int, outcome: Dict[str, Any]) -> None:
        self.state["steps_handled"] += 1
        self.state["cumulative_waste"] += outcome.get("waste", 0.0)
        if outcome.get("rho", 0.0) > 0.10:
            self.state["at_risk_count"] += 1
            if action in (1, 2):
                self.state["routed_count"] += 1

    def generate_messages(
        self, obs: Observation, action: int
    ) -> List[InterAgentMessage]:
        msgs: List[InterAgentMessage] = []
        if obs.rho > 0.25:
            msgs.append(InterAgentMessage(
                sender=self.agent_id,
                recipient="processor_agent",
                msg_type=MessageType.SPOILAGE_ALERT,
                payload={"rho": obs.rho, "temp": obs.temp},
                hour=obs.hour,
            ))
        return msgs


# ---------------------------------------------------------------------------
# ProcessorAgent
# ---------------------------------------------------------------------------
class ProcessorAgent(SupplyChainAgent):
    """Processing efficiency; sends SURPLUS_ALERT when surplus_ratio > 0.5."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="processor_agent",
            role="processor",
            role_bias=np.array([-0.06, +0.14, -0.08]),
        )

    def observe(self, env_state: Dict[str, Any], hour: float) -> Observation:
        return Observation(
            rho=env_state["rho"],
            inv=env_state["inv"],
            temp=env_state["temp"],
            rh=env_state["rh"],
            y_hat=env_state["y_hat"],
            tau=env_state["tau"],
            hour=hour,
            surplus_ratio=env_state.get("surplus_ratio", 0.0),
            raw=env_state,
            messages=self.flush_inbox(),
        )

    def update(self, action: int, outcome: Dict[str, Any]) -> None:
        self.state["steps_handled"] += 1
        self.state["cumulative_waste"] += outcome.get("waste", 0.0)
        if outcome.get("rho", 0.0) > 0.10:
            self.state["at_risk_count"] += 1
            if action in (1, 2):
                self.state["routed_count"] += 1

    def generate_messages(
        self, obs: Observation, action: int
    ) -> List[InterAgentMessage]:
        msgs: List[InterAgentMessage] = []
        if obs.surplus_ratio > 0.5:
            msgs.append(InterAgentMessage(
                sender=self.agent_id,
                recipient="distributor_agent",
                msg_type=MessageType.SURPLUS_ALERT,
                payload={"surplus_ratio": obs.surplus_ratio, "inv": obs.inv},
                hour=obs.hour,
            ))
        return msgs


# ---------------------------------------------------------------------------
# DistributorAgent
# ---------------------------------------------------------------------------
class DistributorAgent(SupplyChainAgent):
    """Community redistribution; sends REROUTE_REQUEST when rho > 0.45."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="distributor_agent",
            role="distributor",
            role_bias=np.array([-0.12, +0.28, -0.16]),
        )

    def observe(self, env_state: Dict[str, Any], hour: float) -> Observation:
        return Observation(
            rho=env_state["rho"],
            inv=env_state["inv"],
            temp=env_state["temp"],
            rh=env_state["rh"],
            y_hat=env_state["y_hat"],
            tau=env_state["tau"],
            hour=hour,
            surplus_ratio=env_state.get("surplus_ratio", 0.0),
            raw=env_state,
            messages=self.flush_inbox(),
        )

    def update(self, action: int, outcome: Dict[str, Any]) -> None:
        self.state["steps_handled"] += 1
        self.state["cumulative_waste"] += outcome.get("waste", 0.0)
        if outcome.get("rho", 0.0) > 0.10:
            self.state["at_risk_count"] += 1
            if action in (1, 2):
                self.state["routed_count"] += 1

    def generate_messages(
        self, obs: Observation, action: int
    ) -> List[InterAgentMessage]:
        msgs: List[InterAgentMessage] = []
        if obs.rho > 0.45:
            msgs.append(InterAgentMessage(
                sender=self.agent_id,
                recipient="recovery_agent",
                msg_type=MessageType.REROUTE_REQUEST,
                payload={"rho": obs.rho, "action": action},
                hour=obs.hour,
            ))
        return msgs


# ---------------------------------------------------------------------------
# CooperativeAgent
# ---------------------------------------------------------------------------
class CooperativeAgent(SupplyChainAgent):
    """Coordinates supply chain actors; broadcasts COORDINATION_UPDATE when
    per-step demand forecast exceeds 0.12% of current inventory (max 60
    per episode)."""

    MAX_COORDINATION_BROADCASTS = 60

    def __init__(self) -> None:
        super().__init__(
            agent_id="cooperative_agent",
            role="cooperative",
            role_bias=np.array([-0.04, +0.10, -0.06]),
        )
        self._coordination_broadcasts = 0

    def observe(self, env_state: Dict[str, Any], hour: float) -> Observation:
        return Observation(
            rho=env_state["rho"],
            inv=env_state["inv"],
            temp=env_state["temp"],
            rh=env_state["rh"],
            y_hat=env_state["y_hat"],
            tau=env_state["tau"],
            hour=hour,
            surplus_ratio=env_state.get("surplus_ratio", 0.0),
            raw=env_state,
            messages=self.flush_inbox(),
        )

    def update(self, action: int, outcome: Dict[str, Any]) -> None:
        self.state["steps_handled"] += 1
        self.state["cumulative_waste"] += outcome.get("waste", 0.0)
        if outcome.get("rho", 0.0) > 0.10:
            self.state["at_risk_count"] += 1
            if action in (1, 2):
                self.state["routed_count"] += 1

    def generate_messages(
        self, obs: Observation, action: int
    ) -> List[InterAgentMessage]:
        msgs: List[InterAgentMessage] = []
        # Broadcast coordination update when per-step demand forecast
        # exceeds 0.12% of current inventory (operational trigger for
        # demand-inventory imbalance detection).
        if (self._coordination_broadcasts < self.MAX_COORDINATION_BROADCASTS
                and obs.y_hat > obs.inv * 0.0012):
            msgs.append(InterAgentMessage(
                sender=self.agent_id,
                recipient="broadcast",
                msg_type=MessageType.COORDINATION_UPDATE,
                payload={
                    "demand_forecast": obs.y_hat,
                    "inventory": obs.inv,
                    "rho": obs.rho,
                },
                hour=obs.hour,
            ))
            self._coordination_broadcasts += 1
        return msgs

    def reset(self) -> None:
        super().reset()
        self._coordination_broadcasts = 0


# ---------------------------------------------------------------------------
# RecoveryAgent
# ---------------------------------------------------------------------------
class RecoveryAgent(SupplyChainAgent):
    """Waste valorization; broadcasts CAPACITY_UPDATE (max 80 per episode)."""

    MAX_CAPACITY_BROADCASTS = 80

    def __init__(self) -> None:
        super().__init__(
            agent_id="recovery_agent",
            role="recovery",
            role_bias=np.array([-0.12, -0.05, +0.17]),
        )
        self._capacity_broadcasts = 0

    def observe(self, env_state: Dict[str, Any], hour: float) -> Observation:
        return Observation(
            rho=env_state["rho"],
            inv=env_state["inv"],
            temp=env_state["temp"],
            rh=env_state["rh"],
            y_hat=env_state["y_hat"],
            tau=env_state["tau"],
            hour=hour,
            surplus_ratio=env_state.get("surplus_ratio", 0.0),
            raw=env_state,
            messages=self.flush_inbox(),
        )

    def update(self, action: int, outcome: Dict[str, Any]) -> None:
        self.state["steps_handled"] += 1
        self.state["cumulative_waste"] += outcome.get("waste", 0.0)
        if outcome.get("rho", 0.0) > 0.10:
            self.state["at_risk_count"] += 1
            if action in (1, 2):
                self.state["routed_count"] += 1

    def generate_messages(
        self, obs: Observation, action: int
    ) -> List[InterAgentMessage]:
        msgs: List[InterAgentMessage] = []
        if self._capacity_broadcasts < self.MAX_CAPACITY_BROADCASTS:
            remaining = self.MAX_CAPACITY_BROADCASTS - self._capacity_broadcasts
            msgs.append(InterAgentMessage(
                sender=self.agent_id,
                recipient="broadcast",
                msg_type=MessageType.CAPACITY_UPDATE,
                payload={"remaining_capacity": remaining},
                hour=obs.hour,
            ))
            self._capacity_broadcasts += 1
        return msgs

    def reset(self) -> None:
        super().reset()
        self._capacity_broadcasts = 0
