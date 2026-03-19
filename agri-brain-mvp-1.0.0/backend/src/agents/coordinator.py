"""Multi-agent coordinator that dispatches decisions to role-specific agents.

The coordinator maps each simulation timestep to the appropriate supply
chain agent based on the lifecycle stage, delegates observation building
and action selection (via ``select_action`` from ``action_selection.py``),
and routes inter-agent messages after each step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Observation, SupplyChainAgent
from .message import InterAgentMessage
from .roles import (
    FarmAgent,
    ProcessorAgent,
    CooperativeAgent,
    DistributorAgent,
    RecoveryAgent,
    stage_for_hour,
)
from ..models.action_selection import select_action


class AgentCoordinator:
    """Orchestrates multi-agent decision-making across the supply chain.

    Parameters
    ----------
    agents : optional pre-configured list of agents.  When *None*,
        one of each role is created with default biases.
    """

    def __init__(
        self,
        agents: Optional[List[SupplyChainAgent]] = None,
    ) -> None:
        if agents is None:
            agent_list: List[SupplyChainAgent] = [
                FarmAgent(),
                ProcessorAgent(),
                CooperativeAgent(),
                DistributorAgent(),
                RecoveryAgent(),
            ]
        else:
            agent_list = agents

        self.agents: Dict[str, SupplyChainAgent] = {
            a.role: a for a in agent_list
        }
        self._message_log: List[InterAgentMessage] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all agents and clear the message log."""
        for agent in self.agents.values():
            agent.reset()
        self._message_log.clear()

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def get_active_agent(self, hour: float) -> SupplyChainAgent:
        """Return the agent responsible for the current lifecycle stage."""
        role = stage_for_hour(hour)
        return self.agents[role]

    def step(
        self,
        env_state: Dict[str, Any],
        hour: float,
        mode: str,
        policy: Any,
        rng: np.random.Generator,
        scenario: str = "baseline",
        rag_context: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Run one decision step through the active agent.

        The cooperative agent additionally participates as an overlay
        during hours 12-30, observing state and generating messages
        alongside the primary stage agent.

        Returns
        -------
        (action_idx, probs, active_agent)
        """
        active = self.get_active_agent(hour)
        obs = active.observe(env_state, hour)

        # Cooperative overlay: observe + generate messages during hours 12-30
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and 12.0 <= hour < 30.0:
            cooperative.observe(env_state, hour)

        # Compute combined role bias: primary agent + cooperative overlay
        combined_bias = active.role_bias.copy()
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and 12.0 <= hour < 30.0:
            combined_bias = combined_bias + cooperative.role_bias

        action_idx, probs = select_action(
            mode=mode,
            rho=obs.rho,
            inv=obs.inv,
            y_hat=obs.y_hat,
            temp=obs.temp,
            tau=obs.tau,
            policy=policy,
            rng=rng,
            scenario=scenario,
            hour=hour,
            role_bias=combined_bias,
            rag_context=rag_context,
        )

        return action_idx, probs, active

    def post_step(
        self,
        agent: SupplyChainAgent,
        action: int,
        obs: Observation,
        outcome: Dict[str, Any],
        hour: float = 0.0,
    ) -> None:
        """Update agent state and route inter-agent messages.

        Parameters
        ----------
        agent : the active agent that took the action.
        action : action index selected.
        obs : the observation built during ``step()``.
        outcome : dict with at least ``waste`` and ``rho`` keys.
        hour : current hour for cooperative overlay check.
        """
        agent.update(action, outcome)

        messages = agent.generate_messages(obs, action)

        # Cooperative overlay: also update and generate messages during hours 12-30
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not agent and 12.0 <= hour < 30.0:
            cooperative.update(action, outcome)
            coop_obs = cooperative.observe(obs.raw, hour)
            messages.extend(cooperative.generate_messages(coop_obs, action))

        for msg in messages:
            self._message_log.append(msg)
            if msg.recipient == "broadcast":
                for other in self.agents.values():
                    if other.agent_id != msg.sender:
                        other.receive_message(msg)
            else:
                for other in self.agents.values():
                    if other.agent_id == msg.recipient:
                        other.receive_message(msg)
                        break

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def message_log(self) -> List[InterAgentMessage]:
        """Full log of all inter-agent messages this episode."""
        return list(self._message_log)

    def agent_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Per-agent summary statistics."""
        return {role: dict(agent.state) for role, agent in self.agents.items()}
