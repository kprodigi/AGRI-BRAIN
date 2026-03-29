"""Abstract base class for supply chain agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .message import InterAgentMessage


VALID_ROLES = ("farm", "processor", "cooperative", "distributor", "recovery")


@dataclass
class Observation:
    """Observation seen by an agent at a single timestep.

    Mirrors the env_state dict plus any inter-agent messages received
    since the last step.
    """
    rho: float
    inv: float
    temp: float
    rh: float
    y_hat: float
    tau: float
    hour: float
    surplus_ratio: float
    raw: Dict[str, Any] = field(default_factory=dict)
    messages: List[InterAgentMessage] = field(default_factory=list)


class SupplyChainAgent(ABC):
    """Abstract supply chain agent with a role-specific bias.

    Parameters
    ----------
    agent_id : unique identifier (e.g. ``"farm_agent"``).
    role : one of :data:`VALID_ROLES`.
    role_bias : logit bias vector of shape ``(3,)`` added to the softmax
        policy in ``select_action()``.
    """

    CONTEXT_ENABLED: bool = True
    """Whether this agent participates in MCP/piRAG context injection."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        role_bias: np.ndarray,
    ) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}; must be one of {VALID_ROLES}")
        self.agent_id = agent_id
        self.role = role
        self.role_bias = np.asarray(role_bias, dtype=np.float64).reshape(3)

        self._inbox: List[InterAgentMessage] = []

        # Local state tracking
        self.state: Dict[str, Any] = {
            "steps_handled": 0,
            "cumulative_waste": 0.0,
            "at_risk_count": 0,
            "routed_count": 0,
        }

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def observe(self, env_state: Dict[str, Any], hour: float) -> Observation:
        """Build an Observation from the current environment state."""

    @abstractmethod
    def update(self, action: int, outcome: Dict[str, Any]) -> None:
        """Update internal state after an action is taken."""

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def receive_message(self, msg: InterAgentMessage) -> None:
        """Enqueue an incoming message."""
        self._inbox.append(msg)

    def flush_inbox(self) -> List[InterAgentMessage]:
        """Return and clear all pending messages."""
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs

    def generate_messages(
        self,
        obs: Observation,
        action: int,
    ) -> List[InterAgentMessage]:
        """Produce outgoing messages (default: none)."""
        return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self._inbox.clear()
        self.state = {
            "steps_handled": 0,
            "cumulative_waste": 0.0,
            "at_risk_count": 0,
            "routed_count": 0,
        }

    # ------------------------------------------------------------------
    # MCP / piRAG context interface
    # ------------------------------------------------------------------

    def invoke_mcp_tools(
        self,
        obs: Observation,
        registry: Optional[Any] = None,
        shared_context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Invoke MCP tools for the current observation.

        Delegates to the role-specific tool dispatch workflow.
        Returns empty dict if MCP infrastructure is unavailable.
        """
        if registry is None:
            return {}
        try:
            from pirag.mcp.tool_dispatch import dispatch_tools
            return dispatch_tools(self.role, obs, registry, shared_context)
        except ImportError:
            return {}

    def get_rag_context(
        self,
        obs: Observation,
        scenario: str,
        mcp_results: Dict[str, Any],
        pipeline: Optional[Any] = None,
        mcp_server: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Retrieve piRAG context for the current observation.

        Returns empty dict if piRAG infrastructure is unavailable.
        """
        if pipeline is None:
            return {}
        try:
            from pirag.context_builder import retrieve_role_context
            return retrieve_role_context(
                self.role, obs, scenario, mcp_results, pipeline, mcp_server,
            )
        except ImportError:
            return {}

    def compute_decision_context(
        self,
        obs: Observation,
        scenario: str,
        registry: Optional[Any] = None,
        shared_context: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        temporal_window: Optional[Any] = None,
        mcp_server: Optional[Any] = None,
        rule_weights: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]:
        """Compute full decision context: MCP results, RAG context, modifier.

        Returns
        -------
        (mcp_results, rag_context, context_modifier)
        """
        mcp_results = self.invoke_mcp_tools(obs, registry, shared_context)
        rag_context = self.get_rag_context(obs, scenario, mcp_results, pipeline, mcp_server)

        context_modifier = None
        try:
            from pirag.context_to_logits import compute_context_modifier
            context_modifier = compute_context_modifier(
                mcp_results, rag_context, obs, temporal_window, rule_weights,
            )
        except ImportError:
            pass

        return mcp_results, rag_context, context_modifier
