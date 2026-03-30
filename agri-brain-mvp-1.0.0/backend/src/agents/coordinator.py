"""Multi-agent coordinator that dispatches decisions to role-specific agents.

The coordinator maps each simulation timestep to the appropriate supply
chain agent based on the lifecycle stage, delegates observation building
and action selection (via ``select_action`` from ``action_selection.py``),
and routes inter-agent messages after each step.

When ``context_enabled=True`` and mode is ``"agribrain"`` (or ablation
modes ``"mcp_only"``/``"pirag_only"``), the coordinator integrates MCP
tool dispatch, piRAG retrieval, physics-informed context modifiers,
online REINFORCE learning, and context quality evaluation.
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

# Context modes that enable MCP/piRAG infrastructure
_CONTEXT_MODES = {"agribrain", "mcp_only", "pirag_only"}

# Map operating mode to context_mode parameter for feature masking
_CONTEXT_MODE_MAP = {
    "agribrain": "full",
    "mcp_only": "mcp_only",
    "pirag_only": "pirag_only",
}


class AgentCoordinator:
    """Orchestrates multi-agent decision-making across the supply chain.

    Parameters
    ----------
    agents : optional pre-configured list of agents.  When *None*,
        one of each role is created with default biases.
    context_enabled : whether to activate MCP/piRAG context injection.
    """

    def __init__(
        self,
        agents: Optional[List[SupplyChainAgent]] = None,
        context_enabled: bool = True,
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
        self.context_enabled = context_enabled

        # Context infrastructure (lazy init, guarded by try/except)
        self._registry = None
        self._mcp_server = None
        self._shared_context = None
        self._temporal_window = None
        self._pirag_pipeline = None
        self._context_learner = None
        self._context_evaluator = None
        self._context_log: List[Dict[str, Any]] = []
        self._decision_history: List[Dict[str, Any]] = []

        # Current-step context (for post_step use)
        self._step_mcp_results: Dict[str, Any] = {}
        self._step_rag_context: Dict[str, Any] = {}
        self._step_context_modifier: Optional[np.ndarray] = None
        self._step_context_features: Optional[np.ndarray] = None
        self._step_probs: Optional[np.ndarray] = None
        self._step_rules_fired: List[int] = []
        self._step_policy: Any = None
        self._step_scenario: str = "baseline"
        self._step_role_bias: Optional[np.ndarray] = None
        self._step_override: bool = False

        if context_enabled:
            self._init_context_infrastructure()

    def _init_context_infrastructure(self) -> None:
        """Initialize MCP/piRAG infrastructure. Fails gracefully."""
        try:
            from pirag.mcp.registry import get_default_registry
            from pirag.mcp.protocol import MCPServer
            from pirag.mcp.resources import register_agent_resources
            from pirag.mcp.prompts import register_prompts
            from pirag.mcp.context_sharing import SharedContextStore
            from pirag.mcp.agent_capabilities import register_all_agent_capabilities
            from pirag.temporal_context import TemporalContextWindow
            from pirag.context_learner import ContextMatrixLearner
            from pirag.context_eval import ContextEvaluator
            from pirag.context_to_logits import THETA_CONTEXT

            self._registry = get_default_registry()
            self._mcp_server = MCPServer(registry=self._registry)

            # Register resources (use a closure for live state)
            self._agent_state_snapshot: Dict[str, Any] = {}
            register_agent_resources(
                self._mcp_server,
                lambda: self._agent_state_snapshot,
            )

            # Register prompts
            register_prompts(self._mcp_server)

            # Register agent capabilities
            register_all_agent_capabilities(self._mcp_server, self.agents)

            self._shared_context = SharedContextStore()
            self._temporal_window = TemporalContextWindow()
            self._context_learner = ContextMatrixLearner(
                initial_theta=THETA_CONTEXT,
                learning_rate=0.003,
            )
            self._context_evaluator = ContextEvaluator()

        except ImportError:
            self.context_enabled = False

        try:
            from pirag.agent_pipeline import PiRAGPipeline
            self._pirag_pipeline = PiRAGPipeline()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all agents, context stores, and logs."""
        for agent in self.agents.values():
            agent.reset()
        self._message_log.clear()
        self._context_log.clear()
        self._decision_history.clear()
        self._step_mcp_results = {}
        self._step_rag_context = {}
        self._step_context_modifier = None
        self._step_context_features = None
        self._step_probs = None
        self._step_rules_fired = []
        self._step_policy = None
        self._step_scenario = "baseline"
        self._step_role_bias = None
        self._step_override = False

        if self._shared_context is not None:
            self._shared_context.reset()
        if self._temporal_window is not None:
            self._temporal_window.reset()
        if self._context_learner is not None:
            self._context_learner.reset()
        if self._context_evaluator is not None:
            self._context_evaluator.reset()
        if self._registry is not None:
            self._registry.clear_cache()

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

        # Context injection for context-enabled modes
        context_modifier = None
        self._step_mcp_results = {}
        self._step_rag_context = {}
        self._step_context_modifier = None
        self._step_context_features = None
        self._step_probs = None
        self._step_rules_fired = []
        self._step_policy = policy
        self._step_scenario = scenario
        self._step_role_bias = combined_bias
        self._step_override = False

        context_mode = _CONTEXT_MODE_MAP.get(mode)
        if (self.context_enabled
                and context_mode is not None
                and self._registry is not None):
            context_modifier = self._compute_step_context(
                active, obs, scenario, hour,
                context_mode=context_mode,
            )

        # Get learned SLCA amp coefficient
        slca_amp = None
        if self._context_learner is not None and hasattr(self._context_learner, 'get_slca_amp'):
            slca_amp = self._context_learner.get_slca_amp()

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
            context_modifier=context_modifier,
            slca_amp_coeff=slca_amp,
        )

        # Store probs for learner update
        self._step_probs = probs

        # Track governance override
        if action_idx == 1 and probs[1] == 1.0 and context_modifier is not None:
            self._step_override = True

        return action_idx, probs, active

    def _compute_step_context(
        self,
        active: SupplyChainAgent,
        obs: Observation,
        scenario: str,
        hour: float,
        context_mode: str = "full",
    ) -> Optional[np.ndarray]:
        """Compute MCP/piRAG context modifier for the current step."""
        try:
            from pirag.mcp.tool_dispatch import dispatch_tools
            from pirag.context_builder import retrieve_role_context
            from pirag.context_to_logits import compute_context_modifier, extract_context_features

            # Update live state snapshot for MCP resources
            self._agent_state_snapshot = {
                "temp": obs.temp, "rh": obs.rh, "inv": obs.inv,
                "rho": obs.rho, "y_hat": obs.y_hat, "tau": obs.tau,
            }

            # MCP tool dispatch
            mcp_results = dispatch_tools(
                active.role, obs, self._registry, self._shared_context,
            )

            # Publish to shared context
            if self._shared_context is not None:
                for tool_name in mcp_results.get("_tools_invoked", []):
                    self._shared_context.publish(
                        active.role, tool_name, mcp_results.get(tool_name), hour,
                    )

            # piRAG retrieval
            rag_context = retrieve_role_context(
                active.role, obs, scenario, mcp_results,
                self._pirag_pipeline, self._mcp_server,
            )

            # Update temporal window
            if self._temporal_window is not None:
                guidance_type = ""
                if rag_context.get("regulatory_guidance"):
                    guidance_type = "regulatory"
                elif rag_context.get("sop_guidance"):
                    guidance_type = "sop"
                elif rag_context.get("waste_hierarchy_guidance"):
                    guidance_type = "waste_hierarchy"
                elif rag_context.get("governance_guidance"):
                    guidance_type = "governance"

                self._temporal_window.add(
                    hour, active.role, rag_context.get("query", ""),
                    rag_context.get("top_doc_id", ""),
                    rag_context.get("top_citation_score", 0.0),
                    guidance_type,
                )

            # Get learned parameters from ContextMatrixLearner
            theta_override = None
            slca_amp_override = None
            temporal_params_override = None
            if self._context_learner is not None and hasattr(self._context_learner, 'get_theta'):
                theta_override = self._context_learner.get_theta()
                slca_amp_override = self._context_learner.get_slca_amp()
                temporal_params_override = self._context_learner.get_temporal_params()

            modifier = compute_context_modifier(
                mcp_results, rag_context, obs,
                self._temporal_window,
                theta_override=theta_override,
                slca_amp_override=slca_amp_override,
                temporal_params_override=temporal_params_override,
                context_mode=context_mode,
            )

            # Cooperative overlay blending during hours 12-30
            cooperative = self.agents.get("cooperative")
            if (cooperative is not None
                    and cooperative is not active
                    and 12.0 <= obs.hour < 30.0):
                try:
                    coop_obs = cooperative.observe(obs.raw, obs.hour)
                    coop_mcp = dispatch_tools(
                        "cooperative", coop_obs, self._registry, self._shared_context,
                    )
                    coop_rag = retrieve_role_context(
                        "cooperative", coop_obs, "", coop_mcp,
                        self._pirag_pipeline, self._mcp_server,
                    )
                    coop_modifier = compute_context_modifier(
                        coop_mcp, coop_rag, coop_obs,
                        self._temporal_window,
                        theta_override=theta_override,
                        temporal_params_override=temporal_params_override,
                        context_mode=context_mode,
                    )
                    modifier = 0.7 * modifier + 0.3 * coop_modifier
                except Exception:
                    pass

            # Track which features are active (non-zero) for the learner
            psi = extract_context_features(mcp_results, rag_context, obs)
            rules_fired = [i for i in range(len(psi)) if psi[i] > 0.01]

            # Store for post_step
            self._step_mcp_results = mcp_results
            self._step_rag_context = rag_context
            self._step_context_modifier = modifier
            self._step_context_features = psi
            self._step_rules_fired = rules_fired

            # Log
            self._context_log.append({
                "hour": obs.hour,
                "role": active.role,
                "mcp_tools_invoked": mcp_results.get("_tools_invoked", []),
                "mcp_tools_skipped": mcp_results.get("_tools_skipped", []),
                "pirag_query": rag_context.get("query", ""),
                "pirag_citations": len(rag_context.get("citations", [])),
                "top_doc_id": rag_context.get("top_doc_id", ""),
                "top_citation_score": rag_context.get("top_citation_score", 0.0),
                "context_modifier": modifier.tolist() if modifier is not None else None,
                "modifier_norm": float(np.linalg.norm(modifier)),
                "guards_passed": rag_context.get("guards_passed", True),
                "rules_fired": self._step_rules_fired,
                "context_mode": context_mode,
                "governance_override": False,  # Updated after select_action
            })

            return modifier

        except ImportError:
            return None

    def post_step(
        self,
        agent: SupplyChainAgent,
        action: int,
        obs: Observation,
        outcome: Dict[str, Any],
        hour: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """Update agent state, enrich and route messages, update learner.

        Parameters
        ----------
        agent : the active agent that took the action.
        action : action index selected.
        obs : the observation built during ``step()``.
        outcome : dict with at least ``waste`` and ``rho`` keys.
        hour : current hour for cooperative overlay check.
        reward : reward received for this step (for context learner).
        """
        agent.update(action, outcome)

        # Tag override on the most recent log entry
        if self._step_override and self._context_log:
            self._context_log[-1]["governance_override"] = True

        messages = agent.generate_messages(obs, action)

        # Cooperative overlay: also update and generate messages during hours 12-30
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not agent and 12.0 <= hour < 30.0:
            cooperative.update(action, outcome)
            coop_obs = cooperative.observe(obs.raw, hour)
            messages.extend(cooperative.generate_messages(coop_obs, action))

        # Enrich messages with piRAG context if available
        if self.context_enabled and self._step_rag_context:
            try:
                from pirag.message_enrichment import enrich_message
                messages = [
                    enrich_message(msg, self._step_rag_context, self._step_mcp_results)
                    for msg in messages
                ]
            except ImportError:
                pass

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

        # Context evaluation and learner update
        if (self.context_enabled
                and self._step_context_modifier is not None
                and self._context_evaluator is not None):
            # Compute counterfactual action (without modifier)
            try:
                from ..models.action_selection import select_action as _sa
                rng_cf = np.random.default_rng(42)
                action_without, _ = _sa(
                    mode="agribrain", rho=obs.rho, inv=obs.inv,
                    y_hat=obs.y_hat, temp=obs.temp, tau=obs.tau,
                    policy=self._step_policy,
                    rng=rng_cf, scenario=self._step_scenario,
                    hour=hour,
                    role_bias=self._step_role_bias,
                    deterministic=True, context_modifier=None,
                )
            except Exception:
                action_without = action

            self._context_evaluator.record(
                hour, agent.role, action_without, action,
                reward, self._step_context_modifier,
            )

            # Update ContextMatrixLearner via REINFORCE
            if (self._context_learner is not None
                    and hasattr(self._context_learner, 'get_theta')
                    and self._step_context_features is not None
                    and self._step_probs is not None):
                self._context_learner.update(
                    psi=self._step_context_features,
                    action=action,
                    probs=self._step_probs,
                    reward=reward,
                    slca_score=outcome.get("slca", 0.0),
                )

        # Decision history for dynamic knowledge ingestion
        if self.context_enabled:
            self._decision_history.append({
                "hour": hour,
                "action": action,
                "role": agent.role,
                "slca": outcome.get("slca", 0.0),
                "carbon_kg": outcome.get("carbon_kg", 0.0),
                "waste": outcome.get("waste", 0.0),
            })

            # Periodic piRAG knowledge ingestion (every 24 steps)
            if len(self._decision_history) % 24 == 0 and self._pirag_pipeline is not None:
                try:
                    from pirag.dynamic_knowledge import ingest_decision_history
                    ingest_decision_history(
                        self._pirag_pipeline,
                        self._decision_history,
                        "simulation",
                    )
                except ImportError:
                    pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def message_log(self) -> List[InterAgentMessage]:
        """Full log of all inter-agent messages this episode."""
        return list(self._message_log)

    @property
    def context_log(self) -> List[Dict[str, Any]]:
        """Full log of context injection events."""
        return list(self._context_log)

    def agent_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Per-agent summary statistics."""
        return {role: dict(agent.state) for role, agent in self.agents.items()}

    def context_summary(self) -> Dict[str, Any]:
        """Summary of MCP and piRAG activity for paper reporting."""
        if not self._context_log:
            return {"total_context_steps": 0}

        per_role: Dict[str, Dict[str, Any]] = {}
        total_tools = 0
        guard_failures = 0

        for entry in self._context_log:
            role = entry["role"]
            if role not in per_role:
                per_role[role] = {
                    "mcp_calls": 0, "pirag_queries": 0,
                    "modifier_magnitudes": [], "guard_failures": 0,
                    "rules_fired_total": 0,
                }
            n_tools = len(entry.get("mcp_tools_invoked", []))
            per_role[role]["mcp_calls"] += n_tools
            per_role[role]["pirag_queries"] += 1 if entry.get("pirag_query") else 0
            per_role[role]["modifier_magnitudes"].append(entry.get("modifier_norm", 0.0))
            if not entry.get("guards_passed", True):
                per_role[role]["guard_failures"] += 1
                guard_failures += 1
            per_role[role]["rules_fired_total"] += len(entry.get("rules_fired", []))
            total_tools += n_tools

        # Compute per-role means
        for role in per_role:
            mags = per_role[role].pop("modifier_magnitudes")
            per_role[role]["mean_modifier_magnitude"] = float(np.mean(mags)) if mags else 0.0
            per_role[role]["nonzero_modifier_count"] = sum(1 for m in mags if m > 1e-9)

        modifiers = [e["modifier_norm"] for e in self._context_log]

        return {
            "total_context_steps": len(self._context_log),
            "total_mcp_tool_calls": total_tools,
            "guard_failures": guard_failures,
            "mean_modifier_magnitude": float(np.mean(modifiers)) if modifiers else 0.0,
            "nonzero_modifier_steps": sum(1 for m in modifiers if m > 1e-9),
            "governance_overrides": sum(1 for e in self._context_log if e.get("governance_override")),
            "per_role": per_role,
        }

    def learner_summary(self) -> Dict[str, Any]:
        """Context learner statistics."""
        if self._context_learner is not None:
            return self._context_learner.summary()
        return {}

    def evaluator_summary(self) -> Dict[str, Any]:
        """Context quality evaluator statistics."""
        if self._context_evaluator is not None:
            return self._context_evaluator.summary()
        return {}
