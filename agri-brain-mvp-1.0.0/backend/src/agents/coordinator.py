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

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

_log = logging.getLogger(__name__)

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

# Cooperative agent overlay window (simulation hours).
# The cooperative agent observes, votes, and contributes to role_bias only
# while hour is in this half-open interval. Chosen to match the midday
# decision window used by the benchmark scenarios.
COOPERATIVE_OVERLAY_START: float = 12.0
COOPERATIVE_OVERLAY_END: float = 30.0


def _cooperative_window_active(hour: float) -> bool:
    return COOPERATIVE_OVERLAY_START <= hour < COOPERATIVE_OVERLAY_END


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
        self._forecast_learner = None
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
        self._step_mode: str = ""
        self._step_scenario: str = "baseline"
        self._step_role_bias: Optional[np.ndarray] = None
        self._step_supply_hat: Optional[float] = None
        self._step_supply_std: Optional[float] = None
        self._step_demand_std: Optional[float] = None
        self._step_rng_state: Optional[Dict[str, Any]] = None
        self._step_phi: Optional[np.ndarray] = None
        self._step_override: bool = False
        self._step_counterfactual_action: int = 0
        self._step_counterfactual_probs: Optional[np.ndarray] = None
        self._step_keywords: Dict[str, Any] = {}
        self._last_explanation: Optional[Dict[str, Any]] = None
        self._step_dispatch_cfg: Dict[str, Any] = {}

        # Trace exporter and protocol recorder for paper evidence
        self._trace_exporter = None
        self._protocol_recorder = None

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
            from pirag.context_learner import ContextMatrixLearner, ForecastWeightsLearner
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

        # The forecast-column learner lives outside the context-enabled
        # guard: it trains THETA[:, 6:9] for every non-static mode, not just
        # the context-enabled ones. This matches its conceptual role as a
        # base-policy learner rather than a context-pipeline learner.
        try:
            from pirag.context_learner import ForecastWeightsLearner as _FWL
            self._forecast_learner = _FWL()
        except ImportError:
            self._forecast_learner = None

        try:
            from pirag.agent_pipeline import PiRAGPipeline
            self._pirag_pipeline = PiRAGPipeline()
        except ImportError:
            pass

        try:
            from pirag.trace_exporter import TraceExporter
            self._trace_exporter = TraceExporter(max_traces=50)
        except ImportError:
            pass

        try:
            from pirag.mcp.protocol_recorder import ProtocolRecorder
            if self._mcp_server is not None:
                self._protocol_recorder = ProtocolRecorder(self._mcp_server, max_records=200)
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
        self._step_mode = ""
        self._step_scenario = "baseline"
        self._step_role_bias = None
        self._step_supply_hat = None
        self._step_supply_std = None
        self._step_demand_std = None
        self._step_rng_state = None
        self._step_phi = None
        self._step_override = False
        self._step_counterfactual_action = 0
        self._step_counterfactual_probs = None
        self._step_keywords = {}
        self._last_explanation = None
        self._step_dispatch_cfg = {}

        if self._protocol_recorder is not None:
            self._protocol_recorder.reset()
        if self._trace_exporter is not None:
            self._trace_exporter.reset()
        if self._shared_context is not None:
            self._shared_context.reset()
        if self._temporal_window is not None:
            self._temporal_window.reset()
        if self._context_learner is not None:
            self._context_learner.reset()
        if self._forecast_learner is not None:
            self._forecast_learner.reset()
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

        # Cooperative overlay: observe + generate messages during the
        # cooperative window.
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and _cooperative_window_active(hour):
            cooperative.observe(env_state, hour)

        # Compute combined role bias: primary agent + cooperative overlay
        combined_bias = active.role_bias.copy()
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and _cooperative_window_active(hour):
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
        self._step_mode = mode
        self._step_scenario = scenario
        self._step_role_bias = combined_bias
        self._step_override = False
        self._step_dispatch_cfg = {
            "enable_qos_routing": bool(getattr(policy, "enable_mcp_qos_routing", False)),
            "enable_reliability": bool(getattr(policy, "enable_mcp_reliability", False)),
            "qos_profile": "heterogeneous" if bool(getattr(policy, "enable_heterogeneous_profiles", False)) else "legacy",
            "retries": 1,
        }
        if bool(getattr(policy, "enable_heterogeneous_profiles", False)):
            role_profile = getattr(active, "profile", {})
            self._step_dispatch_cfg["role_preferred_qos"] = role_profile.get("preferred_qos", "standard")

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

        # Supply and demand forecast quantities flow into the state vector
        # (phi_6..phi_8). They are carried on obs.raw; missing keys default
        # to None, in which case build_feature_vector emits zeros on the
        # corresponding channels.
        raw = getattr(obs, "raw", {}) or {}
        supply_hat = raw.get("supply_hat")
        if isinstance(supply_hat, (list, tuple)) and supply_hat:
            supply_hat = supply_hat[0]
        supply_std = raw.get("supply_std")
        demand_std = raw.get("demand_std")
        self._step_supply_hat = supply_hat
        self._step_supply_std = supply_std
        self._step_demand_std = demand_std

        # Snapshot the RNG state before the live call consumes from it.
        # The counterfactual in post_step() rebuilds a fresh generator from
        # this state so the CF draws the same random variates the live
        # call would have drawn; the only controlled difference is then
        # context_modifier (None in the CF, computed in the live call).
        self._step_rng_state = copy.deepcopy(rng.bit_generator.state)

        # Cache phi (9D state feature vector) for the forecast-column
        # learner update in post_step. Cheap to compute; keeps post_step
        # from having to thread the forecast kwargs a second time.
        from ..models.action_selection import build_feature_vector as _bfv
        self._step_phi = _bfv(
            obs.rho, obs.inv, obs.y_hat, obs.temp,
            supply_hat=supply_hat, supply_std=supply_std, demand_std=demand_std,
        )

        theta_forecast_delta = (
            self._forecast_learner.get_theta_delta()
            if self._forecast_learner is not None else None
        )

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
            context_modifier=context_modifier,
            slca_amp_coeff=slca_amp,
            supply_hat=supply_hat,
            supply_std=supply_std,
            demand_std=demand_std,
            theta_forecast_delta=theta_forecast_delta,
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

            # MCP tool dispatch (route through protocol for recording)
            mcp_results = dispatch_tools(
                active.role, obs, self._registry, self._shared_context,
                mcp_server=self._mcp_server,
                dispatch_config=self._step_dispatch_cfg,
            )
            if isinstance(obs.raw, dict):
                flags = obs.raw.get("policy_flags", {})
                if flags.get("enable_failure_injection", False):
                    # Deterministic injection pattern for reproducibility.
                    if int(hour) % 11 == 0:
                        mcp_results["_fault_injected"] = "drop_tool_results"
                        for tool_name in list(mcp_results.get("_tools_invoked", [])):
                            mcp_results[tool_name] = None

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
            if self._context_evaluator is not None:
                cf = rag_context.get("counterfactual", {})
                if isinstance(cf, dict) and cf:
                    self._context_evaluator.record_retrieval_counterfactual(
                        hour=hour,
                        role=active.role,
                        top_doc_id=rag_context.get("top_doc_id", ""),
                        cf_top_doc_id=cf.get("top_doc_id", ""),
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

            # Cooperative overlay blending during the cooperative window.
            cooperative = self.agents.get("cooperative")
            if (cooperative is not None
                    and cooperative is not active
                    and _cooperative_window_active(obs.hour)):
                try:
                    coop_obs = cooperative.observe(obs.raw, obs.hour)
                    coop_mcp = dispatch_tools(
                        "cooperative", coop_obs, self._registry, self._shared_context,
                        mcp_server=self._mcp_server,
                        dispatch_config=self._step_dispatch_cfg,
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
                except Exception as _exc:
                    _log.debug("cooperative overlay blending skipped: %s", _exc)

            # Track which features are active (non-zero) for the learner
            psi = extract_context_features(mcp_results, rag_context, obs)
            rules_fired = [i for i in range(len(psi)) if psi[i] > 0.01]

            # Store for post_step
            self._step_mcp_results = mcp_results
            self._step_rag_context = rag_context
            self._step_context_modifier = modifier
            self._step_keywords = rag_context.get("keywords", {})
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
                "physics_consistency_score": rag_context.get("physics_consistency_score", 1.0),
                "retrieval_metrics": rag_context.get("retrieval_metrics", {}),
                "retrieval_counterfactual": rag_context.get("counterfactual", {}),
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

        # Cooperative overlay: also update and generate messages during the
        # cooperative window.
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not agent and _cooperative_window_active(hour):
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
            # The counterfactual hardcodes mode="agribrain" because all
            # four context-enabled modes (agribrain, no_context, mcp_only,
            # pirag_only) share the same base-logit branch in
            # select_action. This assertion guards against a future
            # context-enabled mode that lands in a different branch,
            # which would silently make the CF apples-to-oranges.
            assert self._step_mode in _CONTEXT_MODES, (
                f"counterfactual invariant violated: context_modifier is "
                f"set for non-context mode {self._step_mode!r}"
            )
            # Compute counterfactual action and probs (without modifier).
            # The CF rebuilds a generator from the snapshot taken in step()
            # so it draws the same random variates the live call saw; the
            # live call stays stochastic (deterministic omitted) so the
            # CF matches its sampling mode. The only controlled difference
            # is then context_modifier.
            try:
                from ..models.action_selection import select_action as _sa
                rng_cf = np.random.default_rng()
                if self._step_rng_state is not None:
                    rng_cf.bit_generator.state = copy.deepcopy(self._step_rng_state)
                theta_fcast_delta_cf = (
                    self._forecast_learner.get_theta_delta()
                    if self._forecast_learner is not None else None
                )
                action_without, probs_without = _sa(
                    mode="agribrain", rho=obs.rho, inv=obs.inv,
                    y_hat=obs.y_hat, temp=obs.temp, tau=obs.tau,
                    policy=self._step_policy,
                    rng=rng_cf, scenario=self._step_scenario,
                    hour=hour,
                    role_bias=self._step_role_bias,
                    context_modifier=None,
                    supply_hat=self._step_supply_hat,
                    supply_std=self._step_supply_std,
                    demand_std=self._step_demand_std,
                    theta_forecast_delta=theta_fcast_delta_cf,
                )
                self._step_counterfactual_action = action_without
                self._step_counterfactual_probs = probs_without
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

        # Update ForecastWeightsLearner via REINFORCE. Runs for every
        # non-static mode because the forecast channels enter phi on every
        # step regardless of whether the context pipeline is active.
        if (self._forecast_learner is not None
                and self._step_phi is not None
                and self._step_probs is not None):
            self._forecast_learner.update(
                phi=self._step_phi,
                action=action,
                probs=self._step_probs,
                reward=reward,
            )

        # Generate structured explanation and capture trace
        if self.context_enabled and self._step_mcp_results:
            try:
                from pirag.explain_decision import explain_decision
                action_names = ["cold_chain", "local_redistribute", "recovery"]
                cf_action_name = action_names[self._step_counterfactual_action] if self._step_counterfactual_probs is not None else None
                self._last_explanation = explain_decision(
                    action=action_names[action],
                    role=agent.role,
                    hour=hour,
                    obs=obs,
                    mcp_results=self._step_mcp_results,
                    rag_context=self._step_rag_context,
                    slca_score=outcome.get("slca", 0.0),
                    carbon_kg=outcome.get("carbon_kg", 0.0),
                    waste=outcome.get("waste", 0.0),
                    context_features=self._step_context_features,
                    logit_adjustment=self._step_context_modifier,
                    action_probs=self._step_probs,
                    counterfactual_action=cf_action_name,
                    counterfactual_probs=self._step_counterfactual_probs,
                    governance_override=self._step_override,
                    keywords=self._step_keywords,
                )
            except Exception:
                self._last_explanation = None

            # Update context cache for MCP resource reads
            try:
                from pirag.mcp.tools.context_features import update_context_cache
                update_context_cache(
                    features=self._step_context_features,
                    modifier=self._step_context_modifier,
                    explanation=self._last_explanation,
                    hour=hour,
                    override=self._step_override,
                    robustness={
                        "dispatch_profile": self._step_dispatch_cfg.get("qos_profile", "legacy"),
                        "reliability_enabled": bool(self._step_dispatch_cfg.get("enable_reliability", False)),
                        "fault_injected": bool(self._step_mcp_results.get("_fault_injected")),
                    },
                )
            except ImportError:
                pass

            if self._trace_exporter is not None:
                # Determine if context changed the action
                action_without = action
                if (self._context_evaluator is not None
                        and self._context_evaluator._records):
                    last_eval = self._context_evaluator._records[-1]
                    action_without = last_eval.get("action_without", action)

                self._trace_exporter.capture(
                    obs=obs,
                    scenario=self._step_scenario,
                    action=action_names[action],
                    probs=self._step_probs,
                    mcp_results=self._step_mcp_results,
                    rag_context=self._step_rag_context,
                    context_features=self._step_context_features,
                    logit_adjustment=self._step_context_modifier,
                    explanation=self._last_explanation,
                    role=agent.role,
                    action_changed=(action != action_without),
                    governance_override=self._step_override,
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
            dyn_feedback_enabled = True
            if self._step_policy is not None:
                dyn_feedback_enabled = bool(
                    getattr(self._step_policy, "enable_dynamic_knowledge_feedback", True)
                )
            if dyn_feedback_enabled and len(self._decision_history) % 24 == 0 and self._pirag_pipeline is not None:
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
        physics_scores = [float(e.get("physics_consistency_score", 1.0)) for e in self._context_log]
        faithfulness_vals = [
            float((e.get("retrieval_metrics", {}) or {}).get("faithfulness_at_3", 0.0))
            for e in self._context_log
            if e.get("retrieval_metrics")
        ]

        return {
            "total_context_steps": len(self._context_log),
            "total_mcp_tool_calls": total_tools,
            "guard_failures": guard_failures,
            "mean_modifier_magnitude": float(np.mean(modifiers)) if modifiers else 0.0,
            "nonzero_modifier_steps": sum(1 for m in modifiers if m > 1e-9),
            "governance_overrides": sum(1 for e in self._context_log if e.get("governance_override")),
            "mean_physics_consistency": float(np.mean(physics_scores)) if physics_scores else 1.0,
            "mean_retrieval_faithfulness_at_3": float(np.mean(faithfulness_vals)) if faithfulness_vals else 0.0,
            "per_role": per_role,
        }

    @property
    def trace_exporter(self):
        """Trace exporter for paper evidence (None if not initialized)."""
        return self._trace_exporter

    @property
    def protocol_recorder(self):
        """MCP protocol recorder (None if not initialized)."""
        return self._protocol_recorder

    def learner_summary(self) -> Dict[str, Any]:
        """Context learner statistics."""
        if self._context_learner is not None:
            return self._context_learner.summary()
        return {}

    def forecast_learner_summary(self) -> Dict[str, Any]:
        """Forecast-column learner statistics."""
        if self._forecast_learner is not None:
            return self._forecast_learner.summary()
        return {}

    def evaluator_summary(self) -> Dict[str, Any]:
        """Context quality evaluator statistics."""
        if self._context_evaluator is not None:
            return self._context_evaluator.summary()
        return {}
