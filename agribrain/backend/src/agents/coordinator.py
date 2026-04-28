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

_log = logging.getLogger(__name__)

# Context modes that enable MCP/piRAG infrastructure
_CONTEXT_MODES = {
    "agribrain", "mcp_only", "pirag_only",
    # Paper §4.7 ablation modes: zero-init REINFORCE + three perturbation
    # strengths of the hand-calibrated prior. They share agribrain's full
    # context pipeline; only the THETA_CONTEXT initialization differs.
    "agribrain_cold_start",
    "agribrain_pert_10", "agribrain_pert_25", "agribrain_pert_50",
}

# Map operating mode to context_mode parameter for feature masking. Cold-
# start and perturbation ablations use the full ψ (all five components);
# mcp_only / pirag_only still mask to their respective subsets.
_CONTEXT_MODE_MAP = {
    "agribrain": "full",
    "mcp_only": "mcp_only",
    "pirag_only": "pirag_only",
    "agribrain_cold_start": "full",
    "agribrain_pert_10": "full",
    "agribrain_pert_25": "full",
    "agribrain_pert_50": "full",
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
        context_learner_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        context_learner_overrides : optional dict of keyword arguments passed
            verbatim to :class:`ContextMatrixLearner`. Used by the cold-start
            ablation and sensitivity ablation modes to override the default
            ``learning_rate``, ``initial_theta``, ``magnitude_cap_mode`` etc.
            When ``None`` the learner is instantiated with the default
            hand-calibrated initial matrix and production hyperparameters.
        """
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
        self._context_learner_overrides: Dict[str, Any] = dict(
            context_learner_overrides or {}
        )

        # Context infrastructure (lazy init, guarded by try/except)
        self._registry = None
        self._mcp_server = None
        self._shared_context = None
        self._temporal_window = None
        self._pirag_pipeline = None
        self._context_learner = None
        self._theta_learner = None
        self._theta_learners: Dict[str, Any] = {}
        self._reward_shaping_learner = None
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
        self._step_price_signal: Optional[float] = None
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
            learner_kwargs: Dict[str, Any] = {
                "initial_theta": THETA_CONTEXT,
                "learning_rate": 0.02,
                "magnitude_cap_mode": "relative_delta",
                "magnitude_cap_value": 0.5,
                "magnitude_cap_abs_floor": 0.10,
            }
            # Cold-start and sensitivity ablation modes override any of
            # these (e.g. initial_theta=zeros, magnitude_cap_mode="absolute",
            # perturbed initial_theta). Production agribrain runs without
            # overrides, so its behavior is the refined-default above.
            learner_kwargs.update(self._context_learner_overrides)
            self._context_learner = ContextMatrixLearner(**learner_kwargs)
            self._context_evaluator = ContextEvaluator()

        except ImportError:
            self.context_enabled = False

        # Per-role policy-delta learners. The 2026-04 audit pointed out
        # that "online REINFORCE" with a SHARED learner across all five
        # roles makes the multi-agent framing thin: every role's
        # gradients update the same Theta_delta, so role-specific
        # mandates ("preserve freshness", "waste valorization") cannot
        # diverge in their learned corrections. The fix here gives each
        # role its own PolicyDeltaLearner instance, keyed by role name.
        # ``self._theta_learner`` retains the previous singleton API
        # (used by tests, learner_summary export, and the context
        # learner integration) and now points at the *active* role's
        # learner each step. ``self._theta_learners`` is the dict of
        # all five — production code that wants the active learner uses
        # ``_theta_learner``; code that wants the full set uses
        # ``_theta_learners``.
        # IMPORTANT: declared OUTSIDE the try/except so attribute access
        # in reset() / step() is always safe even when the import below
        # fails (the legacy single-learner code path takes over).
        self._theta_learners: Dict[str, Any] = {}
        self._theta_learner = None
        try:
            from pirag.context_learner import PolicyDeltaLearner as _PDL
            from ..models.action_selection import THETA as _INITIAL_THETA
            for _role_name in ("farm", "processor", "cooperative",
                                "distributor", "recovery"):
                self._theta_learners[_role_name] = _PDL(initial_theta=_INITIAL_THETA)
            # Default to the farm-stage learner; ``step`` will set this
            # to the active role's instance per timestep.
            self._theta_learner = self._theta_learners["farm"]
        except ImportError:
            self._theta_learner = None

        # The reward-shaping learner applies the same delta-with-cap
        # pattern to SLCA_BONUS, SLCA_RHO_BONUS, and NO_SLCA_OFFSET so
        # the hand-calibrated reward-shaping constants are no longer the
        # only hand-tuned piece in the policy path. Mode-conditional
        # gradients route per the same rules select_action uses.
        try:
            from pirag.context_learner import RewardShapingLearner as _RSL
            from ..models.action_selection import (
                SLCA_BONUS as _INITIAL_SLCA_BONUS,
                SLCA_RHO_BONUS as _INITIAL_SLCA_RHO_BONUS,
                NO_SLCA_OFFSET as _INITIAL_NO_SLCA_OFFSET,
            )
            self._reward_shaping_learner = _RSL(
                initial_slca_bonus=_INITIAL_SLCA_BONUS,
                initial_slca_rho_bonus=_INITIAL_SLCA_RHO_BONUS,
                initial_no_slca_offset=_INITIAL_NO_SLCA_OFFSET,
            )
        except ImportError:
            self._reward_shaping_learner = None

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
        """Reset all agents, context stores, logs, and per-episode counters."""
        for agent in self.agents.values():
            agent.reset()
        self._message_log.clear()
        self._context_log.clear()
        self._decision_history.clear()
        # Reset the MCP dispatch-id counter so per-episode protocol
        # traces use comparable id ranges (prevents the global counter
        # from growing unboundedly across the simulator's mode/scenario
        # loop).
        try:
            from pirag.mcp.tool_dispatch import reset_dispatch_id_counter
            reset_dispatch_id_counter()
        except Exception:
            pass
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
        self._step_price_signal = None
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
        # Reset every per-role theta learner (not just the active one)
        # so a new episode starts every role from a clean delta.
        for _learner in self._theta_learners.values():
            _learner.reset()
        if self._theta_learner is not None and not self._theta_learners:
            # Legacy single-learner code path (only triggers if the
            # per-role import failed and we fell back to None).
            self._theta_learner.reset()
        if self._reward_shaping_learner is not None:
            self._reward_shaping_learner.reset()
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
        policy_temperature: float = 1.0,
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

        # Route ``_theta_learner`` to the active role's per-role learner
        # so this step's `get_theta_delta()` and `update(...)` calls
        # operate on the role-specific Theta_delta. Cooperative overlay
        # is handled separately below.
        if self._theta_learners and active.role in self._theta_learners:
            self._theta_learner = self._theta_learners[active.role]
        obs = active.observe(env_state, hour)

        # Cooperative overlay: observe + generate messages during the
        # cooperative window.
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and _cooperative_window_active(hour):
            cooperative.observe(env_state, hour)

        # Compute combined role bias: primary agent + cooperative overlay
        # + inter-agent message bias.
        combined_bias = active.role_bias.copy()
        cooperative = self.agents.get("cooperative")
        if cooperative is not None and cooperative is not active and _cooperative_window_active(hour):
            combined_bias = combined_bias + cooperative.role_bias

        # 2026-04 fix: messages received in the active agent's inbox now
        # actually shape the decision. The previous implementation
        # appended messages to ``Observation.messages`` and then did
        # nothing with them, making the documented protocol
        # (SPOILAGE_ALERT / SURPLUS_ALERT / CAPACITY_UPDATE /
        # REROUTE_REQUEST / ACK) non-falsifiable. ``message_bias_from_inbox``
        # converts the flushed inbox into a bounded logit nudge in
        # action space; the bias is added to ``combined_bias`` here so
        # the same code path that consumes role_bias also consumes
        # message-derived bias.
        try:
            from .message import message_bias_from_inbox as _mbias
            inbox_bias = _mbias(getattr(obs, "messages", []) or [])
            combined_bias = combined_bias + inbox_bias
            self._step_message_bias = inbox_bias
        except Exception:
            self._step_message_bias = np.zeros(3)

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
        # (phi_6..phi_8) and the price-volatility proxy feeds phi_9. They
        # are carried on obs.raw; missing keys default to None, in which
        # case build_feature_vector emits zeros on the corresponding
        # channels.
        raw = getattr(obs, "raw", {}) or {}
        supply_hat = raw.get("supply_hat")
        if isinstance(supply_hat, (list, tuple)) and supply_hat:
            supply_hat = supply_hat[0]
        supply_std = raw.get("supply_std")
        demand_std = raw.get("demand_std")
        price_signal = raw.get("price_signal")
        self._step_supply_hat = supply_hat
        self._step_supply_std = supply_std
        self._step_demand_std = demand_std
        self._step_price_signal = price_signal

        # Snapshot the RNG state before the live call consumes from it.
        # The counterfactual in post_step() rebuilds a fresh generator from
        # this state so the CF draws the same random variates the live
        # call would have drawn; the only controlled difference is then
        # context_modifier (None in the CF, computed in the live call).
        self._step_rng_state = copy.deepcopy(rng.bit_generator.state)

        # Cache phi (10D state feature vector) for the forecast-column
        # learner update in post_step. Cheap to compute; keeps post_step
        # from having to thread the forecast kwargs a second time.
        from ..models.action_selection import build_feature_vector as _bfv
        self._step_phi = _bfv(
            obs.rho, obs.inv, obs.y_hat, obs.temp,
            supply_hat=supply_hat, supply_std=supply_std, demand_std=demand_std,
            price_signal=price_signal,
        )

        theta_delta = (
            self._theta_learner.get_theta_delta()
            if self._theta_learner is not None else None
        )
        if self._reward_shaping_learner is not None:
            _slca_bonus_delta = self._reward_shaping_learner.get_slca_bonus_delta()
            _slca_rho_delta = self._reward_shaping_learner.get_slca_rho_delta()
            _no_slca_offset_delta = self._reward_shaping_learner.get_no_slca_offset_delta()
        else:
            _slca_bonus_delta = None
            _slca_rho_delta = None
            _no_slca_offset_delta = None

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
            price_signal=price_signal,
            theta_delta=theta_delta,
            slca_bonus_delta=_slca_bonus_delta,
            slca_rho_delta=_slca_rho_delta,
            no_slca_offset_delta=_no_slca_offset_delta,
            policy_temperature=policy_temperature,
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

            # Cooperative overlay during the 12-30h cooperative window.
            # Two paths: (1) modifier blending (continuous nudge,
            # `0.7*primary + 0.3*coop`); (2) cooperative *veto* — when
            # the cooperative agent's own context analysis surfaces a
            # critical compliance violation that the primary stage
            # missed, the cooperative replaces the primary modifier
            # entirely with a recovery-biased override. Veto is the
            # genuine hierarchical signal the README's "cooperative
            # governance" claim was promising but the previous
            # implementation never delivered.
            cooperative = self.agents.get("cooperative")
            self._step_cooperative_veto = False
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

                    # Veto trigger: cooperative's compliance check
                    # flags a critical violation AND the primary
                    # agent's MCP results did not flag the same. This
                    # is the case where the cooperative stage saw a
                    # piece of governance state the primary missed.
                    coop_compliance = (coop_mcp.get("check_compliance") or {})
                    primary_compliance = (mcp_results.get("check_compliance") or {})
                    coop_critical = bool(
                        not coop_compliance.get("compliant", True)
                        and any(
                            v.get("severity") == "critical"
                            for v in coop_compliance.get("violations", []) or []
                        )
                    )
                    primary_missed = not (
                        not primary_compliance.get("compliant", True)
                        and any(
                            v.get("severity") == "critical"
                            for v in primary_compliance.get("violations", []) or []
                        )
                    )

                    if coop_critical and primary_missed:
                        # Hierarchical override: cooperative's modifier
                        # replaces the primary's. Bias toward local
                        # redistribution (action 1) so the next-step
                        # decision honours the cooperative's safety
                        # signal even when the active agent's local
                        # MCP did not surface it.
                        veto_bias = np.array([-0.20, +0.20, 0.0])
                        modifier = coop_modifier + veto_bias
                        self._step_cooperative_veto = True
                    else:
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
                theta_delta_cf = (
                    self._theta_learner.get_theta_delta()
                    if self._theta_learner is not None else None
                )
                if self._reward_shaping_learner is not None:
                    _slca_bonus_delta_cf = self._reward_shaping_learner.get_slca_bonus_delta()
                    _slca_rho_delta_cf = self._reward_shaping_learner.get_slca_rho_delta()
                    _no_slca_offset_delta_cf = self._reward_shaping_learner.get_no_slca_offset_delta()
                else:
                    _slca_bonus_delta_cf = None
                    _slca_rho_delta_cf = None
                    _no_slca_offset_delta_cf = None
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
                    price_signal=self._step_price_signal,
                    theta_delta=theta_delta_cf,
                    slca_bonus_delta=_slca_bonus_delta_cf,
                    slca_rho_delta=_slca_rho_delta_cf,
                    no_slca_offset_delta=_no_slca_offset_delta_cf,
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

        # Update PolicyDeltaLearner via REINFORCE. Runs for every
        # non-static mode so the (3, 10) delta is trained uniformly
        # across ablations, which keeps the ablation structure coherent
        # (every mode uses the same learner with the same anchor).
        if (self._theta_learner is not None
                and self._step_phi is not None
                and self._step_probs is not None):
            self._theta_learner.update(
                phi=self._step_phi,
                action=action,
                probs=self._step_probs,
                reward=reward,
            )

        # Update RewardShapingLearner via REINFORCE. Mode-conditional
        # gradient routing internally; shrinkage applies on every call.
        if (self._reward_shaping_learner is not None
                and self._step_probs is not None
                and self._step_mode):
            self._reward_shaping_learner.update(
                action=action,
                probs=self._step_probs,
                reward=reward,
                mode=self._step_mode,
                rho=float(getattr(obs, "rho", 0.0)),
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

            # Periodic piRAG knowledge ingestion (every 24 steps).
            # Disabled by default as of 2026-04 because re-ingested
            # documents are autogenerated summary statistics of the
            # agent's own past actions; the loop creates a
            # self-amplification effect that biases retrieval. Enable
            # explicitly via Policy.enable_dynamic_knowledge_feedback or
            # the DYNAMIC_KB_FEEDBACK env var for ablation studies.
            import os as _os_dyn
            # Default-on so the §3.7 blockchain-to-piRAG feedback loop is
            # active in standard runs. Older builds defaulted off; set
            # DYNAMIC_KB_FEEDBACK=false explicitly to reproduce them.
            dyn_feedback_enabled = (
                _os_dyn.environ.get("DYNAMIC_KB_FEEDBACK", "true").lower() == "true"
            )
            if self._step_policy is not None:
                # Policy-level setting takes precedence when set.
                dyn_feedback_enabled = bool(
                    getattr(self._step_policy, "enable_dynamic_knowledge_feedback", False)
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

    def theta_learner_summary(self) -> Dict[str, Any]:
        """Policy-delta learner statistics."""
        if self._theta_learner is not None:
            return self._theta_learner.summary()
        return {}

    def reward_shaping_learner_summary(self) -> Dict[str, Any]:
        """Reward-shaping learner statistics."""
        if self._reward_shaping_learner is not None:
            return self._reward_shaping_learner.summary()
        return {}

    def save_learner_states(self) -> Dict[str, Any]:
        """Serialise all learner states into one JSON-friendly dict.

        Use this at the end of a long HPC episode (or crash-resume point)
        to persist learned weights across runs. The returned dict can be
        written with ``json.dump`` and later restored via
        :meth:`load_learner_states`.
        """
        state: Dict[str, Any] = {}
        if self._context_learner is not None:
            state["context_learner"] = self._context_learner.save_state()
        # Per-role theta learners. Save each role's state under a
        # role-keyed dict so the snapshot survives the coordinator's
        # rotation through the role schedule. The legacy "theta_learner"
        # key is retained pointing at the active role's state for
        # consumers that don't yet know about per-role learners.
        if self._theta_learners:
            state["theta_learners"] = {
                role: lrn.save_state()
                for role, lrn in self._theta_learners.items()
            }
        if self._theta_learner is not None:
            state["theta_learner"] = self._theta_learner.save_state()
        if self._reward_shaping_learner is not None:
            state["reward_shaping_learner"] = self._reward_shaping_learner.save_state()
        return state

    def load_learner_states(self, state: Dict[str, Any]) -> None:
        """Restore learner state produced by :meth:`save_learner_states`.

        Missing keys are tolerated so partial checkpoints (e.g. only the
        theta learner) still work. Attempting to load into a coordinator
        whose learner was never constructed (import-time failure) is a
        no-op for that slot.
        """
        ctx = state.get("context_learner")
        if ctx is not None and self._context_learner is not None:
            self._context_learner.load_state(ctx)
        # Per-role theta learners restored first; the legacy
        # `theta_learner` key (active-role state only) is then applied
        # for back-compat with snapshots produced by the previous
        # singleton-learner code path.
        per_role = state.get("theta_learners") or {}
        if isinstance(per_role, dict):
            for role, role_state in per_role.items():
                if role in self._theta_learners and role_state is not None:
                    self._theta_learners[role].load_state(role_state)
        theta = state.get("theta_learner")
        if theta is not None and self._theta_learner is not None:
            self._theta_learner.load_state(theta)
        rsl = state.get("reward_shaping_learner")
        if rsl is not None and self._reward_shaping_learner is not None:
            self._reward_shaping_learner.load_state(rsl)

    def evaluator_summary(self) -> Dict[str, Any]:
        """Context quality evaluator statistics."""
        if self._context_evaluator is not None:
            return self._context_evaluator.summary()
        return {}
