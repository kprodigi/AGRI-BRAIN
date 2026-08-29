"""Regression tests for the centralized learner/mode taxonomy."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pirag.context_learner import RewardShapingLearner
from pirag.explain_decision import (
    _build_policy_trace_paragraph,
    _governance_predicate_record,
)
from src.agents.coordinator import AgentCoordinator
from src.agents.message import InterAgentMessage, MessageType
from src.models.action_selection import (
    GOVERNANCE_CC_PROB_CEILING,
    GOVERNANCE_LOCAL_ADVANTAGE_MIN,
    NO_SLCA_OFFSET,
    SLCA_BONUS,
    SLCA_RHO_BONUS,
)
from src.models.mode_capabilities import (
    DECISION_OWNER_ROLES,
    MODE_CAPABILITIES,
    MULTI_EPISODE_MODES,
    PUBLICATION_BENCHMARK_MODES,
    REWARD_SHAPING_LEARNING_MODES,
    capabilities_for,
)
from src.models.policy import Policy


def _env() -> dict:
    return {
        "rho": 0.25,
        "inv": 12_000.0,
        "temp": 7.0,
        "rh": 88.0,
        "y_hat": 100.0,
        "tau": 0.0,
        "surplus_ratio": 0.0,
        "supply_hat": 12_000.0,
        "supply_std": 100.0,
        "demand_std": 5.0,
        "price_signal": 0.0,
    }


def _one_update(coordinator: AgentCoordinator, mode: str, hour: float) -> str:
    rng = np.random.default_rng(1000 + int(hour))
    action, _probs, agent = coordinator.step(
        _env(), hour, mode, Policy(), rng, "baseline",
    )
    obs = agent.observe(_env(), hour)
    coordinator.post_step(
        agent, action, obs,
        {"waste": 0.05, "rho": 0.25, "slca": 0.7, "carbon_kg": 1.0},
        hour=hour, reward=0.5,
    )
    return agent.role


def test_mode_capabilities_classify_every_adaptive_arm() -> None:
    assert tuple(MODE_CAPABILITIES) == PUBLICATION_BENCHMARK_MODES
    assert len(MODE_CAPABILITIES) == 11

    assert capabilities_for("hybrid_rl").policy_delta_learning
    assert not capabilities_for("hybrid_rl").reward_shaping_learning
    assert not capabilities_for("hybrid_rl").context_matrix_learning
    assert MULTI_EPISODE_MODES["hybrid_rl"] == 4

    assert capabilities_for("no_pinn").learned
    assert capabilities_for("no_pinn").spoilage_residual is False
    assert capabilities_for("agribrain").spoilage_residual is True

    assert capabilities_for("no_slca").learned
    assert not capabilities_for("no_slca").reward_shaping_learning
    for mode in REWARD_SHAPING_LEARNING_MODES:
        assert MODE_CAPABILITIES[mode].episode_count == 4
    assert not MODE_CAPABILITIES["static"].learned
    assert MODE_CAPABILITIES["static"].episode_count == 1


@pytest.mark.parametrize(
    "retired_mode",
    [
        "agribrain_cold_start",
        "agribrain_pert_10", "agribrain_pert_25", "agribrain_pert_50",
        "agribrain_pert_10_static", "agribrain_pert_25_static",
        "agribrain_pert_50_static", "agribrain_no_bonus",
        "agribrain_theta_pert_10", "agribrain_theta_pert_25",
        "agribrain_theta_pert_50",
    ],
)
def test_pre_final_mode_family_is_not_executable(retired_mode: str) -> None:
    assert retired_mode not in MODE_CAPABILITIES
    with pytest.raises(ValueError, match="unknown operating mode"):
        capabilities_for(retired_mode)


def test_secondary_capabilities_are_declared_one_factor_changes() -> None:
    reference = capabilities_for("agribrain")
    expected_difference = {
        "agribrain_standard_rag": "retrieval_kind",
        "agribrain_no_peer": "peer_messages",
        "agribrain_sign_unconstrained": "sign_constrained_learning",
    }
    for mode, changed_field in expected_difference.items():
        candidate = capabilities_for(mode)
        differences = {
            field
            for field in reference.__dataclass_fields__
            if getattr(reference, field) != getattr(candidate, field)
        }
        assert differences == {changed_field}


@pytest.mark.parametrize(
    ("mode", "expected_kind"),
    [
        ("agribrain", "pirag"),
        ("agribrain_standard_rag", "standard"),
    ],
)
def test_coordinator_routes_declared_retrieval_kind(
    monkeypatch, mode: str, expected_kind: str,
) -> None:
    coordinator = AgentCoordinator(context_enabled=False, mode=mode)
    # Keep this test focused on capability routing: a sentinel registry opens
    # the context branch and the stub captures the coordinator boundary.
    coordinator.context_enabled = True
    coordinator._registry = object()
    captured = {}

    def _capture_context(active, obs, scenario, hour, **kwargs):
        captured.update(kwargs)
        return np.zeros(3)

    monkeypatch.setattr(
        coordinator, "_compute_step_context", _capture_context,
    )
    coordinator.step(
        _env(), 0.0, mode, Policy(), np.random.default_rng(91), "baseline",
    )
    assert captured == {
        "context_mode": "full",
        "retrieval_kind": expected_kind,
    }


def test_standard_rag_full_coordinator_trace_uses_standard_transforms() -> None:
    coordinator = AgentCoordinator(
        context_enabled=True, mode="agribrain_standard_rag",
    )
    coordinator.step(
        _env(), 0.0, "agribrain_standard_rag", Policy(),
        np.random.default_rng(191), "baseline",
    )
    primary = coordinator._step_context_integration_trace["primary"]
    assert coordinator._step_rag_context["retrieval_kind"] == "standard"
    assert primary["retrieval_kind"] == "standard"
    assert primary["temporal_scale"] == pytest.approx(1.0)
    assert primary["physics_scale"] == pytest.approx(1.0)
    assert coordinator.context_log[-1]["retrieval_kind"] == "standard"
    assert coordinator.context_summary()["retrieval_kind"] == "standard"


def test_mcp_only_summary_does_not_report_skipped_retrieval_as_queries_or_failures() -> None:
    coordinator = AgentCoordinator(context_enabled=True, mode="mcp_only")
    # Exercise the cooperative window as well as the primary channel.  Both
    # role-specific skipped-retrieval sentinels must use the same numeric score
    # convention as the decision ledger; ``None`` here used to make the HPC
    # validator reject mcp_only/heatwave at its first record.
    coordinator.step(
        _env(), 18.0, "mcp_only", Policy(),
        np.random.default_rng(192), "heatwave",
    )
    summary = coordinator.context_summary()
    assert summary["total_context_steps"] == 1
    assert summary["total_pirag_queries"] == 0
    assert summary["guard_failures"] == 0
    assert sum(
        role["pirag_queries"] for role in summary["per_role"].values()
    ) == 0
    assert coordinator.context_log[-1]["retrieval_attempted"] is False
    assert coordinator.context_log[-1]["guards_passed"] is None
    evidence = coordinator._step_channel_evidence
    for channel_name in ("primary", "cooperative"):
        retrieval = evidence[channel_name]["retrieval"]
        assert retrieval["attempted"] is False
        assert retrieval["skip_reason"] == "structural_retrieval_ablation"
        assert retrieval["top_citation_score"] == 0.0
        assert retrieval["top_fused_score"] == 0.0
        assert retrieval["top_rerank_score"] == 0.0


def test_no_peer_disables_generation_delivery_consumption_and_bias(
    monkeypatch,
) -> None:
    pending = InterAgentMessage(
        sender="farm_agent",
        recipient="processor_agent",
        msg_type=MessageType.SPOILAGE_ALERT,
        payload={"rho": 0.8},
        hour=17.75,
    )
    coop_pending = InterAgentMessage(
        sender="farm_agent",
        recipient="cooperative_agent",
        msg_type=MessageType.SURPLUS_ALERT,
        payload={"surplus_ratio": 0.9},
        hour=17.75,
    )

    no_peer = AgentCoordinator(
        context_enabled=False, mode="agribrain_no_peer",
    )
    processor = no_peer.agents["processor"]
    cooperative = no_peer.agents["cooperative"]
    processor.receive_message(pending)
    cooperative.receive_message(coop_pending)

    captured_obs = {}
    original_observe = processor.observe

    def _capture_observation(env_state, hour):
        observed = original_observe(env_state, hour)
        captured_obs["value"] = observed
        return observed

    def _generation_must_not_run(*args, **kwargs):
        raise AssertionError("no-peer arm invoked message generation")

    monkeypatch.setattr(processor, "observe", _capture_observation)
    monkeypatch.setattr(
        processor, "generate_messages", _generation_must_not_run,
    )
    monkeypatch.setattr(
        cooperative, "generate_messages", _generation_must_not_run,
    )

    action, _probs, agent = no_peer.step(
        _env(), 18.0, "agribrain_no_peer", Policy(),
        np.random.default_rng(92), "baseline",
    )
    assert agent is processor
    assert captured_obs["value"].messages == []
    assert processor._inbox == [pending]
    assert cooperative._inbox == [coop_pending]
    np.testing.assert_array_equal(no_peer._step_message_bias, np.zeros(3))
    np.testing.assert_array_equal(
        no_peer._step_role_bias,
        processor.role_bias + cooperative.role_bias,
    )

    no_peer.post_step(
        agent, action, captured_obs["value"],
        {"waste": 0.05, "rho": 0.25, "slca": 0.7, "carbon_kg": 1.0},
        hour=18.0, reward=0.5,
    )
    assert processor._inbox == [pending]
    assert cooperative._inbox == [coop_pending]
    assert no_peer.message_log == []

    # The full arm consumes the identical pending message and converts it to
    # the declared logit bias, proving the structural zero is mode-specific.
    reference = AgentCoordinator(context_enabled=False, mode="agribrain")
    reference_processor = reference.agents["processor"]
    reference_processor.receive_message(pending)
    reference.step(
        _env(), 18.0, "agribrain", Policy(),
        np.random.default_rng(92), "baseline",
    )
    assert reference_processor._inbox == []
    np.testing.assert_allclose(
        reference._step_message_bias, [-0.10, 0.10, 0.05],
    )


def test_strict_run_fails_closed_when_peer_bias_conversion_fails(
    monkeypatch,
) -> None:
    import src.agents.message as message_module

    coordinator = AgentCoordinator(context_enabled=False, mode="agribrain")
    monkeypatch.setattr(
        message_module,
        "message_bias_from_inbox",
        lambda _messages: (_ for _ in ()).throw(ValueError("fixture failure")),
    )
    monkeypatch.setenv("STRICT_VALIDATION", "1")

    with pytest.raises(
        RuntimeError,
        match="publication-critical peer-message bias conversion failed",
    ):
        coordinator.step(
            _env(), 18.0, "agribrain", Policy(),
            np.random.default_rng(93), "baseline",
        )


def test_sign_unconstrained_changes_only_the_projection_rail() -> None:
    constrained = AgentCoordinator(context_enabled=True, mode="agribrain")
    unconstrained = AgentCoordinator(
        context_enabled=True, mode="agribrain_sign_unconstrained",
    )

    context_a = constrained._context_learner
    context_b = unconstrained._context_learner
    assert context_a.sign_constrained is True
    assert context_b.sign_constrained is False
    np.testing.assert_array_equal(context_a.initial_theta, context_b.initial_theta)
    np.testing.assert_array_equal(context_a.theta, context_b.theta)
    for attr in (
        "lr", "prior_precision", "grad_clip", "magnitude_cap_mode",
        "magnitude_cap_value", "magnitude_cap_abs_floor",
    ):
        assert getattr(context_a, attr) == getattr(context_b, attr)

    for role in DECISION_OWNER_ROLES:
        policy_a = constrained._theta_learners[role]
        policy_b = unconstrained._theta_learners[role]
        assert policy_a.sign_constrained is True
        assert policy_b.sign_constrained is False
        np.testing.assert_array_equal(
            policy_a.initial_theta, policy_b.initial_theta,
        )
        np.testing.assert_array_equal(
            policy_a.theta_delta, policy_b.theta_delta,
        )
        np.testing.assert_array_equal(
            policy_a._magnitude_bound, policy_b._magnitude_bound,
        )
        for attr in ("lr", "prior_precision", "grad_clip", "cap_fraction"):
            assert getattr(policy_a, attr) == getattr(policy_b, attr)

    reward_a = constrained._reward_shaping_learner
    reward_b = unconstrained._reward_shaping_learner
    assert reward_a.sign_constrained is True
    assert reward_b.sign_constrained is False
    for attr in (
        "initial_slca_bonus", "initial_slca_rho",
        "initial_no_slca_offset", "_bound_bonus", "_bound_rho",
        "_bound_offset",
    ):
        np.testing.assert_array_equal(
            getattr(reward_a, attr), getattr(reward_b, attr),
        )
    for attr in ("lr", "prior_precision", "grad_clip", "cap_fraction"):
        assert getattr(reward_a, attr) == getattr(reward_b, attr)

    assert constrained.learner_summary()["sign_constrained"] is True
    assert unconstrained.learner_summary()["sign_constrained"] is False
    assert all(
        item["sign_constrained"] is False
        for item in unconstrained.theta_learner_summary()["per_role"].values()
    )
    assert unconstrained.reward_shaping_learner_summary()[
        "sign_constrained"
    ] is False
    # Disabling the projection rail is not evidence that a reversal occurred.
    # With the otherwise-identical 50%/25% magnitude caps, all three learners
    # start with zero actual reversals and publication reporting must say so.
    assert unconstrained.learner_summary()["sign_reversal_count"] == 0
    assert unconstrained.learner_summary()[
        "compliance_sign_reversal_count"
    ] == 0
    assert unconstrained.theta_learner_summary()["sign_reversal_count"] == 0
    assert unconstrained.reward_shaping_learner_summary()[
        "sign_reversal_count"
    ] == 0


def test_context_reversal_diagnostics_identify_compliance_coordinate() -> None:
    coordinator = AgentCoordinator(
        context_enabled=True, mode="agribrain_sign_unconstrained",
    )
    learner = coordinator._context_learner
    learner.theta[0, 0] = abs(float(learner.initial_theta[0, 0]))

    summary = coordinator.learner_summary()

    assert summary["sign_reversal_count"] == 1
    assert summary["compliance_sign_reversal_count"] == 1
    assert summary["sign_preserved"] is False
    assert summary["worst_compliance_sign_reversal"] == (
        summary["sign_reversal_coordinates"][0]
    )


def test_hybrid_gets_four_policy_learners_without_context_or_rsl() -> None:
    coordinator = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    assert tuple(coordinator._theta_learners) == DECISION_OWNER_ROLES
    assert "cooperative" not in coordinator._theta_learners
    assert coordinator._context_learner is None
    assert coordinator._reward_shaping_learner is None

    role = _one_update(coordinator, "hybrid_rl", 0.0)
    assert role == "farm"
    assert coordinator._theta_learners["farm"].n_updates == 1
    assert np.linalg.norm(
        coordinator._theta_learners["farm"].get_theta_delta()
    ) > 0.0


def test_probability_gap_substitution_is_not_an_on_policy_sample() -> None:
    coordinator = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    rng = np.random.default_rng(123)
    action, _probs, agent = coordinator.step(
        _env(), 0.0, "hybrid_rl", Policy(), rng, "baseline",
    )
    obs = agent.observe(_env(), 0.0)
    # Simulate the exact post-selection condition: the live action was
    # substituted by the declared probability-gap rule, not sampled from the
    # policy distribution.
    coordinator._step_override = True
    coordinator.post_step(
        agent, action, obs,
        {"waste": 0.05, "rho": 0.25, "slca": 0.7, "carbon_kg": 1.0},
        hour=0.0, reward=0.5,
    )
    assert coordinator._theta_learners["farm"].n_updates == 0
    summary = coordinator.theta_learner_summary()
    assert summary["governance_skipped_learning_steps"] == 1
    snapshot = coordinator.save_learner_states()
    restored = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    restored.load_learner_states(snapshot)
    assert restored.theta_learner_summary()[
        "governance_skipped_learning_steps"
    ] == 1


def test_each_lifecycle_stage_updates_only_its_decision_owner() -> None:
    coordinator = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    observed_roles = [
        _one_update(coordinator, "hybrid_rl", hour)
        for hour in (0.0, 18.0, 36.0, 54.0)
    ]
    assert tuple(observed_roles) == DECISION_OWNER_ROLES
    assert {
        role: learner.n_updates
        for role, learner in coordinator._theta_learners.items()
    } == {role: 1 for role in DECISION_OWNER_ROLES}
    summary = coordinator.theta_learner_summary()
    assert summary["n_updates"] == 4
    assert summary["updates_per_role"] == {
        role: 1 for role in DECISION_OWNER_ROLES
    }


def test_per_role_checkpoint_is_not_overwritten_by_legacy_active_state() -> None:
    source = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    for idx, role in enumerate(DECISION_OWNER_ROLES, start=1):
        learner = source._theta_learners[role]
        learner.theta_delta.fill(idx * 1e-4)
        learner.n_updates = idx
    # Reproduce the old failure mode: the legacy singleton points at Recovery
    # while the snapshot also contains all four role states.
    source._theta_learner = source._theta_learners["recovery"]
    snapshot = source.save_learner_states()

    restored = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    restored.load_learner_states(snapshot)
    for idx, role in enumerate(DECISION_OWNER_ROLES, start=1):
        np.testing.assert_array_equal(
            restored._theta_learners[role].theta_delta,
            np.full((3, 10), idx * 1e-4),
        )
        assert restored._theta_learners[role].n_updates == idx
    assert not np.array_equal(
        restored._theta_learners["farm"].theta_delta,
        restored._theta_learners["recovery"].theta_delta,
    )
    summary = restored.theta_learner_summary()
    assert set(summary["per_role_state_sha256"]) == set(DECISION_OWNER_ROLES)
    assert len(summary["combined_state_sha256"]) == 64


def test_legacy_singleton_checkpoint_migrates_to_every_decision_owner() -> None:
    source = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    singleton = source._theta_learners["farm"]
    singleton.theta_delta.fill(2e-4)
    singleton.n_updates = 7
    legacy_snapshot = {"theta_learner": singleton.save_state()}

    restored = AgentCoordinator(context_enabled=False, mode="hybrid_rl")
    restored.load_learner_states(legacy_snapshot)
    for role in DECISION_OWNER_ROLES:
        np.testing.assert_array_equal(
            restored._theta_learners[role].theta_delta,
            np.full((3, 10), 2e-4),
        )
        assert restored._theta_learners[role].n_updates == 7

    summary = restored.theta_learner_summary()
    # Equal migrated states have equal role hashes, while the combined digest
    # also commits to role names and ordering.
    assert len(set(summary["per_role_state_sha256"].values())) == 1
    assert len(summary["combined_state_sha256"]) == 64


def _new_rsl() -> RewardShapingLearner:
    return RewardShapingLearner(
        initial_slca_bonus=SLCA_BONUS,
        initial_slca_rho_bonus=SLCA_RHO_BONUS,
        initial_no_slca_offset=NO_SLCA_OFFSET,
        learning_rate=0.1,
        prior_precision=0.0,
    )


@pytest.mark.parametrize(
    "mode",
    sorted(REWARD_SHAPING_LEARNING_MODES),
)
def test_reward_shaping_gradients_cover_locked_learning_modes(mode: str) -> None:
    learner = _new_rsl()
    learner.update(
        action=1, probs=np.array([0.3, 0.5, 0.2]),
        reward=1.0, mode=mode, rho=0.3,
    )
    assert learner.n_updates == 1
    assert np.linalg.norm(learner.get_slca_bonus_delta()) > 0.0
    assert np.linalg.norm(learner.get_slca_rho_delta()) > 0.0
    np.testing.assert_array_equal(
        learner.get_no_slca_offset_delta(), np.zeros(3),
    )


@pytest.mark.parametrize(
    "mode",
    ["static", "hybrid_rl", "no_slca"],
)
def test_reward_shaping_is_true_noop_when_vectors_do_not_learn(mode: str) -> None:
    learner = _new_rsl()
    learner.update(
        action=1, probs=np.array([0.3, 0.5, 0.2]),
        reward=1.0, mode=mode, rho=0.3,
    )
    assert learner.n_updates == 0
    np.testing.assert_array_equal(learner.get_slca_bonus_delta(), np.zeros(3))
    np.testing.assert_array_equal(learner.get_slca_rho_delta(), np.zeros(3))


def test_governance_explanation_names_only_probability_predicates() -> None:
    probs = np.array([0.001, 0.90, 0.099])
    record = _governance_predicate_record(probs, True)
    assert record["p_cold_chain"] == pytest.approx(0.001)
    assert record["p_local_minus_cold"] == pytest.approx(0.899)
    assert record["p_cold_chain_ceiling"] == GOVERNANCE_CC_PROB_CEILING
    assert record["p_local_minus_cold_minimum"] == GOVERNANCE_LOCAL_ADVANTAGE_MIN
    assert record["predicate_recomputed"] is True

    paragraph = _build_policy_trace_paragraph(
        "local_redistribute", "local redistribution", "farm", 12.0,
        SimpleNamespace(), {}, {}, None, True, probs,
    )
    assert "p(cold) < 0.005" in paragraph
    assert "p(local)-p(cold) > 0.80" in paragraph
    assert "simultaneous benchmark-envelope violation" not in paragraph
    assert "high modelled spoilage risk triggered" not in paragraph
