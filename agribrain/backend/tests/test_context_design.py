"""Revised context design: weights, features and the acceptance-policy document."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pirag.agent_pipeline import PiRPipeline
from pirag.context_design import (
    ACCEPTANCE_POLICY_DOC_ID,
    RETRIEVAL_FEATURE_INDICES,
    THETA_CONTEXT_REVISED,
    TOOL_FEATURE_INDICES,
    capacity_report,
    extract_revised_context_features,
    parse_acceptance_policy,
    published_criteria,
    render_acceptance_policy,
    retrieve_acceptance_policy,
)
from pirag.context_to_logits import THETA_CONTEXT, compute_context_modifier


def test_revised_weights_keep_three_columns_and_mirror_the_recovery_column() -> None:
    assert THETA_CONTEXT_REVISED.shape == (3, 5)
    assert np.array_equal(THETA_CONTEXT_REVISED[:, [0, 1, 4]], THETA_CONTEXT[:, [0, 1, 4]])
    mirrored = THETA_CONTEXT[[0, 2, 1], 4]  # recovery column with routes 1 and 2 exchanged
    assert np.array_equal(THETA_CONTEXT_REVISED[:, 2], mirrored)
    assert np.array_equal(THETA_CONTEXT_REVISED[:, 3], mirrored)
    assert sorted(TOOL_FEATURE_INDICES + RETRIEVAL_FEATURE_INDICES) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("rho_at", "rho_cut", "strict"), [
    (0.10, 0.50, False), (0.05, 0.40, False), (0.15, 0.60, True), (0.0, 1.0, False),
])
def test_policy_document_round_trip(rho_at, rho_cut, strict) -> None:
    text = render_acceptance_policy(rho_at, rho_cut, strict)
    assert parse_acceptance_policy(text) == {
        "rho_at": rho_at, "rho_cut": rho_cut, "strict_redistribution": strict,
    }


@pytest.mark.parametrize("text", [
    "", "redistribution_maximum_modeled_risk = 0.5",
    "redistribution_maximum_modeled_risk = 0.2\nrecovery_minimum_modeled_risk = 0.4",
    "redistribution_maximum_modeled_risk = 1.5\nrecovery_minimum_modeled_risk = 0.1",
])
def test_unreadable_or_inconsistent_criteria_are_rejected(text) -> None:
    assert parse_acceptance_policy(text) is None


def _pipeline_with_policy(rho_at=0.10, rho_cut=0.50, document_id=ACCEPTANCE_POLICY_DOC_ID):
    pipeline = PiRPipeline()
    pipeline.ingest([{
        "id": document_id,
        "text": render_acceptance_policy(rho_at, rho_cut),
        "metadata": {"source": document_id},
    }])
    return pipeline


def test_policy_is_unavailable_until_the_document_is_in_the_corpus() -> None:
    record = retrieve_acceptance_policy(PiRPipeline())
    assert record["available"] is False
    assert record["reason"] == "criteria_not_in_top_passage"
    assert retrieve_acceptance_policy(None)["available"] is False


def test_policy_is_recognized_by_its_content_not_by_its_identifier() -> None:
    default = retrieve_acceptance_policy(_pipeline_with_policy(0.10, 0.40))
    renamed = retrieve_acceptance_policy(
        _pipeline_with_policy(0.10, 0.40, document_id="outlet_intake_rules"))
    assert renamed["available"] is True and renamed["doc_id"] == "outlet_intake_rules"
    keys = ("rho_at", "rho_cut", "strict_redistribution", "passage_sha256")
    assert {k: renamed[k] for k in keys} == {k: default[k] for k in keys}


def test_published_criteria_follow_the_document_settings() -> None:
    from src.models.policy import Policy

    assert published_criteria(Policy()) == {
        "rho_at": 0.10, "rho_cut": 0.50, "strict_redistribution": False}
    changed = Policy(acceptance_rho_cut=0.60, published_rho_cut=0.40,
                     acceptance_strict_redistribution=True)
    assert published_criteria(changed) == {
        "rho_at": 0.10, "rho_cut": 0.40, "strict_redistribution": True}


@pytest.mark.parametrize(("rho_at", "rho_cut"), [(0.10, 0.50), (0.05, 0.40)])
def test_policy_is_retrieved_top_ranked_and_its_values_are_read(rho_at, rho_cut) -> None:
    record = retrieve_acceptance_policy(_pipeline_with_policy(rho_at, rho_cut))
    assert record["available"] is True and record["reason"] == "retrieved"
    assert record["doc_id"] == ACCEPTANCE_POLICY_DOC_ID
    assert (record["rho_at"], record["rho_cut"]) == (rho_at, rho_cut)
    assert all(record["guard_breakdown"].values())
    assert record["fused_score"] > record["fused_score_floor_exclusive"]


def test_revised_features() -> None:
    policy = retrieve_acceptance_policy(_pipeline_with_policy())
    tools = {
        "check_compliance": {"compliant": False, "violations": [{"severity": "critical"}]},
        "spoilage_forecast": {"urgency": "high"},
        "destination_capacity": {
            "_status": "ok", "dispatch_size": 1.0, "redistribution_capacity": 0.25,
            "recovery_capacity": 2.0, "redistribution_shortfall": 0.75,
            "recovery_shortfall": 0.0,
        },
    }
    marketable = extract_revised_context_features(tools, policy, SimpleNamespace(rho=0.30))
    assert marketable.tolist() == [1.0, 0.7, 0.75, 0.0, 0.0]
    unmarketable = extract_revised_context_features(tools, policy, SimpleNamespace(rho=0.60))
    assert unmarketable.tolist() == [1.0, 0.7, 0.75, 1.0, 0.0]
    # nothing reached the decision side: every component is zero
    assert extract_revised_context_features({}, None, SimpleNamespace(rho=0.9)).tolist() == [0.0] * 5
    assert capacity_report(tools) == {
        "dispatch_size": 1.0, "redistribution_capacity": 0.25, "recovery_capacity": 2.0,
    }
    assert capacity_report({}) is None


def test_modifier_with_supplied_features_gate_and_bound() -> None:
    psi = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    obs = SimpleNamespace(rho=0.6, hour=0.0, raw={})
    kwargs = dict(theta_override=THETA_CONTEXT_REVISED, retrieval_kind="standard",
                  psi_override=psi, pir_feature_indices=RETRIEVAL_FEATURE_INDICES)
    open_gate = compute_context_modifier({}, {}, obs, retrieval_gate_override=True, **kwargs)
    closed_gate = compute_context_modifier({}, {}, obs, retrieval_gate_override=False, **kwargs)
    assert np.allclose(open_gate, THETA_CONTEXT_REVISED @ psi)
    without_retrieval = psi.copy()
    without_retrieval[3] = 0.0
    assert np.allclose(closed_gate, THETA_CONTEXT_REVISED @ without_retrieval)
    # the bound applies unless it is removed
    large = 10.0 * THETA_CONTEXT_REVISED
    bounded = compute_context_modifier({}, {}, obs, retrieval_gate_override=True,
                                       **{**kwargs, "theta_override": large})
    unbounded = compute_context_modifier({}, {}, obs, retrieval_gate_override=True,
                                         clamp=float("inf"),
                                         **{**kwargs, "theta_override": large})
    assert np.all(np.abs(bounded) <= 1.0)
    assert np.allclose(unbounded, large @ psi)
