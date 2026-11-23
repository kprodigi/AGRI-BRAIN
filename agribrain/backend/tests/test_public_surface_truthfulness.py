"""Focused truthfulness contracts for publication-facing runtime surfaces."""
from __future__ import annotations

from types import SimpleNamespace
from io import BytesIO
import json

import numpy as np
import pytest


def test_development_runtime_defaults_fail_safe(monkeypatch):
    from src.settings import load_settings

    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("DEPLOYMENT_PHASE", raising=False)
    monkeypatch.delenv("DYNAMIC_KB_FEEDBACK", raising=False)

    settings = load_settings()
    assert settings.deployment_phase == "monitoring"
    assert settings.dynamic_kb_feedback is False


def test_unknown_scenario_cannot_mutate_active_state_or_runtime_data():
    from fastapi import HTTPException
    from src.routers import scenarios

    previous_active = dict(scenarios.ACTIVE)
    previous_app_state = scenarios._APP_STATE
    sentinel_df = object()
    scenarios.register_app_state({"df": sentinel_df, "df_original": object()})
    scenarios.ACTIVE.update({"name": "heatwave", "intensity": 0.75})

    try:
        with pytest.raises(HTTPException) as exc_info:
            scenarios.run_scenario(
                scenarios.RunRequest(name="not-a-scenario", intensity=1.0)
            )
        assert exc_info.value.status_code == 422
        assert scenarios.ACTIVE == {"name": "heatwave", "intensity": 0.75}
        assert scenarios._APP_STATE["df"] is sentinel_df

        with pytest.raises(HTTPException) as legacy_exc:
            scenarios.legacy_apply(
                body=scenarios.LegacyApplyBody(id="also-not-a-scenario")
            )
        assert legacy_exc.value.status_code == 422
        assert scenarios.ACTIVE == {"name": "heatwave", "intensity": 0.75}
        assert scenarios._APP_STATE["df"] is sentinel_df
    finally:
        scenarios.ACTIVE.clear()
        scenarios.ACTIVE.update(previous_active)
        scenarios._APP_STATE = previous_app_state


def test_audit_mapping_fails_closed_and_preserves_evidence_contract():
    from src.routers.audit import _map_for_pdf

    current = _map_for_pdf({
        "evidence_status": "development_only",
        "publication_evidence": False,
        "execution_contract": "role_selected_single_step_without_peer_overlay",
    })
    assert current["evidence_status"] == "development_only"
    assert current["publication_evidence"] is False
    assert current["execution_contract"] == (
        "role_selected_single_step_without_peer_overlay"
    )

    legacy = _map_for_pdf({"action": "cold_chain"})
    assert legacy["evidence_status"] == "unverified_runtime_output"
    assert legacy["publication_evidence"] is False
    assert legacy["execution_contract"] == "unspecified_runtime_contract"


def test_decisions_api_fails_closed_for_legacy_runtime_memos():
    from src import app

    previous_log = app.state.get("log")
    app.state["log"] = [{"agent": "legacy", "role": "farm"}]
    try:
        memo = app.list_decisions()["decisions"][0]
        assert memo["evidence_status"] == "unverified_runtime_output"
        assert memo["publication_evidence"] is False
        assert memo["execution_contract"] == "unspecified_runtime_contract"
        assert app.last_decision() == memo
    finally:
        app.state["log"] = previous_log if previous_log is not None else []


def test_all_roles_report_selects_latest_across_roles_not_farm():
    from src import app

    logs = [
        {"agent": "farm-agent", "role": "farm"},
        {"agent": "recovery-agent", "role": "recovery"},
    ]
    selected, requested_role, all_roles = app._select_report_memo(logs, "all")
    assert selected["agent"] == "recovery-agent"
    assert requested_role == "all"
    assert all_roles is True

    selected, requested_role, all_roles = app._select_report_memo(logs, "farm")
    assert selected["agent"] == "farm-agent"
    assert requested_role == "farm"
    assert all_roles is False


def test_runtime_pdfs_print_nonpublication_status_and_truthful_chain_scope(
    monkeypatch,
):
    from pypdf import PdfReader
    from src import app
    from src.routers.audit import _render_pdf, _map_for_pdf

    previous_df = app.state.get("df")
    previous_log = app.state.get("log")
    memo = {
        "time": "2026-08-28T12:00:00+00:00",
        "agent": "recovery-agent",
        "role": "recovery",
        "mode": "agribrain",
        "action": "recovery",
        "evidence_status": "development_only",
        "publication_evidence": False,
        "execution_contract": "role_selected_single_step_without_peer_overlay",
        "tx_hash": "0x" + "a" * 64,
    }
    monkeypatch.setattr(app, "kpis", lambda: {})
    app.state["df"] = object()
    app.state["log"] = [memo]
    try:
        response = app.report_pdf(role="all")
    finally:
        app.state["df"] = previous_df
        app.state["log"] = previous_log if previous_log is not None else []

    report_text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(response.body)).pages
    ).lower()
    assert "latest recorded decision across all roles" in report_text
    assert "evidence status: development only" in report_text
    assert "publication evidence: no" in report_text
    assert "not a publication result" in report_text
    assert "optional on-chain decision record" in report_text
    assert "does not anchor the local explanation merkle root" in report_text
    assert "author-declared social-performance proxy" in report_text
    assert "modeled transport-emissions indicator" in report_text

    audit_bytes = _render_pdf({}, _map_for_pdf(memo))
    audit_text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(audit_bytes)).pages
    ).lower()
    assert "evidence status: development only" in audit_text
    assert "publication evidence: no" in audit_text
    assert "runtime output only" in audit_text
    assert "author-declared social-performance proxy" in audit_text
    assert "modeled transport-emissions indicator" in audit_text


def test_trace_export_uses_feature_names_and_marks_legacy_aliases(tmp_path):
    from pirag.trace_exporter import TraceExporter

    exporter = TraceExporter(max_traces=2)
    exporter.capture(
        obs=SimpleNamespace(
            rho=0.2, temp=4.0, rh=90.0, inv=100.0, tau=0.0,
            surplus_ratio=0.0, hour=0.0,
        ),
        scenario="baseline",
        action="cold_chain",
        probs=np.array([0.7, 0.2, 0.1]),
        mcp_results={},
        rag_context={},
        context_features=np.zeros(5),
        logit_adjustment=np.zeros(3),
        explanation={
            "attribution_chain": {
                "primary_feature": "forecast urgency",
                "primary_contribution": 0.12,
            },
            "ablation_delta": {
                "action_without_context": "cold_chain",
                "probability_shift": [0.0, 0.0, 0.0],
                "action_changed": False,
            },
        },
        role="farm",
    )

    output = tmp_path / "trace.json"
    exporter.export_json(str(output))
    attribution = json.loads(output.read_text(encoding="utf-8"))[0][
        "policy_attribution"
    ]
    assert attribution["primary_feature"] == "forecast urgency"
    assert attribution["primary_cause"] == attribution["primary_feature"]
    assert attribution["legacy_aliases"] == {
        "primary_cause": "primary_feature",
        "counterfactual_action": "action_without_context",
    }
    assert "not causal identification" in attribution["attribution_scope"]


def test_explanation_exposes_complete_local_merkle_leaf_inventory():
    from pirag.explain_decision import explain_decision
    from pirag.provenance.merkle import merkle_root

    retrieval_hashes = ["a" * 64, "b" * 64]
    mcp_results = {
        "_tools_invoked": [
            "check_compliance", "spoilage_forecast", "slca_lookup"
        ],
        "check_compliance": {
            "compliant": True,
            "assessment_type": "synthetic_benchmark_operating_envelope",
            "violations": [],
        },
        "spoilage_forecast": {
            "forecast_rho": 0.25,
            "hours_ahead": 6,
            "urgency": "low",
        },
        "slca_lookup": {"product_type": "spinach", "score": 0.5},
    }
    result = explain_decision(
        action="cold_chain",
        role="farm",
        hour=0.0,
        obs=SimpleNamespace(
            rho=0.2, temp=4.0, rh=90.0, inv=100.0, surplus_ratio=0.0
        ),
        mcp_results=mcp_results,
        rag_context={"evidence_hashes": retrieval_hashes},
        slca_score=0.5,
        carbon_kg=1.0,
        waste=0.1,
        action_probs=np.array([0.5, 0.3, 0.2]),
    )

    leaves = result["evidence_hashes"]
    assert leaves[:2] == retrieval_hashes
    assert result["evidence_hash_count"] == len(leaves) == 5
    assert result["evidence_hashes_complete"] is True
    assert result["merkle_root"] == merkle_root(leaves)
    assert set(result["mcp_evidence_hashes"]) == {
        "check_compliance", "spoilage_forecast", "slca_lookup"
    }
    assert result["retrieval_evidence_hashes"] == retrieval_hashes
    assert result["merkle_inclusion_paths_exposed"] is False
    assert result["merkle_root_anchored_on_chain"] is False


def test_rag_api_distinguishes_local_commitment_from_optional_anchor(monkeypatch):
    from pirag.api.routes import rag

    citation = SimpleNamespace(
        doc_id="doc-1", sha256="a" * 64, meta={}, passage="evidence"
    )
    response = SimpleNamespace(
        answer="answer",
        citations=[citation],
        guards_passed=True,
        evidence_hashes=["a" * 64],
        merkle_root="b" * 64,
        chain_tx=None,
    )
    monkeypatch.setattr(
        rag, "enforce_api_key", lambda _request, _x_api_key: None
    )
    monkeypatch.setattr(
        rag._pipe, "ask", lambda *_args, **_kwargs: response
    )

    payload = rag.ask(
        rag.AskReq(question="q", k=1, anchor_on_chain=False),
        request=object(),
    )
    assert payload["evidence_hashes"] == ["a" * 64]
    assert payload["evidence_hash_count"] == 1
    assert payload["evidence_hashes_complete"] is True
    assert payload["commitment_type"] == "local_merkle_root"
    assert payload["merkle_inclusion_paths_exposed"] is False
    assert payload["anchor_requested"] is False
    assert payload["merkle_root_anchored_on_chain"] is False
    assert payload["chain_tx"] is None
