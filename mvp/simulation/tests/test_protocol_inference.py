"""Focused tests for the publication protocol's inferential bookkeeping."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mvp.simulation.analysis.experiment_accounting import (
    PRIMARY_PUBLICATION_MODES,
    build_episode_accounting,
    build_h3_episode_accounting,
    validate_episode_accounting,
)
from mvp.simulation.analysis.protocol_statistics import (
    H1_PRACTICAL_MARGIN,
    H2_DIRECTIONAL_PAIRS,
    equivalence_tost,
    h2_synergy_interaction,
)
from mvp.simulation.validation import validate_publication_artifacts


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "experiment_protocol.json"


def test_locked_protocol_discloses_stochastic_spoilage_and_cooperative_composition():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    spoilage = protocol["spoilage_model"]
    assert spoilage["base_k_ref_per_hour"] == pytest.approx(0.0021)
    assert spoilage["base_ea_over_r_kelvin"] == pytest.approx(8000.0)
    draws = spoilage["episode_parameter_draws"]
    assert draws["k_ref_fractional_std"] == pytest.approx(0.20)
    assert draws["ea_over_r_fractional_std"] == pytest.approx(0.14)
    assert draws["paired_counter_keyed_across_modes"] is True

    overlay = protocol["cooperative_overlay"]
    assert overlay["start_hour_inclusive"] == pytest.approx(12.0)
    assert overlay["end_hour_exclusive"] == pytest.approx(30.0)
    assert overlay["ordinary_context_composition"] == {
        "primary_weight": 0.70,
        "cooperative_weight": 0.30,
    }
    assert overlay["critical_envelope_adjustment"]["fixed_bias"] == [
        -0.20, 0.20, 0.00,
    ]
    assert overlay["composition_clip"] == [-1.0, 1.0]
    assert overlay["distinct_from_probability_gap_rule"] is True


def test_locked_protocol_pins_publication_retrieval_and_guard_boundaries():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    retrieval = protocol["retrieval_protocol"]
    assert retrieval["fixed_corpus_confirmatory"] is True
    assert retrieval["corpus_document_count"] == 20
    assert retrieval["dynamic_decision_history_ingestion"] is False
    base = retrieval["base_retriever"]
    assert base["publication_top_k"] == 4
    assert base["rrf_k"] == 60
    assert base["bm25_k1"] == pytest.approx(1.6)
    assert base["bm25_b"] == pytest.approx(0.72)
    assert base["tfidf_max_features"] == 5000
    guards = retrieval["guards"]
    assert guards["raw_rrf_top_score_operator"] == "strictly_greater_than"
    assert guards["raw_rrf_top_score_floor"] == pytest.approx(1.5 / 61.0)
    assert guards["feasibility_inclusive_range"] == [-1.0e9, 1.0e9]
    assert "not substantive physical validation" in guards[
        "feasibility_interpretation"
    ]


def test_primary_protocol_is_800_cells_but_2900_executed_episodes():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    outcomes = protocol["mode_neutral_outcomes"]
    assert outcomes["same_equations_across_modes"] is True
    assert "no factor of 1000" in outcomes["ari_per_modeled_emissions"]
    route_proxy = outcomes["route_circularity_proxy"]
    assert route_proxy["recovery_excludes_food_bank"] is True
    assert "animal_feed_suitability" in route_proxy["recovery"]
    assert "composting_suitability" in route_proxy["recovery"]
    assert "food_bank_suitability" not in route_proxy["recovery"]

    budgets = {mode: 4 for mode in PRIMARY_PUBLICATION_MODES}
    budgets["static"] = 1
    accounting = build_episode_accounting(
        scenarios=("heatwave", "overproduction", "cyber", "adaptive", "baseline"),
        configured_modes=PRIMARY_PUBLICATION_MODES,
        episode_budget_by_mode=budgets,
        n_seeds=20,
    )
    assert accounting["retained_endpoint_cells_primary"] == 800
    assert accounting["executed_episodes_primary"] == 2900
    assert accounting["simulated_decision_steps_primary"] == 835_200
    validate_episode_accounting(accounting)


def test_episode_accounting_validator_rejects_800_episode_relabelling():
    budgets = {mode: 4 for mode in PRIMARY_PUBLICATION_MODES}
    budgets["static"] = 1
    accounting = build_episode_accounting(
        scenarios=("a", "b", "c", "d", "e"),
        configured_modes=PRIMARY_PUBLICATION_MODES,
        episode_budget_by_mode=budgets,
        n_seeds=20,
    )
    contradicted = copy.deepcopy(accounting)
    contradicted["executed_episodes_primary"] = 800
    with pytest.raises(ValueError, match="executed_episodes_primary"):
        validate_episode_accounting(contradicted)


def test_h3_accounting_distinguishes_reused_and_dedicated_nominal_reference():
    reused = build_h3_episode_accounting(
        n_seeds=20,
        n_scenarios=5,
        n_stressors=5,
        episodes_per_condition=4,
        nominal_reference_reused=True,
    )
    dedicated = build_h3_episode_accounting(
        n_seeds=20,
        n_scenarios=5,
        n_stressors=5,
        episodes_per_condition=4,
        nominal_reference_reused=False,
    )
    assert reused["formal_contrast_cells"] == 25
    assert reused["retained_stressed_endpoint_cells"] == 500
    assert reused["incremental_executed_episodes"] == 2000
    assert dedicated["incremental_executed_episodes"] == 2400


def test_h2_family_contains_all_four_directional_contrasts():
    assert H1_PRACTICAL_MARGIN == pytest.approx(0.005)
    assert H2_DIRECTIONAL_PAIRS == (
        ("mcp_only", "no_context"),
        ("pirag_only", "no_context"),
        ("agribrain", "mcp_only"),
        ("agribrain", "pirag_only"),
    )
    assert len(H2_DIRECTIONAL_PAIRS) * 5 == 20


def test_h2_synergy_is_not_the_same_as_full_beating_each_single_channel():
    # Full exceeds both single channels, yet the combined uplift is
    # sub-additive relative to the No-external-context floor.
    interaction = h2_synergy_interaction(
        full=[0.620, 0.630],
        mcp_only=[0.615, 0.625],
        retrieval_only=[0.614, 0.624],
        no_external_context=[0.600, 0.610],
    )
    assert np.all(interaction < 0.0)


def test_h3_tost_exposes_one_sided_bound_and_passes_clear_equivalence():
    result = equivalence_tost(np.linspace(-0.001, 0.001, 20), margin=0.01)
    assert result["equivalent_alpha_0p05"] is True
    assert result["one_sided_95_bound_below_margin"] is True
    assert result["max_abs_one_sided_95_bound"] < 0.01
    assert result["margin_clearance"] > 0.0
    assert result["one_sided_95_lower_bound"] == pytest.approx(
        result["ci90_low"]
    )
    assert result["one_sided_95_upper_bound"] == pytest.approx(
        result["ci90_high"]
    )


def test_h3_near_margin_point_estimate_fails_when_uncertainty_crosses_margin():
    values = 0.0091 + np.linspace(-0.006, 0.006, 20)
    result = equivalence_tost(values, margin=0.01)
    assert result["mean"] == pytest.approx(0.0091)
    assert result["max_abs_one_sided_95_bound"] > 0.01
    assert result["one_sided_95_bound_below_margin"] is False
    assert result["equivalent_alpha_0p05"] is False


def test_h3_exact_margin_is_not_equivalent_even_without_variance():
    result = equivalence_tost([0.01] * 20, margin=0.01)
    assert result["equivalent_alpha_0p05"] is False
    assert result["one_sided_95_bound_below_margin"] is False


def _locked_h3_summary_meta() -> dict:
    return {
        "max_rows": None,
        "adaptation_episodes_per_stressed_condition": 3,
        "frozen_evaluation_episodes_per_stressed_condition": 1,
        "nominal_reference": "reused_primary_benchmark_episode_3",
        "mcp_reliability_posture": "false",
        "adaptation_posture": (
            "the primary nominal endpoint is reused; each stressed arm adapts "
            "from the same declared priors on episodes 0-2 and retains a "
            "no-update frozen episode 3"
        ),
        "decision_history_posture": (
            "fresh in-memory decision history at every episode; only learner "
            "state persists within an arm"
        ),
    }


def test_publication_h3_gate_accepts_locked_reuse_and_freeze_metadata():
    validate_publication_artifacts._validate_h3_design_meta(
        _locked_h3_summary_meta()
    )


def test_publication_h3_gate_rejects_obsolete_two_arm_metadata():
    stale = _locked_h3_summary_meta()
    stale.pop("adaptation_episodes_per_stressed_condition")
    stale.pop("frozen_evaluation_episodes_per_stressed_condition")
    stale["learning_episodes_per_condition"] = 4
    stale["mcp_reliability_enabled_in_both_arms"] = True
    with pytest.raises(SystemExit):
        validate_publication_artifacts._validate_h3_design_meta(stale)
