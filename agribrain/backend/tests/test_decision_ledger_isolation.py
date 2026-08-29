"""Regression tests for publication-arm decision-history isolation."""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_ROOT = REPO_ROOT / "mvp" / "simulation"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))


def _write_ledger(directory: Path, action: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"hour": 1.0, "action": action, "mode": "test"}
    (directory / "arm.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


def test_explicit_empty_scope_never_falls_back_to_repository_history(
    tmp_path, monkeypatch
):
    from pirag.mcp.tools.chain_query import _read_ledger_jsonl

    empty = tmp_path / "empty_arm"
    empty.mkdir()
    monkeypatch.setenv("DECISION_LEDGER_DIR", str(empty))
    result = _read_ledger_jsonl(10)
    assert result is not None
    assert result["_status"] == "empty"
    assert result["records"] == []
    assert str(empty) in result["_source"]


def test_zero_record_request_is_consistently_empty_for_file_scope(
    tmp_path, monkeypatch
):
    from pirag.mcp.tools.chain_query import _read_ledger_jsonl

    populated = tmp_path / "populated"
    _write_ledger(populated, "recovery")
    monkeypatch.setenv("DECISION_LEDGER_DIR", str(populated))
    result = _read_ledger_jsonl(0)
    assert result is not None
    assert result["_status"] == "empty"
    assert result["records"] == []


def test_active_empty_episode_shadows_populated_files_and_exposes_new_appends(
    tmp_path, monkeypatch
):
    from pirag.mcp.tools.chain_query import query_recent_decisions
    from src.chain.decision_ledger import (
        DecisionLedger,
        decision_ledger_episode_scope,
    )

    stale = tmp_path / "stale"
    _write_ledger(stale, "recovery")
    monkeypatch.setenv("DECISION_LEDGER_DIR", str(stale))
    ledger = DecisionLedger({"mode": "agribrain", "scenario": "heatwave"})

    with decision_ledger_episode_scope(ledger):
        first = query_recent_decisions(10)
        assert first == {
            "_status": "empty",
            "_source": "active_episode_ledger",
            "records": [],
        }
        ledger.append({"hour": 0.25, "action": "cold_chain", "mode": "agribrain"})
        ledger.append({"hour": 0.50, "action": "local_redistribute", "mode": "agribrain"})
        latest = query_recent_decisions(1)
        assert latest["_source"] == "active_episode_ledger"
        assert [record["action"] for record in latest["records"]] == [
            "local_redistribute"
        ]


def test_episode_context_is_fresh_nested_safe_and_cleans_up_on_exception():
    from pirag.mcp.tools.chain_query import query_recent_decisions
    from src.chain.decision_ledger import (
        DecisionLedger,
        decision_ledger_episode_scope,
        get_active_episode_ledger,
    )

    outer = DecisionLedger()
    inner = DecisionLedger()
    outer.append({"action": "recovery"})
    inner.append({"action": "cold_chain"})
    with pytest.raises(RuntimeError):
        with decision_ledger_episode_scope(outer):
            assert query_recent_decisions(1)["records"][0]["action"] == "recovery"
            with decision_ledger_episode_scope(inner):
                assert query_recent_decisions(1)["records"][0]["action"] == "cold_chain"
            assert query_recent_decisions(1)["records"][0]["action"] == "recovery"
            raise RuntimeError("probe cleanup")
    assert get_active_episode_ledger() is None


def test_recent_records_are_defensive_copies():
    from src.chain.decision_ledger import DecisionLedger

    ledger = DecisionLedger()
    original = {"action": "recovery", "probs": [0.1, 0.2, 0.7]}
    ledger.append(original)
    original["probs"][2] = 0.0
    exported = ledger.recent_records(1)
    exported[0]["probs"][2] = 0.0
    assert ledger.recent_records(1)[0]["probs"] == [0.1, 0.2, 0.7]


def test_public_run_episode_wrapper_starts_each_episode_with_empty_history(
    monkeypatch
):
    import generate_results as gr
    from pirag.mcp.tools.chain_query import query_recent_decisions

    observations = []

    def fake_impl(*args, decision_ledger, **kwargs):
        observations.append(query_recent_decisions(10)["_status"])
        decision_ledger.append({"action": "recovery", "mode": args[1]})
        observations.append(query_recent_decisions(10)["records"][-1]["action"])
        return {"ok": True}

    monkeypatch.setattr(gr, "_run_episode_impl", fake_impl)
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"])})
    for _ in range(2):
        result = gr.run_episode(
            frame, "agribrain", gr.Policy(), np.random.default_rng(42)
        )
        assert result["ok"] is True
        receipt = result["episode_runtime_receipt"]
        assert receipt["utc_start"].endswith("Z")
        assert receipt["utc_end"].endswith("Z")
        assert receipt["wall_seconds"] >= 0.0
        assert receipt["process_cpu_seconds"] >= 0.0
    assert observations == ["empty", "recovery", "empty", "recovery"]


def test_scoped_history_is_order_invariant_and_environment_is_restored(
    tmp_path, monkeypatch
):
    from generate_results import decision_ledger_scope
    from pirag.mcp.tools.chain_query import _read_ledger_jsonl
    from src.chain.decision_ledger import get_active_decision_ledger_output_dir

    arm_a = tmp_path / "arm_a"
    arm_b = tmp_path / "arm_b"
    _write_ledger(arm_a, "recovery")
    _write_ledger(arm_b, "cold_chain")
    monkeypatch.setenv("DECISION_LEDGER_DIR", str(tmp_path / "outer"))

    observed = []
    for arm in (arm_a, arm_b, arm_a):
        with decision_ledger_scope(arm):
            assert get_active_decision_ledger_output_dir() == arm.resolve()
            result = _read_ledger_jsonl(10)
            observed.append(result["records"][-1]["action"])
    assert observed == ["recovery", "cold_chain", "recovery"]
    assert os.environ["DECISION_LEDGER_DIR"] == str(tmp_path / "outer")


def test_scope_reset_removes_only_the_current_arms_stale_ledger(tmp_path):
    from generate_results import decision_ledger_scope

    arm = tmp_path / "arm"
    other = tmp_path / "other"
    _write_ledger(arm, "recovery")
    _write_ledger(other, "cold_chain")
    with decision_ledger_scope(arm, reset=True):
        assert not list(arm.glob("*.jsonl"))
        assert list(other.glob("*.jsonl"))


def test_output_scopes_are_thread_isolated_and_do_not_leak_environment(
    tmp_path, monkeypatch
):
    from generate_results import decision_ledger_scope
    from src.chain.decision_ledger import get_active_decision_ledger_output_dir

    inherited = str(tmp_path / "inherited")
    monkeypatch.setenv("DECISION_LEDGER_DIR", inherited)
    barrier = threading.Barrier(2)

    def probe(name):
        target = (tmp_path / name).resolve()
        with decision_ledger_scope(target):
            barrier.wait(timeout=5)
            return get_active_decision_ledger_output_dir(), os.environ["DECISION_LEDGER_DIR"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(probe, "a")
        b = pool.submit(probe, "b")
        assert a.result(timeout=10) == ((tmp_path / "a").resolve(), inherited)
        assert b.result(timeout=10) == ((tmp_path / "b").resolve(), inherited)
    assert os.environ["DECISION_LEDGER_DIR"] == inherited


def test_h3_stress_conditions_and_seeds_receive_distinct_scopes(
    tmp_path, monkeypatch
):
    import benchmarks.run_stress_suite as stress

    monkeypatch.setattr(stress, "STRESS_LEDGER_ROOT", tmp_path / "stress_ledgers")
    monkeypatch.setenv("STRESS_LEARNING_EPISODES", "4")
    monkeypatch.setenv("RUN_TAG", "test_run")
    monkeypatch.setattr(stress, "_ledger_file_binding", lambda *args, **kwargs: {
        "decision_ledger_path": kwargs["canonical_path"],
        "decision_ledger_sha256": "a" * 64,
        "decision_ledger_merkle_root": "b" * 64,
        "decision_ledger_n_records": 288,
    })
    calls: list[Path] = []

    def fake_run_episode(*args, **kwargs):
        from src.chain.decision_ledger import get_active_decision_ledger_output_dir
        from src.models.synthetic_spoilage_dgp import synthetic_dgp_provenance
        scope = get_active_decision_ledger_output_dir()
        assert scope is not None
        calls.append(scope)
        _write_ledger(scope, "recovery")
        episode_index = int(kwargs["episode_index"])
        frozen = episode_index == 3
        return {
            "ari": 0.5, "waste": 0.1, "slca": 0.6, "rle": 0.7,
            "carbon": 10.0, "equity": 0.8,
            "message_count": 0,
            "constraint_violation_rate": 0.0,
            "mean_decision_latency_ms": 1.0,
            "downstream_violation_rate": 0.0,
            "contained_violation_rate": 1.0,
            "trace_schema_version": 5,
            "benchmark_seed": int(kwargs["benchmark_seed"]),
            "episode_index": episode_index,
            "environment_stream_id": str(kwargs["environment_stream_id"]),
            "policy_stream_id": str(kwargs["policy_stream_id"]),
            "stochastic_stream_id": str(kwargs["stochastic_stream_id"]),
            "context_prior_sha256": "c" * 64,
            "policy_theta_initial_sha256": "d" * 64,
            "spoilage_estimator": {
                "kind": "mechanistic_plus_frozen_synthetic_pinn_residual",
                "checkpoint_sha256": "1" * 64,
                "training_dataset_sha256": "2" * 64,
                "training_target_origin": "independent_synthetic_dgp",
                "residual_bound_abs": 0.08,
                "deployment_transform": (
                    "clip_quality_to_unit_interval_then_cumulative_minimum"
                ),
                "synthetic_only": True,
                "external_validation": False,
            },
            "latent_spoilage_model": synthetic_dgp_provenance(),
            "latent_environment_sha256": "a" * 64,
            "observed_policy_input_sha256": "b" * 64,
            "demand_observation_sha256": "e" * 64,
            "demand_forecast_method": "holt_linear",
            "supply_forecast_method": "persistence",
            "learning_enabled": not frozen,
            "episode_phase": (
                "frozen_evaluation" if frozen else "adaptation"
            ),
            "learner_freeze_summary": ({
                "learners_frozen": True,
                "learner_phase": "frozen_evaluation",
                "freeze_reason": "retained_episode_3",
                "context_matrix_frozen": True,
                "reward_shaping_frozen": True,
                "policy_delta_frozen_by_role": {
                    role: True for role in (
                        "farm", "processor", "distributor", "recovery"
                    )
                },
            } if frozen else {}),
            "dispatch_opportunity_count": 1,
            "dispatch_cadence_hours": 0.25,
        }

    monkeypatch.setattr(stress, "run_episode", fake_run_episode)
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"])})
    for condition in ("nominal_reference", "stressed__sensor_noise"):
        for seed in (42, 43):
            stress._run_pair(
                frame, "heatwave", seed=seed, with_faults=False,
                modes=("agribrain",), ledger_condition=condition,
            )

    unique = {path.relative_to(stress.STRESS_LEDGER_ROOT).as_posix() for path in calls}
    assert unique == {
        "heatwave/nominal_reference/seed_42",
        "heatwave/nominal_reference/seed_43",
        "heatwave/stressed__sensor_noise/seed_42",
        "heatwave/stressed__sensor_noise/seed_43",
    }
    assert len(calls) == 16  # four episodes in each of four independent arms


@pytest.mark.slow
def test_publication_ledger_contains_reconstructable_policy_trace(
    tmp_path, monkeypatch
):
    """The final JSONL contains the quantities claimed in the manuscript."""
    import generate_results as gr

    monkeypatch.setenv("DETERMINISTIC_MODE", "true")
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(8)
    # Span the coordinator's 12-30 h cooperative window without making this
    # focused trace test process dozens of additional telemetry rows.
    frame["timestamp"] = frame["timestamp"].iloc[0] + pd.to_timedelta(
        np.arange(len(frame)) * 4, unit="h",
    )
    with gr.decision_ledger_scope(tmp_path / "trace", reset=True):
        episode = gr.run_episode(
            frame, "agribrain", gr.Policy(), np.random.default_rng(42),
            scenario="baseline", stoch=gr._STOCH_DISABLED, seed=42,
        )

    payloads = [
        json.loads(line)
        for line in Path(episode["decision_ledger_path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    records = [record for record in payloads if not record.get("_header")]
    assert len(records) == len(frame)
    record = records[0]
    assert len(record["phi"]) == 10
    assert len(record["peer_message_bias"]) == 3
    assert len(record["psi"]) == 5
    assert len(record["context_modifier"]) == 3
    assert len(record["base_logits"]) == 3
    assert len(record["post_context_logits_pre_override"]) == 3
    assert isinstance(record["retrieval_top_doc_id"], str)
    assert isinstance(record["retrieval_evidence_hashes"], list)

    cooperative_scopes = set()
    for recorded in records:
        theta = np.asarray(recorded["effective_context_theta"], dtype=float)
        feature_allocation = np.asarray(
            recorded["context_feature_contributions"], dtype=float,
        )
        residual = np.asarray(
            recorded["context_nonfeature_residual"], dtype=float,
        )
        modifier = np.asarray(recorded["context_modifier"], dtype=float)
        modifier_jacobian = np.asarray(
            recorded["context_modifier_theta_jacobian"], dtype=float,
        )
        integration = recorded["context_integration"]
        assert theta.shape == (3, 5)
        assert feature_allocation.shape == (3, 5)
        assert modifier_jacobian.shape == (3, 5)
        assert residual.shape == (3,)
        np.testing.assert_allclose(
            feature_allocation.sum(axis=1) + residual,
            modifier, rtol=1e-12, atol=1e-12,
        )
        action_idx = int(recorded["action_idx"])
        np.testing.assert_allclose(
            recorded["chosen_action_context_contributions"],
            feature_allocation[action_idx], rtol=1e-12, atol=1e-12,
        )
        assert recorded["chosen_action_context_residual"] == pytest.approx(
            residual[action_idx], abs=1e-12,
        )
        np.testing.assert_allclose(
            integration["composition"]["modifier_theta_jacobian"],
            modifier_jacobian, rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            integration["composition"]["final_modifier"],
            modifier, rtol=0, atol=1e-12,
        )
        scope = recorded.get("context_attribution_scope")
        if scope and scope.startswith("cooperative_"):
            cooperative_scopes.add(scope)
    assert cooperative_scopes, "expected at least one cooperative-window trace"

    if not record["governance_override"]:
        logits = np.asarray(record["post_context_logits_pre_override"], dtype=float)
        expected = np.exp(logits - logits.max())
        expected /= expected.sum()
        np.testing.assert_allclose(record["probs"], expected, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_learned_context_arms_are_invariant_to_execution_order(
    tmp_path, monkeypatch
):
    """Real small-data episodes: prior arm order cannot change final outputs."""
    import generate_results as gr

    monkeypatch.setenv("DETERMINISTIC_MODE", "true")
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(12)
    metric_keys = ("ari", "waste", "rle", "slca", "carbon", "equity")

    def execute(order, label):
        outcomes = {}
        for mode in order:
            cache = {}
            rng = np.random.default_rng(907)
            episode = None
            with gr.decision_ledger_scope(tmp_path / label / mode, reset=True):
                for _ in range(4):
                    episode = gr.run_episode(
                        frame, mode, gr.Policy(), rng, scenario="baseline",
                        stoch=gr._STOCH_DISABLED, seed=907,
                        learner_state_cache=cache,
                    )
            assert episode is not None
            outcomes[mode] = {
                "metrics": tuple(float(episode[key]) for key in metric_keys),
                "actions": tuple(episode["action_trace"]),
            }
        return outcomes

    forward = execute(("agribrain", "mcp_only"), "forward")
    reverse = execute(("mcp_only", "agribrain"), "reverse")
    assert forward == reverse


@pytest.mark.slow
def test_h3_arm_is_invariant_to_condition_execution_order(tmp_path, monkeypatch):
    """Real H3 driver: identical AGRI-BRAIN conditions remain identical."""
    import benchmarks.run_stress_suite as stress
    import generate_results as gr

    monkeypatch.setenv("DETERMINISTIC_MODE", "true")
    monkeypatch.setenv("STRESS_LEARNING_EPISODES", "4")
    monkeypatch.setenv("RUN_TAG", "test_run")
    monkeypatch.setattr(stress, "STRESS_LEDGER_ROOT", tmp_path / "h3_ledgers")
    monkeypatch.setattr(stress, "_ledger_file_binding", lambda *args, **kwargs: {
        "decision_ledger_path": kwargs["canonical_path"],
        "decision_ledger_sha256": "a" * 64,
        "decision_ledger_merkle_root": "b" * 64,
        "decision_ledger_n_records": 288,
    })
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(8)

    forward = stress._run_pair(
        frame, "baseline", seed=42, with_faults=False,
        modes=("agribrain",), ledger_condition="nominal_first",
    )
    reverse = stress._run_pair(
        frame, "baseline", seed=42, with_faults=False,
        modes=("agribrain",), ledger_condition="stressed_second",
    )
    left = {
        k: v for k, v in forward["agribrain"].items()
        if k != "decision_latency_ms" and not k.startswith("decision_ledger_")
    }
    right = {
        k: v for k, v in reverse["agribrain"].items()
        if k != "decision_latency_ms" and not k.startswith("decision_ledger_")
    }
    assert left == right
