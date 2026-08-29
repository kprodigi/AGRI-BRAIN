"""Regression tests for the paired context-influence rate metric.

Lock the contract that fig 9 panel (c) and the supplementary methods
table both depend on:

* ``select_action`` exposes ``out["base_argmax"]`` as an observer-only
  policy diagnostic; it is not compared with a stochastic live action to
  score context influence.
* The coordinator replays the context-ablated policy from the RNG state saved
  immediately before the live call. Stochastic calls consume the same
  categorical variate, including when the probability-gap rule discards the live
  sampled action.
* ``generate_results.run_episode`` emits both ``context_honor_rate``
  and ``context_influence_rate`` on every episode result, with the
  latter counting active steps where the paired live and context-ablated
  actions differ.
* The two rates share the same denominator
  (``context_active_steps``).
* Modes that bypass the modifier (static, cyber-outage during the
  outage window) contribute zero to both numerators.

These tests run fast (no full ``run_all`` simulator invocation): they
exercise ``select_action`` directly with synthetic inputs.
"""
from __future__ import annotations

import copy
import json
import numpy as np
import pytest

from src.models.action_selection import select_action
from src.models.policy import Policy


def _select(mode: str, *, context_modifier=None, scenario="baseline", hour=0.0,
            out=None, **kwargs):
    """Thin wrapper around select_action with sensible test defaults."""
    policy = kwargs.pop("policy", Policy())
    rng = kwargs.pop("rng", np.random.default_rng(0))
    return select_action(
        mode=mode,
        rho=kwargs.pop("rho", 0.20),
        inv=kwargs.pop("inv", 100.0),
        y_hat=kwargs.pop("y_hat", 100.0),
        temp=kwargs.pop("temp", 5.0),
        tau=kwargs.pop("tau", 0.0),
        policy=policy,
        rng=rng,
        scenario=scenario,
        hour=hour,
        context_modifier=context_modifier,
        deterministic=True,
        out=out,
        **kwargs,
    )


def test_out_dict_populated_when_context_modifier_present():
    """The regular modifier path retains the base-policy diagnostic."""
    out: dict = {}
    modifier = np.array([0.0, 0.5, 0.0])  # nudge toward local_redistribute
    action_idx, probs = _select("agribrain", context_modifier=modifier, out=out)
    assert "base_argmax" in out, (
        "select_action did not populate out['base_argmax'] on the "
        "context-modifier path; channel-attribution diagnostics will be "
        "incomplete."
    )
    assert isinstance(out["base_argmax"], int)
    assert 0 <= out["base_argmax"] < 3


def test_out_dict_unset_when_context_modifier_none():
    """No modifier -> nothing to flip -> base_argmax not populated."""
    out: dict = {}
    _select("hybrid_rl", context_modifier=None, out=out)
    assert "base_argmax" not in out, (
        "select_action populated base_argmax with no modifier present; "
        "the influence-rate gating logic will count spurious flips."
    )


def test_out_dict_unset_on_static_path():
    """Static returns cold_chain before the modifier branch executes."""
    out: dict = {}
    modifier = np.array([0.0, 1.0, 0.0])
    action_idx, _ = _select("static", context_modifier=modifier, out=out)
    assert action_idx == 0
    assert "base_argmax" not in out, (
        "select_action populated base_argmax on the static path; static "
        "is supposed to bypass the modifier branch entirely."
    )


def test_out_dict_populated_during_cyber_outage_normal_policy():
    """Cyber outage no longer bypasses the context-to-policy path."""
    out: dict = {}
    modifier = np.array([0.5, 0.0, 0.0])
    _select("agribrain", context_modifier=modifier,
            scenario="cyber_outage", hour=30.0, out=out)
    assert "base_argmax" in out


def test_modifier_can_flip_chosen_action():
    """In deterministic mode, a large modifier can flip the policy argmax."""
    # First: no modifier baseline.
    base_out: dict = {}
    base_action, _ = _select(
        "agribrain", context_modifier=np.array([0.0, 0.0, 0.0]),
        out=base_out, rho=0.20, inv=100.0, temp=5.0,
    )
    base_argmax = base_out["base_argmax"]
    # Construct a modifier that strongly recommends a different action.
    flip_target = (base_argmax + 1) % 3
    aggressive = np.zeros(3)
    aggressive[flip_target] = 5.0
    flip_out: dict = {}
    flipped_action, _ = _select(
        "agribrain", context_modifier=aggressive, out=flip_out,
        rho=0.20, inv=100.0, temp=5.0,
    )
    assert flip_out["base_argmax"] == base_argmax, (
        "base_argmax depends on the modifier; the metric is broken."
    )
    assert flipped_action != base_argmax, (
        "A modifier with magnitude 5.0 on a non-base action did not flip "
        "the chosen action; the influence rate will undercount real flips."
    )


def test_zero_modifier_does_not_flip():
    """In deterministic mode, a zero modifier preserves the policy argmax."""
    out: dict = {}
    modifier = np.zeros(3)
    action_idx, _ = _select("agribrain", context_modifier=modifier, out=out)
    assert out["base_argmax"] == action_idx, (
        "A zero modifier produced a flipped action; numerical noise is "
        "leaking into the influence-rate counter."
    )


def test_negative_only_modifier_typically_does_not_flip():
    """When all modifier components are negative ('avoid every action a
    little'), argmax(base + modifier) usually equals argmax(base) because
    the relative ranking is preserved. This keeps the observer-only argmax
    diagnostic numerically well behaved.
    """
    # This is a probabilistic property, not a strict invariant: a
    # heterogeneous negative modifier could still re-rank near-tied
    # actions. Test with a uniform negative modifier so the ranking
    # is provably preserved.
    out: dict = {}
    modifier = np.array([-0.3, -0.3, -0.3])
    action_idx, _ = _select(
        "agribrain", context_modifier=modifier, out=out,
        rho=0.20, inv=100.0, temp=5.0,
    )
    assert out["base_argmax"] == action_idx, (
        "Uniform negative modifier unexpectedly changed the deterministic "
        "policy argmax."
    )


def test_same_draw_context_ablation_ignores_sampling_away_from_argmax():
    """Stochastic sampling alone must not count as context influence.

    A uniform active modifier shifts every logit equally, leaving the policy
    distribution unchanged. Seed 2 samples cold-chain even though the common
    policy argmax is local redistribution. The legacy argmax-vs-sample metric
    therefore reported a false change; paired replay must not.
    """
    import sys
    from pathlib import Path

    sim_dir = Path(__file__).resolve().parents[3] / "mvp" / "simulation"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import generate_results as gr  # type: ignore

    policy = Policy()
    live_rng = np.random.default_rng(2)
    saved_state = copy.deepcopy(live_rng.bit_generator.state)
    live_out: dict = {}
    live_action, live_probs = select_action(
        mode="agribrain", rho=0.20, inv=100.0, y_hat=100.0,
        temp=5.0, tau=0.0, policy=policy, rng=live_rng,
        context_modifier=np.full(3, 0.20), out=live_out,
    )
    cf_rng = np.random.default_rng()
    cf_rng.bit_generator.state = saved_state
    cf_action, cf_probs = select_action(
        mode="agribrain", rho=0.20, inv=100.0, y_hat=100.0,
        temp=5.0, tau=0.0, policy=policy, rng=cf_rng,
        context_modifier=None,
    )

    assert live_action != live_out["base_argmax"]
    np.testing.assert_allclose(live_probs, cf_probs)
    assert live_action == cf_action
    assert not gr._paired_context_action_changed(
        live_action, cf_action, cf_probs,
    )


def test_paired_metric_requires_a_successful_context_ablation():
    """Unavailable context-ablated replays are excluded rather than guessed."""
    import sys
    from pathlib import Path

    sim_dir = Path(__file__).resolve().parents[3] / "mvp" / "simulation"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import generate_results as gr  # type: ignore

    assert not gr._paired_context_action_changed(1, 0, None)
    assert gr._paired_context_action_changed(
        1, 0, np.array([0.7, 0.2, 0.1]),
    )


def test_strict_publication_run_aborts_if_paired_replay_fails(monkeypatch):
    """STRICT_VALIDATION must not silently undercount failed replays."""
    import sys
    from pathlib import Path

    sim_dir = Path(__file__).resolve().parents[3] / "mvp" / "simulation"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import generate_results as gr  # type: ignore
    import src.models.action_selection as action_selection_module

    def _fail_context_ablation(*args, **kwargs):
        raise ValueError("synthetic replay failure")

    # AgentCoordinator holds the original live select_action reference, while
    # post_step imports this module attribute for the replay. Patching here
    # therefore fails only the context-ablated replay.
    monkeypatch.setattr(
        action_selection_module, "select_action", _fail_context_ablation,
    )
    monkeypatch.setenv("STRICT_VALIDATION", "1")
    df = gr.pd.read_csv(
        gr.DATA_CSV, parse_dates=["timestamp"],
    ).head(16).reset_index(drop=True)

    with pytest.raises(
        RuntimeError, match="paired pre-selection-state context ablation failed",
    ):
        gr.run_episode(
            df, "agribrain", gr.Policy(), np.random.default_rng(42),
            "baseline", seed=42,
        )


def test_generate_results_emits_both_rates(monkeypatch):
    """run_episode result dict must carry both honor and influence rate
    fields so the supplementary methods table can quote either.
    """
    # Quick integration: run a 16-step episode in deterministic mode and
    # check the result dict shape. Skipped if the simulator import path
    # is broken (e.g. partial install).
    sys = pytest.importorskip("sys")  # always available; placeholder import
    monkeypatch.setenv("DETERMINISTIC_MODE", "true")
    from pathlib import Path
    sim_dir = Path(__file__).resolve().parents[3] / "mvp" / "simulation"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import generate_results as gr  # type: ignore

    df = gr.pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(16).reset_index(drop=True)
    rng = np.random.default_rng(42)
    ep = gr.run_episode(df, "agribrain", gr.Policy(), rng,
                         "baseline", seed=42)

    for key in (
        "context_active_steps",
        "context_honored_steps",
        "context_honor_rate",
        "context_influenced_steps",
        "context_influence_rate",
    ):
        assert key in ep, f"run_episode result dict missing {key!r}"

    assert 0 <= ep["context_influence_rate"] <= 1.0, (
        f"context_influence_rate {ep['context_influence_rate']} outside [0, 1]"
    )
    assert 0 <= ep["context_honor_rate"] <= 1.0, (
        f"context_honor_rate {ep['context_honor_rate']} outside [0, 1]"
    )
    # Same denominator: the influenced count cannot exceed the active count.
    assert ep["context_influenced_steps"] <= ep["context_active_steps"], (
        "influenced_steps > active_steps -- denominator invariant violated."
    )
    # Same denominator: honored count cannot exceed active count either.
    assert ep["context_honored_steps"] <= ep["context_active_steps"], (
        "honored_steps > active_steps -- denominator invariant violated."
    )

    # The per-decision ledger must carry the exact paired intervention used by
    # the episode statistic. This makes the published numerator independently
    # recomputable rather than an untraceable aggregate counter.
    ledger_path = Path(ep["decision_ledger_path"])
    with ledger_path.open("r", encoding="utf-8") as handle:
        ledger_rows = [json.loads(line) for line in handle if line.strip()]
    records = [row for row in ledger_rows if not row.get("_header")]
    assert len(records) == len(df)
    assert all(row["context_counterfactual_probs"] is not None for row in records)
    assert all(len(row["context_counterfactual_probs"]) == 3 for row in records)
    for row in records:
        assert "retrieval_top_fused_score" in row
        assert "retrieval_top_rerank_score" in row
        assert row["retrieval_top_score"] == row["retrieval_top_fused_score"]
        expected_changed = (
            int(row["action_idx"])
            != int(row["context_counterfactual_action_idx"])
        )
        assert row["context_action_changed"] is expected_changed
        assert row["context_influence_counted"] is (
            expected_changed and bool(row["context_influence_active"])
        )
    ledger_changed_count = sum(
        int(bool(row["context_influence_counted"])) for row in records
    )
    assert ledger_changed_count == ep["context_influenced_steps"]


def test_threshold_counters_carry_both_rates(monkeypatch):
    """The per-threshold sensitivity table must carry both rates."""
    sys = pytest.importorskip("sys")
    monkeypatch.setenv("DETERMINISTIC_MODE", "true")
    from pathlib import Path
    sim_dir = Path(__file__).resolve().parents[3] / "mvp" / "simulation"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import generate_results as gr  # type: ignore

    df = gr.pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(16).reset_index(drop=True)
    rng = np.random.default_rng(42)
    ep = gr.run_episode(df, "agribrain", gr.Policy(), rng,
                         "baseline", seed=42)
    counters = ep.get("context_threshold_counters", {})
    assert counters, "context_threshold_counters missing from result dict"
    for thr_key, payload in counters.items():
        for field in ("active", "honored", "influenced",
                      "honor_rate", "influence_rate"):
            assert field in payload, (
                f"threshold {thr_key} payload missing {field!r}; "
                f"got keys {sorted(payload.keys())}"
            )
