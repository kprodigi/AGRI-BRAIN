"""Regression tests for the 2026-05 context-influence rate metric.

Lock the contract that fig 9 panel (c) and the supplementary methods
table both depend on:

* ``select_action`` populates ``out["base_argmax"]`` exactly when a
  context modifier is supplied and the regular logit-construction
  path is taken (NOT on the static path or the cyber-outage Bernoulli
  reroute path).
* The coordinator stashes the value on ``self._step_base_argmax``
  alongside ``self._step_context_modifier``.
* ``generate_results.run_episode`` emits both ``context_honor_rate``
  and ``context_influence_rate`` on every episode result, with the
  latter being the count of steps where the modifier flipped the
  chosen action vs the base argmax.
* The two rates share the same denominator
  (``context_active_steps``).
* Modes that bypass the modifier (static, cyber-outage during the
  outage window) contribute zero to both numerators.

These tests run fast (no full ``run_all`` simulator invocation): they
exercise ``select_action`` directly with synthetic inputs.
"""
from __future__ import annotations

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
    """The regular logit-construction path with a modifier sets base_argmax."""
    out: dict = {}
    modifier = np.array([0.0, 0.5, 0.0])  # nudge toward local_redistribute
    action_idx, probs = _select("agribrain", context_modifier=modifier, out=out)
    assert "base_argmax" in out, (
        "select_action did not populate out['base_argmax'] on the "
        "context-modifier path; the context-influence metric will be "
        "uncomputable."
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


def test_out_dict_unset_during_cyber_outage_bernoulli():
    """Cyber-outage during outage window uses Bernoulli reroute, no modifier."""
    out: dict = {}
    modifier = np.array([0.5, 0.0, 0.0])
    _select("agribrain", context_modifier=modifier,
            scenario="cyber_outage", hour=30.0, out=out)
    assert "base_argmax" not in out, (
        "Cyber-outage Bernoulli path populated base_argmax; the "
        "modifier-vs-base comparison is undefined for those steps."
    )


def test_modifier_can_flip_chosen_action():
    """A large positive modifier on a non-base-argmax action flips the choice."""
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
    """All-zero modifier must yield base_argmax == chosen action."""
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
    the relative ranking is preserved. This is the case where the legacy
    honor-rate metric falsely flagged a non-honor; the new influence
    metric correctly records no flip.
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
        "Uniform negative modifier flipped the action; the metric is "
        "incorrectly counting noise-only signals as influence."
    )


def test_generate_results_emits_both_rates():
    """run_episode result dict must carry both honor and influence rate
    fields so the supplementary methods table can quote either.
    """
    # Quick integration: run a 16-step episode in deterministic mode and
    # check the result dict shape. Skipped if the simulator import path
    # is broken (e.g. partial install).
    sys = pytest.importorskip("sys")  # always available; placeholder import
    import os
    os.environ["DETERMINISTIC_MODE"] = "true"
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


def test_threshold_counters_carry_both_rates():
    """The per-threshold sensitivity table must carry both rates."""
    sys = pytest.importorskip("sys")
    import os
    os.environ["DETERMINISTIC_MODE"] = "true"
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
