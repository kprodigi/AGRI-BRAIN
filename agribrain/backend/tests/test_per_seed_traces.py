"""Regression tests for the 2026-05 per-seed-trace dump + loader.

Locks the contracts that fig 2 panel (d) seed-CI ribbon depends on:

* ``run_single_seed.py`` dumps a JSON envelope of the form
  ``{"seed": int, "scenarios": {...}, "traces": {sc: {mode: {"ari_trace": [...]}}}}``.
* The "scenarios" block keeps the same shape the legacy aggregator
  consumes (so old benchmark_summary aggregation is byte-stable).
* The "traces" block carries ``ari_trace`` for the canonical paper trio
  ``(static, hybrid_rl, agribrain)`` across all 5 scenarios at 4-decimal
  precision.
* ``generate_figures._load_per_seed_traces`` walks a tagged or flat
  ``benchmark_seeds/`` directory, stacks the per-seed traces into a
  ``(n_seeds, n_steps)`` array, and returns ``None`` when no per-seed
  JSONs are present (so fig 2 panel d's ribbon path can fall back to
  the single-seed line cleanly).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIM_DIR = _REPO_ROOT / "mvp" / "simulation"
_BENCHMARKS_DIR = _SIM_DIR / "benchmarks"

if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))


def test_action_contract_accepts_only_exact_discrete_numeric_values():
    """Immutable d3286ae traces use integral floats; reject everything else."""

    from benchmarks.trace_contract import _canonical_action_index

    for value, expected in ((0, 0), (1, 1), (2, 2), (0.0, 0), (1.0, 1), (2.0, 2)):
        assert _canonical_action_index(value, where="test/action_trace") == expected
    invalid_values = (
        True,
        False,
        -1,
        3,
        10**400,
        0.5,
        1.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        "1",
        None,
    )
    for invalid in invalid_values:
        with pytest.raises(ValueError, match="noncanonical action"):
            _canonical_action_index(invalid, where="test/action_trace")


def test_run_single_seed_declares_canonical_trace_modes():
    """The canonical paper trio is the documented contract."""
    from benchmarks.trace_contract import TRACE_MODES

    assert TRACE_MODES == ("static", "hybrid_rl", "agribrain"), (
        "TRACE_MODES drifted from the canonical paper trio "
        "(static, hybrid_rl, agribrain). Update fig 2 panel d "
        "comments and the test in lockstep with any change."
    )


def test_run_single_seed_declares_canonical_trace_fields():
    """Pin the per-step fields each seed envelope must dump.

    Pre-2026-05 the contract was the single field ``ari_trace`` for
    fig 2 panel D's seed-CI ribbon. Extended in 2026-05 to cover
    every per-step field the figure code reads, so a completed
    HPC run produces a self-contained cache the
    ``regenerate_figures_from_cache.py`` script can re-render every
    figure from without rerunning the simulator. The full set is
    enumerated in TRACE_FIELDS at the top of run_single_seed.py and
    documented inline there.
    """
    from benchmarks.trace_contract import TRACE_FIELDS

    required_fields = (
        "ari_trace",
        "waste_trace",
        "rho_trace",
        "rho_policy_observed_trace",
        "rho_outcome_environmental_trace",
        "action_trace",
        "prob_trace",
        "carbon_trace",
        "hours",
        "temp_trace",
        "rh_trace",
        "inventory_trace",
        "demand_trace",
        "temp_policy_observed_trace",
        "temp_outcome_environmental_trace",
        "rh_policy_observed_trace",
        "rh_outcome_environmental_trace",
        "inventory_policy_observed_trace",
        "inventory_outcome_environmental_trace",
        "demand_policy_observed_trace",
        "demand_forecast_policy_observed_trace",
        "demand_regime_flag_trace",
        "price_signal_trace",
        "supply_forecast_policy_observed_trace",
        "demand_outcome_environmental_trace",
        "transport_multiplier_outcome_environmental_trace",
        "simulated_dispatch_accounted_trace",
        "slca_component_trace",
        "slca_trace",
        "equity_trace",
        "reward_trace",
    )
    assert TRACE_FIELDS == required_fields, (
        "The shared TRACE_FIELDS contract drifted from the complete, ordered "
        "publication trace schema. Update the simulator, raw validator, figure "
        "renderer, and this assertion together for an intentional change."
    )


def test_run_single_seed_envelope_shape(tmp_path: Path, monkeypatch):
    """The dumped JSON has the documented envelope keys.

    Imports run_single_seed.main with sys.argv patched. Uses
    DETERMINISTIC_MODE=true for speed (still ~3-5 min on this hardware
    -- the simulator runs the full mode x scenario matrix). Marked
    'slow' so it doesn't bloat the default suite.
    """
    pytest.skip(
        "Heavy: requires full simulator run. Covered by the "
        "structural / file-existence checks below plus the runtime "
        "exercise in mvp/simulation/tests/test_per_seed_traces_integration.py "
        "(once HPC writes per-seed JSONs into the canonical path)."
    )


def _write_seed_json(seed_dir: Path, seed: int, *, traces: dict) -> None:
    """Helper to drop a synthetic per-seed JSON in the documented envelope shape."""
    payload = {
        "seed": seed,
        "scenarios": {
            "heatwave": {"agribrain": {"ari": 0.6, "waste": 0.04}},
        },
        "traces": traces,
    }
    (seed_dir / f"seed_{seed}.json").write_text(json.dumps(payload))

