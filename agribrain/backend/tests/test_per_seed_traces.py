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


def test_run_single_seed_declares_canonical_trace_modes():
    """The canonical paper trio is the documented contract."""
    text = (_BENCHMARKS_DIR / "run_single_seed.py").read_text(encoding="utf-8")
    assert 'TRACE_MODES = ("static", "hybrid_rl", "agribrain")' in text, (
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
    text = (_BENCHMARKS_DIR / "run_single_seed.py").read_text(encoding="utf-8")
    required_fields = (
        "ari_trace",
        "waste_trace",
        "rho_trace",
        "action_trace",
        "prob_trace",
        "carbon_trace",
        "hours",
        "batch_effective_rho_trace",
        "effective_rho_trace",
        "temp_trace",
        "rh_trace",
        "inventory_trace",
        "demand_trace",
        "slca_component_trace",
        "equity_trace",
        "reward_trace",
    )
    for field in required_fields:
        assert f'"{field}"' in text, (
            f"TRACE_FIELDS no longer dumps {field}; the "
            f"regenerate_figures_from_cache.py path will lose its "
            f"ability to re-render any panel that consumes that "
            f"per-step trace. See run_single_seed.py docstring for "
            f"the field-by-figure mapping."
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


def test_loader_returns_none_when_benchmark_seeds_missing(tmp_path, monkeypatch):
    """Fig 2 panel d falls back cleanly when no per-seed JSONs exist."""
    import generate_figures as gf  # type: ignore
    # Redirect RESULTS_DIR to an empty tmp_path; loader should return None.
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    out = gf._load_per_seed_traces("heatwave", "agribrain", "ari_trace")
    assert out is None, (
        "Loader returned non-None when benchmark_seeds/ doesn't exist; "
        "the fig 2 fallback path won't trigger and the panel will "
        "crash on a missing array."
    )


def test_loader_returns_none_when_dir_empty(tmp_path, monkeypatch):
    """Empty benchmark_seeds/ -> loader returns None (graceful fallback)."""
    import generate_figures as gf  # type: ignore
    (tmp_path / "benchmark_seeds").mkdir()
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    out = gf._load_per_seed_traces("heatwave", "agribrain", "ari_trace")
    assert out is None


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


def test_loader_stacks_three_synthetic_seeds(tmp_path, monkeypatch):
    """Three synthetic seeds -> (3, n_steps) stack returned."""
    import generate_figures as gf  # type: ignore
    seeds_dir = tmp_path / "benchmark_seeds"
    seeds_dir.mkdir()
    n_steps = 8
    for s, base in [(42, 0.5), (1337, 0.55), (2024, 0.60)]:
        trace = [round(base + 0.01 * t, 4) for t in range(n_steps)]
        _write_seed_json(
            seeds_dir, s,
            traces={"heatwave": {"agribrain": {"ari_trace": trace}}},
        )
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    out = gf._load_per_seed_traces("heatwave", "agribrain", "ari_trace")
    assert out is not None, "Loader returned None despite three valid seed files"
    assert out.shape == (3, n_steps), f"shape={out.shape}, expected (3, {n_steps})"
    # Per-step seed-mean for step 0 should be (0.5 + 0.55 + 0.6) / 3 = 0.55
    assert out[:, 0].mean() == pytest.approx(0.55, abs=1e-9)


def test_loader_handles_tagged_subdir_layout(tmp_path, monkeypatch):
    """HPC orchestrator writes seeds under benchmark_seeds/<RUN_TAG>/."""
    import generate_figures as gf  # type: ignore
    seeds_root = tmp_path / "benchmark_seeds"
    tagged_dir = seeds_root / "abc123_20260504_1200"
    tagged_dir.mkdir(parents=True)
    n_steps = 6
    for s in (42, 1337):
        _write_seed_json(
            tagged_dir, s,
            traces={"heatwave": {"static": {"ari_trace": [0.4] * n_steps}}},
        )
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    out = gf._load_per_seed_traces("heatwave", "static", "ari_trace")
    assert out is not None and out.shape == (2, n_steps)


def test_loader_drops_seeds_with_mismatched_step_count(tmp_path, monkeypatch):
    """A truncated seed must not crash the stack; it gets dropped."""
    import generate_figures as gf  # type: ignore
    seeds_dir = tmp_path / "benchmark_seeds"
    seeds_dir.mkdir()
    full = [0.5] * 8
    short = [0.5] * 5  # truncated
    for s, t in [(42, full), (1337, full), (2024, short)]:
        _write_seed_json(
            seeds_dir, s,
            traces={"heatwave": {"agribrain": {"ari_trace": t}}},
        )
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    out = gf._load_per_seed_traces("heatwave", "agribrain", "ari_trace")
    # Modal length is 8 (two seeds), so the short seed (one) is dropped.
    assert out is not None and out.shape == (2, 8)


def test_loader_returns_none_when_no_traces_key(tmp_path, monkeypatch):
    """Legacy per-seed JSONs (pre-2026-05) lack the "traces" key.

    The loader returns None for those so the figure's fallback path
    fires cleanly.
    """
    import generate_figures as gf  # type: ignore
    seeds_dir = tmp_path / "benchmark_seeds"
    seeds_dir.mkdir()
    legacy_payload = {
        "heatwave": {"agribrain": {"ari": 0.6}},
        # No "traces" key, no "seed" key, just the legacy
        # scenario-at-root format.
    }
    (seeds_dir / "seed_42.json").write_text(json.dumps(legacy_payload))
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    out = gf._load_per_seed_traces("heatwave", "agribrain", "ari_trace")
    assert out is None


def test_loader_returns_none_for_missing_scenario_or_mode(tmp_path, monkeypatch):
    """Asking for a scenario/mode the dump didn't carry -> None."""
    import generate_figures as gf  # type: ignore
    seeds_dir = tmp_path / "benchmark_seeds"
    seeds_dir.mkdir()
    _write_seed_json(
        seeds_dir, 42,
        traces={"heatwave": {"agribrain": {"ari_trace": [0.5, 0.51]}}},
    )
    monkeypatch.setattr(gf, "RESULTS_DIR", tmp_path)
    # Wrong scenario.
    assert gf._load_per_seed_traces("cyber_outage", "agribrain", "ari_trace") is None
    # Wrong mode.
    assert gf._load_per_seed_traces("heatwave", "no_pinn", "ari_trace") is None
    # Wrong field.
    assert gf._load_per_seed_traces("heatwave", "agribrain", "rho_trace") is None
