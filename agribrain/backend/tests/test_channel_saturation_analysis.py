"""Correctness pins for the §5.8 channel saturation/redundancy analysis
(`mvp/simulation/analysis/channel_saturation_analysis.py`).

Covers the two statistical primitives that back it:
  * `_paired_tost` -> the equivalence verdict on add-second-channel diffs;
  * `_crossfit_moderation` -> the coupling-free saturation slope (and that the
    naive within-seed slope is the coupled one it corrects).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = (Path(__file__).resolve().parents[3] / "mvp" / "simulation" / "analysis")
sys.path.insert(0, str(ANALYSIS_DIR))

pytest.importorskip("scipy")
from channel_saturation_analysis import (  # noqa: E402
    _paired_tost, _crossfit_moderation, _find_canonical_run, _load_ari,
    _linslope, _validate_flat_seed_root, EXPECTED_SEEDS, SESOI,
)


def test_tost_equivalent_bounded_null():
    # Tight, centred near 0 and well inside +/-SESOI -> bounded equivalence.
    diff = 0.0005 * np.array([1, -1] * 10, dtype=float)  # mean 0, tiny sd
    r = _paired_tost(diff)
    assert r["verdict"] == "equivalent_within_margin", r
    assert r["p_tost"] < 0.05 and r["p_two_sided"] > 0.05
    assert r["ci90_low"] > -SESOI and r["ci90_high"] < SESOI


def test_tost_additive_clear_positive():
    # Mean 0.03 ARI, far above the 0.01 margin -> additive, not equivalent.
    diff = 0.03 + 0.001 * np.array([1, -1] * 10, dtype=float)
    r = _paired_tost(diff)
    assert r["verdict"] == "positive_difference", r
    assert r["p_two_sided"] < 0.05 and r["p_tost"] > 0.05
    assert r["mean_diff"] > SESOI


def test_tost_inconclusive_underpowered():
    # Small mean but large spread: neither significantly >0 nor equivalent.
    diff = 0.005 + 0.03 * np.array([1, -1] * 10, dtype=float)
    r = _paired_tost(diff)
    assert r["verdict"] == "inconclusive", r
    assert r["p_two_sided"] > 0.05 and r["p_tost"] > 0.05


def test_tost_reports_clear_negative_difference_not_inconclusive():
    diff = -0.03 + 0.001 * np.array([1, -1] * 10, dtype=float)
    r = _paired_tost(diff)
    assert r["verdict"] == "negative_difference", r
    assert r["p_two_sided"] < 0.05 and r["p_tost"] > 0.05


def test_crossfit_breaks_coupling_artifact():
    # Construct per-scenario ARI where the second channel adds a CONSTANT,
    # MCP-strength-independent marginal (true saturation slope = 0). The naive
    # within-seed slope must be driven negative by the shared mcp_only term,
    # while the cross-fit slope stays near 0.
    # Each mode is an INDEPENDENT noisy draw around base+gain -- this is what
    # creates the shared-term coupling: mcp_only appears in both x=(mcp-nc) and
    # y=(agri-mcp) as the *same* noisy draw, injecting a spurious -var(eps_mcp)
    # into cov(x, y). The true piR marginal is a CONSTANT 0.008 across
    # scenarios, so the genuine saturation slope is 0. The cross-fit removes the
    # coupling *in expectation*; we therefore verify the property over many
    # replications (a single 6-scenario draw is too noisy to assert on, which
    # is exactly why the real 4-scenario estimate is underpowered, p~0.6).
    scenarios = tuple(f"s{i}" for i in range(6))
    gains = np.linspace(0.004, 0.024, len(scenarios))
    # Use enough mode-specific noise for the shared mcp_only error term to be
    # visible at the scenario-mean level.  The goal is to test the direction of
    # the coupling bias, not to mimic the publication effect magnitude.
    base, sd, n_rep = 0.55, 0.030, 60
    naive_slopes, cross_slopes = [], []
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        ari = {}
        for s, g in zip(scenarios, gains):
            ari[s] = {
                "no_context": base + rng.normal(0, sd, 24),
                "mcp_only": base + g + rng.normal(0, sd, 24),
                "agribrain": base + g + 0.008 + rng.normal(0, sd, 24),
                "pirag_only": base + 0.006 + rng.normal(0, sd, 24),
            }
        mod = _crossfit_moderation(ari, scenarios, first="mcp", second="pirag")
        naive_slopes.append(mod["naive_coupled_bound"]["slope"])
        cross_slopes.append(mod["crossfit"]["slope"])
        assert mod["crossfit"]["n"] == len(scenarios)
        assert mod["crossfit"]["p_value"] is None
        assert mod["crossfit"]["inferential"] is False
    naive_mean = float(np.mean(naive_slopes))
    cross_mean = float(np.mean(cross_slopes))
    # Naive is biased clearly negative by the coupling; cross-fit recovers the
    # true ~0 slope, so its mean is above naive and much closer to zero.
    assert naive_mean < -0.1, f"coupled naive slope not negative in mean: {naive_mean}"
    assert cross_mean > naive_mean, (cross_mean, naive_mean)
    assert abs(cross_mean) < 0.5 * abs(naive_mean), (cross_mean, naive_mean)


def test_linslope_recovers_known_line():
    x = np.linspace(0, 1, 30)
    y = 2.0 * x + 0.5
    r = _linslope(x, y)
    assert r["slope"] == pytest.approx(2.0, abs=1e-9)
    assert r["r2"] == pytest.approx(1.0, abs=1e-9)


def test_linslope_constant_response_is_finite():
    r = _linslope(np.arange(4.0), np.ones(4))
    assert r["slope"] == 0.0
    assert r["intercept"] == 1.0
    assert r["r2"] == 0.0
    assert r["estimable"] is True
    assert np.isfinite([r["slope"], r["intercept"], r["r2"]]).all()


def test_linslope_constant_moderator_is_explicitly_unestimable():
    r = _linslope(np.ones(4), np.arange(4.0))
    assert r["estimable"] is False
    assert r["slope"] is None and r["r2"] is None and r["p_value"] is None


def test_explicit_publication_run_tag_wins_over_same_commit_prefix(tmp_path):
    old = tmp_path / "abc1234_old"
    target = tmp_path / "abc1234_target"
    old.mkdir()
    target.mkdir()
    for seed in range(20):
        (old / f"seed_{seed}.json").write_text("{}", encoding="utf-8")
        (target / f"seed_{seed}.json").write_text("{}", encoding="utf-8")
    assert _find_canonical_run(tmp_path, "abc1234_target") == target


def test_explicit_publication_run_tag_rejects_incomplete_panel(tmp_path):
    incomplete = tmp_path / "abc1234_incomplete"
    incomplete.mkdir()
    (incomplete / "seed_42.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="expected the complete 20-seed"):
        _find_canonical_run(tmp_path, "abc1234_incomplete")


def test_flat_seed_root_uses_only_exact_top_level_manifested_panel(tmp_path):
    for seed in EXPECTED_SEEDS:
        (tmp_path / f"seed_{seed}.json").write_text("{}")
    (tmp_path / "old_tagged_cache").mkdir()
    assert _validate_flat_seed_root(tmp_path) == tmp_path.resolve()

    (tmp_path / "seed_999999.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exact flat manifested seed panel"):
        _validate_flat_seed_root(tmp_path)


def test_ari_loader_rejects_modewise_missingness_instead_of_breaking_pairs(tmp_path):
    for seed in EXPECTED_SEEDS:
        scenarios = {}
        for scenario in ("heatwave", "overproduction", "cyber_outage",
                         "adaptive_pricing", "baseline"):
            scenarios[scenario] = {
                mode: {"ari": 0.5}
                for mode in ("agribrain", "mcp_only", "pirag_only", "no_context")
            }
        if seed == EXPECTED_SEEDS[-1]:
            del scenarios["heatwave"]["mcp_only"]["ari"]
        (tmp_path / f"seed_{seed}.json").write_text(
            __import__("json").dumps({"seed": seed, "scenarios": scenarios}),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="incomplete paired ARI panel"):
        _load_ari(tmp_path)


def test_ari_loader_uses_declared_seed_order(tmp_path):
    for seed in reversed(EXPECTED_SEEDS):
        scenarios = {
            scenario: {
                mode: {"ari": float(seed)}
                for mode in ("agribrain", "mcp_only", "pirag_only", "no_context")
            }
            for scenario in (
                "heatwave", "overproduction", "cyber_outage",
                "adaptive_pricing", "baseline",
            )
        }
        (tmp_path / f"seed_{seed}.json").write_text(
            __import__("json").dumps({"seed": seed, "scenarios": scenarios}),
            encoding="utf-8",
        )

    ari, seeds = _load_ari(tmp_path)
    assert tuple(seeds) == EXPECTED_SEEDS
    assert ari["baseline"]["agribrain"].tolist() == [
        float(seed) for seed in EXPECTED_SEEDS
    ]
