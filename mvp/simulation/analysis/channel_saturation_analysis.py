#!/usr/bin/env python3
r"""Channel saturation vs. redundancy analysis (C4 disambiguation).

The channel-decomposition contrasts ``agribrain_vs_{mcp_only,pirag_only}``
test whether *adding* the second context channel on top of one improves ARI
further. In some scenarios these contrasts are null. That null was
previously *asserted* to mean "the single channel saturates this
scenario" rather than "the second channel is unnecessary" -- a framing
choice. This script turns it into two
falsifiable tests, both run on the canonical 20-seed benchmark per-seed
envelopes (``benchmark_seeds/<run>/seed_*.json``) -- no re-run required.

1. TOST equivalence (per scenario + pooled). For the paired per-seed
   "add-the-second-channel" differences ``d = agribrain - single_channel``,
   a two-one-sided-test against an equivalence margin SESOI distinguishes:
     * additive    : d significantly > 0 (the second channel adds value);
     * equivalent  : |d| significantly within +/-SESOI (a *bounded* null --
                     the second channel is genuinely redundant *here*, not
                     merely underpowered);
     * inconclusive: neither (underpowered / can't separate from SESOI).
   SESOI = 0.01 ARI -- the same negligible-effect threshold the manuscript
   pre-registers for H3 robustness, so the margin is not chosen post hoc.

2. Cross-fitted moderation slope (the saturation mechanism). Saturation
   predicts the second channel's marginal value *shrinks where the first
   channel is already strong*. Regressing ``(agribrain - mcp_only)`` on
   ``(mcp_only - no_context)`` directly would be mathematically coupled (both
   share ``mcp_only`` -> spurious negative slope). We break the coupling by
   cross-fitting: the first-channel standalone strength is estimated on one
   disjoint half of the seeds and the second-channel marginal on the other
   half, so the shared-term artefact cannot arise.
     * slope ~ -1  : strong saturation / substitution (one channel's gain
                     displaces the other's);
     * slope ~  0  : independent additivity (second channel adds its own
                     value regardless of the first);
     * -1 < slope < 0 : partial saturation.
   The naive (coupled) slope is also reported, labelled as a coupling-
   inflated bound, for transparency.

Output: ``mvp/simulation/results/channel_saturation_analysis.json``.

Run::

    python mvp/simulation/analysis/channel_saturation_analysis.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover - scipy is a hard dep of the pipeline
    _st = None

SESOI = 0.01           # equivalence margin in ARI units (H3 pre-reg threshold)
ALPHA = 0.05
SCENARIOS = ("heatwave", "overproduction", "cyber_outage",
             "adaptive_pricing", "baseline")
PERTURBED = ("heatwave", "overproduction", "cyber_outage", "adaptive_pricing")
_SPLIT_RNG_SEED = 20260606


def _find_canonical_run(seeds_root: Path) -> Path:
    """Return the benchmark_seeds/<run> dir that matches Table 1.

    Pins to the run whose dir name starts with the short git_commit recorded
    in ``benchmark_summary.json`` (the same run the manuscript reports), so
    this analysis can never silently drift to a newer run dir. Falls back to
    the most-populated / latest dir if the summary or a match is unavailable.
    """
    candidates = []
    for d in sorted(seeds_root.glob("*")):
        if d.is_dir():
            n = len(list(d.glob("seed_*.json")))
            if n:
                candidates.append((n, d.name, d))
    if not candidates:
        raise SystemExit(f"no seed envelopes under {seeds_root}")

    summary = seeds_root.parent / "benchmark_summary.json"
    if summary.exists():
        try:
            commit = json.loads(summary.read_text()).get("_meta", {}).get("git_commit", "")
        except Exception:
            commit = ""
        if commit:
            short = commit[:7]
            for _, name, d in candidates:
                if name.startswith(short):
                    return d
            print(f"WARNING: no benchmark_seeds run matches benchmark_summary "
                  f"git_commit {short}; falling back to latest.")
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def _load_ari(run_dir: Path):
    """Return {scenario: {mode: np.array of per-seed ARI}} and the seed order."""
    files = sorted(run_dir.glob("seed_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    seeds, rows = [], []
    for f in files:
        d = json.loads(f.read_text())
        seeds.append(int(d.get("seed", int(f.stem.split("_")[1]))))
        rows.append(d["scenarios"])
    out = {}
    for scn in SCENARIOS:
        out[scn] = {}
        for mode in ("agribrain", "mcp_only", "pirag_only", "no_context"):
            vals = [r.get(scn, {}).get(mode, {}).get("ari") for r in rows]
            out[scn][mode] = np.array(
                [v for v in vals if v is not None], dtype=float)
    return out, seeds


def _paired_tost(diff: np.ndarray, sesoi: float = SESOI, alpha: float = ALPHA):
    """Paired TOST (two one-sided t-tests) for equivalence of mean(diff) to 0
    within +/-sesoi. Returns the verdict and the supporting statistics."""
    n = diff.size
    mean = float(diff.mean()) if n else 0.0
    if n < 2:
        return {"verdict": "inconclusive", "n": n, "mean_diff": mean}
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    # Two-sided test that the effect differs from 0 (is it additive?).
    if se == 0.0:
        # Degenerate: identical every seed. mean==0 -> equivalent; else additive.
        p_two = 0.0 if mean != 0 else 1.0
        t_lower = t_upper = np.inf if mean != 0 else 0.0
    else:
        t_stat = mean / se
        p_two = float(2 * _st.t.sf(abs(t_stat), df))
        t_lower = (mean - (-sesoi)) / se   # H0: mean <= -sesoi
        t_upper = (mean - (sesoi)) / se    # H0: mean >= +sesoi
    # One-sided p-values for the two TOST nulls.
    p_lower = float(_st.t.sf(t_lower, df)) if se else (0.0 if mean > -sesoi else 1.0)
    p_upper = float(_st.t.cdf(t_upper, df)) if se else (0.0 if mean < sesoi else 1.0)
    p_tost = max(p_lower, p_upper)          # equivalence established iff < alpha
    ci_lo = mean - _st.t.ppf(1 - alpha / 2, df) * se if se else mean
    ci_hi = mean + _st.t.ppf(1 - alpha / 2, df) * se if se else mean

    additive = (p_two < alpha) and (mean > 0)
    equivalent = (p_tost < alpha)
    if additive and not equivalent:
        verdict = "additive"            # second channel adds value here
    elif equivalent and not additive:
        verdict = "equivalent"          # bounded null: genuinely redundant here
    elif additive and equivalent:
        verdict = "additive_small"      # >0 but also within SESOI (tiny but real)
    else:
        verdict = "inconclusive"        # underpowered / cannot separate
    return {
        "verdict": verdict,
        "n": int(n),
        "mean_diff": mean,
        "ci_low": float(ci_lo),
        "ci_high": float(ci_hi),
        "p_two_sided": p_two,
        "p_tost": p_tost,
        "sesoi": sesoi,
    }


def _linslope(x: np.ndarray, y: np.ndarray):
    if _st is None or x.size < 3 or np.allclose(x, x[0]):
        return {"slope": 0.0, "p_value": 1.0, "r2": 0.0, "n": int(x.size)}
    res = _st.linregress(x, y)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "p_value": float(res.pvalue), "r2": float(res.rvalue ** 2),
            "n": int(x.size)}


def _crossfit_moderation(ari, scenarios, first, second):
    """Coupling-free moderation slope of the SECOND channel's marginal value
    on the FIRST channel's standalone strength.

    first/second are 'mcp'/'pirag'. standalone(first) is estimated on seed
    half-1, marginal(second on top of first) on the disjoint half-2, so the
    two never share a seed's ``first_only`` term.
    """
    fmode = "mcp_only" if first == "mcp" else "pirag_only"
    xs, ys, xs_naive, ys_naive = [], [], [], []
    rng = np.random.default_rng(_SPLIT_RNG_SEED)
    for scn in scenarios:
        a = ari[scn]["agribrain"]
        fo = ari[scn][fmode]
        nc = ari[scn]["no_context"]
        n = min(a.size, fo.size, nc.size)
        if n < 4:
            continue
        idx = rng.permutation(n)
        h1, h2 = idx[: n // 2], idx[n // 2:]
        x_scn = float(np.mean(fo[h1] - nc[h1]))      # standalone(first), half-1
        for j in h2:                                  # marginal(second), half-2
            xs.append(x_scn)
            ys.append(float(a[j] - fo[j]))
        # Naive (coupled) within-seed points, for the labelled bound.
        for j in range(n):
            xs_naive.append(float(fo[j] - nc[j]))
            ys_naive.append(float(a[j] - fo[j]))
    return {
        "crossfit": _linslope(np.array(xs), np.array(ys)),
        "naive_coupled_bound": _linslope(np.array(xs_naive), np.array(ys_naive)),
        "interpretation": ("slope ~ -1 substitution/strong saturation; "
                           "slope ~ 0 independent additivity; "
                           "crossfit breaks the shared-term coupling, "
                           "naive is a coupling-inflated bound"),
    }


def _mean_ci(x: np.ndarray):
    n = x.size
    m = float(x.mean()) if n else 0.0
    if _st is None or n < 2:
        return {"mean": m, "ci_low": m, "ci_high": m, "n": int(n)}
    se = float(x.std(ddof=1)) / np.sqrt(n)
    h = _st.t.ppf(1 - ALPHA / 2, n - 1) * se
    return {"mean": m, "ci_low": m - h, "ci_high": m + h, "n": int(n)}


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds-root", type=Path,
                    default=Path("mvp/simulation/results/benchmark_seeds"))
    ap.add_argument("--output", type=Path,
                    default=Path("mvp/simulation/results/channel_saturation_analysis.json"))
    args = ap.parse_args()

    if _st is None:
        raise SystemExit("scipy is required for channel_saturation_analysis")

    run_dir = _find_canonical_run(args.seeds_root.resolve())
    ari, seeds = _load_ari(run_dir)

    by_scn = {}
    for scn in SCENARIOS:
        a, m, p, n = (ari[scn]["agribrain"], ari[scn]["mcp_only"],
                      ari[scn]["pirag_only"], ari[scn]["no_context"])
        k = min(a.size, m.size, p.size, n.size)
        a, m, p, n = a[:k], m[:k], p[:k], n[:k]
        by_scn[scn] = {
            "n_seeds": int(k),
            "standalone_mcp": _mean_ci(m - n),
            "standalone_pirag": _mean_ci(p - n),
            "full_gain": _mean_ci(a - n),
            # "add piRAG on top of MCP" and "add MCP on top of piRAG"
            "add_pirag_on_mcp": _paired_tost(a - m),
            "add_mcp_on_pirag": _paired_tost(a - p),
        }

    def _pool(diff_fn):
        return np.concatenate([
            diff_fn(ari[s]) for s in PERTURBED
            if ari[s]["agribrain"].size])

    pooled = {
        "scenarios": list(PERTURBED),
        "add_pirag_on_mcp": _paired_tost(
            _pool(lambda c: c["agribrain"][:min(c["agribrain"].size, c["mcp_only"].size)]
                  - c["mcp_only"][:min(c["agribrain"].size, c["mcp_only"].size)])),
        "add_mcp_on_pirag": _paired_tost(
            _pool(lambda c: c["agribrain"][:min(c["agribrain"].size, c["pirag_only"].size)]
                  - c["pirag_only"][:min(c["agribrain"].size, c["pirag_only"].size)])),
    }

    moderation = {
        "pirag_marginal_vs_mcp_strength": _crossfit_moderation(
            ari, PERTURBED, first="mcp", second="pirag"),
        "mcp_marginal_vs_pirag_strength": _crossfit_moderation(
            ari, PERTURBED, first="pirag", second="mcp"),
    }

    out = {
        "_meta": {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "benchmark_run": run_dir.name,
            "n_seeds": len(seeds),
            "sesoi_ari": SESOI,
            "alpha": ALPHA,
            "split_rng_seed": _SPLIT_RNG_SEED,
            "source": "benchmark_seeds/<run>/seed_*.json (ari field)",
            "method": ("TOST equivalence on paired add-second-channel diffs "
                       "(SESOI=0.01 ARI = H3 pre-reg threshold) + cross-fitted "
                       "moderation slope (coupling-free)"),
        },
        "by_scenario": by_scn,
        "pooled_perturbed": pooled,
        "moderation": moderation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Saved: {args.output}  (run {run_dir.name}, {len(seeds)} seeds)")

    # Console summary.
    print("\nPer-scenario 'add the second channel' verdicts (TOST, SESOI=0.01):")
    for scn, c in by_scn.items():
        print(f"  {scn:16s} +piRAG/MCP: {c['add_pirag_on_mcp']['verdict']:13s}"
              f" (Δ={c['add_pirag_on_mcp']['mean_diff']:+.4f})"
              f"   +MCP/piRAG: {c['add_mcp_on_pirag']['verdict']:13s}"
              f" (Δ={c['add_mcp_on_pirag']['mean_diff']:+.4f})")
    mo = moderation["pirag_marginal_vs_mcp_strength"]
    print(f"\nCross-fit moderation (piRAG marginal vs MCP strength): "
          f"slope={mo['crossfit']['slope']:+.3f} p={mo['crossfit']['p_value']:.3g} "
          f"(naive bound {mo['naive_coupled_bound']['slope']:+.3f})")


if __name__ == "__main__":
    main()
