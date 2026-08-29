#!/usr/bin/env python3
r"""Channel saturation vs. redundancy analysis (C4 disambiguation).

The channel-decomposition contrasts ``agribrain_vs_{mcp_only,pirag_only}``
test whether *adding* the second context channel on top of one improves ARI
further. In some scenarios these contrasts are null. That null was
previously *asserted* to mean "the single channel saturates this
scenario" rather than "the second channel is unnecessary" -- a framing
choice. This script turns it into two
falsifiable tests, both run on the exact flat, manifested 20-seed benchmark
envelopes (``benchmark_seeds/seed_*.json``) -- no re-run required.

1. TOST equivalence (per scenario + pooled). For the paired per-seed
   "add-the-second-channel" differences ``d = agribrain - single_channel``,
   a two-one-sided-test against an equivalence margin SESOI distinguishes:
     * additive    : d significantly > 0 (the second channel adds value);
     * equivalent  : |d| significantly within +/-SESOI (a *bounded* null --
                     the second channel is genuinely redundant *here*, not
                     merely underpowered);
     * negative    : d significantly below 0 (adding the channel reduced ARI);
     * inconclusive: neither (underpowered / can't separate from SESOI).
   SESOI = 0.01 ARI -- the same negligible-effect threshold the manuscript
   declares for H3 robustness. These scenario-specific H2 diagnostics are
   exploratory and unadjusted.

2. Cross-fitted descriptive moderation slope (a saturation diagnostic).
   Saturation
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
   Only four perturbed scenarios are available.  The cross-fitted and naive
   slopes are therefore reported as four-scenario descriptive diagnostics;
   no regression p-value is presented.  The naive (coupled) slope is retained,
   labelled as a coupling-prone descriptive comparator, for transparency; it
   is not a formal bound.

Output: ``mvp/simulation/results/channel_saturation_analysis.json``.

Run::

    python mvp/simulation/analysis/channel_saturation_analysis.py \
        --seed-root mvp/simulation/results/benchmark_seeds \
        --output mvp/simulation/results/channel_saturation_analysis.json \
        --source-commit "$AGRIBRAIN_GIT_COMMIT" --run-tag "$RUN_TAG"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover - scipy is a hard dep of the pipeline
    _st = None

SESOI = 0.01           # equivalence margin in ARI units (declared H3 threshold)
ALPHA = 0.05
SCENARIOS = ("heatwave", "overproduction", "cyber_outage",
             "adaptive_pricing", "baseline")
PERTURBED = ("heatwave", "overproduction", "cyber_outage", "adaptive_pricing")
EXPECTED_SEEDS = (42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505,
                  606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515)
_SPLIT_RNG_SEED = 20260606


def _find_canonical_run(seeds_root: Path, run_tag: str = "") -> Path:
    """Return the benchmark_seeds/<run> dir that matches Table 1.

    Pins to the run whose directory name starts with the short ``git_commit``
    recorded in ``benchmark_summary.json``. Missing or ambiguous provenance
    fails closed; this helper never chooses a latest or most-populated run.
    """
    if run_tag:
        exact = seeds_root / run_tag
        count = len(list(exact.glob("seed_*.json"))) if exact.is_dir() else 0
        if count != 20:
            raise SystemExit(
                f"ARTIFACT_RUN_TAG={run_tag!r} has {count} seed envelopes; "
                "expected the complete 20-seed publication panel"
            )
        return exact

    candidates = []
    for d in sorted(seeds_root.glob("*")):
        if d.is_dir():
            n = len(list(d.glob("seed_*.json")))
            if n:
                candidates.append((n, d.name, d))
    if not candidates:
        raise SystemExit(f"no seed envelopes under {seeds_root}")

    summary = seeds_root.parent / "benchmark_summary.json"
    if not summary.exists():
        raise SystemExit(
            f"missing provenance file required to select a canonical run: {summary}"
        )
    try:
        commit = json.loads(summary.read_text()).get("_meta", {}).get(
            "git_commit", ""
        )
    except Exception as exc:
        raise SystemExit(f"invalid canonical-run provenance: {summary}") from exc
    if not commit:
        raise SystemExit(f"canonical-run provenance lacks git_commit: {summary}")
    short = commit[:7]
    matches = [d for _, name, d in candidates if name.startswith(short)]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one benchmark_seeds run matching git_commit "
            f"{short}; found {len(matches)}"
        )
    return matches[0]


def _validate_flat_seed_root(seed_root: Path) -> Path:
    """Require only the exact 20 flat publication seed-envelope files.

    Tagged cache directories may coexist below ``benchmark_seeds`` while the
    publisher is running, but this producer never traverses them.  Its only
    inputs are the twenty top-level files retained by the artifact manifest.
    """

    if seed_root.is_symlink():
        raise RuntimeError("channel saturation seed root must not be a symlink")
    root = seed_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("channel saturation seed root must be a real directory")
    expected = {f"seed_{seed}.json" for seed in EXPECTED_SEEDS}
    observed = {
        path.name for path in root.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError(
            "channel saturation requires the exact flat manifested seed panel; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    for name in expected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"channel saturation seed input is irregular: {path}")
    return root


def _load_ari(run_dir: Path):
    """Return {scenario: {mode: np.array of per-seed ARI}} and the seed order."""
    files_by_seed = {
        int(path.stem.split("_")[1]): path
        for path in run_dir.glob("seed_*.json")
    }
    if set(files_by_seed) != set(EXPECTED_SEEDS) or len(files_by_seed) != len(EXPECTED_SEEDS):
        raise RuntimeError(
            "channel analysis requires the exact 20-seed publication inventory; "
            f"found {sorted(files_by_seed)}"
        )
    # Canonical order makes the deterministic cross-fit split invariant to
    # directory enumeration and filename sorting conventions.
    files = [files_by_seed[seed] for seed in EXPECTED_SEEDS]
    seeds, rows = [], []
    for f in files:
        d = json.loads(f.read_text())
        filename_seed = int(f.stem.split("_")[1])
        payload_seed = int(d.get("seed", filename_seed))
        if payload_seed != filename_seed:
            raise RuntimeError(f"seed identity mismatch in {f}")
        seeds.append(payload_seed)
        rows.append(d["scenarios"])
    if tuple(seeds) != EXPECTED_SEEDS:
        raise RuntimeError(
            "channel analysis requires the exact 20-seed publication panel; "
            f"found {seeds}"
        )
    out = {}
    for scn in SCENARIOS:
        out[scn] = {}
        for mode in ("agribrain", "mcp_only", "pirag_only", "no_context"):
            vals = [r.get(scn, {}).get(mode, {}).get("ari") for r in rows]
            if any(v is None for v in vals):
                raise RuntimeError(f"incomplete paired ARI panel for {scn}/{mode}")
            arr = np.asarray(vals, dtype=float)
            if arr.size != len(seeds) or not np.all(np.isfinite(arr)):
                raise RuntimeError(f"invalid paired ARI panel for {scn}/{mode}")
            out[scn][mode] = arr
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
    ci90_lo = mean - _st.t.ppf(1 - alpha, df) * se if se else mean
    ci90_hi = mean + _st.t.ppf(1 - alpha, df) * se if se else mean
    ci95_lo = mean - _st.t.ppf(1 - alpha / 2, df) * se if se else mean
    ci95_hi = mean + _st.t.ppf(1 - alpha / 2, df) * se if se else mean

    positive = (p_two < alpha) and (mean > 0)
    negative = (p_two < alpha) and (mean < 0)
    equivalent = (p_tost < alpha)
    if equivalent and positive:
        verdict = "positive_but_equivalent"  # nonzero, but within the SESOI
    elif equivalent and negative:
        verdict = "negative_but_equivalent"
    elif equivalent:
        verdict = "equivalent_within_margin"
    elif positive:
        verdict = "positive_difference"
    elif negative:
        verdict = "negative_difference"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "n": int(n),
        "mean_diff": mean,
        "ci90_low": float(ci90_lo),
        "ci90_high": float(ci90_hi),
        "ci95_low": float(ci95_lo),
        "ci95_high": float(ci95_hi),
        "p_two_sided": p_two,
        "p_tost": p_tost,
        "sesoi": sesoi,
    }


def _linslope(x: np.ndarray, y: np.ndarray):
    if _st is None or x.size < 3 or np.allclose(x, x[0]):
        return {
            "slope": None,
            "intercept": float(np.mean(y)) if y.size else 0.0,
            "p_value": None,
            "r2": None,
            "n": int(x.size),
            "estimable": False,
            "not_estimable_reason": (
                "fewer than three points or no variation in the moderator"
            ),
        }
    if np.allclose(y, y[0]):
        # scipy.stats.linregress returns NaN r/p values for a constant
        # response.  The descriptive slope is nevertheless exactly zero; emit
        # a finite record so strict JSON cannot silently contain NaN.
        return {
            "slope": 0.0,
            "intercept": float(y[0]),
            "p_value": 1.0,
            "r2": 0.0,
            "n": int(x.size),
            "estimable": True,
        }
    res = _st.linregress(x, y)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "p_value": float(res.pvalue), "r2": float(res.rvalue ** 2),
            "n": int(x.size), "estimable": True}


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
        # One coupling-free point per scenario: the x and y estimates use
        # disjoint seeds.  Repeating x_scn once per half-2 seed would falsely
        # turn four scenario groups into ~40 independent regression rows.
        xs.append(float(np.mean(fo[h1] - nc[h1])))
        ys.append(float(np.mean(a[h2] - fo[h2])))
        # Coupled comparator, also reduced to one point per scenario.
        xs_naive.append(float(np.mean(fo - nc)))
        ys_naive.append(float(np.mean(a - fo)))

    def _descriptive_fit(x, y):
        fit = _linslope(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        # Four scenario points cannot support a credible generalizable
        # moderation test.  Preserve slope/r2 but suppress the nominal OLS
        # p-value and label the estimand explicitly.
        fit["p_value"] = None
        fit["inferential"] = False
        fit["unit"] = "scenario"
        fit["interpretation_limit"] = (
            "descriptive across four perturbed scenarios; no inferential "
            "moderation claim"
        )
        return fit
    return {
        "crossfit": _descriptive_fit(xs, ys),
        # Keep the historical key for artifact-schema compatibility.  Its
        # interpretation is a coupling-prone comparator, not a mathematical
        # upper or lower bound.
        "naive_coupled_bound": _descriptive_fit(xs_naive, ys_naive),
        "interpretation": ("slope ~ -1 substitution/strong saturation; "
                           "slope ~ 0 independent additivity; "
                           "crossfit breaks the shared-term coupling, "
                           "naive is a coupling-prone descriptive comparator, "
                           "not a formal bound"),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seed-root", type=Path, required=True,
        help="Exact flat manifested benchmark_seeds directory.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--source-commit",
        default=os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip(),
    )
    ap.add_argument(
        "--run-tag", default=os.environ.get("RUN_TAG", "").strip(),
    )
    args = ap.parse_args(argv)

    if _st is None:
        raise SystemExit("scipy is required for channel_saturation_analysis")

    seed_root = _validate_flat_seed_root(args.seed_root)
    source_commit = str(args.source_commit).strip() or _git_commit()
    run_tag = str(args.run_tag).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("channel saturation requires a full source commit")
    if not run_tag:
        raise RuntimeError("channel saturation requires a run tag")
    head = _git_commit()
    if head != "unknown" and head != source_commit:
        raise RuntimeError("channel saturation source commit differs from Git HEAD")
    ari, seeds = _load_ari(seed_root)

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

    def _pool_by_seed(diff_fn):
        """Average scenario differences within seed before pooled inference."""
        matrix = np.vstack([diff_fn(ari[s]) for s in PERTURBED])
        if matrix.shape != (len(PERTURBED), len(seeds)):
            raise RuntimeError(f"unexpected pooled paired panel shape {matrix.shape}")
        return np.mean(matrix, axis=0)

    pooled = {
        "scenarios": list(PERTURBED),
        "inferential_unit": "seed",
        "scenario_aggregation": "mean paired difference across four scenarios within seed",
        "add_pirag_on_mcp": _paired_tost(
            _pool_by_seed(lambda c: c["agribrain"] - c["mcp_only"])),
        "add_mcp_on_pirag": _paired_tost(
            _pool_by_seed(lambda c: c["agribrain"] - c["pirag_only"])),
    }

    moderation = {
        "pirag_marginal_vs_mcp_strength": _crossfit_moderation(
            ari, PERTURBED, first="mcp", second="pirag"),
        "mcp_marginal_vs_pirag_strength": _crossfit_moderation(
            ari, PERTURBED, first="pirag", second="mcp"),
    }

    out = {
        "_meta": {
            "git_commit": source_commit,
            "benchmark_run": run_tag,
            "n_seeds": len(seeds),
            "sesoi_ari": SESOI,
            "alpha": ALPHA,
            "split_rng_seed": _SPLIT_RNG_SEED,
            "source": "benchmark_seeds/seed_*.json (ari field)",
            "seed_order": seeds,
            "multiplicity": (
                "scenario-specific H2 TOST and two-sided diagnostics are "
                "exploratory and unadjusted; pooled estimates average the "
                "four perturbed scenarios within seed before inference"
            ),
            "method": ("TOST equivalence on paired add-second-channel diffs "
                       "(SESOI=0.01 ARI) with seed-level scenario aggregation "
                       "+ descriptive four-scenario cross-fitted moderation slope"),
        },
        "by_scenario": by_scn,
        "pooled_perturbed": pooled,
        "moderation": moderation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(f"Saved: {args.output}  (run {run_tag}, {len(seeds)} seeds)")

    # Console summary.
    print("\nPer-scenario 'add the second channel' verdicts (TOST, SESOI=0.01):")
    for scn, c in by_scn.items():
        print(f"  {scn:16s} +piRAG/MCP: {c['add_pirag_on_mcp']['verdict']:13s}"
              f" (Δ={c['add_pirag_on_mcp']['mean_diff']:+.4f})"
              f"   +MCP/piRAG: {c['add_mcp_on_pirag']['verdict']:13s}"
              f" (Δ={c['add_mcp_on_pirag']['mean_diff']:+.4f})")
    mo = moderation["pirag_marginal_vs_mcp_strength"]
    def _slope_text(value):
        return "not estimable" if value is None else f"{float(value):+.3f}"
    print(f"\nDescriptive four-scenario cross-fit moderation "
          f"(piRAG marginal vs MCP strength): "
          f"slope={_slope_text(mo['crossfit']['slope'])} "
          f"(naive coupled comparator "
          f"{_slope_text(mo['naive_coupled_bound']['slope'])}; "
          "no inferential p-value)")


if __name__ == "__main__":
    main()
