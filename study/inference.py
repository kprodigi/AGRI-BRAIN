"""Interval estimates and tests used in the analysis of the study.

The procedure is the one used for the submitted study
(``analysis/primary_statistics.py``): the unit is the seed; every interval is a
BCa interval from 10,000 resamples of whole seeds, drawn with a generator
seeded from a BLAKE2b hash of the cell key; tests are exact Wilcoxon
signed-rank tests with Holm adjustment. The functions of that module are used
directly where they apply.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import primary_statistics as submitted  # noqa: E402

N_BOOT = submitted.N_BOOT
holm_adjust = submitted.holm_adjust
paired_interval = submitted.paired_interval


def mean_interval(values, key, alpha=0.05):
    """BCa interval of the mean of seed-level values: ``(low, high, method)``.

    With ``alpha=0.05`` this is ``primary_statistics.mean_interval``.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected a vector of at least two finite seed-level values')
    theta = float(np.mean(x))
    if float(np.std(x, ddof=1)) == 0.0:
        return theta, theta, 'deterministic'
    rng = np.random.default_rng(submitted.cell_seed('bootstrap_ci', key))
    boots = np.array([float(np.mean(rng.choice(x, len(x), replace=True)))
                      for _ in range(N_BOOT)])
    jacks = np.array([float(np.mean(np.delete(x, i))) for i in range(len(x))])
    return submitted.bca_interval(boots, theta, jacks, alpha)


def estimate(values, key, alpha=0.05):
    """Mean of seed-level values with its interval and sign counts."""
    x = np.asarray(values, dtype=float)
    low, high, method = mean_interval(x, key, alpha)
    return {'mean': float(x.mean()), 'low': low, 'high': high, 'ci_method': method,
            'confidence': 1.0 - alpha, 'n': int(len(x)), 'sd': float(x.std(ddof=1)),
            'positive_seeds': int((x > 0).sum()), 'negative_seeds': int((x < 0).sum()),
            'zero_seeds': int((x == 0).sum())}


def signed_rank(differences, alternative):
    """Exact Wilcoxon signed-rank p-value: ``(p, method)``.

    Without zeros or tied magnitudes this is the exact test of
    ``scipy.stats.wilcoxon``. Otherwise zeros are dropped, tied magnitudes
    share their mean rank, and the p-value is taken from the exact distribution
    of the rank sum over all sign assignments.
    """
    if alternative not in ('greater', 'less', 'two-sided'):
        raise ValueError('Unknown alternative')
    d = np.asarray(differences, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite difference')
    nonzero = d[d != 0.0]
    if len(nonzero) == 0:
        return 1.0, 'all differences zero'
    magnitudes = np.abs(nonzero)
    if len(nonzero) == len(d) and len(np.unique(magnitudes)) == len(magnitudes):
        return float(wilcoxon(d.tolist(), alternative=alternative, method='exact').pvalue), 'exact'
    doubled = np.rint(2.0 * rankdata(magnitudes)).astype(int)
    counts = np.zeros(int(doubled.sum()) + 1, dtype=float)
    counts[0] = 1.0
    for rank in doubled:
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:len(counts) - rank]
        counts = counts + shifted
    observed = int(doubled[nonzero > 0].sum())
    total = float(counts.sum())
    upper, lower = float(counts[observed:].sum()) / total, float(counts[:observed + 1].sum()) / total
    p = {'greater': upper, 'less': lower, 'two-sided': min(1.0, 2.0 * min(upper, lower))}[alternative]
    return p, 'exact, zeros dropped and ties ranked by their mean'
