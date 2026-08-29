"""Pure statistical primitives for the prespecified H2 and H3 protocols."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


H1_PRACTICAL_MARGIN = 0.005
H2_DIRECTIONAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("mcp_only", "no_context"),
    ("pirag_only", "no_context"),
    ("agribrain", "mcp_only"),
    ("agribrain", "pirag_only"),
)


def h2_synergy_interaction(
    full: Iterable[float],
    mcp_only: Iterable[float],
    retrieval_only: Iterable[float],
    no_external_context: Iterable[float],
) -> np.ndarray:
    """Return paired superadditivity values ``Full - MCP - Retrieval + None``.

    A positive value is a stricter construct than Full merely exceeding both
    single-channel arms.  Inputs must use the same seed order.
    """

    arrays = tuple(
        np.asarray(list(values), dtype=float)
        for values in (full, mcp_only, retrieval_only, no_external_context)
    )
    if not arrays[0].size or any(arr.shape != arrays[0].shape for arr in arrays):
        raise ValueError("H2 interaction inputs must be non-empty and paired")
    if any(not np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("H2 interaction inputs must be finite")
    return arrays[0] - arrays[1] - arrays[2] + arrays[3]


def equivalence_tost(
    values: Iterable[float], margin: float,
) -> dict[str, float | bool | int]:
    """One-sample TOST for a mean paired difference within ``±margin``.

    The 90% two-sided interval is the pair of one-sided 95% confidence bounds
    used by TOST.  ``max_abs_one_sided_95_bound`` makes the strict margin check
    directly inspectable, including for point estimates close to the margin.
    """

    from scipy.stats import t as student_t

    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("TOST margin must be finite and positive")
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 2 or not np.all(np.isfinite(arr)):
        raise ValueError("TOST requires at least two finite seed-level values")
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / float(np.sqrt(len(arr)))
    df = len(arr) - 1
    if se == 0.0:
        equivalent = abs(mean) < margin
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        ci90 = (mean, mean)
        ci95 = (mean, mean)
    else:
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = float(1.0 - student_t.cdf(t_lower, df))
        p_upper = float(student_t.cdf(t_upper, df))
        crit90 = float(student_t.ppf(0.95, df))
        crit95 = float(student_t.ppf(0.975, df))
        ci90 = (mean - crit90 * se, mean + crit90 * se)
        ci95 = (mean - crit95 * se, mean + crit95 * se)
        equivalent = max(p_lower, p_upper) < 0.05

    one_sided_95_lower = float(ci90[0])
    one_sided_95_upper = float(ci90[1])
    max_abs_one_sided_bound = float(max(
        -one_sided_95_lower,
        one_sided_95_upper,
    ))
    bound_below_margin = bool(max_abs_one_sided_bound < margin)
    if bound_below_margin != bool(equivalent):
        raise RuntimeError(
            "TOST result contradicts its one-sided 95% confidence bounds"
        )
    result: dict[str, Any] = {
        "n": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "se": se,
        "margin": float(margin),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": float(max(p_lower, p_upper)),
        "ci90_low": float(ci90[0]),
        "ci90_high": float(ci90[1]),
        "ci95_low": float(ci95[0]),
        "ci95_high": float(ci95[1]),
        "one_sided_95_lower_bound": one_sided_95_lower,
        "one_sided_95_upper_bound": one_sided_95_upper,
        "max_abs_one_sided_95_bound": max_abs_one_sided_bound,
        "one_sided_95_bound_below_margin": bound_below_margin,
        "margin_clearance": float(margin - max_abs_one_sided_bound),
        "equivalent_alpha_0p05": bool(equivalent),
    }
    return result
