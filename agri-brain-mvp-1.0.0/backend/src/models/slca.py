"""
Full 4-component Social Life-Cycle Assessment (SLCA) scorer.

Components
----------
C  - Carbon reduction          : C = max(0, 1 - carbon_kg / 50)
L  - Labour fairness           : per-action base score
R  - Community resilience      : per-action base score
P  - Price transparency        : per-action base score

Per-action base scores (L, R, P):
    ColdChain          : L=0.50, R=0.40, P=0.45
    LocalRedistribute  : L=0.92, R=0.88, P=0.85
    Recovery           : L=0.72, R=0.75, P=0.70

Composite:
    S = w_c*C + w_l*L + w_r*R + w_p*P
with default weights  w_c=0.30, w_l=0.20, w_r=0.25, w_p=0.25.
"""
from __future__ import annotations

from typing import Dict, Optional


# Per-action base scores keyed by canonical action family
_ACTION_BASES: Dict[str, Dict[str, float]] = {
    "coldchain":          {"L": 0.50, "R": 0.40, "P": 0.45},
    "local_redistribute": {"L": 0.92, "R": 0.88, "P": 0.85},
    "recovery":           {"L": 0.72, "R": 0.75, "P": 0.70},
}

# Alias mapping so various action strings resolve correctly
_ACTION_ALIASES: Dict[str, str] = {
    "standard_cold_chain":    "coldchain",
    "cold_chain":             "coldchain",
    "expedite_to_retail":     "coldchain",
    "reroute_to_near_dc":     "coldchain",
    "local_redistribution":   "local_redistribute",
    "localredistribute":      "local_redistribute",
    "redistribute_or_recover":"local_redistribute",
    "price_adjusted_route":   "local_redistribute",
    "recovery":               "recovery",
    "recover":                "recovery",
}


def _resolve_action(action: str) -> str:
    """Map an action string to a canonical base-score key."""
    key = action.strip().lower().replace(" ", "_")
    if key in _ACTION_BASES:
        return key
    return _ACTION_ALIASES.get(key, "coldchain")


def slca_score(
    carbon_kg: float,
    action: str = "coldchain",
    *,
    w_c: float = 0.30,
    w_l: float = 0.20,
    w_r: float = 0.25,
    w_p: float = 0.25,
    carbon_cap: float = 50.0,
    fairness: Optional[float] = None,
    resilience: Optional[float] = None,
    transparency: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the full 4-component SLCA score.

    Parameters
    ----------
    carbon_kg : total carbon footprint for the action in kg CO2-eq.
    action : routing decision string (resolved via alias table).
    w_c, w_l, w_r, w_p : component weights (should sum to 1).
    carbon_cap : denominator for carbon normalisation (default 50 kg).
    fairness, resilience, transparency :
        Optional overrides for L, R, P (use per-action defaults when None).

    Returns
    -------
    dict with keys ``C``, ``L``, ``R``, ``P``, ``composite``, ``action_family``.
    """
    family = _resolve_action(action)
    bases = _ACTION_BASES[family]

    C = max(0.0, 1.0 - carbon_kg / carbon_cap)
    L = fairness if fairness is not None else bases["L"]
    R = resilience if resilience is not None else bases["R"]
    P = transparency if transparency is not None else bases["P"]

    composite = w_c * C + w_l * L + w_r * R + w_p * P
    composite = float(max(0.0, min(1.0, composite)))

    return {
        "C": round(C, 4),
        "L": round(L, 4),
        "R": round(R, 4),
        "P": round(P, 4),
        "composite": round(composite, 4),
        "action_family": family,
    }
