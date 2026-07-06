"""SLCA weight lookup tool for the MCP server.

Returns Social Life-Cycle Assessment weights and base scores for
different product types and routing actions.

Implementation note: 2025-04 single-source-of-truth fix.
Prior to this revision the MCP tool kept its own copy of base scores
({L=0.50, R=0.40, P=0.45} for cold_chain etc.) which had drifted from
the simulator's `_ACTION_BASES` after the 2025-04 SLCA recalibration
(the simulator now uses {L=0.60, R=0.55, P=0.55}). Agents querying MCP
got values that disagreed with the values driving the actual policy
calculations. This module now imports `_ACTION_BASES` directly from
`src.models.slca` so there is exactly one source of truth for the
base scores; product-specific scaling factors are applied on top.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

# Single source of truth for base scores. The simulator's `slca_score`
# uses these directly; this MCP tool exposes them to agents.
from src.models.slca import _ACTION_BASES as _CANONICAL_ACTION_BASES


# Per-product *modifiers* applied multiplicatively to the canonical bases.
# A modifier of 1.0 means "use the canonical base"; lettuce sits slightly
# below spinach on labour fairness because of harsher harvest mechanics
# documented in Arcese et al. (2018). The default produce profile uses
# the canonical bases unmodified.
_PRODUCT_MODIFIERS: Dict[str, Dict[str, Dict[str, float]]] = {
    "spinach": {
        "cold_chain":         {"L": 1.00, "R": 1.00, "P": 1.00},
        "local_redistribute": {"L": 1.00, "R": 1.00, "P": 1.00},
        "recovery":           {"L": 1.00, "R": 1.00, "P": 1.00},
    },
    "lettuce": {
        "cold_chain":         {"L": 0.95, "R": 0.95, "P": 0.95},
        "local_redistribute": {"L": 0.97, "R": 0.97, "P": 0.97},
        "recovery":           {"L": 0.97, "R": 0.97, "P": 0.97},
    },
}

_DEFAULT_WEIGHTS = {"w_c": 0.30, "w_l": 0.20, "w_r": 0.25, "w_p": 0.25}
_DEFAULT_CARBON_CAP = 50.0


def _apply_modifier(base: Dict[str, float], modifier: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(base[k]) * float(modifier.get(k, 1.0)), 4)
            for k in base}


def lookup_slca_weights(product_type: str = "spinach") -> Dict[str, Any]:
    """Look up SLCA weights and base scores for a product type.

    Parameters
    ----------
    product_type : produce type (spinach, lettuce, ...). Unknown types
        fall through to the canonical (unmodified) base scores.

    Returns
    -------
    Dict with weights, base_scores per action, carbon_cap, and a
    ``_source`` field documenting the provenance of the base scores
    (always ``"src.models.slca._ACTION_BASES"``).
    """
    pt = (product_type or "spinach").lower()
    modifier = _PRODUCT_MODIFIERS.get(pt)
    canonical = deepcopy(_CANONICAL_ACTION_BASES)
    if modifier is None:
        scores = canonical
    else:
        scores = {action: _apply_modifier(canonical[action], modifier[action])
                  for action in canonical}
    return {
        "product_type": product_type,
        "weights": dict(_DEFAULT_WEIGHTS),
        "base_scores": scores,
        "carbon_cap": _DEFAULT_CARBON_CAP,
        "_source": "src.models.slca._ACTION_BASES",
    }
