"""
Shared action alias resolution for the AGRI-BRAIN routing decision system.

All model modules (slca.py, waste.py, resilience.py, action_selection.py)
import from this single source of truth to ensure consistent alias handling.

Canonical action families
-------------------------
    cold_chain          — standard cold-chain routing (farm -> DC -> retail)
    local_redistribute  — local redistribution to food banks / community markets
    recovery            — diversion to composting, bioenergy, or animal feed

Alias mapping covers the various string forms that appear in the system:
    - camelCase, snake_case, and concatenated forms
    - legacy endpoint names and simulation descriptors
    - synonyms (e.g. "recover" -> "recovery")
"""
from __future__ import annotations

from typing import Dict


CANONICAL_ACTIONS: list[str] = ["cold_chain", "local_redistribute", "recovery"]
"""The three canonical action family names used throughout the system."""

ACTION_ALIASES: Dict[str, str] = {
    # cold_chain variants
    "coldchain":              "cold_chain",
    "cold_chain":             "cold_chain",
    "standard_cold_chain":    "cold_chain",
    "expedite_to_retail":     "cold_chain",
    "reroute_to_near_dc":     "cold_chain",
    # local_redistribute variants
    "local_redistribute":     "local_redistribute",
    "local_redistribution":   "local_redistribute",
    "localredistribute":      "local_redistribute",
    "redistribute_or_recover":"local_redistribute",
    "price_adjusted_route":   "local_redistribute",
    # recovery variants
    "recovery":               "recovery",
    "recover":                "recovery",
}


def resolve_action(action: str) -> str:
    """Normalize an action string to its canonical family key.

    Parameters
    ----------
    action : raw action string from any caller.

    Returns
    -------
    One of ``cold_chain``, ``local_redistribute``, or ``recovery``.
    Falls back to ``cold_chain`` for unrecognised strings (safest default).
    """
    key = action.strip().lower().replace(" ", "_")
    if key in CANONICAL_ACTIONS:
        return key
    return ACTION_ALIASES.get(key, "cold_chain")
