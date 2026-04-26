"""Circular reverse logistics module for waste recovery pathway evaluation.

Scores composting, animal feed, and food bank pathways based on current
spoilage risk, inventory levels, and temperature conditions. Computes
a circular economy score for Recovery and LocalRedistribute actions.

References
----------
    - Ellen MacArthur Foundation (2019). Cities and Circular Economy
      for Food.
    - Papargyropoulou, E. et al. (2014). The food waste hierarchy as
      a framework for the management of food surplus and food waste.
      Journal of Cleaner Production, 76, 106-115.
"""
from __future__ import annotations

from typing import Any, Dict


def evaluate_recovery_options(
    spoilage_risk: float,
    inventory: float,
    temperature: float,
) -> Dict[str, Any]:
    """Score recovery pathway options based on current conditions.

    Parameters
    ----------
    spoilage_risk : current rho value (0-1).
    inventory : current inventory units.
    temperature : current temperature in Celsius.

    Returns
    -------
    Dict with scores for each recovery pathway (0-1, higher = more suitable)
    and the recommended pathway.
    """
    # Food bank: viable when spoilage is low-moderate and quality sufficient
    food_bank = max(0.0, 1.0 - spoilage_risk * 2.0)
    if temperature > 10.0:
        food_bank *= 0.5  # reduced if cold chain compromised

    # Animal feed: viable across wider spoilage range
    animal_feed = min(1.0, 0.3 + spoilage_risk * 0.8)
    if inventory > 15000:
        animal_feed = min(1.0, animal_feed * 1.2)  # more feed from surplus

    # Composting: always viable, preferred at high spoilage
    composting = min(1.0, 0.2 + spoilage_risk * 1.0)

    # Normalize scores
    total = food_bank + animal_feed + composting
    if total > 0:
        food_bank /= total
        animal_feed /= total
        composting /= total

    # Select recommended pathway
    scores = {"food_bank": food_bank, "animal_feed": animal_feed, "composting": composting}
    recommended = max(scores, key=scores.get)

    return {
        "food_bank": round(food_bank, 4),
        "animal_feed": round(animal_feed, 4),
        "composting": round(composting, 4),
        "recommended": recommended,
        "conditions": {
            "spoilage_risk": spoilage_risk,
            "inventory": inventory,
            "temperature": temperature,
        },
    }


def compute_circular_economy_score(
    action: str,
    recovery_options: Dict[str, Any],
) -> float:
    """Compute a circular economy score for the taken action.

    Parameters
    ----------
    action : routing action name (cold_chain, local_redistribute, recovery).
    recovery_options : output from evaluate_recovery_options().

    Returns
    -------
    Score in [0, 1] where 0 = standard cold chain (linear) and
    1.0 = optimal circular recovery.
    """
    if action == "cold_chain":
        # Cold chain is the linear economy baseline
        return 0.0

    if action == "local_redistribute":
        # Local redistribution contributes to circularity through
        # surplus redistribution and community food access
        fb = recovery_options.get("food_bank", 0.0)
        return round(min(1.0, 0.4 + fb * 0.6), 4)

    if action == "recovery":
        # Full recovery: score based on best available pathway
        best = max(
            recovery_options.get("food_bank", 0.0),
            recovery_options.get("animal_feed", 0.0),
            recovery_options.get("composting", 0.0),
        )
        return round(min(1.0, 0.5 + best * 0.5), 4)

    return 0.0
