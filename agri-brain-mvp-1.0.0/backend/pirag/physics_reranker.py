"""Physics-informed retrieval: query expansion and passage re-ranking.

Two mechanisms make piRAG genuinely "physics-informed":

1. **Physics query expansion**: appends domain-specific terms to the base
   query based on current physical conditions (temperature, spoilage, decay
   rate). This steers retrieval toward physically relevant passages.

2. **Physics plausibility re-ranking**: adjusts passage scores based on
   physical relevance (temperature proximity, spoilage-stage matching,
   urgency). The physics bonus is clamped to [0, 0.30] and added to the
   text retrieval score.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query_with_physics(
    base_query: str,
    rho: float,
    temperature: float,
    k_eff: float = 0.0,
) -> str:
    """Expand a piRAG query with physics-informed terms.

    Parameters
    ----------
    base_query : original query string.
    rho : current spoilage risk.
    temperature : current temperature in Celsius.
    k_eff : effective decay rate (from Arrhenius model).

    Returns
    -------
    Expanded query string with appended physics terms.
    """
    expansions: List[str] = []

    if temperature > 10.0:
        expansions.append("accelerated thermal degradation")
    if rho > 0.50:
        expansions.append("advanced spoilage requiring immediate diversion")
    elif rho > 0.15:
        expansions.append("early quality decline during lag phase transition")
    if k_eff > 0.005:
        expansions.append("exponential decay acceleration beyond reference rate")

    if not expansions:
        return base_query

    return base_query + " " + " ".join(expansions)


# ---------------------------------------------------------------------------
# Physics re-ranking
# ---------------------------------------------------------------------------

_TEMP_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:degrees?\s*(?:Celsius|C)|°C)", re.IGNORECASE)
_SPOILAGE_KEYWORDS = {"spoilage", "decay", "degradation", "deterioration", "rot", "decomposition"}
_FRESHNESS_KEYWORDS = {"fresh", "preservation", "storage", "maintain", "shelf life"}
_URGENCY_KEYWORDS = {"emergency", "urgent", "critical", "immediate", "rapid", "time-critical"}


def _extract_temperatures(text: str) -> List[float]:
    """Extract temperature values mentioned in a passage."""
    return [float(m.group(1)) for m in _TEMP_PATTERN.finditer(text)]


def _keyword_density(text: str, keywords: set) -> float:
    """Fraction of keywords found in the text."""
    words = set(text.lower().split())
    return len(words & keywords) / max(len(keywords), 1)


def physics_rerank(
    passages: List[Dict[str, Any]],
    temperature: float,
    rho: float,
    humidity: float,
) -> List[Dict[str, Any]]:
    """Re-rank passages using physics plausibility scoring.

    Parameters
    ----------
    passages : list of dicts with "text", "score", "id", "meta" keys.
    temperature : current ambient temperature in Celsius.
    rho : current spoilage risk.
    humidity : current relative humidity in percent.

    Returns
    -------
    Passages sorted by (original_score + physics_bonus) descending.
    """
    scored = []
    for passage in passages:
        text = passage.get("text", "")
        base_score = passage.get("score", 0.0)
        physics_bonus = 0.0

        # Temperature proximity bonus
        mentioned_temps = _extract_temperatures(text)
        if mentioned_temps:
            min_diff = min(abs(t - temperature) for t in mentioned_temps)
            temp_bonus = max(0.0, 0.10 - min_diff * 0.005)
            physics_bonus += temp_bonus

        # Spoilage-stage matching
        if rho > 0.30:
            physics_bonus += _keyword_density(text, _SPOILAGE_KEYWORDS) * 0.15
        else:
            physics_bonus += _keyword_density(text, _FRESHNESS_KEYWORDS) * 0.10

        # Urgency bonus for high spoilage
        if rho > 0.40:
            physics_bonus += _keyword_density(text, _URGENCY_KEYWORDS) * 0.10

        # Clamp physics bonus to [0, 0.30]
        physics_bonus = min(max(physics_bonus, 0.0), 0.30)

        scored.append({
            **passage,
            "score": base_score + physics_bonus,
            "physics_bonus": round(physics_bonus, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
