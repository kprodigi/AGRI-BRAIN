"""Lexical + physics retrieval: query expansion and passage re-ranking.

This module reranks retrieval hits using a mix of lexical evidence
(temperature numbers extracted by regex, keyword sets for spoilage /
freshness / urgency) and a single genuine-physics component
(Arrhenius-consistency: agreement between the temperature mentioned in
the passage and the current ``k_eff`` derived from the simulator's
upstream Arrhenius rate). Honest framing: most of the bonus is
keyword-density based and would not survive a vocabulary swap; only
the Arrhenius-consistency term has a thermodynamic basis.

Two mechanisms:

1. **Query expansion**: appends physics-informed terms (temperature
   regime, spoilage stage, decay rate) to the base query so BM25/TF-IDF
   has more discriminative ground.

2. **Lexical + Arrhenius reranking**: applies a per-passage bonus that
   is the sum of (i) lexical-density terms and (ii) an
   Arrhenius-consistency term that compares the passage's temperature
   mention to the simulator's ``k_eff`` and penalises mismatches in the
   wrong direction (passage talks about cold storage when k_eff is
   high, or vice versa). The combined bonus is clamped to [0, 0.30]
   and added to the text retrieval score.

Earlier revisions of this docstring described the rerank as
"physics plausibility scoring" without distinguishing the lexical
density from the Arrhenius term, which overstated the thermodynamic
content. The function is renamed in the public API as
``lexical_arrhenius_rerank`` and ``physics_rerank`` is retained as a
deprecated alias.
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


def lexical_arrhenius_rerank(
    passages: List[Dict[str, Any]],
    temperature: float,
    rho: float,
    humidity: float,
    k_eff: float = 0.0,
) -> List[Dict[str, Any]]:
    """Re-rank passages using a lexical+Arrhenius bonus.

    The bonus has two clearly separated components:

    - **Lexical density terms** (most of the bonus mass): temperature
      regex match proximity, spoilage / freshness / urgency keyword
      density, humidity mention. These would not survive a vocabulary
      swap to a different domain.
    - **Arrhenius-consistency term** (the only thermodynamic component):
      compares the temperature(s) mentioned in the passage to the
      current Arrhenius rate ``k_eff``. Passages whose mentioned
      temperature implies a vastly different decay rate are
      down-ranked. This is the only ranker component that uses real
      physics rather than keywords.

    Parameters
    ----------
    passages : list of dicts with "text", "score", "id", "meta" keys.
    temperature : current ambient temperature in Celsius.
    rho : current spoilage risk.
    humidity : current relative humidity in percent.
    k_eff : current Arrhenius effective rate (h^-1). When supplied,
        enables the Arrhenius-consistency term; when 0 (legacy callers),
        only the lexical terms are applied.

    Returns
    -------
    Passages sorted by (original_score + bonus) descending. Each
    record carries ``lexical_bonus`` and ``arrhenius_consistency``
    fields so downstream scoring can audit which component drove the
    rank change.
    """
    scored = []
    for passage in passages:
        text = passage.get("text", "")
        base_score = passage.get("score", 0.0)
        lexical_bonus = 0.0
        consistency = 1.0
        arrhenius_consistency = 1.0

        # Temperature proximity (lexical)
        mentioned_temps = _extract_temperatures(text)
        if mentioned_temps:
            min_diff = min(abs(t - temperature) for t in mentioned_temps)
            lexical_bonus += max(0.0, 0.10 - min_diff * 0.005)
            if min_diff > 12.0:
                consistency *= 0.65

        # Spoilage-stage / freshness keyword density (lexical)
        if rho > 0.30:
            lexical_bonus += _keyword_density(text, _SPOILAGE_KEYWORDS) * 0.15
        else:
            lexical_bonus += _keyword_density(text, _FRESHNESS_KEYWORDS) * 0.10

        # Urgency keyword density (lexical)
        if rho > 0.40:
            lexical_bonus += _keyword_density(text, _URGENCY_KEYWORDS) * 0.10
            if _keyword_density(text, _URGENCY_KEYWORDS) < 0.01:
                consistency *= 0.8

        # Humidity mention consistency (lexical)
        if humidity > 92 and "humidity" not in text.lower() and "relative humidity" not in text.lower():
            consistency *= 0.9

        # Arrhenius-consistency term (the physics component). The
        # passage's mentioned temperature implies a decay rate
        # k_passage = k_ref * exp(Ea_R * (1/T_ref - 1/T_pass)) (without
        # humidity coupling, for a clean term-by-term audit). We
        # compare it to the current k_eff from the simulator and
        # penalise the bonus when the magnitudes differ by more than
        # a factor of 2 (one order of magnitude in log space).
        if k_eff > 0.0 and mentioned_temps:
            import math as _math
            k_ref = 0.0021
            Ea_R = 8000.0
            T_ref_K = 277.15
            log_ratios = []
            for t_pass in mentioned_temps:
                T_K = t_pass + 273.15
                k_pass = k_ref * _math.exp(Ea_R * (1.0 / T_ref_K - 1.0 / T_K))
                if k_pass > 0:
                    log_ratios.append(abs(_math.log(max(k_eff, 1e-9) / max(k_pass, 1e-9))))
            if log_ratios:
                # log_ratio < ln(2) ~= 0.69 -> within factor 2 -> full bonus
                # log_ratio >= ln(10) ~= 2.30 -> off by 10x -> zero bonus
                min_log_ratio = min(log_ratios)
                arrhenius_consistency = max(0.0, 1.0 - max(0.0, min_log_ratio - 0.69) / 1.61)

        # Clamp the lexical bonus separately from the Arrhenius factor.
        lexical_bonus = min(max(lexical_bonus, 0.0), 0.30)
        consistency = min(max(consistency, 0.0), 1.0)
        arrhenius_consistency = min(max(arrhenius_consistency, 0.0), 1.0)
        # Arrhenius-consistency multiplies the lexical bonus so a
        # temperature-mismatched passage receives less of the keyword-
        # driven boost.
        bonus = lexical_bonus * arrhenius_consistency
        adjusted_score = (base_score + bonus) * consistency

        scored.append({
            **passage,
            "score": adjusted_score,
            "lexical_bonus": round(lexical_bonus, 4),
            "arrhenius_consistency": round(arrhenius_consistency, 4),
            # Backward-compatible aggregate field; some downstream
            # code reads physics_bonus directly.
            "physics_bonus": round(bonus, 4),
            "physics_consistency": round(consistency, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# Backward-compatible alias. Deprecated; use lexical_arrhenius_rerank.
def physics_rerank(
    passages: List[Dict[str, Any]],
    temperature: float,
    rho: float,
    humidity: float,
    k_eff: float = 0.0,
) -> List[Dict[str, Any]]:
    """Deprecated alias for :func:`lexical_arrhenius_rerank`."""
    return lexical_arrhenius_rerank(passages, temperature, rho, humidity, k_eff)
