"""Sliding context window for temporal piRAG retrieval history.

Maintains a bounded window of recent retrieval events, enabling temporal
awareness in context modulation. The continuity score measures whether
the same documents are being retrieved repeatedly (stable situation) or
different documents each time (volatile situation).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


class TemporalContextWindow:
    """Sliding window of recent piRAG retrievals.

    Parameters
    ----------
    max_entries : maximum entries retained.
    horizon_hours : maximum age of entries in hours.
    """

    def __init__(
        self,
        max_entries: int = 20,
        horizon_hours: float = 6.0,
    ) -> None:
        self._max_entries = max_entries
        self._horizon = horizon_hours
        self._entries: List[Dict[str, Any]] = []

    def add(
        self,
        hour: float,
        role: str,
        query: str,
        top_doc_id: str,
        top_score: float,
        guidance_type: str,
    ) -> None:
        """Record a piRAG retrieval event."""
        self._entries.append({
            "hour": hour,
            "role": role,
            "query": query,
            "top_doc_id": top_doc_id,
            "top_score": top_score,
            "guidance_type": guidance_type,
        })
        # Enforce max entries
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def _recent(self, current_hour: float) -> List[Dict[str, Any]]:
        """Return entries within the time horizon."""
        cutoff = current_hour - self._horizon
        return [e for e in self._entries if e["hour"] >= cutoff]

    def get_recent(self, current_hour: float, n: int = 5) -> List[Dict[str, Any]]:
        """Get the ``n`` most recent entries within the horizon."""
        recent = self._recent(current_hour)
        return recent[-n:]

    def get_role_history(self, role: str, current_hour: float) -> List[Dict[str, Any]]:
        """Get retrieval history for a specific role within the horizon."""
        return [e for e in self._recent(current_hour) if e["role"] == role]

    def dominant_guidance_type(self, current_hour: float) -> Optional[str]:
        """Most-retrieved guidance type recently.

        ``'emergency_rerouting_sop'`` dominant indicates crisis mode.
        Returns None if no recent entries.
        """
        recent = self._recent(current_hour)
        if not recent:
            return None
        counts = Counter(e["guidance_type"] for e in recent if e["guidance_type"])
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def context_continuity_score(self, current_hour: float) -> float:
        """Measure retrieval stability.

        Returns a score in [0, 1]:
        - 0.0 = random/diverse retrievals (volatile situation)
        - 1.0 = same document retrieved every time (stable situation)

        Used in Task 13 to modulate context modifier strength.
        """
        recent = self._recent(current_hour)
        if len(recent) < 2:
            return 0.5  # neutral default

        doc_ids = [e["top_doc_id"] for e in recent if e["top_doc_id"]]
        if not doc_ids:
            return 0.5

        counts = Counter(doc_ids)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(doc_ids)

    def recency_weighted_doc_scores(self, current_hour: float, half_life_hours: float = 2.0) -> Dict[str, float]:
        """Return recency-weighted score per top_doc_id."""
        recent = self._recent(current_hour)
        if not recent:
            return {}
        scores: Dict[str, float] = {}
        for e in recent:
            doc = e.get("top_doc_id", "")
            if not doc:
                continue
            age = max(0.0, current_hour - float(e["hour"]))
            weight = 0.5 ** (age / max(half_life_hours, 1e-6))
            scores[doc] = scores.get(doc, 0.0) + weight
        return scores

    def stale_context_ratio(self, current_hour: float, stale_after_hours: float = 3.0) -> float:
        recent = self._recent(current_hour)
        if not recent:
            return 0.0
        stale = sum(
            1 for e in recent if (current_hour - float(e.get("hour", current_hour))) > stale_after_hours
        )
        return stale / len(recent)

    def reset(self) -> None:
        """Clear all entries (call between episodes)."""
        self._entries.clear()
