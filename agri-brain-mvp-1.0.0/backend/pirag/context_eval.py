"""Context quality evaluator.

Tracks whether context injection changed routing decisions and whether
those changes improved outcomes. Provides summary statistics for paper
reporting.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ContextEvaluator:
    """Track the impact of context injection on routing decisions."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        hour: float,
        role: str,
        action_without: int,
        action_with: int,
        reward: float,
        modifier: np.ndarray,
    ) -> None:
        """Record a decision step for context quality evaluation.

        Parameters
        ----------
        hour : simulation hour.
        role : active agent role.
        action_without : action that would have been taken without context.
        action_with : action actually taken with context.
        reward : reward received for the action_with decision.
        modifier : context modifier vector applied.
        """
        self._records.append({
            "hour": hour,
            "role": role,
            "action_without": action_without,
            "action_with": action_with,
            "changed": action_without != action_with,
            "reward": reward,
            "modifier_magnitude": float(np.linalg.norm(modifier)),
        })

    def summary(self) -> Dict[str, Any]:
        """Summary statistics for paper reporting.

        Returns
        -------
        Dict with total_steps, context_changed_action_count,
        context_change_rate, mean_modifier_magnitude,
        mean_reward_when_changed, mean_reward_when_unchanged.
        """
        total = len(self._records)
        if total == 0:
            return {
                "total_steps": 0,
                "context_changed_action_count": 0,
                "context_change_rate": 0.0,
                "mean_modifier_magnitude": 0.0,
                "mean_reward_when_changed": 0.0,
                "mean_reward_when_unchanged": 0.0,
            }

        changed = [r for r in self._records if r["changed"]]
        unchanged = [r for r in self._records if not r["changed"]]

        return {
            "total_steps": total,
            "context_changed_action_count": len(changed),
            "context_change_rate": len(changed) / total,
            "mean_modifier_magnitude": float(np.mean([r["modifier_magnitude"] for r in self._records])),
            "mean_reward_when_changed": float(np.mean([r["reward"] for r in changed])) if changed else 0.0,
            "mean_reward_when_unchanged": float(np.mean([r["reward"] for r in unchanged])) if unchanged else 0.0,
        }

    def reset(self) -> None:
        """Clear all records."""
        self._records.clear()
