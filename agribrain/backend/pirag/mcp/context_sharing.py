"""Inter-agent context store for MCP result sharing.

Episode-scoped shared context that allows agents to publish tool results
and downstream agents to query them. This avoids redundant tool invocations
(e.g., processor skips compliance check when farm already ran it) and
enables context propagation across lifecycle stages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class SharedContextStore:
    """Episode-scoped shared context. Agents publish; downstream agents query."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def publish(
        self,
        role: str,
        tool_name: str,
        result: Any,
        hour: float,
    ) -> None:
        """Publish a tool result to shared context."""
        self._entries.append({
            "role": role,
            "tool_name": tool_name,
            "result": result,
            "hour": hour,
        })

    def query(
        self,
        role: Optional[str] = None,
        tool_name: Optional[str] = None,
        max_age_hours: float = 4.0,
        current_hour: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Query published results with optional filters.

        Parameters
        ----------
        role : filter by publishing agent role.
        tool_name : filter by tool name.
        max_age_hours : maximum age of results in hours.
        current_hour : current simulation hour for age filtering.

        Returns
        -------
        List of matching entries (newest first).
        """
        results = []
        for entry in reversed(self._entries):
            age = current_hour - entry["hour"]
            if age > max_age_hours:
                continue
            if role is not None and entry["role"] != role:
                continue
            if tool_name is not None and entry["tool_name"] != tool_name:
                continue
            results.append(entry)
        return results

    def get_upstream_compliance(self, current_hour: float) -> Optional[Dict[str, Any]]:
        """Get the most recent compliance check from an upstream agent.

        Specifically looks for farm-published compliance results within
        the last 4 hours.
        """
        entries = self.query(
            role="farm", tool_name="check_compliance",
            max_age_hours=4.0, current_hour=current_hour,
        )
        if entries:
            return entries[0]["result"]
        return None

    def get_upstream_spoilage_forecast(self, current_hour: float) -> Optional[Dict[str, Any]]:
        """Get the most recent spoilage forecast from any upstream agent."""
        entries = self.query(
            tool_name="spoilage_forecast",
            max_age_hours=4.0, current_hour=current_hour,
        )
        if entries:
            return entries[0]["result"]
        return None

    def reset(self) -> None:
        """Clear all entries (call between episodes)."""
        self._entries.clear()

    @property
    def summary(self) -> Dict[str, int]:
        """Per-role entry counts."""
        counts: Dict[str, int] = {}
        for entry in self._entries:
            role = entry["role"]
            counts[role] = counts.get(role, 0) + 1
        return counts
