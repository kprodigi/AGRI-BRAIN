"""MCP tool: read the current context feature vector and modifier.

Enables external monitoring systems to observe what information the
MCP/piRAG context layer is injecting into routing decisions.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Module-level cache updated by the coordinator each step
_cache: Dict[str, Any] = {}


def update_context_cache(
    features: Optional[np.ndarray] = None,
    modifier: Optional[np.ndarray] = None,
    explanation: Optional[Dict[str, Any]] = None,
    hour: float = 0.0,
    override: bool = False,
    robustness: Optional[Dict[str, Any]] = None,
) -> None:
    """Update the module-level cache with current context state."""
    _cache.clear()

    feature_names = [
        "compliance_severity", "forecast_urgency",
        "retrieval_confidence", "regulatory_pressure", "recovery_saturation",
        "supply_uncertainty",
    ]

    if features is not None:
        for i, name in enumerate(feature_names):
            _cache[f"psi_{i}"] = float(features[i])
        dominant_idx = int(np.argmax(np.abs(features)))
        _cache["dominant"] = feature_names[dominant_idx]
    else:
        _cache["dominant"] = "none"

    if modifier is not None:
        _cache["mod_0"] = float(modifier[0])
        _cache["mod_1"] = float(modifier[1])
        _cache["mod_2"] = float(modifier[2])
        _cache["mod_norm"] = float(np.linalg.norm(modifier))
    else:
        _cache["mod_norm"] = 0.0

    _cache["hour"] = hour
    _cache["override"] = override
    _cache["explanation"] = explanation or {}
    _cache["robustness"] = robustness or {}
    _cache["features"] = {
        name: _cache.get(f"psi_{i}", 0.0) for i, name in enumerate(feature_names)
    }
    _cache["modifier"] = {
        "cold_chain": _cache.get("mod_0", 0.0),
        "local_redistribute": _cache.get("mod_1", 0.0),
        "recovery": _cache.get("mod_2", 0.0),
    }


def read_context_features() -> Dict[str, Any]:
    """Read the most recently computed context feature vector.

    Returns the 6D feature vector, the 3D logit modifier, feature names,
    dominant feature, and modifier magnitude. The sixth entry is supply
    uncertainty from the Path B yield_query integration; suppressed when
    ``context_mode="no_yield"``.
    """
    return {
        "features": _cache.get("features", {}),
        "modifier": _cache.get("modifier", {}),
        "modifier_magnitude": _cache.get("mod_norm", 0.0),
        "dominant_feature": _cache.get("dominant", "none"),
        "governance_override_active": _cache.get("override", False),
        "timestamp_hour": _cache.get("hour", 0.0),
    }


def get_context_cache() -> Dict[str, Any]:
    """Return the full cache dict (used by MCP resources)."""
    return dict(_cache)
