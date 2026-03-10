"""SLCA weight lookup tool for the MCP server.

Returns Social Life-Cycle Assessment weights and base scores for
different product types and routing actions.
"""
from __future__ import annotations

from typing import Any, Dict


_SLCA_CONFIG = {
    "spinach": {
        "weights": {"w_c": 0.30, "w_l": 0.20, "w_r": 0.25, "w_p": 0.25},
        "base_scores": {
            "cold_chain":         {"L": 0.50, "R": 0.40, "P": 0.45},
            "local_redistribute": {"L": 0.92, "R": 0.88, "P": 0.85},
            "recovery":           {"L": 0.72, "R": 0.75, "P": 0.70},
        },
        "carbon_cap": 50.0,
    },
    "lettuce": {
        "weights": {"w_c": 0.30, "w_l": 0.20, "w_r": 0.25, "w_p": 0.25},
        "base_scores": {
            "cold_chain":         {"L": 0.48, "R": 0.38, "P": 0.42},
            "local_redistribute": {"L": 0.90, "R": 0.85, "P": 0.82},
            "recovery":           {"L": 0.70, "R": 0.72, "P": 0.68},
        },
        "carbon_cap": 50.0,
    },
    "default": {
        "weights": {"w_c": 0.25, "w_l": 0.25, "w_r": 0.25, "w_p": 0.25},
        "base_scores": {
            "cold_chain":         {"L": 0.50, "R": 0.40, "P": 0.45},
            "local_redistribute": {"L": 0.85, "R": 0.80, "P": 0.80},
            "recovery":           {"L": 0.70, "R": 0.70, "P": 0.65},
        },
        "carbon_cap": 50.0,
    },
}


def lookup_slca_weights(product_type: str = "spinach") -> Dict[str, Any]:
    """Look up SLCA weights and base scores for a product type.

    Parameters
    ----------
    product_type : produce type (spinach, lettuce, default).

    Returns
    -------
    Dict with weights, base_scores per action, and carbon_cap.
    """
    config = _SLCA_CONFIG.get(product_type.lower(), _SLCA_CONFIG["default"])
    return {
        "product_type": product_type,
        "weights": config["weights"],
        "base_scores": config["base_scores"],
        "carbon_cap": config["carbon_cap"],
    }
