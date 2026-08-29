"""Declared operating-envelope check for the MCP server.

The legacy tool name is ``check_compliance``, but this function is not a legal
or food-safety determination. It compares readings with author-specified
benchmark temperature/humidity envelopes.
"""
from __future__ import annotations

from typing import Any, Dict


# Synthetic operating envelopes per product type. Aligned with the
# dataset's legacy ``regulatory_temp_max`` column (8 degC for leafy greens) so
# the check uses the same benchmark cold-chain ceiling that
# ``temp_violation`` uses in generate_results.py. These values are not
# attributed to FDA, and the humidity limits are not legal
# limits. FDA's 5 degC retail guidance for cut leafy greens is preserved in the
# source-labelled knowledge-base note, not converted here into a universal law.
_BENCHMARK_ENVELOPES = {
    "spinach": {"temp_max_c": 8.0, "rh_min": 85.0, "rh_max": 95.0},
    "lettuce": {"temp_max_c": 8.0, "rh_min": 90.0, "rh_max": 98.0},
    "berries": {"temp_max_c": 4.0, "rh_min": 90.0, "rh_max": 95.0},
    "default": {"temp_max_c": 8.0, "rh_min": 80.0, "rh_max": 95.0},
}


def check_compliance(
    temperature: float,
    humidity: float,
    product_type: str = "spinach",
) -> Dict[str, Any]:
    """Check readings against the declared benchmark operating envelope.

    Parameters
    ----------
    temperature : current temperature in Celsius.
    humidity : current relative humidity in percent.
    product_type : produce type (spinach, lettuce, berries, default).

    Returns
    -------
    Dict with compliance status, violations list, and thresholds used.
    """
    limits = _BENCHMARK_ENVELOPES.get(
        product_type.lower(), _BENCHMARK_ENVELOPES["default"]
    )
    violations = []

    if temperature > limits["temp_max_c"]:
        violations.append({
            "parameter": "temperature",
            "value": temperature,
            "limit": limits["temp_max_c"],
            "severity": "critical" if temperature > limits["temp_max_c"] + 3 else "warning",
            "message": f"Temperature {temperature:.1f}C exceeds limit of {limits['temp_max_c']:.1f}C",
        })

    if humidity < limits["rh_min"]:
        violations.append({
            "parameter": "humidity_low",
            "value": humidity,
            "limit": limits["rh_min"],
            "severity": "warning",
            "message": f"Humidity {humidity:.1f}% below minimum {limits['rh_min']:.1f}%",
        })

    if humidity > limits["rh_max"]:
        violations.append({
            "parameter": "humidity_high",
            "value": humidity,
            "limit": limits["rh_max"],
            "severity": "warning",
            "message": f"Humidity {humidity:.1f}% above maximum {limits['rh_max']:.1f}%",
        })

    return {
        "compliant": len(violations) == 0,
        "assessment_type": "synthetic_benchmark_operating_envelope",
        "is_regulatory_determination": False,
        "product_type": product_type,
        "violations": violations,
        "thresholds": limits,
        "readings": {"temperature": temperature, "humidity": humidity},
    }
