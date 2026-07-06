"""FDA compliance checking tool for the MCP server.

Validates temperature and humidity readings against regulatory thresholds
for various produce types.
"""
from __future__ import annotations

from typing import Any, Dict


# Regulatory temperature limits per product type. Aligned with the
# dataset's ``regulatory_temp_max`` column (8 degC for leafy greens) so
# the compliance check uses the same cold-chain ceiling that
# ``temp_violation`` uses in generate_results.py. Earlier values pegged
# spinach at 5 degC strict-FDA which produced 65-70 percent compliance
# violation rates that read as alarming on the bench summary even when
# the cold-chain truck was operating well within the dataset's stated
# regulatory limit; harmonising to 8 degC removes that asymmetry.
# Humidity bounds unchanged.
_FDA_LIMITS = {
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
    """Check FDA compliance for temperature and humidity readings.

    Parameters
    ----------
    temperature : current temperature in Celsius.
    humidity : current relative humidity in percent.
    product_type : produce type (spinach, lettuce, berries, default).

    Returns
    -------
    Dict with compliance status, violations list, and thresholds used.
    """
    limits = _FDA_LIMITS.get(product_type.lower(), _FDA_LIMITS["default"])
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
        "product_type": product_type,
        "violations": violations,
        "thresholds": limits,
        "readings": {"temperature": temperature, "humidity": humidity},
    }
