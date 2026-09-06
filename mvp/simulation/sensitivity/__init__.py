"""Seed-locked structural sensitivity support for the publication benchmark.

This package deliberately uses the term *structural sensitivity*.  Its Latin-
hypercube coordinates are a space-filling design over declared deterministic
bounds; they are not draws from calibrated probability distributions.
"""

from .design import build_design, build_task_manifest
from .parameters import PARAMETERS, validate_parameter_registry

__all__ = [
    "PARAMETERS",
    "build_design",
    "build_task_manifest",
    "validate_parameter_registry",
]
