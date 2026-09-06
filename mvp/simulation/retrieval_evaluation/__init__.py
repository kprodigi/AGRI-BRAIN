"""Independent human evaluation of retrieval-ranking quality.

This package is intentionally separate from the stochastic benchmark and its
downstream Adaptive Resilience Index (ARI) analyses.
"""

from .validate_retrieval_evaluation import (
    EvaluationValidationError,
    ValidatedEvaluationBundle,
    validate_bundle,
)

__all__ = [
    "EvaluationValidationError",
    "ValidatedEvaluationBundle",
    "validate_bundle",
]
