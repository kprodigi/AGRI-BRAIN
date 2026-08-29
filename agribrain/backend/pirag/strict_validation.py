"""Fail-closed handling for publication-critical piRAG operations."""
from __future__ import annotations

import logging
import os


def strict_validation_enabled() -> bool:
    """Return whether the canonical fail-closed validation contract is active."""
    return os.environ.get("STRICT_VALIDATION", "0") == "1"


def handle_unexpected_failure(
    component: str,
    exc: BaseException,
    logger: logging.Logger,
) -> None:
    """Raise in strict mode; otherwise warn and allow a declared fallback.

    This helper is for unexpected execution failures, not for ordinary
    retrieval/units/feasibility guard rejections. Guards return explicit
    booleans and remain valid policy outcomes.
    """
    message = f"publication-critical {component} failed: {exc}"
    if strict_validation_enabled():
        raise RuntimeError(message) from exc
    logger.warning("%s; using non-strict fallback", message)
