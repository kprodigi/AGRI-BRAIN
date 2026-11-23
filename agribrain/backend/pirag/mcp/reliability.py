"""Reliability helpers for MCP tool invocation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float = 0.0
    is_open: bool = False


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_after_s: float = 5.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._state: Dict[str, CircuitState] = {}

    def allow(self, key: str) -> bool:
        st = self._state.setdefault(key, CircuitState())
        if not st.is_open:
            return True
        if (time.time() - st.opened_at) >= self.reset_after_s:
            st.is_open = False
            st.failures = 0
            return True
        return False

    def on_success(self, key: str) -> None:
        st = self._state.setdefault(key, CircuitState())
        st.failures = 0
        st.is_open = False

    def on_failure(self, key: str) -> None:
        st = self._state.setdefault(key, CircuitState())
        st.failures += 1
        if st.failures >= self.failure_threshold:
            st.is_open = True
            st.opened_at = time.time()


def invoke_with_retry(fn: Callable[[], Any], retries: int = 1, backoff_s: float = 0.05) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_s * (2 ** attempt))
    if last_error is not None:
        raise last_error
    return None


def invoke_with_fallback(primary: Callable[[], Any], fallbacks: Iterable[Callable[[], Any]]) -> Any:
    try:
        return primary()
    except Exception:
        for fb in fallbacks:
            try:
                return fb()
            except Exception:
                continue
    return None


def quorum_success(results: Iterable[Any], min_success: int = 1) -> bool:
    ok = sum(1 for r in results if r is not None)
    return ok >= min_success

