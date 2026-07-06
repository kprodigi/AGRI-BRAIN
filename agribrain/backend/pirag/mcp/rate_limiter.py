"""Per-tool token-bucket rate limiter for the MCP layer.

Reads ``rate_limits`` from ``configs/policy.yaml``. Each entry is a
``"<tool_name>": "<count>/<period>"`` mapping where ``<period>`` is one
of ``s``, ``sec``, ``min``, or ``hour``. Tools without an entry are
unlimited (the registry already gates registration).

The 2026-05 hardening makes this advisory-only configuration *real*:
previously the policy.yaml file declared rate limits that nothing read.
Operators can now lift the limits by dialling them up in the config or
disable enforcement entirely with ``MCP_RATE_LIMITS=disabled``.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import yaml


_log = logging.getLogger(__name__)

_PERIOD_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "m": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "hour": 3600.0,
    "hr": 3600.0,
    "h": 3600.0,
    "hours": 3600.0,
}


_RATE_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*([a-zA-Z]+)\s*$")


def _parse_rate(raw: str) -> Optional[float]:
    """Return a fractional capacity ``count/period_s``, or None on parse error."""
    if not isinstance(raw, str):
        return None
    m = _RATE_PATTERN.match(raw)
    if m is None:
        return None
    count = int(m.group(1))
    period = m.group(2).lower()
    seconds = _PERIOD_TO_SECONDS.get(period)
    if seconds is None or seconds <= 0:
        return None
    return float(count) / float(seconds)


@dataclass
class _Bucket:
    capacity: float          # tokens per second (the configured rate)
    period_capacity: float   # full bucket size (count over the period)
    tokens: float
    last_refill: float

    def consume(self) -> bool:
        """Try to consume one token. Returns True on success."""
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_refill)
        # Refill at the configured rate, capped at the period capacity.
        self.tokens = min(self.period_capacity,
                          self.tokens + elapsed * self.capacity)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitExceeded(RuntimeError):
    """Raised when a tool invocation is rejected by the rate limiter."""


class RateLimiter:
    """Per-tool token bucket loaded from ``policy.yaml``.

    Construction is cheap; the limiter caches the policy file's mtime
    and reloads when it changes so that operators editing
    ``policy.yaml`` at runtime see the new caps without a restart.
    """

    def __init__(self, policy_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[str, _Bucket] = {}
        self._policy_mtime: float = 0.0
        if policy_path is None:
            policy_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "configs", "policy.yaml",
            ))
        self._path = policy_path
        # Default OFF for in-process registry calls (the simulator hot
        # path calls invoke() ~288 * 5 * 8 * N_seeds times -- saturating
        # 120/min would break the benchmark) and ON for the public-
        # facing MCP transport. Operators flip the default with
        # MCP_RATE_LIMITS=enabled (force-on everywhere) or
        # MCP_RATE_LIMITS=disabled (force-off everywhere). The default
        # "transport" mode is the documented production posture.
        mode = os.environ.get("MCP_RATE_LIMITS", "transport").lower()
        if mode in ("disabled", "off", "false"):
            self._enabled = False
            self._transport_only = False
        elif mode in ("enabled", "on", "true"):
            self._enabled = True
            self._transport_only = False
        else:
            # transport (default): enforced from the MCP HTTP/JSON-RPC
            # boundary; in-process registry calls bypass the bucket.
            self._enabled = True
            self._transport_only = True

    def _maybe_reload(self) -> Dict[str, str]:
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return {}
        if mtime == self._policy_mtime and self._buckets:
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            _log.warning("rate limiter: failed to read %s: %s", self._path, exc)
            return {}
        raw = data.get("rate_limits") or {}
        if not isinstance(raw, dict):
            _log.warning("rate limiter: rate_limits in %s must be a mapping", self._path)
            return {}
        configs: Dict[str, str] = {}
        for tool_name, raw_value in raw.items():
            if not isinstance(tool_name, str) or not isinstance(raw_value, str):
                continue
            configs[tool_name] = raw_value
        # Rebuild buckets atomically; preserve token counts when possible.
        with self._lock:
            new_buckets: Dict[str, _Bucket] = {}
            for tool_name, raw_value in configs.items():
                rate = _parse_rate(raw_value)
                if rate is None:
                    _log.warning("rate limiter: ignoring unparseable rate "
                                 "%r for tool %s", raw_value, tool_name)
                    continue
                period_capacity = float(rate)
                period_match = _RATE_PATTERN.match(raw_value)
                if period_match:
                    count = int(period_match.group(1))
                    period_capacity = float(count)
                existing = self._buckets.get(tool_name)
                tokens = existing.tokens if existing else period_capacity
                new_buckets[tool_name] = _Bucket(
                    capacity=rate,
                    period_capacity=period_capacity,
                    tokens=tokens,
                    last_refill=time.monotonic(),
                )
            self._buckets = new_buckets
            self._policy_mtime = mtime
        return configs

    def check(self, tool_name: str, *, source: str = "registry") -> None:
        """Raise :class:`RateLimitExceeded` if the bucket is empty.

        ``source`` distinguishes in-process registry calls from
        public-facing transport calls. In the default "transport" mode
        only ``source="transport"`` consumes a token; the simulator's
        in-process invocations (``source="registry"``) bypass the
        bucket so the benchmark hot path is not throttled. Force-on
        and force-off modes ignore ``source``.

        No-op when limits are disabled or the tool has no
        configured entry.
        """
        if not self._enabled:
            return
        if self._transport_only and source != "transport":
            return
        self._maybe_reload()
        with self._lock:
            bucket = self._buckets.get(tool_name)
            if bucket is None:
                return
            if not bucket.consume():
                raise RateLimitExceeded(
                    f"rate limit exceeded for tool {tool_name!r}"
                )

    def configured_limits(self) -> Dict[str, str]:
        """Return the parsed configured limits (for diagnostics)."""
        return self._maybe_reload()


_DEFAULT_LIMITER: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Return the process-wide rate limiter (created lazily)."""
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        _DEFAULT_LIMITER = RateLimiter()
    return _DEFAULT_LIMITER
