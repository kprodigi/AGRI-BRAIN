"""Shared security helpers for HTTP and websocket routes."""
from __future__ import annotations

import base64
import hmac
import time
from typing import Iterable, Optional

from fastapi import Header, HTTPException, Request, WebSocket

from .settings import SETTINGS


# Scope -> SETTINGS attribute. Each scope's key falls back to APP_API_KEY
# in load_settings(), so single-key deployments keep working unchanged.
# The scoped key is *additionally* accepted; the global key is still
# valid so admin tooling can authenticate against any scope. This
# preserves backward compatibility while letting operators rotate
# scope-specific keys independently (the canonical use case is
# limiting which credentials can mutate governance / chain / phase /
# MCP state).
_SCOPE_KEYS = {
    "governance": "governance_api_key",
    "chain": "chain_api_key",
    "phase": "phase_api_key",
    "mcp": "mcp_api_key",
}


def _is_local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _accept_keys(scope: Optional[str]) -> Iterable[str]:
    """Return the set of API keys accepted for ``scope``.

    The global ``SETTINGS.api_key`` is always accepted (so admin tooling
    can hold a single super-key); the scope-specific key is accepted in
    addition when it is configured. Unset scopes fall back to the
    global key alone.
    """
    keys = []
    if SETTINGS.api_key:
        keys.append(SETTINGS.api_key)
    if scope is None:
        return keys
    attr = _SCOPE_KEYS.get(scope)
    if not attr:
        return keys
    scoped = getattr(SETTINGS, attr, "") or ""
    # When the scoped key matches the global key it is already in the
    # list; only append distinct values.
    if scoped and scoped not in keys:
        keys.append(scoped)
    return keys


def enforce_api_key(request: Request, x_api_key: str | None,
                    *, scope: Optional[str] = None) -> None:
    """Enforce that ``x_api_key`` matches a key valid for ``scope``.

    ``scope`` selects between the global (None) policy and the scoped
    policy used by routers that handle privileged state (governance,
    chain, phase, mcp). When scoped keys are configured they are
    accepted *in addition to* the global key -- never instead -- so
    that a single credential rotation does not require coordinating
    every scope at once. When the scoped key is not configured (the
    common single-key deployment), behaviour is identical to the
    pre-2026-05 path.
    """
    if not SETTINGS.require_api_key:
        return
    # Check X-Forwarded-For first; if present, the real client is remote
    # and loopback bypass should not apply (reverse-proxy scenario).
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        host = request.client.host if request.client else ""
        if SETTINGS.allow_local_without_api_key and _is_local_host(host):
            return
    accepted = list(_accept_keys(scope))
    if not accepted:
        raise HTTPException(status_code=503, detail="Server API key not configured")
    presented = x_api_key or ""
    for k in accepted:
        if hmac.compare_digest(presented, k):
            return
    raise HTTPException(status_code=401, detail="Invalid API key")


def require_scope_api_key(scope: str):
    """FastAPI dependency factory: require a scoped API key.

    Use as ``Depends(require_scope_api_key("governance"))`` on a router
    or individual route. The dependency raises 401 / 503 the same way
    :func:`enforce_api_key` does and is a no-op when REQUIRE_API_KEY is
    false (preserving the dev experience).
    """
    if scope not in _SCOPE_KEYS:
        raise ValueError(f"unknown scope: {scope}")

    def _dep(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        enforce_api_key(request, x_api_key, scope=scope)

    return _dep


def websocket_auth_ok(ws: WebSocket) -> bool:
    if not SETTINGS.websocket_require_api_key:
        return True
    host = ws.client.host if ws.client else ""
    if SETTINGS.allow_local_without_api_key and _is_local_host(host):
        return True
    token = ws.query_params.get("ws_token")
    if token and validate_ws_token(token):
        return True
    api_key = ws.query_params.get("api_key") or ws.headers.get("x-api-key")
    if not SETTINGS.websocket_api_key:
        return False
    return hmac.compare_digest(api_key or "", SETTINGS.websocket_api_key)


def _ws_secret() -> str:
    return SETTINGS.websocket_api_key or SETTINGS.api_key


def issue_ws_token(ttl_seconds: int = 120) -> str:
    """Issue short-lived websocket token derived from server secret."""
    secret = _ws_secret()
    if not secret:
        raise HTTPException(status_code=503, detail="WebSocket auth key not configured")
    exp = int(time.time()) + max(int(ttl_seconds), 1)
    payload = str(exp).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), payload, "sha256").hexdigest()
    raw = f"{exp}.{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def validate_ws_token(token: str) -> bool:
    secret = _ws_secret()
    if not secret:
        return False
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        exp_s, sig = raw.split(".", 1)
        exp = int(exp_s)
    except Exception:
        return False
    if exp < int(time.time()):
        return False
    expected = hmac.new(secret.encode("utf-8"), str(exp).encode("utf-8"), "sha256").hexdigest()
    return hmac.compare_digest(sig, expected)

