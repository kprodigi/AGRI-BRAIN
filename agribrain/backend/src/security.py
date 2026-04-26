"""Shared security helpers for HTTP and websocket routes."""
from __future__ import annotations

import base64
import hmac
import time

from fastapi import HTTPException, Request, WebSocket

from .settings import SETTINGS


def _is_local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def enforce_api_key(request: Request, x_api_key: str | None) -> None:
    if not SETTINGS.require_api_key:
        return
    # Check X-Forwarded-For first; if present, the real client is remote
    # and loopback bypass should not apply (reverse-proxy scenario).
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        host = request.client.host if request.client else ""
        if SETTINGS.allow_local_without_api_key and _is_local_host(host):
            return
    if not SETTINGS.api_key:
        raise HTTPException(status_code=503, detail="Server API key not configured")
    if not hmac.compare_digest(x_api_key or "", SETTINGS.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


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

