"""Shared security helpers for HTTP and websocket routes."""
from __future__ import annotations

from fastapi import Header, HTTPException, Request, WebSocket

from .settings import SETTINGS


def _is_local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def enforce_api_key(request: Request, x_api_key: str | None) -> None:
    if not SETTINGS.require_api_key:
        return
    host = request.client.host if request.client else ""
    if SETTINGS.allow_local_without_api_key and _is_local_host(host):
        return
    if not SETTINGS.api_key:
        raise HTTPException(status_code=503, detail="Server API key not configured")
    if x_api_key != SETTINGS.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def websocket_auth_ok(ws: WebSocket) -> bool:
    if not SETTINGS.websocket_require_api_key:
        return True
    host = ws.client.host if ws.client else ""
    if SETTINGS.allow_local_without_api_key and _is_local_host(host):
        return True
    token = ws.query_params.get("api_key") or ws.headers.get("x-api-key")
    if not SETTINGS.websocket_api_key:
        return False
    return token == SETTINGS.websocket_api_key

