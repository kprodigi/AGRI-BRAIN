# backend/src/routers/stream.py
import asyncio, json
from fastapi import APIRouter, Header, Request, WebSocket, WebSocketDisconnect
from starlette.responses import Response
from src.agents.bus import BUS
from src.security import enforce_api_key, issue_ws_token, websocket_auth_ok

router = APIRouter()


@router.get("/stream/token")
def stream_token(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    """Mint a short-lived websocket token so clients needn't send API keys in URL."""
    enforce_api_key(request, x_api_key)
    return {"token": issue_ws_token(ttl_seconds=120)}


@router.websocket("/stream")
async def stream(ws: WebSocket):
    if not websocket_auth_ok(ws):
        # Reject at WebSocket level: close without accepting so clients
        # see a failed upgrade (HTTP 403) rather than a connected-then-closed
        # sequence.  Starlette supports this via close() before accept().
        try:
            await ws.close(code=1008)
        except RuntimeError:
            # Fallback: accept, send error, close (for older Starlette)
            await ws.accept()
            await ws.send_text(json.dumps({"type": "error", "payload": {"detail": "unauthorized"}}))
            await ws.close(code=1008)
        return
    await BUS.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "payload": {"ok": True}}))
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        await BUS.disconnect(ws)
