# backend/src/routers/stream.py
import asyncio, json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.agents.bus import BUS

router = APIRouter()

@router.websocket("/stream")
async def stream(ws: WebSocket):
    await BUS.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "payload": {"ok": True}}))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        await BUS.disconnect(ws)
