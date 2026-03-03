# backend/src/agents/bus.py
from typing import Set
from fastapi import WebSocket
import asyncio, json

class BroadcastBus:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def emit(self, type: str, payload):
        msg = json.dumps({"type": type, "payload": payload})
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                await self.disconnect(ws)

BUS = BroadcastBus()
