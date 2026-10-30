"""
WebSocket connection manager for real-time event broadcasting.

Maintains a set of active WebSocket connections and provides both async
and thread-safe broadcast methods.  The video pipeline (which runs in a
background thread) uses ``broadcast_json_threadsafe`` to push detection
events to every connected dashboard client.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    """
    Thread-safe manager for WebSocket connections.

    Keeps an ``asyncio.Lock``-protected set of sockets and exposes
    ``broadcast_json`` (async) and ``broadcast_json_threadsafe``
    (callable from worker threads).
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()  # Active client sockets
        self._lock = asyncio.Lock()                # Guards _connections
        self._logger = logging.getLogger("sentinel.websocket")
        self._loop: asyncio.AbstractEventLoop | None = None  # Main event loop reference

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store a reference to the running event loop (called at startup)."""
        self._loop = loop

    async def connect(self, websocket: WebSocket, *, accepted: bool = False) -> None:
        """Accept a new WebSocket connection and register it."""
        if not accepted:
            await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection (idempotent)."""
        async with self._lock:
            self._connections.discard(websocket)

    def has_connections(self) -> bool:
        """True when at least one dashboard is subscribed."""
        return bool(self._connections)

    async def broadcast_json(self, message: dict[str, Any]) -> None:
        """
        Send a JSON message to every connected client.

        Disconnects any client that raises an exception during send.
        """
        async with self._lock:
            sockets = list(self._connections)

        for websocket in sockets:
            try:
                await websocket.send_json(message)
            except Exception:
                self._logger.exception("WebSocket broadcast failed")
                await self.disconnect(websocket)

    def broadcast_json_threadsafe(self, message: dict[str, Any]) -> None:
        """
        Schedule a JSON broadcast from a non-async worker thread.

        Uses ``asyncio.run_coroutine_threadsafe`` to push the coroutine
        onto the main event loop.  No-ops gracefully if the loop hasn't
        been attached yet.
        """
        if self._loop is None:
            self._logger.debug("No event loop attached for websocket broadcast")
            return
        asyncio.run_coroutine_threadsafe(self.broadcast_json(message), self._loop)
