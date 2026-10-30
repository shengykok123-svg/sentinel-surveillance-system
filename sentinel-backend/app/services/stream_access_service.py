"""Short-lived, single-use camera tickets for browser image requests."""
from __future__ import annotations

import secrets
import threading
import time


class StreamAccessService:
    def __init__(self):
        self._tickets: dict[str, tuple[str, str, str, float]] = {}
        self._lock = threading.Lock()

    def issue(self, user_id: str, camera_id: str, access_token: str) -> str:
        with self._lock:
            now = time.monotonic()
            self._tickets = {key: value for key, value in self._tickets.items() if value[3] > now}
            # Bound abandoned requests, including reconnecting browser tabs.
            if sum(value[0] == user_id for value in self._tickets.values()) >= 32 or len(self._tickets) >= 1024:
                raise ValueError("Too many pending stream connections; retry in 30 seconds")
            ticket = secrets.token_urlsafe(32)
            self._tickets[ticket] = (user_id, camera_id, access_token, now + 30)
            return ticket

    def consume(self, ticket: str, camera_id: str) -> str:
        with self._lock:
            value = self._tickets.pop(ticket, None)
        if value is None or value[1] != camera_id or value[3] <= time.monotonic():
            raise ValueError("Invalid or expired stream ticket")
        return value[2]
