"""Keep long-lived streams bounded by the operator's login lifetime."""
from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from starlette.concurrency import run_in_threadpool

from app.core.security import decode_access_token
from app.services.auth_service import AuthError


async def authenticated_frames(frames, auth_service, token):
    expires_at = float(decode_access_token(auth_service.settings, token)["exp"])
    pending = None
    try:
        while time.time() < expires_at:
            try:
                await run_in_threadpool(auth_service.get_current_user, token)
            except (AuthError, ValueError):
                return
            check_at = min(expires_at, time.time() + 15)
            while time.time() < check_at:
                if pending is None:
                    pending = asyncio.create_task(anext(frames))
                done, _ = await asyncio.wait({pending}, timeout=max(0, check_at - time.time()))
                if not done:
                    break
                try:
                    frame = pending.result()
                except StopAsyncIteration:
                    return
                pending = None
                if time.time() >= expires_at:
                    return
                yield frame
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        await frames.aclose()
