"""
Detection event API routes and WebSocket endpoint for the Sentinel backend.

This module groups event-related endpoints under the ``events`` tag.  Unlike
most other route modules it does **not** use a shared router prefix because
it mixes regular HTTP paths (``/api/events``) with a WebSocket path
(``/ws/events``).

Endpoints:

* **GET    /api/events** -- Retrieve a paginated list of stored detection
  events (limit/offset query parameters).
* **DELETE /api/events** -- Clear **all** stored events (auth required).
* **WS     /ws/events**  -- WebSocket connection for real-time event
  broadcasting.  The server pushes new events as they are detected;
  client messages are consumed but not processed (keep-alive).
"""

from __future__ import annotations

import asyncio
import time
from starlette.concurrency import run_in_threadpool
from app.core.security import decode_access_token

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse

from app.api.deps import get_current_user, get_event_service
from app.services.auth_service import AuthenticatedUser, AuthError
from app.services.event_service import EventService


router = APIRouter(tags=["events"])


# Plain ``def`` so FastAPI runs it in the threadpool: the synchronous database
# work must not stall the event loop that sends the camera frame streams.
@router.get("/api/events", dependencies=[Depends(get_current_user)])
def list_events(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_service: EventService = Depends(get_event_service),
):
    """Return a paginated list of detection events.

    **HTTP Method:** GET
    **Path:** /api/events
    **Auth required:** Yes (Bearer token)

    Events are returned in reverse chronological order (newest first).
    Pagination is controlled via ``limit`` and ``offset`` query parameters.

    Args:
        limit: Maximum number of events to return (1--200, default 50).
        offset: Number of events to skip from the beginning (default 0).

    Returns:
        list[dict]: A JSON array of event objects.
    """
    return event_service.list_events(limit=limit, offset=offset)


@router.delete("/api/events", status_code=status.HTTP_204_NO_CONTENT)
async def clear_events(
    event_service: EventService = Depends(get_event_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Delete all stored detection events.

    **HTTP Method:** DELETE
    **Path:** /api/events
    **Auth required:** Yes (Bearer token)

    This is a destructive bulk operation intended for administrative use
    (e.g. clearing test data or resetting the event log).

    Returns:
        None: 204 No Content on success.

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    event_service.clear_events()


@router.get("/api/recognition-records")
async def list_recognition_records(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_service: EventService = Depends(get_event_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Return recognised-person records with their saved evidence availability."""
    return event_service.list_recognition_records(limit=limit, offset=offset)


@router.get("/api/recognition-records/{event_id}/snapshot")
async def get_recognition_record_snapshot(
    event_id: str,
    event_service: EventService = Depends(get_event_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Return the protected, annotated evidence image for one record."""
    path = event_service.get_snapshot_path(event_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recognition record image not found")
    return FileResponse(path, media_type="image/jpeg", filename=f"recognition-record-{event_id}.jpg")


@router.websocket("/ws/events")
async def events_ws(websocket: WebSocket):
    """Authenticate the first message before subscribing to any events."""
    manager = websocket.app.state.websocket_manager
    auth = websocket.app.state.auth_service
    await websocket.accept()
    try:
        message = await asyncio.wait_for(websocket.receive_json(), timeout=5)
        if not isinstance(message, dict) or message.get("type") != "authenticate":
            raise ValueError("Authentication required")
        token = message.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("Authentication required")
        await run_in_threadpool(auth.get_current_user, token)
        expires_at = float(decode_access_token(auth.settings, token)["exp"])
        await manager.connect(websocket, accepted=True)
        await websocket.send_json({"type": "authenticated"})
        while True:
            remaining = expires_at - time.time()
            if remaining <= 0:
                raise ValueError("Session expired")
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=min(15, remaining))
            except asyncio.TimeoutError:
                pass
            await run_in_threadpool(auth.get_current_user, token)
    except WebSocketDisconnect:
        pass
    except (ValueError, AuthError, asyncio.TimeoutError):
        await websocket.close(code=4401, reason="Authentication required or expired")
    finally:
        await manager.disconnect(websocket)
