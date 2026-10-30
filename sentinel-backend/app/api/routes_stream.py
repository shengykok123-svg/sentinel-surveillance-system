"""
Video streaming and live metrics API routes for the Sentinel backend.

Router prefix: ``/api``

This module provides endpoints for consuming live video feeds and retrieving
real-time performance/tracking metrics from the video pipeline:

* **GET /api/stream/{cam_id}**  -- MJPEG video stream for a specific camera.
* **GET /api/counts**           -- Current face detection counts per camera.
* **GET /api/fps**              -- Output and inference FPS per camera.
* **GET /api/tracks/active**    -- All currently active face tracks across
  all cameras.
"""

from __future__ import annotations

from contextlib import suppress

from fastapi import APIRouter, Depends, HTTPException, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_video_service, get_current_user, get_auth_service
from app.api.live_access import authenticated_frames
from app.services.auth_service import AuthService, AuthError, AuthenticatedUser
from app.services.video_service import VideoService


router = APIRouter(prefix="/api", tags=["stream"])


@router.post("/stream/{cam_id}/ticket")
async def create_stream_ticket(
    cam_id: str, request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    video_service: VideoService = Depends(get_video_service),
):
    if cam_id not in {camera["id"] for camera in video_service.list_cameras()}:
        raise HTTPException(404, "Camera not found")
    token = request.headers["authorization"].split(None, 1)[1]
    try:
        ticket = request.app.state.stream_access_service.issue(current_user.id, cam_id, token)
    except ValueError as exc:
        raise HTTPException(429, str(exc)) from exc
    return JSONResponse({"ticket": ticket, "expires_in": 30}, headers={"Cache-Control": "no-store"})


@router.get("/stream/{cam_id}")
async def video_stream(
    cam_id: str, request: Request, ticket: str | None = Query(default=None),
    video_service: VideoService = Depends(get_video_service),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Stream live MJPEG video from the specified camera.

    **HTTP Method:** GET
    **Path:** /api/stream/{cam_id}
    **Auth required:** Bearer login token or single-use camera ticket.

    Returns a ``multipart/x-mixed-replace`` streaming response where each
    part is a JPEG-encoded video frame.  The browser or client renders
    these frames sequentially to produce a live video feed.

    Args:
        cam_id: Unique identifier of the camera to stream from.

    Returns:
        StreamingResponse: An MJPEG stream with content type
            ``multipart/x-mixed-replace; boundary=frame``.

    Raises:
        HTTPException 404: If no camera matches ``cam_id``.
    """
    try:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization.split(None, 1)[1]
        else:
            token = request.app.state.stream_access_service.consume(ticket or "", cam_id)
        await run_in_threadpool(auth_service.get_current_user, token)
    except (ValueError, AuthError, IndexError) as exc:
        raise HTTPException(401, "Valid stream authentication required", headers={"WWW-Authenticate": "Bearer"}) from exc
    # Build a set of valid camera IDs to validate the requested cam_id
    camera_ids = {camera["id"] for camera in video_service.list_cameras()}
    if cam_id not in camera_ids:
        raise HTTPException(404, f"Camera '{cam_id}' not found")
    # Return a streaming response that yields JPEG frames as they become
    # available from the video processing pipeline.
    return StreamingResponse(
        authenticated_frames(video_service.generate_frames(cam_id), auth_service, token),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "private, no-store, max-age=0", "Referrer-Policy": "no-referrer"},
    )


@router.get("/counts", dependencies=[Depends(get_current_user)])
async def get_counts(video_service: VideoService = Depends(get_video_service)):
    """Return current face detection counts per camera.

    **HTTP Method:** GET
    **Path:** /api/counts
    **Auth required:** Yes (Bearer token)

    Returns a mapping of camera IDs to the number of faces currently
    detected in each camera's latest processed frame.

    Returns:
        dict: ``{camera_id: face_count, ...}`` for all active cameras.
    """
    return video_service.get_counts()


@router.get("/fps", dependencies=[Depends(get_current_user)])
async def get_fps(video_service: VideoService = Depends(get_video_service)):
    """Return output and inference FPS metrics per camera.

    **HTTP Method:** GET
    **Path:** /api/fps
    **Auth required:** Yes (Bearer token)

    Provides both the display frame rate (output FPS) and the model
    inference frame rate for each camera's processing pipeline.

    Returns:
        dict: Per-camera FPS metrics.
    """
    return video_service.get_fps()


@router.get("/tracks/active", dependencies=[Depends(get_current_user)])
async def get_active_tracks(video_service: VideoService = Depends(get_video_service)):
    """Return all currently active face tracks across all cameras.

    **HTTP Method:** GET
    **Path:** /api/tracks/active
    **Auth required:** Yes (Bearer token)

    An active track represents a face that is currently being followed
    by the tracker in at least one camera feed.  Each track contains
    information such as the bounding box, track ID, associated person
    (if recognised), and the camera it belongs to.

    Returns:
        list[dict]: JSON array of active track objects, each serialised
            via ``model_dump()``.
    """
    return [track.model_dump() for track in video_service.get_active_tracks()]


# ── WebSocket frame stream ─────────────────────────────────────────────
#
# Registered without the ``/api`` prefix so the Vite dev proxy (which only
# upgrades ``/ws``) and the browser's per-origin HTTP/1.1 connection budget
# (six sockets, three of which MJPEG streams used to pin) both stay out of
# the way. One binary message per JPEG frame; the client keeps a small
# jitter buffer and draws to a canvas.
ws_router = APIRouter(tags=["stream"])


@ws_router.websocket("/ws/stream/{cam_id}")
async def video_stream_ws(websocket: WebSocket, cam_id: str, ticket: str = Query(default="")):
    """Stream JPEG frames for one camera over a WebSocket.

    Auth: a single-use camera ticket from ``POST /api/stream/{cam_id}/ticket``
    in the ``ticket`` query parameter. The login token behind the ticket is
    re-validated every 15 seconds and the socket closes when it expires.
    """
    app_state = websocket.app.state
    video_service: VideoService = app_state.video_service
    auth_service: AuthService = app_state.auth_service
    try:
        token = app_state.stream_access_service.consume(ticket, cam_id)
        await run_in_threadpool(auth_service.get_current_user, token)
    except (ValueError, AuthError):
        await websocket.close(code=4401, reason="Valid stream authentication required")
        return
    if cam_id not in {camera["id"] for camera in video_service.list_cameras()}:
        await websocket.close(code=4404, reason="Camera not found")
        return

    await websocket.accept()
    frames = authenticated_frames(video_service.iter_encoded_frames(cam_id), auth_service, token)
    try:
        async for _sequence, jpeg in frames:
            await websocket.send_bytes(jpeg)
    except WebSocketDisconnect:
        return
    except (AuthError, ValueError):
        pass
    except RuntimeError:
        # Send on a socket the client already closed.
        return
    finally:
        await frames.aclose()
    with suppress(RuntimeError):
        await websocket.close(code=4401, reason="Session expired")
