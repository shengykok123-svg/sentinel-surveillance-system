"""
Camera CRUD API routes for the Sentinel backend.

Router prefix: ``/api/cameras``

This module provides endpoints for managing camera configuration records and
keeping the live video pipeline in sync after every mutation:

* **GET    /api/cameras**              -- List all cameras with live metrics
  (face counts, FPS, stream status) merged from the VideoService.
* **POST   /api/cameras**             -- Register a new camera (auth required).
* **PUT    /api/cameras/{camera_id}** -- Update an existing camera's settings
  (auth required).
* **DELETE /api/cameras/{camera_id}** -- Remove a camera (auth required).

After every write operation (create / update / delete), the VideoService is
instructed to refresh its internal camera pipeline so that streams reflect
the latest configuration without a server restart.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_camera_service, get_current_user, get_video_service
from app.models.schemas import CameraCreate, CameraUpdate
from app.services.auth_service import AuthenticatedUser
from app.services.camera_service import CameraService
from app.services.video_service import VideoService


router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", dependencies=[Depends(get_current_user)])
async def list_cameras(
    camera_service: CameraService = Depends(get_camera_service),
    video_service: VideoService = Depends(get_video_service),
):
    """List all registered cameras with live pipeline metrics.

    **HTTP Method:** GET
    **Path:** /api/cameras
    **Auth required:** Yes (Bearer token)

    The listing is served by :meth:`VideoService.list_cameras` which merges
    the persisted camera configuration with real-time information such as
    connection state, face counts, and FPS.

    Returns:
        list[dict]: A JSON array of camera objects, each containing both
            static configuration fields and live metrics.
    """
    # camera_service is injected but not used directly -- the VideoService
    # already aggregates DB records with live metrics.  We delete the
    # reference to silence linters about the unused variable while keeping
    # Depends() wiring explicit for documentation purposes.
    del camera_service
    return video_service.list_cameras()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_camera(
    payload: CameraCreate,
    camera_service: CameraService = Depends(get_camera_service),
    video_service: VideoService = Depends(get_video_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Register a new camera and start its video pipeline.

    **HTTP Method:** POST
    **Path:** /api/cameras
    **Auth required:** Yes (Bearer token)

    After persisting the camera record via :class:`CameraService`, the
    VideoService is refreshed so that a new processing pipeline is
    created for the camera immediately.

    Args:
        payload: Camera creation payload (name, RTSP URL, etc.).

    Returns:
        dict: The newly created camera record serialised as a dictionary.

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    created = camera_service.create_camera(payload)
    # Trigger the video service to pick up the new camera and start its stream
    video_service.refresh_cameras()
    return created.model_dump()


@router.put("/{camera_id}")
async def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    camera_service: CameraService = Depends(get_camera_service),
    video_service: VideoService = Depends(get_video_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Update an existing camera's configuration.

    **HTTP Method:** PUT
    **Path:** /api/cameras/{camera_id}
    **Auth required:** Yes (Bearer token)

    If the camera is found and updated successfully, the VideoService
    pipeline is refreshed to apply the new settings (e.g. a changed
    RTSP URL) without a restart.

    Args:
        camera_id: Unique identifier of the camera to update.
        payload: Fields to update (partial update supported via
            :class:`CameraUpdate`).

    Returns:
        dict: The updated camera record serialised as a dictionary.

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
        HTTPException 404: If no camera matches ``camera_id``.
    """
    updated = camera_service.update_camera(camera_id, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    # Refresh the video pipeline so the updated settings take effect immediately
    video_service.refresh_cameras()
    return updated.model_dump()


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: str,
    camera_service: CameraService = Depends(get_camera_service),
    video_service: VideoService = Depends(get_video_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Delete a camera and tear down its video pipeline.

    **HTTP Method:** DELETE
    **Path:** /api/cameras/{camera_id}
    **Auth required:** Yes (Bearer token)

    Removes the camera record from the database.  The VideoService is then
    refreshed so the corresponding processing pipeline is stopped and
    cleaned up.

    Args:
        camera_id: Unique identifier of the camera to delete.

    Returns:
        None: 204 No Content on success.

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
        HTTPException 404: If no camera matches ``camera_id``.
    """
    deleted = camera_service.delete_camera(camera_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    # Stop and clean up the pipeline for the removed camera
    video_service.refresh_cameras()
