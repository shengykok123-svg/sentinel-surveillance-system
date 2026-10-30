"""
Person of Interest (POI) management and enrollment API routes.

Router prefix: ``/api/poi``

This module provides endpoints for managing persons of interest -- the
individuals whose face embeddings are stored in the FAISS index for
real-time recognition:

* **GET    /api/poi**              -- List all enrolled persons with their
  current live status (e.g. whether they are visible in any active camera
  track).
* **POST   /api/poi/enroll**       -- Enroll a new person by uploading one
  or more face images (multipart form, auth required).
* **PUT    /api/poi/{person_id}**  -- Update an existing person's metadata
  (auth required).
* **DELETE /api/poi/{person_id}**  -- Remove a person and their embeddings
  (auth required).
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    get_current_user, get_enrollment_service, get_recognition_service, get_video_service,
    get_reference_image_repository, get_reference_image_service,
    get_reference_image_access_repository, require_reference_image_access,
)
from app.models.schemas import PersonUpdate
from app.services.auth_service import AuthenticatedUser
from app.services.enrollment_service import (
    EnrollmentImagePayload,
    EnrollmentService,
    EnrollmentValidationError,
)
from app.services.recognition_service import RecognitionService
from app.services.video_service import VideoService
from app.repositories.reference_image_repository import ReferenceImageRepository
from app.repositories.reference_image_access_repository import ReferenceImageAccessRepository
from app.services.reference_image_service import ReferenceImageService


router = APIRouter(prefix="/api/poi", tags=["poi"])


# Plain ``def`` so FastAPI runs it in the threadpool: the synchronous database
# work must not stall the event loop that sends the camera frame streams.
@router.get("", dependencies=[Depends(get_current_user)])
def list_people(
    recognition_service: RecognitionService = Depends(get_recognition_service),
    video_service: VideoService = Depends(get_video_service),
):
    """List all enrolled persons of interest with live status.

    **HTTP Method:** GET
    **Path:** /api/poi
    **Auth required:** Yes (Bearer token)

    Each person record is enriched with a ``live_status`` field that
    indicates whether the person is currently visible in one or more
    active camera tracks.

    Returns:
        list[dict]: JSON array of person objects, each serialised via
            ``model_dump()``.
    """
    # Fetch all currently active tracks so the recognition service can
    # determine which persons are "live" (i.e. currently being tracked).
    active_tracks = video_service.get_active_tracks()
    return [person.model_dump() for person in recognition_service.list_people(active_tracks=active_tracks)]


@router.post("/assess-quality")
async def assess_upload_quality(
    image: UploadFile = File(...),
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Authenticated, non-persistent per-photo quality preview."""
    payload = EnrollmentImagePayload(image.filename or "image", await image.read())
    try:
        return await run_in_threadpool(enrollment_service.assess_image_quality, payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Quality assessment is unavailable. Please retry.") from exc


@router.post("/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_person(
    request: Request,
    full_name: str = Form(...),
    person_type: str = Form(...),
    is_active: bool = Form(default=True),
    embedding_only_mode: bool = Form(default=False),
    retention_expires_at: datetime | None = Form(default=None),
    notes: str | None = Form(default=None),
    images: list[UploadFile] = File(...),
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
    job_id: UUID | None = None,
):
    """Enroll a new person of interest with one or more face images.

    **HTTP Method:** POST
    **Path:** /api/poi/enroll
    **Auth required:** Yes (Bearer token)
    **Content-Type:** multipart/form-data

    The endpoint accepts metadata via form fields and one or more face
    image files.  The enrollment service detects faces in each image,
    extracts embeddings, and stores them in the FAISS index alongside a
    new person record in the database.

    Args:
        full_name: Display name for the person.
        person_type: Category label (e.g. "whitelist", "blacklist").
        is_active: Whether the person should be actively matched
            (default ``True``).
        embedding_only_mode: If ``True``, store embeddings but do not
            persist the original face images (default ``False``).
        retention_expires_at: Optional expiry date after which the
            person's data may be purged.
        notes: Free-text notes about the person.
        images: One or more face image files (JPEG/PNG).

    Returns:
        dict: The newly created person record, including generated ID and
            enrollment metadata.

    Raises:
        HTTPException 400: If face detection/validation fails for one or
            more images (detail includes ``rejected_images`` list), or if
            any other input validation error occurs.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        # Convert each uploaded file into an EnrollmentImagePayload with
        # its filename and raw byte contents for downstream processing.
        if job_id is not None:
            existing = request.app.state.enrollment_job_service.get(str(job_id), current_user.id)
            if existing is not None:
                return JSONResponse(existing, status_code=202, headers={"Cache-Control": "no-store"})
            if len(images) > 16:
                raise HTTPException(413, "Upload at most 16 reference images per enrollment")
        payloads = []
        remaining_bytes = 20 * 1024 * 1024
        for index, image in enumerate(images):
            contents = await image.read(remaining_bytes + 1) if job_id is not None else await image.read()
            if job_id is not None:
                if len(contents) > remaining_bytes:
                    raise HTTPException(413, "Reference images must total at most 20 MB")
                remaining_bytes -= len(contents)
            payloads.append(EnrollmentImagePayload(
                filename=image.filename or f"image-{index + 1}.jpg", contents=contents))
        if job_id is not None:
            try:
                job = request.app.state.enrollment_job_service.submit(
                    str(job_id), current_user.id, enrollment_service.enroll_person,
                    full_name=full_name, person_type=person_type, is_active=is_active,
                    embedding_only_mode=embedding_only_mode, retention_expires_at=retention_expires_at,
                    notes=notes, images=payloads, enrolled_by_user_id=current_user.id,
                )
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return JSONResponse(job, status_code=202, headers={"Cache-Control": "no-store"})
        result = await run_in_threadpool(enrollment_service.enroll_person,
            full_name=full_name,
            person_type=person_type,
            is_active=is_active,
            embedding_only_mode=embedding_only_mode,
            retention_expires_at=retention_expires_at,
            notes=notes,
            images=payloads,
            enrolled_by_user_id=current_user.id,
        )
    except EnrollmentValidationError as exc:
        # One or more images failed validation (e.g. no face detected).
        # Return a structured error with per-image rejection details.
        raise HTTPException(
            status_code=400,
            detail={
                "code": exc.code,
                "message": str(exc),
                "rejected_images": exc.rejected_images,
            },
        ) from exc
    except ValueError as exc:
        # Generic validation failure (e.g. invalid person_type)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@router.get("/enrollment-jobs/{job_id}")
async def get_enrollment_job(job_id: UUID, request: Request,
                             current_user: AuthenticatedUser = Depends(get_current_user)):
    job = request.app.state.enrollment_job_service.get(str(job_id), current_user.id)
    if job is None:
        raise HTTPException(404, "Job unavailable. The backend may have restarted; check the person list before submitting again.")
    return JSONResponse(job, headers={"Cache-Control": "no-store"})


@router.post("/{person_id}/references", status_code=status.HTTP_201_CREATED)
async def add_reference_images(
    person_id: str,
    images: list[UploadFile] = File(...),
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Append one or more validated reference images to an existing POI."""
    try:
        result = await run_in_threadpool(enrollment_service.add_reference_images,
            person_id=person_id,
            images=[
                EnrollmentImagePayload(
                    filename=image.filename or f"reference-{index + 1}.jpg",
                    contents=await image.read(),
                )
                for index, image in enumerate(images)
            ],
        )
    except EnrollmentValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc), "rejected_images": exc.rejected_images},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.get("/{person_id}/references")
async def list_reference_images(
    person_id: str,
    repository: ReferenceImageRepository = Depends(get_reference_image_repository),
    _current_user: AuthenticatedUser = Depends(require_reference_image_access),
):
    """List privately stored reference-image metadata for a POI profile."""
    return [asdict(image) for image in repository.list_for_person(person_id)]


@router.post("/{person_id}/generate-references")
async def generate_retained_references(
    person_id: str,
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
    _current_user: AuthenticatedUser = Depends(require_reference_image_access),
):
    """An administrator with fresh reauthentication can enhance retained originals."""
    try:
        return await run_in_threadpool(enrollment_service.generate_reference_images, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{person_id}/references/{reference_id}/image")
async def get_reference_image(
    person_id: str,
    reference_id: str,
    request: Request,
    repository: ReferenceImageRepository = Depends(get_reference_image_repository),
    storage: ReferenceImageService = Depends(get_reference_image_service),
    access_repository: ReferenceImageAccessRepository = Depends(get_reference_image_access_repository),
    current_user: AuthenticatedUser = Depends(require_reference_image_access),
):
    """Serve one reference image only after a protected metadata lookup."""
    relative_path = repository.storage_path_for(person_id, reference_id)
    path = storage.resolve(relative_path) if relative_path else None
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reference image not found")
    access_repository.log_view(
        actor_user_id=current_user.id,
        person_id=person_id,
        reference_image_id=reference_id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=path.name,
        headers={"Cache-Control": "private, no-store, max-age=0", "Pragma": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.put("/{person_id}")
async def update_person(
    person_id: str,
    payload: PersonUpdate,
    recognition_service: RecognitionService = Depends(get_recognition_service),
    video_service: VideoService = Depends(get_video_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Update an existing person's metadata.

    **HTTP Method:** PUT
    **Path:** /api/poi/{person_id}
    **Auth required:** Yes (Bearer token)

    Only mutable metadata fields (name, type, active status, notes, etc.)
    can be changed.  Face embeddings are **not** modified by this endpoint;
    re-enrollment is required to update them.

    Args:
        person_id: Unique identifier of the person to update.
        payload: Fields to update (partial update via :class:`PersonUpdate`).

    Returns:
        dict: The updated person record, including live status computed
            from active tracks.

    Raises:
        HTTPException 400: If the update payload fails validation.
        HTTPException 401: If the Bearer token is missing or invalid.
        HTTPException 404: If no person matches ``person_id``.
    """
    try:
        # Pass active tracks so the returned person object includes
        # up-to-date live status information.
        person = recognition_service.update_person(
            person_id,
            payload,
            active_tracks=video_service.get_active_tracks(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if person is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")
    return person.model_dump()


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(
    person_id: str,
    recognition_service: RecognitionService = Depends(get_recognition_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Delete a person of interest and their associated embeddings.

    **HTTP Method:** DELETE
    **Path:** /api/poi/{person_id}
    **Auth required:** Yes (Bearer token)

    Removes the person record from the database and purges their face
    embeddings from the FAISS index.

    Args:
        person_id: Unique identifier of the person to delete.

    Returns:
        None: 204 No Content on success.

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
        HTTPException 404: If no person matches ``person_id``.
    """
    deleted = recognition_service.delete_person(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")
