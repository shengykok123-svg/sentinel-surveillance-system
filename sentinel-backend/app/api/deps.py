"""
FastAPI dependency injection functions for the Sentinel backend.

This module provides factory functions that extract shared service instances
from ``request.app.state`` so they can be injected into route handlers via
FastAPI's ``Depends()`` mechanism.  Each ``get_*_service`` function maps to
exactly one service singleton that was initialised during application startup.

It also exposes :func:`get_current_user`, the standard authentication
dependency that validates a Bearer JWT token and returns the authenticated
user, raising HTTP 401 if the token is missing or invalid.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import bearer_scheme, decode_access_token, extract_bearer_token
from app.services.analytics_service import AnalyticsService
from app.services.camera_service import CameraService
from app.services.enrollment_service import EnrollmentService
from app.services.event_service import EventService
from app.services.face_alignment_service import FaceAlignmentService
from app.services.auth_service import AuthService, AuthenticatedUser, AuthError
from app.services.recognition_service import RecognitionService
from app.services.resource_service import ResourceService
from app.repositories.reference_image_repository import ReferenceImageRepository
from app.repositories.reference_image_access_repository import ReferenceImageAccessRepository
from app.services.reference_image_service import ReferenceImageService
from app.services.video_service import VideoService


# ---------------------------------------------------------------------------
# Service provider dependencies
# ---------------------------------------------------------------------------
# Each function below retrieves a service singleton that was attached to
# ``app.state`` at application startup.  They are intended to be used with
# ``Depends(get_<name>_service)`` in route signatures.
# ---------------------------------------------------------------------------


def get_video_service(request: Request) -> VideoService:
    """Return the :class:`VideoService` instance from application state.

    The VideoService manages camera pipelines, frame generation, and
    real-time tracking/counting metrics.
    """
    return request.app.state.video_service


def get_resource_service(request: Request) -> ResourceService:
    """Return the :class:`ResourceService` instance from application state.

    The ResourceService collects system-level metrics such as CPU, RAM,
    GPU utilisation, disk, and network statistics.
    """
    return request.app.state.resource_service


def get_event_service(request: Request) -> EventService:
    """Return the :class:`EventService` instance from application state.

    The EventService handles storage and retrieval of detection events
    (e.g. recognised faces, alerts).
    """
    return request.app.state.event_service


def get_analytics_service(request: Request) -> AnalyticsService:
    """Return the :class:`AnalyticsService` instance from application state.

    The AnalyticsService aggregates the detection-event history into the
    recognition reports rendered by the Analytics dashboard.
    """
    return request.app.state.analytics_service


def get_camera_service(request: Request) -> CameraService:
    """Return the :class:`CameraService` instance from application state.

    The CameraService provides CRUD operations for camera configuration
    records persisted in the database.
    """
    return request.app.state.camera_service


def get_recognition_service(request: Request) -> RecognitionService:
    """Return the :class:`RecognitionService` instance from application state.

    The RecognitionService manages the FAISS vector index, person records,
    face search, and embedding statistics.
    """
    return request.app.state.recognition_service


def get_enrollment_service(request: Request) -> EnrollmentService:
    """Return the :class:`EnrollmentService` instance from application state.

    The EnrollmentService orchestrates person enrollment, including face
    detection, alignment, embedding extraction, and database persistence.
    """
    return request.app.state.enrollment_service


def get_face_alignment_service(request: Request) -> FaceAlignmentService:
    """Return the :class:`FaceAlignmentService` instance from application state.

    The FaceAlignmentService performs face detection and alignment
    preprocessing prior to embedding extraction.
    """
    return request.app.state.face_alignment_service


def get_auth_service(request: Request) -> AuthService:
    """Return the :class:`AuthService` instance from application state.

    The AuthService handles user authentication, JWT token generation and
    validation, and bootstrap (first-admin) logic.
    """
    return request.app.state.auth_service


def get_reference_image_repository(request: Request) -> ReferenceImageRepository:
    return request.app.state.reference_image_repository


def get_reference_image_service(request: Request) -> ReferenceImageService:
    return request.app.state.reference_image_service


def get_reference_image_access_repository(request: Request) -> ReferenceImageAccessRepository:
    return request.app.state.reference_image_access_repository


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthenticatedUser:
    """Validate the incoming Bearer JWT and return the authenticated user.

    This is the primary authentication dependency used by protected
    endpoints.  It should be included via ``Depends(get_current_user)``
    in any route that requires a logged-in user.

    Flow:
        1. ``bearer_scheme`` extracts the ``Authorization: Bearer <token>``
           header (or ``None`` if absent).
        2. ``extract_bearer_token`` unwraps the credential, raising
           ``ValueError`` when the header is missing or malformed.
        3. ``auth_service.get_current_user`` decodes and validates the JWT,
           raising ``AuthError`` on expiry or invalid signature.
        4. Any failure is converted to an HTTP 401 response with a
           ``WWW-Authenticate: Bearer`` header.

    Returns:
        AuthenticatedUser: The user associated with the valid token.

    Raises:
        HTTPException: 401 Unauthorized if the token is absent, malformed,
            expired, or otherwise invalid.
    """
    try:
        # Extract the raw token string from the Authorization header
        token = extract_bearer_token(credentials)
        # Decode the JWT and look up the corresponding user record
        return auth_service.get_current_user(token)
    except (ValueError, AuthError) as exc:
        # Convert auth failures into a standard 401 response
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_reference_image_access(
    sensitive_access: str | None = Header(default=None, alias="X-Sensitive-Access"),
    current_user: AuthenticatedUser = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthenticatedUser:
    """Require an admin plus a valid, short-lived reference-image scope token."""
    if "admin" not in {role.lower() for role in current_user.roles}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role is required for reference images")
    if not sensitive_access:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sensitive access token is required")
    try:
        payload = decode_access_token(auth_service.settings, sensitive_access)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sensitive access token is invalid or expired") from exc
    if (
        str(payload.get("sub")) != current_user.id
        or payload.get("scope") != "reference_images"
        or payload.get("reauthenticated") is not True
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Sensitive access token is not valid for this user")
    return current_user
