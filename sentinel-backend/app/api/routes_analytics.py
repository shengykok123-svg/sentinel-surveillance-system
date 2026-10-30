"""
Recognition analytics API routes for the Sentinel backend.

Router prefix: ``/api/analytics``

These endpoints aggregate the stored detection-event history into the reports
rendered by the Analytics dashboard:

* **GET /api/analytics/summary**              -- Headline counters plus
  period-over-period deltas.
* **GET /api/analytics/timeline**             -- Activity per hour or day,
  gap-filled, plus a weekday/hour heatmap.
* **GET /api/analytics/similarity-histogram** -- Match-score distribution on
  both sides of the recognition threshold.
* **GET /api/analytics/face-size-quality**    -- Recognition outcome as a
  function of detected face size.
* **GET /api/analytics/cameras**              -- Per-camera comparison,
  including cameras that produced nothing.
* **GET /api/analytics/persons/top**          -- Most frequently recognised
  persons and the enrolment-type mix.

Every endpoint requires authentication, as do live and historical event APIs.

All of them share two query parameters:

``range``
    One of ``24h``, ``7d``, ``30d``, ``all``.

``tz_offset``
    Minutes east of UTC, used to align hour and day boundaries to the viewer's
    local time.  Events are stored in UTC, so without it a dashboard in UTC+8
    would report activity peaks shifted by eight hours.  Browsers obtain the
    correct value with ``-new Date().getTimezoneOffset()``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_analytics_service, get_current_user
from app.models.schemas import (
    AnalyticsSummary,
    AnalyticsTimeline,
    CameraAnalyticsReport,
    FaceSizeQuality,
    PersonAnalyticsReport,
    SimilarityHistogram,
)
from app.services.analytics_service import AnalyticsService
from app.services.auth_service import AuthenticatedUser


router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# Shared query parameter definitions, declared once so every endpoint documents
# and validates the window identically.
RangeQuery = Query(default="7d", pattern="^(24h|7d|30d|all)$", description="Reporting window")
TzOffsetQuery = Query(
    default=0,
    ge=-720,
    le=840,
    description="Minutes east of UTC used to align bucket boundaries to local time",
)


def _bad_request(exc: ValueError) -> HTTPException:
    """Translate a rejected reporting parameter into HTTP 400."""
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/summary", response_model=AnalyticsSummary)
async def get_summary(
    range: str = Query(default="24h", pattern="^(24h|7d|30d|all)$", description="Reporting window"),
    tz_offset: int = TzOffsetQuery,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
) -> AnalyticsSummary:
    """Return headline recognition counters for the window.

    **HTTP Method:** GET
    **Path:** /api/analytics/summary
    **Auth required:** Yes (Bearer token)

    Includes ``delta_vs_previous``, the fractional change of each counter
    against the equally long window immediately before this one.  The ``all``
    range has no such baseline and reports no deltas.

    Returns:
        AnalyticsSummary: Counters, recognition rate, and deltas.

    Raises:
        HTTPException 400: If the requested window is not supported.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        return analytics_service.summary(range_key=range, tz_offset=tz_offset)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/timeline", response_model=AnalyticsTimeline)
async def get_timeline(
    range: str = RangeQuery,
    bucket: str = Query(default="hour", pattern="^(hour|day)$", description="Bucket width"),
    tz_offset: int = TzOffsetQuery,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
) -> AnalyticsTimeline:
    """Return recognition activity over time plus a weekday/hour heatmap.

    **HTTP Method:** GET
    **Path:** /api/analytics/timeline
    **Auth required:** Yes (Bearer token)

    Buckets are gap-filled across the whole window, so quiet periods are
    reported as zeros instead of being skipped.  The heatmap is always a full
    7 x 24 grid.

    Returns:
        AnalyticsTimeline: Chronological buckets and heatmap cells.

    Raises:
        HTTPException 400: If the window or bucket width is not supported.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        return analytics_service.timeline(range_key=range, bucket=bucket, tz_offset=tz_offset)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/similarity-histogram", response_model=SimilarityHistogram)
async def get_similarity_histogram(
    range: str = RangeQuery,
    bins: int = Query(default=20, ge=1, le=100, description="Number of histogram bins across 0..1"),
    tz_offset: int = TzOffsetQuery,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
) -> SimilarityHistogram:
    """Return the match-score distribution either side of the threshold.

    **HTTP Method:** GET
    **Path:** /api/analytics/similarity-histogram
    **Auth required:** Yes (Bearer token)

    Each bin separates scores that cleared ``recognition_threshold`` from
    those that were rejected, and the configured threshold is returned
    alongside so it can be drawn on the chart.  Detections recorded without a
    score are excluded.

    Returns:
        SimilarityHistogram: Bins, the threshold, and the scored-event count.

    Raises:
        HTTPException 400: If the requested window is not supported.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        return analytics_service.similarity_histogram(range_key=range, bins=bins, tz_offset=tz_offset)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/face-size-quality", response_model=FaceSizeQuality)
async def get_face_size_quality(
    range: str = RangeQuery,
    tz_offset: int = TzOffsetQuery,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
) -> FaceSizeQuality:
    """Return recognition outcome grouped by detected face size.

    **HTTP Method:** GET
    **Path:** /api/analytics/face-size-quality
    **Auth required:** Yes (Bearer token)

    Faces are banded by their equivalent square edge in pixels, which is what
    determines how far a subject can be from a camera and still be identified.

    Returns:
        FaceSizeQuality: One entry per size band with rate and mean score.

    Raises:
        HTTPException 400: If the requested window is not supported.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        return analytics_service.face_size_quality(range_key=range, tz_offset=tz_offset)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/cameras", response_model=CameraAnalyticsReport)
async def get_camera_analytics(
    range: str = RangeQuery,
    tz_offset: int = TzOffsetQuery,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
) -> CameraAnalyticsReport:
    """Return per-camera recognition activity, busiest camera first.

    **HTTP Method:** GET
    **Path:** /api/analytics/cameras
    **Auth required:** Yes (Bearer token)

    Cameras that produced no events appear as zero rows, so a camera that has
    gone quiet is visible rather than simply missing.

    Returns:
        CameraAnalyticsReport: One entry per camera.

    Raises:
        HTTPException 400: If the requested window is not supported.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        return analytics_service.cameras(range_key=range, tz_offset=tz_offset)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/persons/top", response_model=PersonAnalyticsReport)
async def get_top_persons(
    range: str = RangeQuery,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of persons to rank"),
    tz_offset: int = TzOffsetQuery,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    _current_user: AuthenticatedUser = Depends(get_current_user),
) -> PersonAnalyticsReport:
    """Return the most frequently recognised persons in the window.

    **HTTP Method:** GET
    **Path:** /api/analytics/persons/top
    **Auth required:** Yes (Bearer token)

    Only recognised events are ranked, so unknown detections cannot distort
    the ordering.  ``type_breakdown`` totals recognitions per person type
    across the ranked entries.

    Returns:
        PersonAnalyticsReport: Ranked persons and the enrolment-type mix.

    Raises:
        HTTPException 400: If the requested window is not supported.
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    try:
        return analytics_service.top_persons(range_key=range, limit=limit, tz_offset=tz_offset)
    except ValueError as exc:
        raise _bad_request(exc) from exc
