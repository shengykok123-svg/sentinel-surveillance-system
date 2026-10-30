"""
Analytics repository module for Sentinel.

Aggregates stored detection events into the summaries that back the
Analytics dashboard: recognition rates, similarity distributions, per-camera
comparisons, and per-person activity.

Two conventions from the write path shape every query here:

* ``confidence_score`` holds ``-1.0`` as a sentinel meaning "no similarity was
  available" (an unknown detection with an empty index behind it).  Every
  aggregate over similarity must exclude those rows or the average is dragged
  below zero, so they are filtered with ``confidence_score >= 0``.
* ``person_id`` is NULL exactly when the detection was *not* matched to an
  enrolled person, which makes it the recognised / unknown discriminator.

Time bucketing is deliberately performed in Python rather than SQL.  The
project runs on SQLite in development and PostgreSQL in deployment, and the
two disagree on both the truncation function (``strftime`` vs ``date_trunc``)
and on how ``CAST(... AS INTEGER)`` rounds -- which would silently place rows
in different buckets depending on the backend.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import Float, case, cast, func, select

from app.core.database import DatabaseManager
from app.models.entities import CameraEntity, DetectionEventEntity, PersonEntity


@dataclass(slots=True)
class EventTotals:
    """Headline counts over a single time window.

    Attributes:
        total: Number of detection events recorded in the window.
        recognized: Events matched to an enrolled person.
        unknown: Events whose best candidate was rejected.
        avg_similarity: Mean similarity of recognised events, or ``None``
            when the window holds no scored match.
        watchlist_hits: Recognised events whose person is on the watchlist.
        active_persons: Distinct persons recognised in the window.
        active_cameras: Distinct cameras that produced events.
    """

    total: int = 0
    recognized: int = 0
    unknown: int = 0
    avg_similarity: float | None = None
    watchlist_hits: int = 0
    active_persons: int = 0
    active_cameras: int = 0


@dataclass(slots=True)
class TimelinePoint:
    """One time bucket of recognition activity."""

    bucket: datetime
    recognized: int = 0
    unknown: int = 0
    watchlist: int = 0


@dataclass(slots=True)
class SimilarityBin:
    """One bin of the similarity distribution.

    ``accepted`` and ``rejected`` are kept apart so the threshold can be drawn
    between two visibly distinct populations rather than over a single blended
    histogram.
    """

    lower: float
    upper: float
    accepted: int = 0
    rejected: int = 0


@dataclass(slots=True)
class FaceSizeBin:
    """Recognition quality for one band of detected face sizes.

    ``label`` describes the band in terms of the equivalent square edge in
    pixels (the square root of the bounding-box area), which reads more
    naturally than raw area when reasoning about camera placement.
    """

    label: str
    lower: int
    upper: int | None
    samples: int = 0
    recognized: int = 0
    avg_similarity: float | None = None


@dataclass(slots=True)
class CameraActivity:
    """Per-camera recognition activity over the window."""

    camera_id: str
    camera_name: str
    location: str | None
    total: int = 0
    recognized: int = 0
    unknown: int = 0
    avg_similarity: float | None = None
    avg_face_size: float | None = None
    last_event_at: datetime | None = None


@dataclass(slots=True)
class PersonActivity:
    """Per-person recognition activity over the window."""

    person_id: str
    person_name: str | None
    person_type: str | None
    events: int = 0
    avg_similarity: float | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    cameras: list[str] = field(default_factory=list)


class AnalyticsRepository:
    """Read-only aggregation queries over the detection-event history.

    All public methods open their own database sessions via the injected
    ``DatabaseManager`` and accept an optional ``since`` bound; passing
    ``None`` aggregates over the entire retained history.
    """

    # Similarity values below this are the "no score available" sentinel
    # written by the event repository, not real measurements.
    _SCORED = DetectionEventEntity.confidence_score >= 0

    def __init__(self, database: DatabaseManager) -> None:
        """Initialise the repository with a database manager.

        Args:
            database: The ``DatabaseManager`` used to obtain sessions.
        """
        self.database = database

    @staticmethod
    def _window(stmt, since: datetime | None, until: datetime | None):
        """Apply the optional time bounds of a window to a statement."""
        if since is not None:
            stmt = stmt.where(DetectionEventEntity.detected_at >= since)
        if until is not None:
            stmt = stmt.where(DetectionEventEntity.detected_at < until)
        return stmt

    def totals(self, since: datetime | None = None, until: datetime | None = None) -> EventTotals:
        """Return headline counts for one time window.

        Args:
            since: Inclusive lower bound, or ``None`` for all history.
            until: Exclusive upper bound, or ``None`` for "up to now".

        Returns:
            An ``EventTotals`` describing the window.
        """
        recognized_flag = case((DetectionEventEntity.person_id.is_not(None), 1), else_=0)

        with self.database.session() as session:
            stmt = select(
                func.count(),
                func.coalesce(func.sum(recognized_flag), 0),
                func.count(func.distinct(DetectionEventEntity.person_id)),
                func.count(func.distinct(DetectionEventEntity.camera_id)),
            ).select_from(DetectionEventEntity)
            total, recognized, persons, cameras = session.execute(self._window(stmt, since, until)).one()

            # Averaged over recognised events only: this reports how confident
            # the accepted matches are, not how the rejections scored.
            similarity_stmt = (
                select(func.avg(cast(DetectionEventEntity.confidence_score, Float)))
                .select_from(DetectionEventEntity)
                .where(self._SCORED, DetectionEventEntity.person_id.is_not(None))
            )
            avg_similarity = session.scalar(self._window(similarity_stmt, since, until))

            watchlist_stmt = (
                select(func.count())
                .select_from(DetectionEventEntity)
                .join(PersonEntity, PersonEntity.id == DetectionEventEntity.person_id)
                .where(PersonEntity.person_type == "watchlist")
            )
            watchlist_hits = session.scalar(self._window(watchlist_stmt, since, until)) or 0

        total = int(total or 0)
        recognized = int(recognized or 0)
        return EventTotals(
            total=total,
            recognized=recognized,
            unknown=total - recognized,
            avg_similarity=float(avg_similarity) if avg_similarity is not None else None,
            watchlist_hits=int(watchlist_hits),
            active_persons=int(persons or 0),
            active_cameras=int(cameras or 0),
        )

    def timeline_rows(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[tuple[datetime, bool, bool]]:
        """Return ``(detected_at, recognized, watchlist)`` for the window.

        Bucketing is left to the caller so it can be done in Python, where the
        result does not depend on which database backend is in use.
        """
        with self.database.session() as session:
            stmt = (
                select(
                    DetectionEventEntity.detected_at,
                    DetectionEventEntity.person_id,
                    PersonEntity.person_type,
                )
                .select_from(DetectionEventEntity)
                .outerjoin(PersonEntity, PersonEntity.id == DetectionEventEntity.person_id)
                .order_by(DetectionEventEntity.detected_at)
            )
            rows = session.execute(self._window(stmt, since, until)).all()

        return [
            (detected_at, person_id is not None, person_type == "watchlist")
            for detected_at, person_id, person_type in rows
        ]

    def similarity_rows(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[tuple[float, bool]]:
        """Return ``(similarity, accepted)`` for every scored event in the window.

        Sentinel rows carrying no similarity are excluded, so the caller can
        bin the result directly.
        """
        with self.database.session() as session:
            stmt = (
                select(DetectionEventEntity.confidence_score, DetectionEventEntity.person_id)
                .select_from(DetectionEventEntity)
                .where(self._SCORED)
            )
            rows = session.execute(self._window(stmt, since, until)).all()

        return [(float(score), person_id is not None) for score, person_id in rows]

    def face_size_rows(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[tuple[int, int, bool, float | None]]:
        """Return ``(bbox_w, bbox_h, recognized, similarity)`` for the window.

        The similarity is ``None`` for sentinel rows, letting the caller count
        the detection while leaving it out of the average.
        """
        with self.database.session() as session:
            stmt = select(
                DetectionEventEntity.bbox_w,
                DetectionEventEntity.bbox_h,
                DetectionEventEntity.person_id,
                DetectionEventEntity.confidence_score,
            ).select_from(DetectionEventEntity)
            rows = session.execute(self._window(stmt, since, until)).all()

        return [
            (int(width), int(height), person_id is not None, float(score) if score >= 0 else None)
            for width, height, person_id, score in rows
        ]

    def camera_activity(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[CameraActivity]:
        """Return per-camera activity, busiest camera first.

        Cameras that produced no events in the window are omitted; the caller
        is expected to merge in the full camera roster when it wants to show
        idle cameras too.
        """
        recognized_flag = case((DetectionEventEntity.person_id.is_not(None), 1), else_=0)
        # Averaging the sentinel would understate every camera, so scored rows
        # are summed and counted explicitly instead of using AVG directly.
        scored_flag = case((DetectionEventEntity.confidence_score >= 0, 1), else_=0)
        scored_value = case(
            (DetectionEventEntity.confidence_score >= 0, cast(DetectionEventEntity.confidence_score, Float)),
            else_=0.0,
        )
        # sqrt() is not portable (SQLite exposes it only when built with the
        # math extension), so the mean area is averaged in SQL and the edge is
        # derived in Python. The result is the quadratic mean of the edge.
        face_area = cast(DetectionEventEntity.bbox_w, Float) * cast(DetectionEventEntity.bbox_h, Float)

        with self.database.session() as session:
            stmt = (
                select(
                    DetectionEventEntity.camera_id,
                    CameraEntity.name,
                    CameraEntity.location,
                    func.count(),
                    func.coalesce(func.sum(recognized_flag), 0),
                    func.coalesce(func.sum(scored_flag), 0),
                    func.coalesce(func.sum(scored_value), 0.0),
                    func.avg(face_area),
                    func.max(DetectionEventEntity.detected_at),
                )
                .select_from(DetectionEventEntity)
                .join(CameraEntity, CameraEntity.id == DetectionEventEntity.camera_id)
                .group_by(DetectionEventEntity.camera_id, CameraEntity.name, CameraEntity.location)
                .order_by(func.count().desc())
            )
            rows = session.execute(self._window(stmt, since, until)).all()

        activity: list[CameraActivity] = []
        for camera_id, name, location, total, recognized, scored, scored_sum, avg_area, last_event in rows:
            total = int(total or 0)
            recognized = int(recognized or 0)
            scored = int(scored or 0)
            activity.append(
                CameraActivity(
                    camera_id=str(camera_id),
                    camera_name=name,
                    location=location,
                    total=total,
                    recognized=recognized,
                    unknown=total - recognized,
                    avg_similarity=float(scored_sum) / scored if scored else None,
                    avg_face_size=math.sqrt(float(avg_area)) if avg_area else None,
                    last_event_at=last_event,
                )
            )
        return activity

    def camera_roster(self) -> list[tuple[str, str, str | None]]:
        """Return ``(id, name, location)`` for every non-deleted camera.

        Used to fold cameras that produced no events into per-camera reports,
        so an idle or offline camera shows as a zero row rather than vanishing.
        """
        with self.database.session() as session:
            stmt = (
                select(CameraEntity.id, CameraEntity.name, CameraEntity.location)
                .where(CameraEntity.deleted_at.is_(None))
                .order_by(CameraEntity.created_at, CameraEntity.name)
            )
            return [(str(camera_id), name, location) for camera_id, name, location in session.execute(stmt)]

    def person_activity(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 10,
    ) -> list[PersonActivity]:
        """Return the most frequently recognised persons in the window.

        Only recognised events participate, so unknown detections cannot
        distort the ranking.

        Args:
            since: Inclusive lower bound, or ``None`` for all history.
            until: Exclusive upper bound, or ``None`` for "up to now".
            limit: Maximum number of persons to return.
        """
        scored_flag = case((DetectionEventEntity.confidence_score >= 0, 1), else_=0)
        scored_value = case(
            (DetectionEventEntity.confidence_score >= 0, cast(DetectionEventEntity.confidence_score, Float)),
            else_=0.0,
        )

        with self.database.session() as session:
            stmt = (
                select(
                    DetectionEventEntity.person_id,
                    PersonEntity.full_name,
                    PersonEntity.person_type,
                    func.count(),
                    func.coalesce(func.sum(scored_flag), 0),
                    func.coalesce(func.sum(scored_value), 0.0),
                    func.min(DetectionEventEntity.detected_at),
                    func.max(DetectionEventEntity.detected_at),
                )
                .select_from(DetectionEventEntity)
                .outerjoin(PersonEntity, PersonEntity.id == DetectionEventEntity.person_id)
                .where(DetectionEventEntity.person_id.is_not(None))
                .group_by(DetectionEventEntity.person_id, PersonEntity.full_name, PersonEntity.person_type)
                .order_by(func.count().desc())
                .limit(limit)
            )
            rows = session.execute(self._window(stmt, since, until)).all()

            people = [
                PersonActivity(
                    person_id=str(person_id),
                    person_name=full_name,
                    person_type=person_type,
                    events=int(count or 0),
                    avg_similarity=float(scored_sum) / int(scored) if scored else None,
                    first_seen=first_seen,
                    last_seen=last_seen,
                )
                for person_id, full_name, person_type, count, scored, scored_sum, first_seen, last_seen in rows
            ]

            # Resolve which cameras saw each of the ranked persons. Done as a
            # second pass restricted to the ranked ids so the grouped query
            # above stays a single row per person.
            if people:
                ranked_ids = [uuid.UUID(person.person_id) for person in people]
                camera_stmt = (
                    select(DetectionEventEntity.person_id, CameraEntity.name)
                    .select_from(DetectionEventEntity)
                    .join(CameraEntity, CameraEntity.id == DetectionEventEntity.camera_id)
                    .where(DetectionEventEntity.person_id.in_(ranked_ids))
                    .group_by(DetectionEventEntity.person_id, CameraEntity.name)
                    .order_by(CameraEntity.name)
                )
                seen: dict[str, list[str]] = {}
                for person_id, camera_name in session.execute(self._window(camera_stmt, since, until)):
                    seen.setdefault(str(person_id), []).append(camera_name)
                for person in people:
                    person.cameras = seen.get(person.person_id, [])

        return people
