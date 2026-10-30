"""
Recognition analytics service for the Sentinel dashboard.

Turns the raw aggregates produced by :class:`AnalyticsRepository` into the
reports the Analytics page renders: a KPI summary with period-over-period
deltas, an activity timeline and weekday/hour heatmap, the similarity
distribution either side of the recognition threshold, recognition quality as
a function of face size, and per-camera / per-person breakdowns.

Two things are handled here rather than in SQL:

* **Bucketing.** Hour and day boundaries are computed in Python so the result
  does not depend on whether the deployment runs SQLite or PostgreSQL.
* **Timezone.** Events are stored in UTC, but "activity by hour of the day"
  is only meaningful in the viewer's local time. Callers pass ``tz_offset``
  in minutes east of UTC; bucket boundaries and heatmap cells are aligned to
  local time while the emitted timestamps stay UTC.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from app.core.config import Settings
from app.models.schemas import (
    AnalyticsSummary,
    AnalyticsTimeline,
    AnalyticsWindow,
    CameraAnalytics,
    CameraAnalyticsReport,
    FaceSizeBin,
    FaceSizeQuality,
    HeatmapCell,
    PersonAnalytics,
    PersonAnalyticsReport,
    SimilarityBin,
    SimilarityHistogram,
    TimelineBucket,
)
from app.repositories.analytics_repository import AnalyticsRepository


# Supported range keys and the span each covers. ``all`` has no lower bound.
RANGE_SPANS: dict[str, timedelta | None] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}

# Face-size bands, as the equivalent square edge in pixels. The open final band
# collects everything above the last boundary.
FACE_SIZE_EDGES: list[int] = [0, 32, 64, 96, 128]


class AnalyticsService:
    """Builds recognition analytics reports from stored detection events."""

    def __init__(self, repository: AnalyticsRepository, settings: Settings) -> None:
        """Initialise the service.

        Args:
            repository: Aggregation queries over the detection-event history.
            settings: Application configuration, read for the recognition
                threshold that the similarity histogram is drawn against.
        """
        self.repository = repository
        self.settings = settings
        self.logger = logging.getLogger("sentinel.analytics")

    # ── Window helpers ───────────────────────────────────────────────

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        """Normalise a stored timestamp to an aware UTC datetime.

        SQLite hands back naive datetimes where PostgreSQL returns aware ones,
        so every timestamp is coerced before arithmetic.
        """
        if value is None:
            return None
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    def _resolve_window(self, range_key: str, tz_offset: int) -> tuple[datetime | None, datetime, AnalyticsWindow]:
        """Return the ``(since, until, window)`` triple for a range key.

        Args:
            range_key: One of :data:`RANGE_SPANS`.
            tz_offset: Minutes east of UTC used for bucket alignment.

        Returns:
            The inclusive lower bound (``None`` for all history), the exclusive
            upper bound, and the ``AnalyticsWindow`` describing both.

        Raises:
            ValueError: If *range_key* is not a supported range.
        """
        if range_key not in RANGE_SPANS:
            raise ValueError(f"Unsupported range '{range_key}'")

        until = datetime.now(timezone.utc)
        span = RANGE_SPANS[range_key]
        since = until - span if span is not None else None
        window = AnalyticsWindow(range=range_key, since=since, until=until, tz_offset=tz_offset)
        return since, until, window

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        """Return ``numerator / denominator``, or ``None`` for an empty set."""
        return numerator / denominator if denominator else None

    @staticmethod
    def _fractional_change(current: float | None, previous: float | None) -> float | None:
        """Return the fractional change from *previous* to *current*.

        ``None`` when there is no meaningful baseline -- either value missing,
        or a previous value of zero, which would make the change infinite.
        """
        if current is None or previous is None or previous == 0:
            return None
        return (current - previous) / previous

    def _floor_to_bucket(self, moment: datetime, bucket: str, tz_offset: int) -> datetime:
        """Round a UTC timestamp down to its bucket start, aligned to local time.

        The returned value is still UTC; only the boundary it snaps to is
        chosen in the caller's timezone.
        """
        shift = timedelta(minutes=tz_offset)
        local = moment + shift
        if bucket == "day":
            local = local.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            local = local.replace(minute=0, second=0, microsecond=0)
        return local - shift

    # ── Reports ──────────────────────────────────────────────────────

    def summary(self, range_key: str = "24h", tz_offset: int = 0) -> AnalyticsSummary:
        """Return headline counters for the window, with period-over-period deltas.

        The comparison window is the equally long span immediately preceding
        the current one. ``all`` has no such baseline, so its deltas are empty.
        """
        since, until, window = self._resolve_window(range_key, tz_offset)
        totals = self.repository.totals(since=since, until=until)

        deltas: dict[str, float] = {}
        span = RANGE_SPANS[range_key]
        if span is not None and since is not None:
            previous = self.repository.totals(since=since - span, until=since)
            candidates = {
                "total_events": (totals.total, previous.total),
                "recognized": (totals.recognized, previous.recognized),
                "unknown": (totals.unknown, previous.unknown),
                "watchlist_hits": (totals.watchlist_hits, previous.watchlist_hits),
                "recognition_rate": (
                    self._ratio(totals.recognized, totals.total),
                    self._ratio(previous.recognized, previous.total),
                ),
                "avg_similarity": (totals.avg_similarity, previous.avg_similarity),
            }
            for key, (current_value, previous_value) in candidates.items():
                change = self._fractional_change(current_value, previous_value)
                if change is not None:
                    deltas[key] = change

        return AnalyticsSummary(
            window=window,
            total_events=totals.total,
            recognized=totals.recognized,
            unknown=totals.unknown,
            recognition_rate=self._ratio(totals.recognized, totals.total),
            avg_similarity=totals.avg_similarity,
            watchlist_hits=totals.watchlist_hits,
            active_persons=totals.active_persons,
            active_cameras=totals.active_cameras,
            delta_vs_previous=deltas,
        )

    def timeline(self, range_key: str = "7d", bucket: str = "hour", tz_offset: int = 0) -> AnalyticsTimeline:
        """Return gap-filled activity buckets plus a weekday/hour heatmap.

        Empty buckets are emitted explicitly so the chart shows quiet periods
        as gaps at zero rather than joining across them.
        """
        if bucket not in ("hour", "day"):
            raise ValueError(f"Unsupported bucket '{bucket}'")

        since, until, window = self._resolve_window(range_key, tz_offset)
        rows = self.repository.timeline_rows(since=since, until=until)

        tallies: dict[datetime, dict[str, int]] = defaultdict(lambda: {"recognized": 0, "unknown": 0, "watchlist": 0})
        heatmap: dict[tuple[int, int], int] = defaultdict(int)
        shift = timedelta(minutes=tz_offset)

        for detected_at, recognized, watchlist in rows:
            moment = self._as_utc(detected_at)
            if moment is None:
                continue
            slot = tallies[self._floor_to_bucket(moment, bucket, tz_offset)]
            slot["recognized" if recognized else "unknown"] += 1
            if watchlist:
                slot["watchlist"] += 1
            local = moment + shift
            heatmap[(local.weekday(), local.hour)] += 1

        # Anchor an all-history report at its first event; a bounded range
        # starts at its own lower bound so quiet leading periods still show.
        start = since
        if start is None:
            observed = [self._as_utc(row[0]) for row in rows]
            start = min((moment for moment in observed if moment is not None), default=until)

        step = timedelta(days=1) if bucket == "day" else timedelta(hours=1)
        cursor = self._floor_to_bucket(start, bucket, tz_offset)
        last = self._floor_to_bucket(until, bucket, tz_offset)
        buckets: list[TimelineBucket] = []
        while cursor <= last:
            slot = tallies.get(cursor)
            buckets.append(
                TimelineBucket(
                    ts=cursor,
                    recognized=slot["recognized"] if slot else 0,
                    unknown=slot["unknown"] if slot else 0,
                    watchlist=slot["watchlist"] if slot else 0,
                )
            )
            cursor += step

        return AnalyticsTimeline(
            window=window,
            bucket=bucket,
            buckets=buckets,
            # Always a full 7 x 24 grid so the frontend can render it without
            # having to fill in the cells that saw no activity.
            heatmap=[
                HeatmapCell(weekday=weekday, hour=hour, count=heatmap.get((weekday, hour), 0))
                for weekday in range(7)
                for hour in range(24)
            ],
        )

    def similarity_histogram(self, range_key: str = "7d", bins: int = 20, tz_offset: int = 0) -> SimilarityHistogram:
        """Return the match-score distribution, split by threshold outcome.

        Accepted and rejected scores are counted separately so the threshold
        can be drawn between two distinct populations. Only events carrying a
        real score take part; sentinel rows are already excluded upstream.
        """
        if bins < 1:
            raise ValueError("bins must be at least 1")

        since, until, window = self._resolve_window(range_key, tz_offset)
        rows = self.repository.similarity_rows(since=since, until=until)

        accepted = [0] * bins
        rejected = [0] * bins
        for score, was_accepted in rows:
            # Clamp so a score of exactly 1.0 lands in the final bin rather
            # than one past the end.
            index = min(int(max(score, 0.0) * bins), bins - 1)
            if was_accepted:
                accepted[index] += 1
            else:
                rejected[index] += 1

        width = 1.0 / bins
        return SimilarityHistogram(
            window=window,
            threshold=self.settings.recognition_threshold,
            total_scored=len(rows),
            bins=[
                SimilarityBin(
                    lower=round(index * width, 6),
                    upper=round((index + 1) * width, 6),
                    accepted=accepted[index],
                    rejected=rejected[index],
                )
                for index in range(bins)
            ],
        )

    def face_size_quality(self, range_key: str = "7d", tz_offset: int = 0) -> FaceSizeQuality:
        """Return recognition outcome as a function of detected face size.

        Faces are grouped by the equivalent square edge (the square root of the
        bounding-box area), which relates directly to how far a subject can be
        from a camera and still be identified.
        """
        since, until, window = self._resolve_window(range_key, tz_offset)
        rows = self.repository.face_size_rows(since=since, until=until)

        samples = [0] * len(FACE_SIZE_EDGES)
        recognized = [0] * len(FACE_SIZE_EDGES)
        scored_sum = [0.0] * len(FACE_SIZE_EDGES)
        scored_count = [0] * len(FACE_SIZE_EDGES)

        for width, height, was_recognized, similarity in rows:
            edge = math.sqrt(max(width, 0) * max(height, 0))
            index = 0
            for position, boundary in enumerate(FACE_SIZE_EDGES):
                if edge >= boundary:
                    index = position
            samples[index] += 1
            if was_recognized:
                recognized[index] += 1
            if similarity is not None:
                scored_sum[index] += similarity
                scored_count[index] += 1

        result: list[FaceSizeBin] = []
        for index, lower in enumerate(FACE_SIZE_EDGES):
            upper = FACE_SIZE_EDGES[index + 1] if index + 1 < len(FACE_SIZE_EDGES) else None
            label = f"<{upper}px" if index == 0 else f"{lower}px+" if upper is None else f"{lower}-{upper}px"
            result.append(
                FaceSizeBin(
                    label=label,
                    lower=lower,
                    upper=upper,
                    samples=samples[index],
                    recognized=recognized[index],
                    recognition_rate=self._ratio(recognized[index], samples[index]),
                    avg_similarity=(scored_sum[index] / scored_count[index]) if scored_count[index] else None,
                )
            )
        return FaceSizeQuality(window=window, bins=result)

    def cameras(self, range_key: str = "7d", tz_offset: int = 0) -> CameraAnalyticsReport:
        """Return per-camera activity, busiest first, including idle cameras.

        A camera with no events in the window is reported as a zero row so an
        offline or misaimed camera is visible by its silence.
        """
        since, until, window = self._resolve_window(range_key, tz_offset)
        activity = self.repository.camera_activity(since=since, until=until)
        active_ids = {item.camera_id for item in activity}

        cameras = [
            CameraAnalytics(
                camera_id=item.camera_id,
                camera_name=item.camera_name,
                location=item.location,
                total_events=item.total,
                recognized=item.recognized,
                unknown=item.unknown,
                recognition_rate=self._ratio(item.recognized, item.total),
                avg_similarity=item.avg_similarity,
                avg_face_size=item.avg_face_size,
                last_event_at=self._as_utc(item.last_event_at),
            )
            for item in activity
        ]
        cameras.extend(
            CameraAnalytics(camera_id=camera_id, camera_name=name, location=location)
            for camera_id, name, location in self.repository.camera_roster()
            if camera_id not in active_ids
        )
        return CameraAnalyticsReport(window=window, cameras=cameras)

    def top_persons(self, range_key: str = "7d", limit: int = 10, tz_offset: int = 0) -> PersonAnalyticsReport:
        """Return the most frequently recognised persons and the type mix.

        ``type_breakdown`` counts recognitions per ``person_type`` across the
        ranked persons, which is what the enrolment-mix pie chart plots.
        """
        since, until, window = self._resolve_window(range_key, tz_offset)
        activity = self.repository.person_activity(since=since, until=until, limit=limit)

        breakdown: dict[str, int] = defaultdict(int)
        persons: list[PersonAnalytics] = []
        for item in activity:
            breakdown[item.person_type or "unknown"] += item.events
            persons.append(
                PersonAnalytics(
                    person_id=item.person_id,
                    person_name=item.person_name,
                    person_type=item.person_type,
                    events=item.events,
                    avg_similarity=item.avg_similarity,
                    first_seen=self._as_utc(item.first_seen),
                    last_seen=self._as_utc(item.last_seen),
                    cameras=item.cameras,
                )
            )

        return PersonAnalyticsReport(window=window, persons=persons, type_breakdown=dict(breakdown))
