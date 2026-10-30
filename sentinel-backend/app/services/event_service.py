"""
Recognition event service for publishing and broadcasting face recognition events.

This module persists recognition events to the database and broadcasts them
in real-time to connected dashboard clients via WebSocket. It handles:
    - Paginated event listing for the event log.
    - Publishing new recognition events (persist + broadcast).
    - Severity classification based on recognition status and person type:
        - "alert": recognised person on the watchlist.
        - "ok": recognised known person (not on watchlist).
        - "warn": unknown/unrecognised person.
    - Async-safe broadcasting with fallback to threadsafe mode when no
      asyncio event loop is running.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timezone

from app.core.websocket_manager import WebSocketManager
from app.models.schemas import RecognitionEventRecord
from app.repositories.recognition_event_repository import EventRow, RecognitionEventRepository
from app.services.recognition_snapshot_service import RecognitionSnapshotService

# Service for managing and broadcasting face recognition events.
# Combines database persistence with real-time WebSocket notifications to keep the dashboard event log updated as recognition events occur.
class EventService:

    def __init__(
        self,
        repository: RecognitionEventRepository,
        websocket_manager: WebSocketManager,
        snapshot_service: RecognitionSnapshotService | None = None,
    ) -> None:
        self.repository = repository
        self.websocket_manager = websocket_manager
        self.snapshot_service = snapshot_service
        self.logger = logging.getLogger("sentinel.events")

# Retrieve a paginated list of recognition events, including total count for pagination and serialised event records for display in the dashboard event log.
    def list_events(self, limit: int = 50, offset: int = 0) -> dict:
        total, rows = self.repository.list(limit=limit, offset=offset)
        events = [self._serialize_event(row) for row in rows]
        return {"total": total, "events": [event.model_dump() for event in events]}

    def list_recognition_records(self, limit: int = 50, offset: int = 0) -> dict:
        """Return successful recognition events, newest first."""
        total, rows = self.repository.list(limit=limit, offset=offset, recognized_only=True)
        events = [self._serialize_event(row) for row in rows]
        return {"total": total, "records": [event.model_dump() for event in events]}

# Delete all recognition events from the database, returning the number of events that were deleted. This can be used for clearing the event log from the dashboard.
    def clear_events(self) -> int:
        snapshot_paths = self.repository.snapshot_paths()
        total = self.repository.clear()
        if self.snapshot_service is not None:
            for path in snapshot_paths:
                self.snapshot_service.delete(path)
        return total

    def get_snapshot_path(self, event_id: str):
        """Resolve one protected evidence image for the records page."""
        if self.snapshot_service is None:
            return None
        path = self.repository.snapshot_path_for_event(event_id)
        return self.snapshot_service.resolve(path) if path else None

# Publish a new recognition event by persisting it to the database and broadcasting it to all connected WebSocket clients. 
# This is the main method called by the recognition pipeline when a face is detected or recognised.
    def publish_recognition_event(
        self,
        *,
        camera_id: str,
        camera_name: str,
        track_id: int,
        status: str,
        event_type: str,
        similarity: float | None,
        bbox: list[int] | None,
        person_id: str | None = None,
        person_name: str | None = None,
        person_type: str | None = None,
        detection_confidence: float | None = None,
        snapshot_frame=None,
    ) -> RecognitionEventRecord:
        snapshot = None
        if person_id and snapshot_frame is not None and bbox is not None and self.snapshot_service is not None:
            snapshot = self.snapshot_service.save(
                snapshot_frame,
                bbox,
                person_name=person_name,
                person_type=person_type,
                similarity=similarity,
            )
        try:
            row = self.repository.add(
                person_id=person_id,
                camera_id=camera_id,
                track_id=track_id,
                similarity=similarity,
                detection_confidence=detection_confidence,
                status=status,
                event_type=event_type,
                bbox=bbox,
                snapshot_path=snapshot.path if snapshot else None,
                snapshot_width=snapshot.width if snapshot else None,
                snapshot_height=snapshot.height if snapshot else None,
            )
        except Exception:
            if snapshot is not None:
                self.snapshot_service.delete(snapshot.path)
            raise
        # Construct the full event row with caller-provided name overrides
        # (the repository may not have joined the camera/person names)
        payload = self._serialize_event(
            EventRow(
                id=row.id,
                camera_id=row.camera_id,
                camera_name=camera_name or row.camera_name,
                person_id=person_id or row.person_id,
                person_name=person_name or row.person_name,
                person_type=row.person_type,
                similarity=row.similarity,
                track_id=row.track_id,
                status=row.status,
                bbox=row.bbox,
                timestamp=row.timestamp,
                event_type=row.event_type,
                snapshot_path=row.snapshot_path,
                snapshot_width=row.snapshot_width,
                snapshot_height=row.snapshot_height,
            )
        )
        # Broadcast to connected WebSocket dashboard clients
        self._schedule_broadcast(payload.model_dump(mode="json"))
        return payload

    @staticmethod
    # Helper method to convert an EventRow database record to a RecognitionEventRecord schema, applying severity classification and formatting for display in the dashboard event log.
    def _serialize_event(row: EventRow) -> RecognitionEventRecord:
        
        # Severity classification based on recognition status and person type
        if row.status == "recognized" and row.person_type == "watchlist":
            severity = "alert"
        elif row.status == "recognized":
            severity = "ok"
        else:
            severity = "warn"

        # Format confidence for display: "0.923" or "--" if not available
        confidence_text = f"{row.similarity:.3f}" if row.similarity is not None else "--"

        # Ensure timestamp is timezone-aware (default to UTC if naive)
        timestamp = row.timestamp if row.timestamp.tzinfo else row.timestamp.replace(tzinfo=timezone.utc)
        # Convert to local timezone for display
        local_timestamp = timestamp.astimezone()

        return RecognitionEventRecord(
            id=row.id,
            severity=severity,
            type=row.event_type,
            camera=row.camera_name,
            time=local_timestamp.strftime("%H:%M:%S"),
            conf=confidence_text,
            track_id=row.track_id,
            bbox=row.bbox,
            person_id=row.person_id,
            person_name=row.person_name,
            person_type=row.person_type,
            similarity=row.similarity,
            status=row.status,
            timestamp=timestamp,
            camera_id=row.camera_id,
            event_type=row.event_type,
            has_snapshot=bool(row.snapshot_path),
        )

# Schedule a WebSocket broadcast of the event payload to all clients, attempting to use the current asyncio event loop for non-blocking broadcast, 
# with a fallback to the threadsafe variant if no loop is running (e.g., called from a sync thread).
    def _schedule_broadcast(self, payload: dict) -> None:
        try:
            # Try async broadcast if an event loop is running
            loop = asyncio.get_running_loop()
            loop.create_task(self.websocket_manager.broadcast_json(payload))
        except RuntimeError:
            # No running event loop -- use the threadsafe fallback
            self.websocket_manager.broadcast_json_threadsafe(payload)
