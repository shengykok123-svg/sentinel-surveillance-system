"""
Pydantic request / response schemas for the Sentinel REST API.

These models define the JSON shape that the frontend sends and receives.
They are intentionally decoupled from the SQLAlchemy entities so that
the API surface can evolve independently of the database schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Camera Schemas ──────────────────────────────────────────────────


class CameraBase(BaseModel):
    """Shared fields for camera create / read operations."""
    name: str                                              # Human-readable camera label
    source_type: Literal["file", "rtsp", "webcam"]         # Input source kind
    source_uri: str                                        # File path, RTSP URL, or device index
    is_active: bool = True                                 # Whether the camera should stream


class CameraCreate(CameraBase):
    """Payload for registering a new camera."""
    id: str                                                # Client-supplied unique camera ID


class CameraUpdate(BaseModel):
    """Partial-update payload – only non-None fields are applied."""
    name: str | None = None
    source_type: Literal["file", "rtsp", "webcam"] | None = None
    source_uri: str | None = None
    is_active: bool | None = None


class CameraRead(CameraBase):
    """Camera details returned to the dashboard, including live metrics."""
    model_config = ConfigDict(from_attributes=True)        # Allow from ORM entity

    id: str
    online: bool = False                                   # Is the pipeline currently streaming?
    active_tracks: int = 0                                 # Number of active DeepSORT tracks
    fps: int = 0                                           # Current output FPS
    face_count: int = 0                                    # Faces detected in the latest frame

# ── Person (POI) Schemas ────────────────────────────────────────────


class PersonRead(BaseModel):
    """Full read-model for an enrolled person (Person of Interest)."""
    id: str
    name: str
    person_type: Literal["staff", "vip", "watchlist", "visitor"] = "watchlist"
    notes: str | None = None
    is_active: bool = True
    embedding_only_mode: bool = False                      # True = no face images stored
    enrolled_at: datetime | None = None
    retention_expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict) # Additional key-value pairs
    total_embeddings: int = 0                              # How many embedding vectors exist
    last_seen_camera: str | None = None                    # Camera where last detected
    last_seen_time: datetime | None = None
    status: Literal["active", "inactive"] = "inactive"     # Derived from recent detections


class PersonUpdate(BaseModel):
    """Partial-update payload for an enrolled person."""
    name: str | None = None
    person_type: Literal["staff", "vip", "watchlist", "visitor"] | None = None
    notes: str | None = None
    is_active: bool | None = None
    embedding_only_mode: bool | None = None
    retention_expires_at: datetime | None = None

# ── Enrollment Schemas ──────────────────────────────────────────────


class EnrollmentResponse(BaseModel):
    """Returned after successfully enrolling a person with face images."""
    person_id: str
    full_name: str
    embeddings_count: int                                  # Number of embeddings created
    status: Literal["success"]
    rejected_images: list[str] = Field(default_factory=list)  # Filenames that failed processing
    enhancement: dict = Field(default_factory=dict)

# ── Face Search Schemas ─────────────────────────────────────────────


class SearchMatch(BaseModel):
    """A single match from a FAISS similarity search."""
    person_id: str
    name: str
    similarity: float                                      # Cosine similarity score (0–1)
    embedding_id: str
    camera: str | None = None                              # Camera where the match occurred
    timestamp: datetime | None = None


class SearchResponse(BaseModel):
    """Full response for a face search query."""
    query_bbox: list[int]                                  # [x, y, w, h] of the query face
    results: list[SearchMatch]                             # Ranked matches
    search_ms: float                                       # FAISS query latency in milliseconds

# ── Embedding Stats ─────────────────────────────────────────────────


class EmbeddingStats(BaseModel):
    """Summary statistics about the FAISS index and stored embeddings."""
    total_vectors: int                                     # Vectors currently in FAISS
    total_persons: int                                     # Distinct enrolled persons
    embedding_dim: int                                     # Dimension (always 512)
    index_size_mb: float                                   # FAISS index file size on disk
    last_updated: datetime | None                          # Last time the index was rebuilt

# ── Active Track Schemas ────────────────────────────────────────────


class ActiveTrackRead(BaseModel):
    recognition_phase: str = 'collecting'
    recognition_attempts: int = 0
    best_similarity: float | None = None
    quality_checks: int = 0
    quality_rejections: int = 0
    last_quality_score: float | None = None
    """Live tracking info for a single face currently visible on a camera."""
    camera_id: str
    camera_name: str
    track_id: int                                          # DeepSORT track ID
    bbox: list[int]                                        # [x, y, w, h] bounding box
    frame_width: int                                       # Source frame width (for normalising bbox)
    frame_height: int                                      # Source frame height
    status: Literal["unknown", "recognized"]               # Whether the face has been identified
    person_id: str | None = None                           # Matched person (if recognised)
    person_name: str | None = None
    person_type: str | None = None
    similarity: float | None = None                        # Cosine similarity of best match
    last_seen: datetime                                    # Timestamp of last frame appearance
    last_recognition: datetime | None = None               # Timestamp of last recognition attempt

# ── Recognition Event Schemas ───────────────────────────────────────


class RecognitionEventRecord(BaseModel):
    """A detection / recognition event pushed to the dashboard event log."""
    id: str                                                # Unique event ID
    severity: Literal["ok", "warn", "alert"]               # Visual severity indicator
    type: str                                              # Event type label
    camera: str                                            # Camera display name
    time: str                                              # Human-readable timestamp string
    conf: str                                              # Detection confidence as display string
    track_id: int | None = None                            # Associated DeepSORT track
    bbox: list[int] | None = None                          # Bounding box coordinates
    person_id: str | None = None                           # Matched person ID (if any)
    person_name: str | None = None
    person_type: str | None = None
    similarity: float | None = None                        # Recognition similarity score
    status: Literal["unknown", "recognized"]               # Recognition outcome
    timestamp: datetime                                    # Machine-readable UTC timestamp
    camera_id: str                                         # Camera ID for linking
    event_type: str                                        # Canonical event type string
    has_snapshot: bool = False                             # Whether an evidence image is available


# ── Analytics ────────────────────────────────────────────────────────


class AnalyticsWindow(BaseModel):
    """The time window a report covers."""
    range: str                                             # Requested range key ("24h", "7d", "30d", "all")
    since: datetime | None                                 # Inclusive lower bound; None for all history
    until: datetime                                        # Exclusive upper bound (report generation time)
    tz_offset: int = 0                                     # Minutes east of UTC used for bucket boundaries


class AnalyticsSummary(BaseModel):
    """Headline recognition counters for the Analytics KPI row."""
    window: AnalyticsWindow
    total_events: int                                      # Detections recorded in the window
    recognized: int                                        # Matched to an enrolled person
    unknown: int                                           # Best candidate rejected
    recognition_rate: float | None                         # recognized / total; None when no events
    avg_similarity: float | None                           # Mean similarity of accepted matches
    watchlist_hits: int                                    # Recognitions of watchlist persons
    active_persons: int                                    # Distinct persons recognised
    active_cameras: int                                    # Distinct cameras producing events
    # Fractional change against the preceding window of equal length. Keys are
    # absent when the previous window holds nothing to compare against.
    delta_vs_previous: dict[str, float] = Field(default_factory=dict)


class TimelineBucket(BaseModel):
    """One time bucket of recognition activity."""
    ts: datetime                                           # Bucket start (UTC)
    recognized: int = 0
    unknown: int = 0
    watchlist: int = 0


class HeatmapCell(BaseModel):
    """Activity for one weekday / hour cell, in the requested timezone."""
    weekday: int                                           # 0 = Monday .. 6 = Sunday
    hour: int                                              # 0..23
    count: int = 0


class AnalyticsTimeline(BaseModel):
    """Recognition activity over time, plus a weekday/hour heatmap."""
    window: AnalyticsWindow
    bucket: Literal["hour", "day"]                         # Bucket width
    buckets: list[TimelineBucket]                          # Gap-filled, chronological
    heatmap: list[HeatmapCell]                             # Always 7 x 24 cells


class SimilarityBin(BaseModel):
    """One bin of the similarity distribution."""
    lower: float                                           # Inclusive lower edge
    upper: float                                           # Exclusive upper edge (inclusive at 1.0)
    accepted: int = 0                                      # Events that cleared the threshold
    rejected: int = 0                                      # Events whose candidate was rejected


class SimilarityHistogram(BaseModel):
    """Distribution of match scores either side of the recognition threshold."""
    window: AnalyticsWindow
    threshold: float                                       # Configured recognition threshold
    total_scored: int                                      # Events carrying a real similarity
    bins: list[SimilarityBin]


class FaceSizeBin(BaseModel):
    """Recognition quality for one band of detected face sizes."""
    label: str                                             # Human-readable band ("64-96px")
    lower: int                                             # Inclusive lower edge, in pixels
    upper: int | None                                      # Exclusive upper edge; None for the open band
    samples: int = 0                                       # Detections in this band
    recognized: int = 0                                    # Of those, matched to a person
    recognition_rate: float | None = None                  # recognized / samples
    avg_similarity: float | None = None                    # Mean of the scored detections


class FaceSizeQuality(BaseModel):
    """Recognition outcome as a function of detected face size."""
    window: AnalyticsWindow
    bins: list[FaceSizeBin]


class CameraAnalytics(BaseModel):
    """Per-camera recognition activity."""
    camera_id: str
    camera_name: str
    location: str | None = None
    total_events: int = 0
    recognized: int = 0
    unknown: int = 0
    recognition_rate: float | None = None
    avg_similarity: float | None = None
    avg_face_size: float | None = None                     # Quadratic mean of the bbox edge, in pixels
    last_event_at: datetime | None = None


class CameraAnalyticsReport(BaseModel):
    """Per-camera comparison, busiest first, including idle cameras."""
    window: AnalyticsWindow
    cameras: list[CameraAnalytics]


class PersonAnalytics(BaseModel):
    """Per-person recognition activity."""
    person_id: str
    person_name: str | None = None
    person_type: str | None = None
    events: int = 0
    avg_similarity: float | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    cameras: list[str] = Field(default_factory=list)       # Camera names that saw this person


class PersonAnalyticsReport(BaseModel):
    """Most frequently recognised persons, plus the enrolment type mix."""
    window: AnalyticsWindow
    persons: list[PersonAnalytics]
    type_breakdown: dict[str, int] = Field(default_factory=dict)  # person_type -> recognition count
