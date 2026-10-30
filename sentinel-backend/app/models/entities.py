"""
SQLAlchemy ORM entity definitions for the Sentinel database.

Contains all database table mappings:
  - PersonEntity       → ``enrolled_persons`` – people registered for recognition
  - UserEntity         → ``users``            – dashboard operator accounts
  - UserCredentialEntity → ``user_credentials`` – hashed passwords
  - RoleEntity         → ``roles``            – RBAC role definitions
  - UserRoleEntity     → ``user_roles``       – many-to-many user↔role join
  - FaceEmbeddingEntity → ``face_embeddings`` – 512-d face vectors (pgvector / SQLite text)
  - CameraEntity       → ``cameras``          – surveillance camera records
  - TrackEntity        → ``tracks``           – DeepSORT tracking sessions
  - DetectionEventEntity → ``detection_events`` – individual face detection logs
  - TrackDetectionEventEntity → ``track_detection_events`` – track↔event join

Also defines custom SQLAlchemy column types for storing 512-dimensional
embedding vectors in both PostgreSQL (via pgvector) and SQLite (as TEXT).
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TEXT, TypeDecorator, UserDefinedType

from app.core.database import Base


def utcnow() -> datetime:
    """Return the current UTC datetime (used as column default)."""
    return datetime.now(timezone.utc)


# All face embeddings are 512-dimensional (AdaFace IR-SE-50 output).
EMBEDDING_VECTOR_DIMENSION = 512


def validate_embedding_vector(value, *, field_name: str = "embedding_vector") -> list[float]:
    """
    Validate and normalise an embedding vector to a list of 512 finite floats.

    Accepts numpy arrays, lists, tuples, or any iterable of numeric values.
    Raises ``ValueError`` with a descriptive message on bad input.
    """
    if value is None:
        raise ValueError(f"{field_name} is required")

    candidate = value.tolist() if hasattr(value, "tolist") and not isinstance(value, (list, tuple)) else value
    if isinstance(candidate, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a numeric vector of length {EMBEDDING_VECTOR_DIMENSION}, not a string")

    try:
        items = list(candidate)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a numeric vector of length {EMBEDDING_VECTOR_DIMENSION}") from exc

    if len(items) != EMBEDDING_VECTOR_DIMENSION:
        raise ValueError(f"{field_name} must contain exactly {EMBEDDING_VECTOR_DIMENSION} numeric values")

    normalized: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, (str, bytes, bytearray, list, tuple, dict, set)):
            raise ValueError(f"{field_name}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{index}] must be finite")
        normalized.append(number)
    return normalized


class Vector512(UserDefinedType):
    """
    PostgreSQL-only column type backed by the ``pgvector`` extension.

    Stores a 512-dimensional float vector using the ``vector(512)`` type.
    Bind/result processors convert between Python lists and pgvector's
    text representation ``[0.123,0.456,...]``.
    """
    cache_ok = True

    def get_col_spec(self, **kw) -> str:
        return "vector(512)"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            normalized = validate_embedding_vector(value)
            return "[" + ",".join(f"{item:.10f}" for item in normalized) + "]"

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return [float(item) for item in value]
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            if not text:
                return []
            return [float(item) for item in text.split(",")]

        return process


class SQLiteCompatibleVector(TypeDecorator):
    """
    Dual-dialect column type for embedding vectors.

    - **PostgreSQL** → delegates to ``Vector512`` (pgvector).
    - **SQLite**     → stores as a JSON-like TEXT string ``[0.1,0.2,...]``.

    This lets the same ORM model work for both production (Postgres) and
    lightweight testing (SQLite) without schema changes.
    """
    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector512())
        return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        normalized = validate_embedding_vector(value)
        if dialect.name == "postgresql":
            # PostgreSQL delegates final serialization to the wrapped Vector512 type.
            # Returning a pre-stringified "[...]" here causes the inner bind processor
            # to receive a string instead of a numeric vector.
            return normalized
        return "[" + ",".join(f"{item:.10f}" for item in normalized) + "]"

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            return []
        return [float(item) for item in text.split(",")]


# ── ORM Entities ────────────────────────────────────────────────────


class PersonEntity(Base):
    """An enrolled person (staff / VIP / watchlist / visitor) who can be recognised."""
    __tablename__ = "enrolled_persons"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name: Mapped[str] = mapped_column(sa.String(255), nullable=False)                  # Display name
    person_type: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="watchlist")  # staff|vip|watchlist|visitor
    notes: Mapped[str | None] = mapped_column(sa.Text, nullable=True)                       # Free-text notes
    is_active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)       # Soft-disable flag
    enrolled_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    enrolled_by: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), nullable=False)   # User who enrolled
    retention_expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)  # Data retention deadline
    embedding_only_mode: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)  # Store embeddings only (no photos)
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)  # Soft-delete timestamp

    embeddings: Mapped[list["FaceEmbeddingEntity"]] = relationship(back_populates="person")  # 1:N face embeddings


class UserEntity(Base):
    """A dashboard operator account (admin, operator, viewer, etc.)."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(sa.String(255), nullable=False, unique=True)       # Login handle
    email: Mapped[str | None] = mapped_column(sa.String(255), nullable=True, unique=True)    # Optional email
    is_active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)        # Account enabled
    is_locked: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)       # Locked after failed logins
    account_locked_reason: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    failed_login_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)   # Consecutive failures
    locked_until: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)  # Auto-unlock time
    last_login_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)    # Soft-delete
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,  # Auto-updates on row modification
    )

    credential: Mapped["UserCredentialEntity | None"] = relationship(back_populates="user", uselist=False)  # 1:1 password
    user_roles: Mapped[list["UserRoleEntity"]] = relationship(back_populates="user")  # Assigned roles


class UserCredentialEntity(Base):
    """Stores the hashed password for a single user (1:1 with UserEntity)."""
    __tablename__ = "user_credentials"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    password_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    hash_algorithm: Mapped[str] = mapped_column(sa.String(50), nullable=False, default="bcrypt")
    password_changed_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    must_change_password: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)

    user: Mapped[UserEntity] = relationship(back_populates="credential")


class RoleEntity(Base):
    """A named role (e.g. 'admin', 'operator') used for RBAC."""
    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(sa.String(128), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    is_system_role: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)

    user_roles: Mapped[list["UserRoleEntity"]] = relationship(back_populates="role")


class UserRoleEntity(Base):
    """Many-to-many join table between users and roles, with assignment metadata."""
    __tablename__ = "user_roles"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assigned_by: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)

    __table_args__ = (sa.UniqueConstraint("user_id", "role_id"),)

    user: Mapped[UserEntity] = relationship(back_populates="user_roles")
    role: Mapped[RoleEntity] = relationship(back_populates="user_roles")


class FaceEmbeddingEntity(Base):
    """A single 512-d face embedding vector linked to an enrolled person."""
    __tablename__ = "face_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("enrolled_persons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    faiss_index_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, unique=True)  # Position in FAISS index
    embedding_vector: Mapped[list[float]] = mapped_column(SQLiteCompatibleVector(), nullable=False)  # 512-d float vector
    source_type: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="manual_upload")  # manual_upload | auto_capture
    quality_score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)   # Optional image quality metric
    is_primary: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)  # Representative embedding
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)  # Data retention

    person: Mapped[PersonEntity] = relationship(back_populates="embeddings")  # Parent person


class FaceReferenceImageEntity(Base):
    """A viewable uploaded reference image linked to one face embedding."""
    __tablename__ = "face_reference_images"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("enrolled_persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("face_embeddings.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    storage_path: Mapped[str] = mapped_column(sa.String(512), nullable=False, unique=True)
    original_filename: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    width: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    height: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)


class GeneratedReferenceImageEntity(Base):
    """Synthetic descendants; rejected/budgeted images have no embedding."""
    __tablename__ = "generated_reference_images"
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("enrolled_persons.id", ondelete="CASCADE"), index=True)
    original_reference_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("face_reference_images.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("generated_reference_images.id", ondelete="CASCADE"), nullable=True)
    embedding_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("face_embeddings.id", ondelete="SET NULL"), nullable=True)
    storage_path: Mapped[str] = mapped_column(sa.String(512), unique=True)
    original_filename: Mapped[str] = mapped_column(sa.String(255))
    width: Mapped[int] = mapped_column(sa.Integer)
    height: Mapped[int] = mapped_column(sa.Integer)
    image_kind: Mapped[str] = mapped_column(sa.String(24))
    generation: Mapped[dict] = mapped_column(sa.JSON)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)


class ReferenceImageAccessLogEntity(Base):
    """Append-only record of each retained reference image that is opened."""
    __tablename__ = "reference_image_access_logs"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("enrolled_persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reference_image_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("face_reference_images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ip_address: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    generated_image_id: Mapped[str | None] = mapped_column(sa.String(36), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(sa.String(512), nullable=True)
    accessed_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow, index=True)


class CameraEntity(Base):
    """A surveillance camera source (file / RTSP / webcam)."""
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(sa.String(128), nullable=False)                       # Human-readable label
    location: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)              # Physical location description
    rtsp_url: Mapped[str] = mapped_column(sa.Text, nullable=False)                           # Source URI (file path / RTSP / device)
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False, default="inactive")   # active | inactive | error
    config: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True, default=dict)        # Per-camera settings override
    last_active_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    last_frame_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)   # Timestamp of last decoded frame
    stream_fps: Mapped[float | None] = mapped_column(sa.Float, nullable=True)                # Measured output FPS
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)                # Last error if status == 'error'
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)       # Soft-delete
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)


class TrackEntity(Base):
    """
    A DeepSORT tracking session for a single person across video frames.

    Each track belongs to one camera and may optionally be linked to an
    enrolled person once recognition succeeds.
    """
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("enrolled_persons.id", ondelete="SET NULL"),
        nullable=True,              # NULL until the face is recognised
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deepsort_track_id: Mapped[int] = mapped_column(sa.Integer, nullable=False)      # ID assigned by DeepSORT tracker
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False, default="tentative")  # tentative | confirmed | deleted
    started_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, default=utcnow)
    frame_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)  # Frames this track has appeared in
    trajectory: Mapped[list] = mapped_column(sa.JSON, nullable=True, default=list)   # List of bbox positions over time

    # A camera can only have one active track per DeepSORT ID.
    __table_args__ = (sa.UniqueConstraint("camera_id", "deepsort_track_id"),)


class DetectionEventEntity(Base):
    """
    A single face detection / recognition event.

    Partitioned by ``detected_at`` in PostgreSQL for efficient time-range
    queries and retention-based pruning.
    """
    __tablename__ = "detection_events"
    __table_args__ = {"postgresql_partition_by": "RANGE (detected_at)"}

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("enrolled_persons.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    face_embedding: Mapped[list[float] | None] = mapped_column(SQLiteCompatibleVector(), nullable=True)
    confidence_score: Mapped[float] = mapped_column(sa.Float, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(sa.Float, nullable=False)
    bbox_x: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    snapshot_path: Mapped[str | None] = mapped_column(sa.String(512), nullable=True)
    snapshot_width: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    snapshot_height: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), primary_key=True, nullable=False, default=utcnow)
    is_anonymized: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)


class TrackDetectionEventEntity(Base):
    """Join table linking tracks to the detection events they generated."""
    __tablename__ = "track_detection_events"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_id: Mapped[uuid.UUID] = mapped_column(sa.Uuid(as_uuid=True), nullable=False)
    event_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (sa.UniqueConstraint("track_id", "event_id"),)
