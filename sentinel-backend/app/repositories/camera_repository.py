"""
Camera repository module for Sentinel.

Provides data-access operations (CRUD) for camera entities, including
listing, creating, updating, soft-deleting, and seeding default cameras
from application configuration.  All cameras are identified by UUIDs;
external/human-readable IDs are converted to deterministic UUID5 values
so that the same external ID always maps to the same database key.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.database import DatabaseManager
from app.models.entities import CameraEntity


# Deterministic UUID5 namespace derived from the DNS namespace.
# Used by ``normalize_uuid`` to generate reproducible camera IDs
# from arbitrary string identifiers (e.g. "cam-lobby-01").
CAMERA_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "sentinel-camera")


def normalize_uuid(value: str) -> uuid.UUID:
    """Convert an arbitrary string identifier into a UUID.

    If *value* is already a valid UUID string it is returned as-is.
    Otherwise a deterministic UUID5 is generated using
    ``CAMERA_NAMESPACE`` so the same input always produces the same
    UUID.

    Args:
        value: A string that may be a canonical UUID or any external
            camera identifier.

    Returns:
        A ``uuid.UUID`` instance suitable for use as a database
        primary key.
    """
    try:
        return uuid.UUID(str(value))
    except ValueError:
        # Not a valid UUID string -- derive a deterministic UUID5
        return uuid.uuid5(CAMERA_NAMESPACE, str(value))


class CameraRepository:
    """Repository handling persistence operations for camera entities.

    All public methods open their own database sessions via the
    injected ``DatabaseManager``.  Cameras use a **soft-delete**
    pattern: a non-null ``deleted_at`` timestamp marks a camera as
    removed without physically deleting its row.
    """

    def __init__(self, database: DatabaseManager) -> None:
        """Initialise the repository with a database manager.

        Args:
            database: The ``DatabaseManager`` instance used to obtain
                database sessions.
        """
        self.database = database

    def list_all(self) -> list[CameraEntity]:
        """Return every non-deleted camera, ordered by creation date and name.

        Returns:
            A list of ``CameraEntity`` objects that have not been
            soft-deleted.
        """
        with self.database.session() as session:
            stmt = (
                select(CameraEntity)
                .where(CameraEntity.deleted_at.is_(None))
                .order_by(CameraEntity.created_at, CameraEntity.name)
            )
            return list(session.scalars(stmt))

    def list_active(self) -> list[CameraEntity]:
        """Return only cameras whose status is ``"active"`` and that are not deleted.

        Returns:
            A list of active, non-deleted ``CameraEntity`` objects
            ordered by creation date and name.
        """
        with self.database.session() as session:
            stmt = (
                select(CameraEntity)
                .where(CameraEntity.deleted_at.is_(None), CameraEntity.status == "active")
                .order_by(CameraEntity.created_at, CameraEntity.name)
            )
            return list(session.scalars(stmt))

    def get(self, camera_id: str) -> CameraEntity | None:
        """Fetch a single camera by its identifier.

        The identifier is normalised to a UUID before lookup.  If the
        camera does not exist or has been soft-deleted, ``None`` is
        returned.

        Args:
            camera_id: A UUID string or external camera identifier.

        Returns:
            The matching ``CameraEntity``, or ``None`` if not found or
            deleted.
        """
        with self.database.session() as session:
            entity = session.get(CameraEntity, normalize_uuid(camera_id))
            if entity is None or entity.deleted_at is not None:
                return None
            return entity

    def create(
        self,
        *,
        id: str,
        name: str,
        source_type: str,
        source_uri: str,
        is_active: bool,
    ) -> CameraEntity:
        """Create and persist a new camera entity.

        The camera ID is normalised to a deterministic UUID.  Source
        metadata (type, URI, original external ID) is stored in the
        ``config`` JSON column for future reference.

        Args:
            id: An external identifier for the camera (will be
                converted to a UUID5).
            name: Human-readable display name.
            source_type: The kind of video source (e.g. ``"rtsp"``,
                ``"file"``).
            source_uri: URI or file path pointing to the video source.
            is_active: Whether the camera should start in the active
                state.

        Returns:
            The newly created and flushed ``CameraEntity``.
        """
        entity = CameraEntity(
            id=normalize_uuid(id),
            name=name,
            rtsp_url=source_uri,
            status="active" if is_active else "inactive",
            config={
                "source_type": source_type,
                "source_uri": source_uri,
                "external_id": id,
            },
            # Record the initial activation timestamp only when active
            last_active_at=datetime.now(timezone.utc) if is_active else None,
        )
        with self.database.session() as session:
            session.add(entity)
            session.flush()
            session.refresh(entity)
            return entity

    def update(self, camera_id: str, **changes) -> CameraEntity | None:
        """Apply partial updates to an existing camera.

        Only the keys present in *changes* are modified.  Recognised
        keys are ``name``, ``source_uri``, ``source_type``, and
        ``is_active``.  The ``config`` JSON blob is kept in sync with
        the relational columns so both representations stay consistent.

        Args:
            camera_id: The UUID or external identifier of the camera to
                update.
            **changes: Keyword arguments whose keys correspond to the
                updatable fields.

        Returns:
            The updated ``CameraEntity``, or ``None`` if the camera was
            not found or has been soft-deleted.
        """
        with self.database.session() as session:
            entity = session.get(CameraEntity, normalize_uuid(camera_id))
            if entity is None or entity.deleted_at is not None:
                return None

            # Work on a mutable copy of the config dict so changes are
            # properly tracked by SQLAlchemy's mutation detection.
            config = dict(entity.config or {})
            config["managed_in_settings"] = True
            if "name" in changes:
                entity.name = changes["name"]
            if "source_uri" in changes:
                entity.rtsp_url = changes["source_uri"]
                config["source_uri"] = changes["source_uri"]
            if "source_type" in changes:
                config["source_type"] = changes["source_type"]
            if "is_active" in changes:
                entity.status = "active" if changes["is_active"] else "inactive"
                # Only update the last-active timestamp when activating;
                # preserve the previous value when deactivating.
                entity.last_active_at = datetime.now(timezone.utc) if changes["is_active"] else entity.last_active_at
            entity.config = config
            session.add(entity)
            session.flush()
            session.refresh(entity)
            return entity

    def delete(self, camera_id: str) -> bool:
        """Soft-delete a camera by setting its ``deleted_at`` timestamp.

        The camera is also marked as ``"inactive"`` so it will not
        appear in active-camera queries.

        Args:
            camera_id: The UUID or external identifier of the camera.

        Returns:
            ``True`` if the camera was found and deleted; ``False`` if
            it was already deleted or does not exist.
        """
        with self.database.session() as session:
            entity = session.get(CameraEntity, normalize_uuid(camera_id))
            if entity is None or entity.deleted_at is not None:
                return False
            entity.deleted_at = datetime.now(timezone.utc)
            entity.status = "inactive"
            session.add(entity)
            return True

    def seed_defaults(self, cameras: Iterable[dict]) -> list[CameraEntity]:
        """Create or refresh bootstrap cameras from configuration.

        Each camera dict must contain at least ``id``, ``name``, and
        ``source_uri``. If a camera with the same external ID already exists,
        its bootstrap metadata is refreshed so changed bundled video sources
        and display names take effect after an application restart.

        Args:
            cameras: An iterable of dictionaries, each describing a
                camera to seed (keys: ``id``, ``name``, ``source_uri``,
                and optionally ``source_type`` and ``is_active``).

        Returns:
            The full list of non-deleted cameras after seeding,
            ordered by creation date.
        """
        with self.database.session() as session:
            # Match bootstrap entries by their stable external IDs, allowing
            # their configured source and display name to change in place.
            existing = list(session.scalars(select(CameraEntity).order_by(CameraEntity.created_at)))
            existing_by_external_id = {
                str((entity.config or {}).get("external_id")): entity
                for entity in existing
                if (entity.config or {}).get("external_id") is not None
            }

            entities = []
            for camera in cameras:
                external_id = str(camera["id"])
                source_type = str(camera.get("source_type", "file"))
                source_uri = str(camera["source_uri"])
                is_active = bool(camera.get("is_active", True))
                existing_entity = existing_by_external_id.get(external_id)
                if existing_entity is not None:
                    if (existing_entity.config or {}).get("managed_in_settings"):
                        continue
                    config = dict(existing_entity.config or {})
                    config.update(
                        {
                            "source_type": source_type,
                            "source_uri": source_uri,
                            "external_id": external_id,
                        }
                    )
                    existing_entity.name = str(camera["name"])
                    existing_entity.rtsp_url = source_uri
                    existing_entity.status = "active" if is_active else "inactive"
                    existing_entity.config = config
                    if is_active and existing_entity.last_active_at is None:
                        existing_entity.last_active_at = datetime.now(timezone.utc)
                    continue
                entities.append(
                    CameraEntity(
                        id=normalize_uuid(external_id),
                        name=str(camera["name"]),
                        rtsp_url=source_uri,
                        status="active" if is_active else "inactive",
                        config={
                            "source_type": source_type,
                            "source_uri": source_uri,
                            "external_id": external_id,
                        },
                        last_active_at=datetime.now(timezone.utc) if is_active else None,
                    )
                )
            if entities:
                session.add_all(entities)
                session.flush()

            # Return the complete set of non-deleted cameras after seeding
            return list(
                session.scalars(
                    select(CameraEntity).where(CameraEntity.deleted_at.is_(None)).order_by(CameraEntity.created_at)
                )
            )
