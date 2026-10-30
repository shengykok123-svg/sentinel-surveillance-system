"""
Person repository module for Sentinel.

Provides data-access operations for enrolled-person entities, including
creating, reading, updating, soft-deleting, and counting persons.
Persons represent individuals whose faces are enrolled in the
recognition system (staff, VIPs, watchlist targets, visitors).

This module also handles notes serialisation (string, dict, or list to
a storage-friendly format) and validates that ``person_type`` values
belong to the allowed set.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from app.core.database import DatabaseManager
from app.models.entities import FaceEmbeddingEntity, PersonEntity


# The closed set of valid person classification types.
ALLOWED_PERSON_TYPES = ("staff", "vip", "watchlist", "visitor")

# Sentinel object used as the default for optional keyword arguments in
# ``update_person``.  This distinguishes "the caller did not supply a
# value" from "the caller explicitly passed None", enabling true
# partial updates.
_UNSET = object()


def normalize_person_type(value: str) -> str:
    """Validate and normalise a person-type string.

    The input is stripped, lowercased, and checked against
    ``ALLOWED_PERSON_TYPES``.

    Args:
        value: The raw person-type string to validate.

    Returns:
        The normalised (lowercase, stripped) person-type string.

    Raises:
        ValueError: If *value* is not one of the allowed types.
    """
    normalized = str(value).strip().lower()
    if normalized not in ALLOWED_PERSON_TYPES:
        allowed = ", ".join(ALLOWED_PERSON_TYPES)
        raise ValueError(f"person_type must be one of: {allowed}")
    return normalized


class PersonRepository:
    """Repository handling persistence operations for enrolled persons.

    All public methods open their own database sessions via the
    injected ``DatabaseManager``.  Persons use a **soft-delete**
    pattern: a non-null ``deleted_at`` timestamp marks a person as
    removed without physically deleting the row.
    """

    def __init__(self, database: DatabaseManager) -> None:
        """Initialise the repository with a database manager.

        Args:
            database: The ``DatabaseManager`` instance used to obtain
                database sessions.
        """
        self.database = database

    def create_person(
        self,
        *,
        full_name: str,
        person_type: str,
        notes: str | dict | list | None,
        is_active: bool,
        embedding_only_mode: bool,
        enrolled_by_user_id: str,
        retention_expires_at: datetime | None = None,
    ) -> PersonEntity:
        """Create and persist a new enrolled person.

        Args:
            full_name: The person's display name.
            person_type: Classification (must be one of
                ``ALLOWED_PERSON_TYPES``).
            notes: Free-form notes -- may be a plain string, a dict,
                a list, or ``None``.
            is_active: Whether the person is currently active for
                recognition.
            embedding_only_mode: If ``True``, the person is used only
                for embedding matching and may not appear in full
                person listings.
            enrolled_by_user_id: UUID string of the user who is
                enrolling this person.
            retention_expires_at: Optional datetime after which the
                person record may be purged for data-retention
                compliance.

        Returns:
            The newly created and flushed ``PersonEntity``.
        """
        with self.database.session() as session:
            person = PersonEntity(
                full_name=full_name,
                person_type=normalize_person_type(person_type),
                notes=self._serialize_notes(notes),
                enrolled_by=uuid.UUID(str(enrolled_by_user_id)),
                is_active=bool(is_active),
                embedding_only_mode=bool(embedding_only_mode),
                retention_expires_at=retention_expires_at,
            )
            session.add(person)
            session.flush()
            session.refresh(person)
            return person

    def create(self, name: str, metadata: dict, enrolled_by_user_id: str) -> PersonEntity:
        """Simplified creation interface that extracts fields from a metadata dict.

        This is a convenience wrapper around ``create_person`` for
        callers that pass person attributes inside a single dictionary.

        Args:
            name: The person's display name.
            metadata: Dictionary containing optional keys such as
                ``person_type``, ``is_active``,
                ``embedding_only_mode``, and
                ``retention_expires_at``.
            enrolled_by_user_id: UUID string of the enrolling user.

        Returns:
            The newly created ``PersonEntity``.
        """
        return self.create_person(
            full_name=name,
            person_type=normalize_person_type(str(metadata.get("person_type", "watchlist"))),
            notes=metadata,
            is_active=bool(metadata.get("is_active", True)),
            embedding_only_mode=bool(metadata.get("embedding_only_mode", True)),
            enrolled_by_user_id=enrolled_by_user_id,
            retention_expires_at=metadata.get("retention_expires_at"),
        )

    def get_person_by_id(self, person_id: str) -> PersonEntity | None:
        """Fetch a single person by primary key, excluding soft-deleted records.

        Args:
            person_id: UUID string of the person to retrieve.

        Returns:
            The matching ``PersonEntity``, or ``None`` if not found or
            soft-deleted.
        """
        with self.database.session() as session:
            entity = session.get(PersonEntity, uuid.UUID(str(person_id)))
            if entity is None or entity.deleted_at is not None:
                return None
            return entity

    def get(self, person_id: str) -> PersonEntity | None:
        """Alias for ``get_person_by_id`` for shorter call-sites.

        Args:
            person_id: UUID string of the person.

        Returns:
            The matching ``PersonEntity`` or ``None``.
        """
        return self.get_person_by_id(person_id)

    def get_with_embedding_count(self, person_id: str) -> tuple[PersonEntity, int] | None:
        """Fetch a person together with the count of their face embeddings.

        Uses an outer join so persons with zero embeddings are still
        returned.

        Args:
            person_id: UUID string of the person.

        Returns:
            A ``(PersonEntity, embedding_count)`` tuple, or ``None``
            if the person is not found or has been soft-deleted.
        """
        with self.database.session() as session:
            stmt = (
                select(PersonEntity, func.count(FaceEmbeddingEntity.id))
                .outerjoin(FaceEmbeddingEntity, FaceEmbeddingEntity.person_id == PersonEntity.id)
                .where(
                    PersonEntity.id == uuid.UUID(str(person_id)),
                    PersonEntity.deleted_at.is_(None),
                )
                .group_by(PersonEntity.id)
            )
            row = session.execute(stmt).one_or_none()
            if row is None:
                return None
            entity, embedding_count = row
            return entity, int(embedding_count or 0)

    def list_all(self) -> list[PersonEntity]:
        """Return all non-deleted persons with their embeddings eagerly loaded.

        Results are ordered alphabetically by full name.

        Returns:
            A list of ``PersonEntity`` objects with the ``embeddings``
            relationship pre-loaded.
        """
        with self.database.session() as session:
            stmt = (
                select(PersonEntity)
                .where(PersonEntity.deleted_at.is_(None))
                .options(joinedload(PersonEntity.embeddings))
                .order_by(PersonEntity.full_name)
            )
            # .unique() is needed because joinedload can produce
            # duplicate ORM instances when a person has multiple
            # embeddings.
            return list(session.scalars(stmt).unique())

    def update_person(
        self,
        person_id: str,
        *,
        full_name: str | None | object = _UNSET,
        person_type: str | None | object = _UNSET,
        notes: str | dict | list | None | object = _UNSET,
        is_active: bool | None | object = _UNSET,
        embedding_only_mode: bool | None | object = _UNSET,
        retention_expires_at: datetime | None | object = _UNSET,
    ) -> PersonEntity | None:
        """Apply a partial update to an existing person.

        Only parameters whose values differ from the ``_UNSET``
        sentinel are written.  This enables callers to update a single
        field without affecting the others.

        Args:
            person_id: UUID string of the person to update.
            full_name: New display name (must not be blank).
            person_type: New classification type.
            notes: Replacement notes (string, dict, list, or ``None``).
            is_active: New active flag.
            embedding_only_mode: New embedding-only flag.
            retention_expires_at: New retention expiry datetime.

        Returns:
            The updated ``PersonEntity``, or ``None`` if the person was
            not found or has been soft-deleted.

        Raises:
            ValueError: If *full_name* is provided but empty/blank.
        """
        with self.database.session() as session:
            entity = session.get(PersonEntity, uuid.UUID(str(person_id)))
            if entity is None or entity.deleted_at is not None:
                return None

            # Each field is updated only when the caller explicitly
            # provided a value (i.e. the argument is not _UNSET).
            if full_name is not _UNSET:
                normalized_name = str(full_name).strip()
                if not normalized_name:
                    raise ValueError("name is required")
                entity.full_name = normalized_name
            if person_type is not _UNSET:
                entity.person_type = normalize_person_type(person_type)
            if is_active is not _UNSET:
                entity.is_active = bool(is_active)
            if embedding_only_mode is not _UNSET:
                entity.embedding_only_mode = bool(embedding_only_mode)
            if retention_expires_at is not _UNSET:
                entity.retention_expires_at = retention_expires_at
            if notes is not _UNSET:
                entity.notes = self._serialize_notes(notes)

            session.add(entity)
            session.flush()
            session.refresh(entity)
            return entity

    def delete(self, person_id: str) -> bool:
        """Soft-delete a person by setting ``deleted_at`` and deactivating them.

        Args:
            person_id: UUID string of the person to delete.

        Returns:
            ``True`` if the person was found and marked as deleted;
            ``False`` if the person does not exist or was already
            deleted.
        """
        with self.database.session() as session:
            entity = session.get(PersonEntity, uuid.UUID(str(person_id)))
            if entity is None or entity.deleted_at is not None:
                return False
            entity.deleted_at = datetime.now(timezone.utc)
            entity.is_active = False
            session.add(entity)
            return True

    def count(self) -> int:
        """Return the number of non-deleted persons in the database.

        Returns:
            An integer count of active (non-soft-deleted) person
            records.
        """
        with self.database.session() as session:
            stmt = select(func.count()).select_from(PersonEntity).where(PersonEntity.deleted_at.is_(None))
            return int(session.scalar(stmt) or 0)

    def list_with_embedding_counts(self) -> list[tuple[PersonEntity, int]]:
        """Return all non-deleted persons paired with their embedding counts.

        Uses an outer join so persons with zero embeddings are
        included.  Results are ordered alphabetically by full name.

        Returns:
            A list of ``(PersonEntity, embedding_count)`` tuples.
        """
        with self.database.session() as session:
            stmt = (
                select(PersonEntity, func.count(FaceEmbeddingEntity.id))
                .outerjoin(FaceEmbeddingEntity, FaceEmbeddingEntity.person_id == PersonEntity.id)
                .where(PersonEntity.deleted_at.is_(None))
                .group_by(PersonEntity.id)
                .order_by(PersonEntity.full_name)
            )
            return list(session.execute(stmt))

    @staticmethod
    def _serialize_notes(notes: str | dict | list | None) -> str | None:
        """Convert notes into a storage-ready string representation.

        - ``None`` is passed through as-is.
        - Plain strings are stripped; empty strings become ``None``.
        - Dicts and lists are JSON-serialised.

        Args:
            notes: The notes value to serialise.

        Returns:
            A string suitable for database storage, or ``None``.
        """
        if notes is None:
            return None
        if isinstance(notes, str):
            value = notes.strip()
            # Treat blank strings as no notes
            return value or None
        # Dicts and lists are stored as JSON strings
        return json.dumps(notes)
