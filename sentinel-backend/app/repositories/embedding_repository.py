"""
Embedding repository module for Sentinel.

Provides data-access operations for face-embedding vectors associated
with enrolled persons.  Responsibilities include saving new embeddings
(with automatic primary-flag assignment), listing all active embeddings,
bulk-deleting embeddings for a given person, and returning aggregate
statistics used by dashboard endpoints.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from app.core.database import DatabaseManager
from app.models.entities import FaceEmbeddingEntity, PersonEntity, validate_embedding_vector


class EmbeddingRepository:
    """Repository responsible for face-embedding persistence.

    Each embedding vector belongs to exactly one ``PersonEntity``.
    The first embedding saved for a person is automatically marked as
    the **primary** embedding; subsequent embeddings are secondary.
    """

    def __init__(self, database: DatabaseManager) -> None:
        """Initialise the repository with a database manager.

        Args:
            database: The ``DatabaseManager`` instance used to obtain
                database sessions.
        """
        self.database = database
        self.logger = logging.getLogger("sentinel.embedding_repository")

    def save_embedding(
        self,
        person_id: str,
        vector,
        *,
        source_type: str = "manual_upload",
        quality_score: float | None = 1.0,
        expires_at=None,
    ) -> FaceEmbeddingEntity:
        """Validate, normalise, and persist a face-embedding vector.

        The raw *vector* (which may be a list, numpy array, or other
        sequence) is first passed through ``validate_embedding_vector``
        to ensure it meets length and type constraints.  A short preview
        of the first five components is logged for traceability.

        If no existing primary embedding exists for the person, the new
        embedding is automatically flagged as primary.

        Args:
            person_id: UUID (as string) of the person this embedding
                belongs to.
            vector: The raw embedding vector (any sequence of floats).
            source_type: Origin of the embedding, e.g.
                ``"manual_upload"`` or ``"auto_capture"``.
            quality_score: Optional source-specific quality metric. New registration
                uploads use raw CR-FIQA scores, which may exceed 1; historical
                records may contain detector confidence. Defaults to ``1.0``.
            expires_at: Optional expiry datetime after which the
                embedding should be considered stale.

        Returns:
            The newly created ``FaceEmbeddingEntity`` after flush.
        """
        # Validate and normalise the raw vector into a plain Python list
        normalized_vector = validate_embedding_vector(vector)

        # Log a short preview of the vector for debugging purposes
        preview = ", ".join(f"{value:.4f}" for value in normalized_vector[:5])
        self.logger.info(
            "Saving embedding vector type=%s length=%s preview=[%s]",
            type(vector).__name__,
            len(normalized_vector),
            preview,
        )
        with self.database.session() as session:
            # Count how many primary embeddings the person already has
            existing_primary = session.scalar(
                select(func.count())
                .select_from(FaceEmbeddingEntity)
                .where(FaceEmbeddingEntity.person_id == uuid.UUID(str(person_id)), FaceEmbeddingEntity.is_primary.is_(True))
            )
            entity = FaceEmbeddingEntity(
                person_id=uuid.UUID(str(person_id)),
                embedding_vector=normalized_vector,
                source_type=source_type,
                quality_score=quality_score,
                # Auto-mark as primary when no primary embedding exists yet
                is_primary=(existing_primary or 0) == 0,
                expires_at=expires_at,
            )
            session.add(entity)
            session.flush()
            session.refresh(entity)
            return entity

    def add(self, person_id: str, embedding: list[float]) -> FaceEmbeddingEntity:
        """Convenience wrapper around ``save_embedding`` with default options.

        This thin adapter exists so callers that only have a person ID
        and a plain float list can persist an embedding without
        specifying optional parameters.

        Args:
            person_id: UUID string of the owning person.
            embedding: The embedding vector as a list of floats.

        Returns:
            The saved ``FaceEmbeddingEntity``.
        """
        return self.save_embedding(person_id, embedding)

    def list_all(self) -> list[FaceEmbeddingEntity]:
        """Return all embeddings whose owning person has not been deleted.

        An inner join against ``PersonEntity`` ensures embeddings
        belonging to soft-deleted persons are excluded.  The related
        ``person`` relationship is eagerly loaded to avoid N+1 queries.

        Returns:
            A list of ``FaceEmbeddingEntity`` objects ordered by
            creation date, with the ``person`` relationship populated.
        """
        with self.database.session() as session:
            stmt = (
                select(FaceEmbeddingEntity)
                # Inner join excludes embeddings whose person is deleted
                .join(PersonEntity, FaceEmbeddingEntity.person_id == PersonEntity.id)
                .where(PersonEntity.deleted_at.is_(None))
                .options(joinedload(FaceEmbeddingEntity.person))
                .order_by(FaceEmbeddingEntity.created_at)
            )
            # .unique() is required because joinedload may produce
            # duplicate rows in the result set.
            return list(session.scalars(stmt).unique())

    def list_for_person(self, person_id: str) -> list[FaceEmbeddingEntity]:
        """Return all stored vectors for one active person profile."""
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(FaceEmbeddingEntity).where(FaceEmbeddingEntity.person_id == uuid.UUID(str(person_id)))
                )
            )

    def delete_for_person(self, person_id: str) -> int:
        """Hard-delete all embedding records belonging to a person.

        Unlike person soft-deletion, embeddings are physically removed
        from the database because they contain sensitive biometric data.

        Args:
            person_id: UUID string of the person whose embeddings
                should be removed.

        Returns:
            The number of embedding rows that were deleted.
        """
        with self.database.session() as session:
            stmt = select(FaceEmbeddingEntity).where(FaceEmbeddingEntity.person_id == uuid.UUID(str(person_id)))
            entities = list(session.scalars(stmt))
            deleted = len(entities)
            for entity in entities:
                session.delete(entity)
            return deleted

    def stats(self) -> tuple[int, int]:
        """Return aggregate counts for the embedding store.

        Returns:
            A two-element tuple of ``(total_vectors, total_persons)``
            where *total_vectors* is the number of embedding rows in
            the database and *total_persons* is the number of
            non-deleted person records.
        """
        with self.database.session() as session:
            total_vectors = int(session.scalar(select(func.count()).select_from(FaceEmbeddingEntity)) or 0)
            total_persons = int(
                session.scalar(
                    select(func.count())
                    .select_from(PersonEntity)
                    .where(PersonEntity.deleted_at.is_(None))
                )
                or 0
            )
            return total_vectors, total_persons
