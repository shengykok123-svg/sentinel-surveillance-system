"""Persistence for uploaded reference-image metadata."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.database import DatabaseManager
from app.models.entities import FaceReferenceImageEntity, GeneratedReferenceImageEntity, PersonEntity


@dataclass(frozen=True, slots=True)
class ReferenceImageRow:
    id: str
    person_id: str
    embedding_id: str | None
    original_filename: str
    width: int
    height: int
    created_at: datetime
    image_kind: str = "original"
    parent_id: str | None = None
    original_reference_id: str | None = None
    used_for_recognition: bool = True
    generation: dict | None = None


class ReferenceImageRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def add(
        self, *, person_id: str, embedding_id: str, storage_path: str,
        original_filename: str, width: int, height: int,
    ) -> ReferenceImageRow:
        with self.database.session() as session:
            image = FaceReferenceImageEntity(
                person_id=uuid.UUID(person_id),
                embedding_id=uuid.UUID(embedding_id),
                storage_path=storage_path,
                original_filename=original_filename[:255],
                width=width,
                height=height,
            )
            session.add(image)
            session.flush()
            return self._row(image)

    def list_for_person(self, person_id: str) -> list[ReferenceImageRow]:
        with self.database.session() as session:
            if not self._visible(session, person_id):
                return []
            images = session.scalars(
                select(FaceReferenceImageEntity)
                .where(FaceReferenceImageEntity.person_id == uuid.UUID(person_id))
                .order_by(FaceReferenceImageEntity.created_at)
            )
            rows = [self._row(image) for image in images]
            generated = session.scalars(select(GeneratedReferenceImageEntity)
                .where(GeneratedReferenceImageEntity.person_id == uuid.UUID(person_id))
                .order_by(GeneratedReferenceImageEntity.created_at))
            return rows + [self._generated_row(image) for image in generated]

    def storage_path_for(self, person_id: str, reference_id: str) -> str | None:
        with self.database.session() as session:
            if not self._visible(session, person_id):
                return None
            original = session.scalar(
                select(FaceReferenceImageEntity.storage_path).where(
                    FaceReferenceImageEntity.person_id == uuid.UUID(person_id),
                    FaceReferenceImageEntity.id == uuid.UUID(reference_id),
                )
            )
            if original:
                return original
            return session.scalar(select(GeneratedReferenceImageEntity.storage_path).where(
                GeneratedReferenceImageEntity.person_id == uuid.UUID(person_id),
                GeneratedReferenceImageEntity.id == uuid.UUID(reference_id)))

    @staticmethod
    def _visible(session, person_id):
        person = session.get(PersonEntity, uuid.UUID(person_id))
        if person is None or person.deleted_at is not None:
            return False
        expiry = person.retention_expires_at
        return expiry is None or expiry.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)

    @staticmethod
    def _generated_row(image):
        return ReferenceImageRow(id=str(image.id), person_id=str(image.person_id),
            embedding_id=str(image.embedding_id) if image.embedding_id else None,
            original_filename=image.original_filename, width=image.width, height=image.height,
            created_at=image.created_at, image_kind=image.image_kind,
            parent_id=str(image.parent_id or image.original_reference_id),
            original_reference_id=str(image.original_reference_id),
            used_for_recognition=image.embedding_id is not None, generation=image.generation)

    @staticmethod
    def _row(image: FaceReferenceImageEntity) -> ReferenceImageRow:
        return ReferenceImageRow(
            id=str(image.id), person_id=str(image.person_id), embedding_id=str(image.embedding_id),
            original_filename=image.original_filename, width=image.width, height=image.height,
            created_at=image.created_at,
        )
