"""Append-only audit persistence for retained reference-image views."""

from __future__ import annotations

import uuid

from app.core.database import DatabaseManager
from app.models.entities import ReferenceImageAccessLogEntity, GeneratedReferenceImageEntity


class ReferenceImageAccessRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def log_view(
        self, *, actor_user_id: str, person_id: str, reference_image_id: str,
        ip_address: str | None, user_agent: str | None,
    ) -> None:
        with self.database.session() as session:
            generated = session.get(GeneratedReferenceImageEntity, uuid.UUID(reference_image_id))
            if generated is not None and str(generated.person_id) != person_id:
                raise ValueError("Reference does not belong to this person")
            session.add(ReferenceImageAccessLogEntity(
                actor_user_id=uuid.UUID(actor_user_id),
                person_id=uuid.UUID(person_id),
                reference_image_id=generated.original_reference_id if generated else uuid.UUID(reference_image_id),
                generated_image_id=reference_image_id if generated else None,
                ip_address=ip_address[:64] if ip_address else None,
                user_agent=user_agent[:512] if user_agent else None,
            ))
