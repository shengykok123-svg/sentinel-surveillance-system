"""
Camera service layer providing CRUD operations over camera entities.

This module acts as a thin service layer between the API routes and the
``CameraRepository``. It converts ORM ``CameraEntity`` objects into Pydantic
``CameraRead`` schemas suitable for API responses, and delegates all
persistence operations to the underlying repository.

It also provides a ``ensure_bootstrap_cameras`` method to seed default camera
configurations from the application settings during first-time startup.
"""

from __future__ import annotations

from app.core.config import Settings
from app.models.schemas import CameraCreate, CameraRead, CameraUpdate
from app.repositories.camera_repository import CameraRepository

#── CameraService class definition ─────────────────────────────────────────
class CameraService:
    
# Initialise the camera service.
    def __init__(self, repository: CameraRepository, settings: Settings) -> None:
        
        self.repository = repository
        self.settings = settings

# Ensure that the default bootstrap cameras from settings are present in the database, seeding them if necessary. 
# This is typically called during application startup to guarantee that the configured cameras are available for use.
    def ensure_bootstrap_cameras(self) -> list[CameraRead]:
        entities = self.repository.seed_defaults(self.settings.bootstrap_cameras)
        return [self._to_schema(entity) for entity in entities]
    
# Retrieve all cameras regardless of status.
    def list_cameras(self) -> list[CameraRead]:
        return [self._to_schema(entity) for entity in self.repository.list_all()]

# Retrieve only cameras with an active status.
    def list_active_cameras(self) -> list[CameraRead]:
        
        return [self._to_schema(entity) for entity in self.repository.list_active()]

# Retrieve a single camera by its unique identifier, returning None if not found.
    def get_camera(self, camera_id: str) -> CameraRead | None:
        entity = self.repository.get(camera_id)
        return self._to_schema(entity) if entity is not None else None

# Create a new camera entity from the provided payload, returning the created camera as a schema.
    def create_camera(self, payload: CameraCreate) -> CameraRead:
    
        entity = self.repository.create(**payload.model_dump())
        return self._to_schema(entity)

# Update an existing camera entity with the provided fields, returning the updated camera as a schema or None if the camera does not exist.
    def update_camera(self, camera_id: str, payload: CameraUpdate) -> CameraRead | None:
        
        entity = self.repository.update(camera_id, **payload.model_dump(exclude_none=True))
        return self._to_schema(entity) if entity is not None else None

# Delete a camera entity by its unique identifier, returning True if the camera was found and deleted, or False if it was not found.
    def delete_camera(self, camera_id: str) -> bool:
        return self.repository.delete(camera_id)

    @staticmethod
    # Convert a CameraEntity ORM object to a CameraRead Pydantic schema, extracting source configuration from the JSON config field and populating runtime status fields with default values.
    def _to_schema(entity) -> CameraRead:
    
        # Extract source configuration from the JSON config column
        config = dict(entity.config or {})
        source_type = str(config.get("source_type", "rtsp"))
        source_uri = str(config.get("source_uri", entity.rtsp_url))
        return CameraRead(
            id=str(entity.id),
            name=entity.name,
            source_type=source_type,
            source_uri=source_uri,
            is_active=entity.status == "active",
            online=False,            # Actual online status set by video service at runtime
            active_tracks=0,         # Track count populated at runtime
            fps=int(entity.stream_fps or 0),
            face_count=0,            # Face count populated at runtime
        )
