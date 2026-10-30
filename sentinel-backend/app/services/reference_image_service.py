"""Private filesystem storage for uploaded POI reference images."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class StoredReferenceImage:
    path: str
    width: int
    height: int


class ReferenceImageService:
    """Normalise uploaded references to JPEG files outside the public web root."""

    def __init__(self, settings: Settings) -> None:
        self.root = Path(settings.reference_image_dir)
        self.jpeg_quality = settings.reference_image_jpeg_quality

    def save(self, frame: np.ndarray, person_id: str) -> StoredReferenceImage:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Reference image must be a BGR image")
        height, width = frame.shape[:2]
        relative = Path(person_id) / f"{uuid.uuid4()}.jpg"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("Unable to encode reference image")
        destination.write_bytes(encoded.tobytes())
        return StoredReferenceImage(relative.as_posix(), width, height)

    def resolve(self, relative_path: str) -> Path | None:
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    def delete(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        path = self.resolve(relative_path)
        if path is not None:
            path.unlink(missing_ok=True)
