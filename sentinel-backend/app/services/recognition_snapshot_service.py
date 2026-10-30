"""Create and manage evidence snapshots for recognised-person records."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class RecognitionSnapshot:
    """Metadata for a persisted annotated recognition image."""

    path: str
    width: int
    height: int


class RecognitionSnapshotService:
    """Persists full-frame JPEG evidence with the accepted face box drawn on it."""

    def __init__(self, settings: Settings) -> None:
        self.root = Path(settings.recognition_record_snapshot_dir)
        self.jpeg_quality = settings.recognition_record_jpeg_quality
        self.logger = logging.getLogger("sentinel.recognition_snapshot")

    def save(
        self,
        frame: np.ndarray,
        bbox: list[int],
        *,
        person_name: str | None,
        person_type: str | None,
        similarity: float | None,
    ) -> RecognitionSnapshot:
        """Write an annotated JPEG and return a path relative to ``root``."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Recognition snapshot frame must be a BGR image")

        image = frame.copy()
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Recognition snapshot bbox is outside the frame")

        colour = (38, 38, 220) if (person_type or "").lower() == "watchlist" else (74, 222, 128)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        label = person_name or "Recognized person"
        if similarity is not None:
            label = f"{label}  {similarity:.3f}"
        label_y = y1 - 8 if y1 >= 24 else min(y1 + 20, height - 8)
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

        now = datetime.now(timezone.utc)
        relative = Path(now.strftime("%Y")) / now.strftime("%m") / f"{uuid.uuid4()}.jpg"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            raise RuntimeError("Unable to encode recognition snapshot")
        destination.write_bytes(encoded.tobytes())
        return RecognitionSnapshot(path=relative.as_posix(), width=width, height=height)

    def resolve(self, relative_path: str) -> Path | None:
        """Resolve an evidence path while preventing traversal outside the store."""
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    def delete(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        path = self.resolve(relative_path)
        if path is None:
            return
        try:
            path.unlink()
        except OSError:
            self.logger.warning("Unable to remove recognition snapshot %s", path)
