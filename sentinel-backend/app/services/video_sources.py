from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CameraSource:
    id: str
    name: str
    source_type: str
    source_uri: str


class OpenCVVideoSource:
    def __init__(self, config: CameraSource) -> None:
        self.config = config
        self.logger = logging.getLogger(f"sentinel.source.{config.id}")
        self._capture: cv2.VideoCapture | None = None
        self.ended = False
        self.replay_count = 0

    def open(self) -> None:
        if self.ended:
            return
        if self._capture is not None and self._capture.isOpened():
            return

        source: str | int = self.config.source_uri
        if self.config.source_type == "webcam":
            source = int(self.config.source_uri)

        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_native_fps(self) -> float | None:
        self.open()
        if self._capture is None:
            return None
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.ended:
            return False, None
        self.open()
        if self._capture is None:
            return False, None

        ok, frame = self._capture.read()
        if ok:
            return True, frame

        if self.config.source_type == "file":
            # Replay at EOF while preserving the reader's native-FPS pacing.
            if self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                ok, frame = self._capture.read()
                if ok:
                    self.replay_count += 1
                    self.logger.info("Replaying video %s from the beginning", self.config.id)
                    return True, frame
            # Some codecs cannot seek. Reopen once; do not spin on empty or
            # corrupt files that also fail to yield a frame after reopening.
            self.close()
            self.open()
            if self._capture is not None:
                ok, frame = self._capture.read()
                if ok:
                    self.replay_count += 1
                    self.logger.info("Reopened video %s for replay", self.config.id)
                    return True, frame
            self.ended = True
            self.close()
            self.logger.warning("Video %s cannot be replayed: no readable frames", self.config.id)
            return False, None

        self.close()
        time.sleep(1.0)
        return False, None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
