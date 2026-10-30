"""Persist the supported runtime controls with atomic file replacement."""
import json
import os
from pathlib import Path
from threading import RLock
from pydantic import BaseModel, ConfigDict, Field

class RecognitionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    recognition_threshold: float = Field(ge=0, le=1)
    live_quality_threshold: float = Field(ge=0, le=10)

class SettingsService:
    def __init__(self, settings, path):
        self.settings = settings
        self.path = Path(path)
        self.lock = RLock()
        self.defaults = self.current()
        if self.path.exists():
            saved = RecognitionSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
            self._apply(saved)

    def current(self):
        return RecognitionSettings(**{key: getattr(self.settings, key) for key in RecognitionSettings.model_fields}).model_dump()

    def _apply(self, values):
        for key, value in values.model_dump().items():
            setattr(self.settings, key, value)

    def save(self, values):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    json.dump(values.model_dump(), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            self._apply(values)
            return self.current()
