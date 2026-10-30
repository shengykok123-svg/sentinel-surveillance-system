"""Shared CR-FIQA(S) registration gate and live ranking; raw scores, not probabilities."""
from pathlib import Path
import math
import threading

import cv2
import numpy as np

from app.core.config import Settings


class FaceQualityService:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not math.isfinite(settings.enrollment_quality_threshold):
            raise ValueError('Enrollment quality threshold must be finite')
        self._lock = threading.RLock()
        self._model = None
        self._device = None

    def start(self):
        with self._lock:
            if self._model is not None:
                return
            import torch
            from app.services.crfiqa_backbone import iresnet50
            path = Path(self.settings.face_quality_model_path)
            if not path.is_file():
                raise RuntimeError(f'CR-FIQA(S) weights missing: {path}')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = iresnet50(num_features=512, qs=1, use_se=False)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            self._model = model.eval().to(device)
            self._device = device

    @staticmethod
    def _to_input(aligned_bgr):
        if aligned_bgr.shape != (112, 112, 3) or aligned_bgr.dtype != np.uint8:
            raise ValueError('CR-FIQA requires an aligned 112x112 uint8 BGR face')
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return (rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0 - .5) / .5

    def assess(self, aligned_bgr: np.ndarray) -> float:
        return self.assess_batch([aligned_bgr])[0]

    def assess_batch(self, faces: list[np.ndarray]) -> list[float]:
        if not faces:
            return []
        inputs = np.ascontiguousarray(np.concatenate([self._to_input(face) for face in faces]))
        import torch
        with self._lock:
            self.start()
            with torch.inference_mode():
                scores = self._model(torch.from_numpy(inputs).to(self._device))[1].detach().cpu().numpy().reshape(-1).tolist()
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError('CR-FIQA returned a non-finite quality score')
        return scores

    def stop(self):
        with self._lock:
            self._model = None
            self._device = None
