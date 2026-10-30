"""
AdaFace model service for face embedding extraction.

This module loads the AdaFace model checkpoint, manages the model lifecycle,
and provides methods for face cropping, preprocessing, and 512-dimensional
embedding extraction. It supports both unaligned face crops (which are
resized to 112x112) and pre-aligned 112x112 face images.

The service is thread-safe: a threading.Lock guards model inference so that
concurrent callers do not corrupt internal model state.

Typical usage flow:
    1. Instantiate ``AdaFaceService`` with application settings.
    2. Call ``start()`` to load the model checkpoint (lazy-loaded on first use).
    3. Call ``extract_embedding(face_bgr)`` or ``extract_embedding_aligned(aligned_bgr)``
       to obtain a (vector, norm) tuple.
    4. Call ``stop()`` to release the model from memory.

Important:
    Enrolled embeddings must be regenerated after any change to the alignment
    pipeline. See the NOTE comment near FACE_CROP_MARGIN for details.
"""

from __future__ import annotations

import logging
import pathlib
import platform
import threading
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.services.adaface_backbone import build_adaface_backbone


# NOTE: Enrolled embeddings must be re-generated after any change to the
# alignment or colour pipeline. Build a separate versioned gallery from retained
# originals and verify it before switching indexes; preserve person records.

# Fractional margin added around the detected face bounding box when cropping.
# A value of 0.20 means the crop extends 20% beyond each side of the bbox.
FACE_CROP_MARGIN = 0.20


@contextmanager
def _windows_path_compat():
    #Context manager to handle Windows/Linux path compatibility when loading checkpoints.
    if platform.system() == "Windows":
        # No patching needed on Windows -- WindowsPath is natively available
        yield
        return
    # On Linux/macOS: temporarily replace WindowsPath with PureWindowsPath
    original = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PureWindowsPath
    try:
        yield
    finally:
        # Restore the original class to avoid side-effects
        pathlib.WindowsPath = original

# The AdaFaceService class encapsulates all logic related to the AdaFace model, including loading the model checkpoint, preprocessing face images, and extracting embeddings. 
# It ensures thread-safe access to the model for concurrent inference calls.
class AdaFaceService:
    

# Initialise the AdaFace service
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger("sentinel.adaface")
        self._torch = None        # Lazily imported torch module
        self._model = None         # The loaded AdaFaceBackbone model
        self._device = None        # torch.device (cpu or cuda)
        self._model_lock = threading.Lock()  # Guards concurrent inference calls

# Load the AdaFace model from a checkpoint file and move it to the compute device
    def start(self) -> None:
        if self._model is not None:
            return

        import torch

        model_path = Path(self.settings.recognition_model_path)
        # Load checkpoint with cross-platform path compatibility
        with _windows_path_compat():
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract the state dict -- checkpoints may store it under different keys
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        # Strip common training-wrapper prefixes to match the standalone
        # backbone module. The official IR101 WebFace12M checkpoint uses
        # ``model.``; the previous checkpoint used ``backbone.``.
        state_dict = {
            key.removeprefix("module.").removeprefix("model.").removeprefix("backbone."): value
            for key, value in state_dict.items()
        }

        model = build_adaface_backbone(self.settings.recognition_architecture)
        # The official AdaFace checkpoints also bundle the training head. It
        # is not part of the embedding backbone and must not be passed to the
        # strict backbone-state validation below.
        model_keys = set(model.state_dict())
        ignored_keys = set(state_dict) - model_keys
        if ignored_keys:
            self.logger.debug("Ignoring non-backbone checkpoint keys: %s", sorted(ignored_keys))
            state_dict = {key: value for key, value in state_dict.items() if key in model_keys}
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode (disables dropout, fixes BN stats)

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self.logger.info("AdaFace model loaded on %s from %s", self._device, model_path)

    def stop(self) -> None:
        """Release the model and free associated resources.

        After calling this method, ``start()`` must be called again before
        any embedding extraction.
        """
        self._model = None
        self._device = None
        self._torch = None

    @staticmethod
    def crop_face(
        frame: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
        margin: float = FACE_CROP_MARGIN,
    ) -> tuple[np.ndarray, list[int]]:
        """Crop a square region around the detected face bounding box with margin.

        The crop is expanded by ``margin`` (as a fraction of the larger bbox
        dimension) to include forehead, chin, and ear context. The crop is
        then forced to be square and clamped within the frame boundaries.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        # Ensure minimum box dimensions of 1 pixel to avoid division issues
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        # Compute square crop size from the larger dimension, expanded by margin
        size = int(max(box_w, box_h) * (1.0 + margin))
        # Centre the square crop on the bounding box centre
        center_x = x1 + box_w / 2.0
        center_y = y1 + box_h / 2.0

        left = int(round(center_x - size / 2.0))
        top = int(round(center_y - size / 2.0))
        right = left + size
        bottom = top + size

        # Clamp to frame boundaries
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, width)
        bottom = min(bottom, height)

        # After clamping, ensure the crop remains square by taking the smaller side
        final_size = min(right - left, bottom - top)
        if final_size < 40:
            raise ValueError("Face crop too small for recognition")

        # Re-centre the square crop after clamping
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        half = final_size / 2.0
        left = int(round(center_x - half))
        top = int(round(center_y - half))
        right = left + final_size
        bottom = top + final_size

        # Shift the crop back inside frame boundaries if re-centring pushed it out
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > width:
            shift = right - width
            left = max(left - shift, 0)
            right = width
        if bottom > height:
            shift = bottom - height
            top = max(top - shift, 0)
            bottom = height

        # Final validation after all adjustments
        final_size = min(right - left, bottom - top)
        if final_size < 40:
            raise ValueError("Face crop too small for recognition")

        # Force exact square dimensions
        right = left + final_size
        bottom = top + final_size
        crop = frame[top:bottom, left:right]
        return crop, [left, top, right, bottom]

    @staticmethod
    # Preprocess an unaligned face crop for model inference, applying resizing, colour conversion, normalisation, and formatting to match the model's training input.
    def _to_input(face_bgr: np.ndarray) -> np.ndarray:
        
        # Preserve the RGB order used by the existing gallery and experiments.
        # The earlier custom training notebook does not establish the contract
        # of the current official IR101 checkpoint. Official AdaFace inference
        # uses BGR; switching requires a separately validated gallery migration.
        resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalise from [0, 255] -> [0, 1] -> [-1, 1] to match training transforms
        normalized = ((rgb.astype(np.float32) / 255.0) - 0.5) / 0.5

        # HWC -> CHW, then add batch dimension: (3, 112, 112) -> (1, 3, 112, 112)
        return np.transpose(normalized, (2, 0, 1))[None, ...]

    @staticmethod
    # Preprocess a pre-aligned 112x112 BGR face image for model inference, applying the same normalisation and formatting as the unaligned version but without resizing or cropping.
    def _to_input_aligned(aligned_bgr: np.ndarray) -> np.ndarray:
        assert aligned_bgr.shape == (112, 112, 3), f"aligned_bgr must be 112x112x3, got {aligned_bgr.shape}"
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # Same normalisation as _to_input: [0,255] -> [-1,1]
        normalized = ((rgb.astype(np.float32) / 255.0) - 0.5) / 0.5

        # HWC -> CHW with batch dimension. The backbone flattens with .view(),
        # which requires a contiguous tensor.
        return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None, ...])

    @classmethod
    def _to_input_aligned_batch(cls, aligned_batch: list[np.ndarray]) -> np.ndarray:
        """Stack several pre-aligned 112x112 BGR faces into one NCHW batch."""
        return np.ascontiguousarray(
            np.concatenate([cls._to_input_aligned(face) for face in aligned_batch], axis=0)
        )

# Extract a 512-d face embedding from an unaligned BGR face crop, applying the full preprocessing pipeline and thread-safe model inference. Returns a tuple of (embedding vector, pre-normalisation L2 norm).
    def extract_embedding(self, face_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        self.start()  # Ensure the model is loaded (no-op if already started)
        if self._model is None or self._torch is None or self._device is None:
            raise RuntimeError("AdaFace model is not available")

        inputs = self._torch.from_numpy(self._to_input(face_bgr)).to(self._device)

        # Thread-safe inference with no gradient computation (eval mode)
        with self._model_lock:
            with self._torch.no_grad():
                embedding, norm = self._model(inputs)

        # Move result to CPU and convert to numpy
        vector = embedding[0].detach().cpu().numpy().astype(np.float32)

        # Verify and enforce unit norm (the model should produce ~unit vectors,
        # but numerical drift can cause small deviations)
        unit_norm = float(np.linalg.norm(vector))
        if abs(unit_norm - 1.0) >= 1e-3:
            vector = vector / max(unit_norm, 1e-6)
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3, "AdaFace embedding should be unit norm"
        return vector, float(norm[0].item())

# Extract a 512-d face embedding from a pre-aligned 112x112 BGR face image.
    def extract_embedding_aligned(self, aligned_bgr: np.ndarray) -> tuple[np.ndarray, float]:

        vectors, norms = self.extract_embeddings_aligned([aligned_bgr])
        return vectors[0], norms[0]

    def extract_embeddings_aligned(
        self,
        aligned_batch: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Embed several pre-aligned 112x112 faces in a single forward pass.

        One face at a time leaves the GPU idle between per-call host-to-device
        copies and synchronisations: on an RTX 4060 a single face costs ~11.6 ms
        while nine faces batched together cost ~17 ms in total (~1.9 ms each).
        Callers that have more than one face to embed in the same cycle should
        always use this method.

        Returns:
            A list of unit-norm 512-d vectors and a list of the corresponding
            pre-normalisation feature norms (AdaFace's quality proxy).
        """
        if not aligned_batch:
            return [], []

        self.start()  # Ensure the model is loaded
        if self._model is None or self._torch is None or self._device is None:
            raise RuntimeError("AdaFace model is not available")

        inputs = self._torch.from_numpy(self._to_input_aligned_batch(aligned_batch)).to(self._device)

        # Thread-safe inference
        with self._model_lock:
            with self._torch.no_grad():
                embeddings, norms = self._model(inputs)

        # One device->host transfer for the whole batch rather than one per face
        matrix = embeddings.detach().cpu().numpy().astype(np.float32)
        norm_values = norms.detach().cpu().numpy().reshape(-1).astype(np.float32)

        vectors: list[np.ndarray] = []
        for row in matrix:
            # Verify and enforce unit norm (numerical drift can cause small deviations)
            unit_norm = float(np.linalg.norm(row))
            if abs(unit_norm - 1.0) >= 1e-3:
                row = row / max(unit_norm, 1e-6)
            assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-3, "AdaFace embedding should be unit norm"
            vectors.append(row)

        return vectors, [float(value) for value in norm_values]
