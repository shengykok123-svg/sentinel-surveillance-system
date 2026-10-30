"""
Face alignment service using 5-point facial landmark detection and affine
transformation.

This module provides face alignment capabilities using InsightFace's landmark
detection models. Alignment normalises a detected face to a canonical 112x112
template by:
    1. Detecting 5 facial landmarks (left eye, right eye, nose tip, left mouth
       corner, right mouth corner) within an expanded ROI around the face bbox.
    2. Computing a partial affine transformation from the detected landmarks
       to canonical reference positions.
    3. Warping the original image to produce a consistently oriented 112x112
       face crop.

This alignment step significantly improves face recognition accuracy by
removing variations in head pose, scale, and translation.

Key constants:
    - ``REFERENCE_LANDMARKS_112``: Canonical 5-point landmark positions for
      the 112x112 alignment template (derived from the ArcFace standard).
    - ``_ROI_MARGIN``: Fractional expansion (0.35) around the detection bbox
      when extracting the landmark search ROI.
    - ``_MIN_ROI_SIDE``: Minimum ROI dimension (64 pixels) below which
      landmark detection is skipped.
    - ``_MIN_DETECTION_SIDE``: Minimum detection bbox side (20 pixels) for
      a valid landmark result.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from app.core.config import Settings


# Fractional margin applied around the face detection bbox to create the
# landmark detection ROI. A larger margin helps capture the full face region
# including forehead and chin for more reliable landmark detection.
_ROI_MARGIN = 0.35

# Minimum side length (pixels) for the extracted ROI. Below this, landmark
# detection is unreliable and is skipped.
_MIN_ROI_SIDE = 64

# Minimum side length (pixels) for a valid detection within the ROI.
# Detections smaller than this are discarded as too small for reliable landmarks.
_MIN_DETECTION_SIDE = 20.0

# Service class definition for face alignment using 5-point landmark detection and affine warping.
class FaceAlignmentService:

    # Canonical 112x112 template landmarks derived from the ArcFace standard.
    # These define the target positions for the 5 facial keypoints after alignment.
    # Order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
    REFERENCE_LANDMARKS_112 = np.array(
        [
            [38.2946, 51.6963],   # Left eye centre
            [73.5318, 51.5014],   # Right eye centre
            [56.0252, 71.7366],   # Nose tip
            [41.5493, 92.3655],   # Left mouth corner
            [70.7299, 92.2041],   # Right mouth corner
        ],
        dtype=np.float32,
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger("sentinel.alignment")
        self._model = None                   # InsightFace detection model with landmark support
        self._model_path: Path | None = None  # Path to the loaded ONNX model file
        self._device_label = "cpu"           # "cpu" or "cuda" label for logging
        self._start_lock = threading.Lock()  # Guards model loading (one-time init)
        self._infer_lock = threading.Lock()  # Guards concurrent inference calls

# Start the service by loading the InsightFace model pack and preparing the landmark detection model.
    def start(self) -> None:
       
        if not self.settings.face_alignment_enabled or self._model is not None:
            return

        # Double-checked locking for thread-safe singleton initialisation
        with self._start_lock:
            if not self.settings.face_alignment_enabled or self._model is not None:
                return

            from insightface.model_zoo import get_model
            from insightface.utils import ensure_available

            # Download or locate the model pack directory
            model_dir = Path(
                ensure_available("models", self.settings.face_alignment_model_pack, root="~/.insightface")
            )
            providers = self._providers()
            # ctx_id=0 for GPU, ctx_id=-1 for CPU (InsightFace convention)
            ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1

            detection_model = None
            detection_path: Path | None = None
            # Prioritise dedicated detector files (det*.onnx) over generic models
            det_paths = sorted(model_dir.glob("det*.onnx"))
            other_paths = [path for path in sorted(model_dir.glob("*.onnx")) if path not in det_paths]
            candidate_paths = det_paths + other_paths
            for onnx_file in candidate_paths:
                model = get_model(str(onnx_file), providers=providers)
                if model is None:
                    continue
                # Look for a detection model that supports keypoint (landmark) output
                if getattr(model, "taskname", None) == "detection" and getattr(model, "use_kps", False):
                    detection_model = model
                    detection_path = onnx_file
                    break

            if detection_model is None or detection_path is None:
                raise RuntimeError(
                    f"No 5-point landmark-capable detector found in model pack '{self.settings.face_alignment_model_pack}'"
                )

            # Prepare the model with the configured minimum detection confidence
            detection_model.prepare(ctx_id, det_thresh=self.settings.face_alignment_min_confidence)
            self._model = detection_model
            self._model_path = detection_path
            self._device_label = "cuda" if ctx_id == 0 else "cpu"
            self.logger.info(
                "Landmark model loaded on %s from %s",
                self._device_label,
                detection_path,
            )

# Stop the service by releasing the model and resetting state.
    def stop(self) -> None:
        self._model = None
        self._model_path = None
        self._device_label = "cpu"

# Get 5-point facial landmarks for a given face bounding box in the input frame, returning the landmark coordinates in the original frame's coordinate system.
    def get_5pt_landmarks(self, frame_bgr: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        if not self.settings.face_alignment_enabled:
            return None

        self.start()  # Ensure the model is loaded (no-op if already started)
        if self._model is None:
            return None

        # Extract an expanded ROI around the bbox for landmark detection
        roi, offsets = self._extract_roi(frame_bgr, bbox)
        if roi is None:
            return None

        # Determine the model input size (padded to multiples of 32)
        detect_size = self._detect_input_size(roi.shape[1], roi.shape[0])
        # Thread-safe inference
        with self._infer_lock:
            detections, kpss = self._model.detect(roi, input_size=detect_size, max_num=5)

        if detections is None or kpss is None or len(detections) == 0 or len(kpss) == 0:
            return None

        # Map ROI-local coordinates back to the original frame
        left, top = offsets
        landmarks = self._select_best_landmarks(
            detections=np.asarray(detections, dtype=np.float32),
            kpss=np.asarray(kpss, dtype=np.float32),
            original_bbox=np.asarray(bbox, dtype=np.float32),
            offset_x=left,
            offset_y=top,
        )
        if landmarks is None:
            return None
        return landmarks.astype(np.float32)

# Align the face in the input frame according to the detected landmarks, returning a 112x112 aligned face crop in BGR format. Returns None if alignment fails or is disabled.
    def align(self, frame_bgr: np.ndarray, bbox: list[int]) -> np.ndarray | None:

        if not self.settings.face_alignment_enabled:
            return None

        landmarks = self.get_5pt_landmarks(frame_bgr, bbox)
        if landmarks is None or landmarks.shape != (5, 2):
            return None

        # Compute partial affine (similarity) transform: rotation + scale + translation
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.REFERENCE_LANDMARKS_112,
            method=cv2.LMEDS,  # Least Median of Squares for robustness to outliers
        )
        if matrix is None:
            return None

        # Warp the full frame to the canonical 112x112 alignment template
        return cv2.warpAffine(frame_bgr, matrix, (112, 112), borderValue=0.0)

    def align_from_keypoints(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray | None:
        """Align using pre-computed 5-point keypoints — no re-detection needed.

        This is a pure geometric operation: compute the affine transform from
        the provided keypoints to ``REFERENCE_LANDMARKS_112`` and warp the
        frame.  It does NOT use the SCRFD 500M model and requires no lock.

        Args:
            frame_bgr: Input BGR image as a NumPy array (H, W, 3).
            kps:       Pre-computed 5-point keypoints of shape (5, 2).

        Returns:
            A 112x112 aligned face crop, or ``None`` if the transform fails.
        """
        if kps is None or kps.shape != (5, 2):
            return None

        landmarks = np.asarray(kps, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.REFERENCE_LANDMARKS_112,
            method=cv2.LMEDS,
        )
        if matrix is None:
            return None

        return cv2.warpAffine(frame_bgr, matrix, (112, 112), borderValue=0.0)

    @staticmethod
    # Helper method to determine the available ONNX Runtime execution providers
    def _providers() -> list[str]:

        available = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    # Helper method to compute the diagonal length of a bounding box, used for normalising distance penalties in landmark selection.
    def _detect_input_size(width: int, height: int) -> tuple[int, int]:
        
        side = max(width, height)
        # Round up to nearest multiple of 32 (required by most detection models)
        side = max(_MIN_ROI_SIDE * 2, int(math.ceil(side / 32.0) * 32))
        # Cap at 640 to limit computation
        side = min(side, 640)
        return side, side

    @staticmethod
    # Extract an expanded Region of Interest (ROI) around the face bbox.
    def _extract_roi(frame_bgr: np.ndarray, bbox: list[int]) -> tuple[np.ndarray | None, tuple[int, int]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        # Compute padding based on the larger bbox dimension
        pad = int(round(max(box_w, box_h) * _ROI_MARGIN))

        # Expand and clamp to frame boundaries
        left = max(x1 - pad, 0)
        top = max(y1 - pad, 0)
        right = min(x2 + pad, frame_w)
        bottom = min(y2 + pad, frame_h)

        # Validate minimum ROI size for reliable landmark detection
        if right - left < _MIN_ROI_SIDE or bottom - top < _MIN_ROI_SIDE:
            return None, (left, top)
        return frame_bgr[top:bottom, left:right], (left, top)

# Helper method to select the best set of 5-point landmarks from multiple detections within the ROI, scoring them based on IoU with the original bbox, detection confidence, and centre distance penalty.
    def _select_best_landmarks(
        self,
        *,
        detections: np.ndarray,
        kpss: np.ndarray,
        original_bbox: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> np.ndarray | None:
       
        best_landmarks: np.ndarray | None = None
        best_score = float("-inf")
        # Diagonal of the original bbox, used to normalise distance penalties
        target_diag = self._bbox_diagonal(original_bbox)

        for det, kps in zip(detections, kpss, strict=False):
            # Extract detection confidence score
            score = float(det[4]) if det.shape[0] >= 5 else 0.0
            if score < self.settings.face_alignment_min_confidence:
                continue

            # Convert ROI-local detection bbox to frame coordinates
            det_box = np.array(
                [
                    det[0] + offset_x,
                    det[1] + offset_y,
                    det[2] + offset_x,
                    det[3] + offset_y,
                ],
                dtype=np.float32,
            )
            det_w = det_box[2] - det_box[0]
            det_h = det_box[3] - det_box[1]
            # Skip detections that are too small for reliable landmarks
            if min(det_w, det_h) < _MIN_DETECTION_SIDE:
                continue

            # Compute spatial overlap and distance metrics
            iou = self._bbox_iou(det_box, original_bbox)
            center_distance = self._bbox_center_distance(det_box, original_bbox)
            # Skip detections with no overlap and too far from the target
            if iou <= 0.0 and center_distance > max(target_diag * 1.25, 32.0):
                continue

            # Composite scoring: IoU-weighted + confidence - distance penalty
            match_score = (iou * 2.5) + score - min(center_distance / max(target_diag, 1.0), 2.0) * 0.25
            if match_score > best_score:
                best_score = match_score
                # Copy landmarks and translate from ROI-local to frame coordinates
                best_landmarks = np.asarray(kps, dtype=np.float32).copy()
                best_landmarks[:, 0] += offset_x
                best_landmarks[:, 1] += offset_y

        return best_landmarks

    @staticmethod
    # Helper method to compute Intersection over Union (IoU) between two bounding boxes, used for scoring landmark detections against the original bbox.
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        
        # Compute intersection rectangle
        inter_x1 = max(float(box_a[0]), float(box_b[0]))
        inter_y1 = max(float(box_a[1]), float(box_b[1]))
        inter_x2 = min(float(box_a[2]), float(box_b[2]))
        inter_y2 = min(float(box_a[3]), float(box_b[3]))
        inter_w = max(inter_x2 - inter_x1, 0.0)
        inter_h = max(inter_y2 - inter_y1, 0.0)
        if inter_w == 0.0 or inter_h == 0.0:
            return 0.0
        inter = inter_w * inter_h
        # Compute individual areas (clamped to minimum of 1.0 to avoid division by zero)
        area_a = max(float(box_a[2] - box_a[0]), 1.0) * max(float(box_a[3] - box_a[1]), 1.0)
        area_b = max(float(box_b[2] - box_b[0]), 1.0) * max(float(box_b[3] - box_b[1]), 1.0)
        # IoU = intersection / union
        return inter / max(area_a + area_b - inter, 1.0)

    @staticmethod
    # Helper method to compute the Euclidean distance between the centres of two bounding boxes, used for penalising landmark detections that are far from the original bbox.
    def _bbox_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
        
        acx = (float(box_a[0]) + float(box_a[2])) / 2.0
        acy = (float(box_a[1]) + float(box_a[3])) / 2.0
        bcx = (float(box_b[0]) + float(box_b[2])) / 2.0
        bcy = (float(box_b[1]) + float(box_b[3])) / 2.0
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)

    @staticmethod
    # Helper method to compute the diagonal length of a bounding box, used for normalising distance penalties in landmark selection.
    def _bbox_diagonal(box: np.ndarray) -> float:
        width = max(float(box[2] - box[0]), 1.0)
        height = max(float(box[3] - box[1]), 1.0)
        return float((width**2 + height**2) ** 0.5)
