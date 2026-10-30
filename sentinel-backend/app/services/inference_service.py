"""YOLOv5-face ONNX detection with boxes and five alignment landmarks."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.core.config import Settings


@dataclass(slots=True)
class DetectionResult:
    """Faces found in a single frame.

    Attributes:
        bboxes: ``(N, 5)`` float32 array of ``[x1, y1, x2, y2, score]`` in
            original-frame pixel coordinates, sorted by descending score.
        kpss: ``(N, 5, 2)`` float32 array of 5-point landmarks (left eye,
            right eye, nose, left mouth corner, right mouth corner) in the
            same coordinate space.
    """

    bboxes: np.ndarray
    kpss: np.ndarray

    def __len__(self) -> int:
        return int(self.bboxes.shape[0])

    @classmethod
    def empty(cls) -> "DetectionResult":
        return cls(
            bboxes=np.zeros((0, 5), dtype=np.float32),
            kpss=np.zeros((0, 5, 2), dtype=np.float32),
        )


class InferenceService:
    """YOLOv5-face detector producing boxes and 5-point keypoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger("sentinel.inference")
        self._session: Any = None
        self._input_name: str | None = None
        self._output_names: list[str] = []
        self._input_size: int = settings.infer_size
        # The IO binding reuses one device input buffer, so a session can only
        # serve one caller at a time. Live cameras therefore get a session
        # each (see ``VideoService.detector_factory``); this lock only guards
        # the shared instance used by enrollment and search.
        self._infer_lock = threading.Lock()
        self._io_binding: Any = None
        self._device_input: Any = None
        self._start_lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def start(self) -> None:
        """Load the ONNX model, bind device buffers, and warm up the session."""
        if self._session is not None:
            return

        with self._start_lock:
            if self._session is not None:
                return

            import torch

            # ONNX Runtime resolves CUDA/cuDNN from PATH on Windows; the copies
            # shipped inside the torch wheel are the ones we know match.
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Left unbounded, ORT spawns one thread per core and fights the
            # OpenCV pool and the per-camera pipeline threads for the same CPUs.
            sess_options.intra_op_num_threads = self.settings.ort_intra_op_threads

            session = ort.InferenceSession(
                self.settings.detection_model_path,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            model_input = session.get_inputs()[0]
            self._input_name = model_input.name
            self._output_names = [output.name for output in session.get_outputs()]

            output_shape = session.get_outputs()[0].shape
            if len(self._output_names) != 1 or len(output_shape) != 3 or output_shape[-1] != 16:
                raise RuntimeError("Expected YOLOv5-face output [1, N, 16] (box, objectness, landmarks, class)")

            # Honour the size baked into the graph; these models are exported
            # with a fixed square input.
            shape = model_input.shape
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self._input_size = int(shape[2])
                if shape[2] != shape[3]:
                    raise RuntimeError(f"YOLOv5-face input must be square, got {shape}")
            else:
                self._input_size = self.settings.infer_size

            self._session = session

            # Persistent device-side input buffer, reused for every frame.
            self._device_input = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32),
                "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu",
                0,
            )
            self._io_binding = session.io_binding()
            self._io_binding.bind_ortvalue_input(self._input_name, self._device_input)
            for name in self._output_names:
                self._io_binding.bind_output(name, "cpu")

            # Warm-up: allocate workspace and trigger kernel autotuning so the
            # first real frame is not 10x slower than steady state.
            dummy = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
            for _ in range(3):
                self._run(dummy)

            self.logger.info(
                "Detection ONNX provider ready: %s (input=%s, model=%s)",
                session.get_providers()[0],
                self._input_size,
                os.path.basename(self.settings.detection_model_path),
            )

    def stop(self) -> None:
        """Release the ONNX session and all device-side buffers."""
        self._session = None
        self._input_name = None
        self._output_names = []
        self._io_binding = None
        self._device_input = None

    # -------------------------------------------------------------- inference

    def _run(self, blob: np.ndarray) -> list[np.ndarray]:
        """Execute the session on a preprocessed NCHW blob."""
        with self._infer_lock:
            self._device_input.update_inplace(np.ascontiguousarray(blob))
            self._session.run_with_iobinding(self._io_binding)
            return self._io_binding.copy_outputs_to_cpu()

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Centre-letterbox to fixed ONNX size, then convert BGR to RGB / 255."""
        size = self._input_size
        height, width = frame.shape[:2]
        scale = min(size / height, size / width)
        new_w, new_h = round(width * scale), round(height * scale)
        left = (size - new_w) // 2
        top = (size - new_h) // 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[top:top + new_h, left:left + new_w] = resized
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (size, size), swapRB=True)
        return blob, scale, (left, top)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self.start()
        blob, scale, padding = self.preprocess(frame)
        return self._decode(self._run(blob), scale, padding, frame.shape[1], frame.shape[0])

    def detect_faces(self, frame: np.ndarray) -> DetectionResult:
        return self.detect(frame)

    def _decode(self, outputs: list[np.ndarray], scale: float,
                padding: tuple[int, int], orig_w: int, orig_h: int) -> DetectionResult:
        # Export already decodes anchors: xywh, objectness, 5 xy landmarks, class.
        predictions = np.asarray(outputs[0])[0]
        scores = predictions[:, 4] * predictions[:, 15]
        valid = (scores >= self.settings.conf_thresh) & np.isfinite(predictions).all(axis=1)
        predictions, scores = predictions[valid], scores[valid]
        if not len(predictions):
            return DetectionResult.empty()
        boxes = np.concatenate((predictions[:, :2] - predictions[:, 2:4] / 2,
                                predictions[:, :2] + predictions[:, 2:4] / 2), axis=1)
        landmarks = predictions[:, 5:15].reshape(-1, 5, 2).copy()
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
        keep = self._nms(boxes, scores, self.settings.detection_nms_thresh)
        boxes, landmarks, scores = boxes[keep], landmarks[keep], scores[keep]
        offset = np.asarray(padding, dtype=np.float32)
        boxes = (boxes - np.tile(offset, 2)) / scale
        landmarks = (landmarks - offset) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
        return DetectionResult(np.column_stack((boxes, scores)).astype(np.float32),
                               landmarks.astype(np.float32))

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
        """Greedy NMS. Assumes ``boxes``/``scores`` are already score-sorted."""
        if len(boxes) == 0:
            return np.zeros((0,), dtype=np.int64)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        order = np.arange(len(boxes))
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thresh]

        return np.asarray(keep, dtype=np.int64)
