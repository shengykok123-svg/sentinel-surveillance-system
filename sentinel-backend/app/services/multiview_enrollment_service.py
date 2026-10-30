"""Optional offline LivePortrait view generation; never writes enrollment records.

Run in the isolated LivePortrait environment. Input/output images use BGR, as
does Sentinel. The generator itself receives RGB. No probe images are inputs.
"""
from pathlib import Path
import sys
import cv2
import numpy as np


def anchored_templates(original, synthetic, original_weight=0.75):
    """One original-anchored unit template per synthetic view."""
    if not 0 <= original_weight <= 1:
        raise ValueError("original_weight must be in [0, 1]")
    original = np.asarray(original, dtype=np.float32)
    original = original / max(float(np.linalg.norm(original)), 1e-12)
    synthetic = np.asarray(synthetic, dtype=np.float32).reshape(-1, original.size)
    if not len(synthetic):
        return original[None]
    synthetic = synthetic / np.maximum(np.linalg.norm(synthetic, axis=1, keepdims=True), 1e-12)
    mixed = original_weight * original + (1 - original_weight) * synthetic
    mixed /= np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)
    return np.vstack([original, mixed])


class MultiViewEnrollmentService:
    """Lazy generator using official human models and Sentinel's five landmarks.

    Angles are target yaw commands in LivePortrait's coordinate convention.
    Reported output angles are self-estimates, not ground-truth measurements.
    """
    def __init__(self, repository: Path):
        self.repository = Path(repository).resolve()
        self.wrapper = None

    def start(self):
        if self.wrapper is not None:
            return
        if not (self.repository / "src/live_portrait_wrapper.py").exists():
            raise FileNotFoundError("Install the official LivePortrait repository first")
        sys.path.insert(0, str(self.repository))
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        self.wrapper = LivePortraitWrapper(InferenceConfig(flag_use_half_precision=True))

    def generate(self, frame_bgr, landmarks, yaws=(-45, -30, -15, 0, 15, 30, 45)):
        self.start()
        from src.utils.crop import crop_image
        import torch
        if np.asarray(landmarks).shape != (5, 2):
            raise ValueError("Expected five Sentinel face landmarks")
        # Five landmarks span eyes/mouth rather than the full face contour.
        # Expand to a portrait crop, retaining forehead/chin around that span.
        crop = crop_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                          landmarks, dsize=512, scale=3.0, vy_ratio=-0.1)['img_crop']
        w = self.wrapper
        with torch.inference_mode():
            source = w.prepare_source(crop)
            info = w.get_kp_info(source)
            feature = w.extract_feature_3d(source)
            source_kp = w.transform_keypoint(info)
            outputs = []
            for yaw in yaws:
                target = {k: v.clone() for k, v in info.items()}
                target['yaw'] = torch.full_like(info['yaw'], float(yaw))
                driving = w.transform_keypoint(target)
                # Pose-only: keep source expression, pitch, roll, scale and translation.
                image_rgb = w.parse_output(w.warp_decode(feature, source_kp, driving)['out'])[0]
                measured = w.get_kp_info(w.prepare_source(image_rgb))
                outputs.append((cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), dict(
                    target_yaw=float(yaw), source_yaw=float(info['yaw'].item()),
                    estimated_output_yaw=float(measured['yaw'].item()))))
        return outputs

    def stop(self):
        self.wrapper = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
