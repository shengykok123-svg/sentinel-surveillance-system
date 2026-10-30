from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BACKEND_ROOT.parent
FYP_ROOT = WORKSPACE_ROOT.parent
DATA_ROOT = BACKEND_ROOT / "data"

load_dotenv(BACKEND_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise ValueError(
        f"{name} is not set. Create {BACKEND_ROOT / '.env'} from "
        f"{BACKEND_ROOT / '.env.example'} and define {name}."
    )


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _default_detection_model_path() -> str:
    return str(WORKSPACE_ROOT / "chokePoint Dataset" / "Detection_Model" / "yolov5s-face.onnx")


# Backwards-compatible alias: the detector was renamed from "scrfd_*" to
# "detection_*" when it became the single source of both boxes and keypoints.
_default_scrfd_model_path = _default_detection_model_path


def _default_recognition_model_path() -> str:
    return str(FYP_ROOT / "Recognition_Model" / "adaface_ir101_webface12m.ckpt")


def _default_camera_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "cam1",
            "name": "P2E_S1_C1.1.mp4",
            "source_type": "file",
            "source_uri": str(WORKSPACE_ROOT / "chokePoint Dataset" / "mp4" / "P2E_S1_C1.1.mp4"),
            "is_active": True,
        },
        {
            "id": "cam2",
            "name": "P2E_S5_C1.1.mp4",
            "source_type": "file",
            "source_uri": str(WORKSPACE_ROOT / "chokePoint Dataset" / "mp4" / "P2E_S5_C1.1.mp4"),
            "is_active": True,
        },
        {
            "id": "cam3",
            "name": "P2L_S5_C1.1.mp4",
            "source_type": "file",
            "source_uri": str(WORKSPACE_ROOT / "chokePoint Dataset" / "mp4" / "P2L_S5_C1.1.mp4"),
            "is_active": True,
        },
    ]


def _parse_cors_origins(raw: str | None) -> list[str]:
    if not raw:
        return ["*"]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in raw.split(",") if part.strip()] or ["*"]


def _parse_bootstrap_cameras(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return _default_camera_payload()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            cameras = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                cameras.append(
                    {
                        "id": str(item["id"]),
                        "name": str(item.get("name", item["id"])),
                        "source_type": str(item.get("source_type", "file")),
                        "source_uri": str(item.get("source_uri", "")),
                        "is_active": bool(item.get("is_active", True)),
                    }
                )
            if cameras:
                return cameras
        if isinstance(parsed, dict):
            return [
                {
                    "id": str(key),
                    "name": str(key).upper(),
                    "source_type": "file",
                    "source_uri": str(value),
                    "is_active": True,
                }
                for key, value in parsed.items()
            ]
    except json.JSONDecodeError:
        pass

    cameras: list[dict[str, Any]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        cam_id, source = item.split("=", 1)
        cam_id = cam_id.strip()
        cameras.append(
            {
                "id": cam_id,
                "name": cam_id.upper(),
                "source_type": "file",
                "source_uri": source.strip(),
                "is_active": True,
            }
        )
    return cameras or _default_camera_payload()


@dataclass(slots=True)
class Settings:
    app_name: str = "Sentinel ISS API"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    database_url: str = ""
    database_echo: bool = False
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    sensitive_access_minutes: int = 5
    detection_model_path: str = field(default_factory=_default_detection_model_path)
    scrfd_det_thresh: float = 0.5
    scrfd_input_size: int = 640
    recognition_model_path: str = field(default_factory=_default_recognition_model_path)
    recognition_architecture: str = "ir_101"
    recognition_record_snapshot_dir: str = str(DATA_ROOT / "recognition_records")
    recognition_record_jpeg_quality: int = 88
    reference_image_dir: str = str(DATA_ROOT / "reference_images")
    reference_image_jpeg_quality: int = 92
    enrollment_enhancement_enabled: bool = True
    liveportrait_python: str = str(WORKSPACE_ROOT / ".external/liveportrait-env/Scripts/python.exe")
    liveportrait_repository: str = str(WORKSPACE_ROOT / ".external/LivePortrait")
    enrollment_generation_timeout: int = 180
    face_alignment_model_pack: str = "buffalo_sc"
    face_alignment_enabled: bool = True
    face_alignment_min_confidence: float = 0.5
    infer_size: int = 640
    output_width: int = 1280
    output_height: int = 720
    jpeg_quality: int = 80
    target_fps: int = 30
    infer_every: int = 2
    # Output ring buffer. Every buffered frame is latency the viewer sees, so
    # this stays small; it exists only to absorb jitter, not to store seconds
    # of video.
    buffer_frames: int = 8
    conf_thresh: float = 0.45
    iou_thresh: float = 0.45
    # NMS IoU threshold applied to YOLOv5-face proposals.
    detection_nms_thresh: float = 0.4
    # Number of OpenCV worker threads. OpenCV defaults to one per core, which
    # oversubscribes badly once ONNX Runtime, torch, and the per-camera
    # pipeline threads are all competing for the same cores.
    cv_num_threads: int = 4
    # ONNX Runtime intra-op threads for the detector session.
    ort_intra_op_threads: int = 2
    # Faces recognised per batched AdaFace forward pass.
    recognition_batch_size: int = 8
    # Bound on the recognition backlog; jobs are dropped rather than queued
    # unboundedly when recognition cannot keep up with detection. Each queued
    # job pins the frame it was measured on, so this also caps memory.
    recognition_queue_size: int = 32
    # JPEG encode workers shared across cameras.
    encode_workers: int = 3
    enable_pipeline: bool = True
    preload_models: bool = True
    log_level: str = "INFO"
    faiss_index_path: str = str(DATA_ROOT / "faiss.index")
    faiss_metadata_path: str = str(DATA_ROOT / "faiss.meta.json")
    recognition_threshold: float = 0.53
    recognition_top_k: int = 5
    recognition_unknown_cooldown_seconds: float = 0.5
    recognition_event_cooldown_seconds: float = 30.0
    recognition_reverify_seconds: float = 60.0
    recognition_min_face_size: int = 32
    enrollment_min_face_size: int = 48
    recognition_min_blur_var: float = 40.0
    recognition_collect_seconds: float = 0.8
    recognition_sample_seconds: float = 0.2
    recognition_vote_seconds: float = 2.0
    recognition_identity_margin: float = 0.05
    live_quality_threshold: float = 1.23059
    live_quality_retry_seconds: float = 0.2
    # CR-FIQA(S) is the registration quality gate; blur remains a video-track setting.
    face_quality_model_path: str = str(BACKEND_ROOT / "models" / "crfiqa_s" / "32572backbone.pth")
    enrollment_quality_threshold: float = 1.8852214813232422
    recognition_min_det_score: float = 0.60
    recognition_template_size: int = 5
    recognition_max_good_misses: int = 4
    recognition_confirm_matches: int = 1
    recognition_reopen_margin: float = 1.25
    recognition_shadow_mode: bool = True
    # Persist detections whose best candidate fell below recognition_threshold.
    # Without these rows the event table only ever contains successful matches,
    # so the recognition rate is trivially 100% and the similarity distribution
    # is truncated at the threshold.
    publish_unknown_events: bool = True
    track_stale_seconds: float = 5.0
    track_max_age: int = 30
    # One hit to confirm keeps the pipeline responsive when infer_every > 1.
    # Raising this to 2 or 3 only makes sense at high, per-frame detection
    # rates where spurious single-frame detections are the dominant noise.
    track_n_init: int = 1
    track_max_iou_distance: float = 0.7
    track_max_cosine_distance: float = 0.6
    # Appearance descriptor used for DeepSORT association:
    #   "histogram" - HS colour histogram of the face crop. Effectively free.
    #   "mobilenet" - deep_sort_realtime's MobileNetV2 re-ID network. ~15 ms
    #                 per frame for 9 faces, which is more than the detector,
    #                 the tracker, and the JPEG encoder combined.
    #   "none"      - Kalman + IoU association only.
    # Faces detected at >=15 Hz have strong frame-to-frame IoU continuity, so
    # the deep descriptor buys very little here. See DEFAULT_COSINE_DISTANCE.
    track_embedder: str = "histogram"
    bootstrap_cameras: list[dict[str, Any]] = field(default_factory=_default_camera_payload)


# Sane max_cosine_distance per appearance descriptor. MobileNet re-ID features
# are far more discriminative than a colour histogram, so they tolerate a much
# tighter gate. Applied only when SENTINEL_TRACK_MAX_COSINE_DISTANCE is unset.
DEFAULT_COSINE_DISTANCE = {
    "mobilenet": 0.3,
    "histogram": 0.6,
    "none": 1.0,
}


def _resolve_cosine_distance(embedder: str) -> float:
    raw = os.getenv("SENTINEL_TRACK_MAX_COSINE_DISTANCE")
    if raw is not None:
        return float(raw)
    return DEFAULT_COSINE_DISTANCE.get(embedder, 0.6)


def get_settings() -> Settings:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    track_embedder = os.getenv("SENTINEL_TRACK_EMBEDDER", "histogram").strip().lower()
    return Settings(
        app_name=os.getenv("SENTINEL_APP_NAME", "Sentinel ISS API"),
        cors_origins=_parse_cors_origins(os.getenv("SENTINEL_CORS_ORIGINS")),
        database_url=_require_env("SENTINEL_DATABASE_URL"),
        database_echo=_parse_bool(os.getenv("SENTINEL_DATABASE_ECHO"), False),
        jwt_secret=_require_env("SENTINEL_JWT_SECRET"),
        jwt_algorithm=os.getenv("SENTINEL_JWT_ALGORITHM", "HS256"),
        jwt_expire_minutes=max(_parse_int(os.getenv("SENTINEL_JWT_EXPIRE_MINUTES"), 60), 1),
        sensitive_access_minutes=5,
        detection_model_path=os.getenv(
            "SENTINEL_DETECTION_MODEL_PATH",
            os.getenv("SENTINEL_SCRFD_MODEL_PATH", _default_detection_model_path()),
        ),
        scrfd_det_thresh=_parse_float(os.getenv("SENTINEL_SCRFD_DET_THRESH"), 0.5),
        scrfd_input_size=_parse_int(os.getenv("SENTINEL_INFER_SIZE"), 640),
        recognition_model_path=os.getenv(
            "SENTINEL_RECOGNITION_MODEL_PATH",
            _default_recognition_model_path(),
        ),
        recognition_architecture=os.getenv("SENTINEL_RECOGNITION_ARCH", "ir_101"),
        recognition_record_snapshot_dir=os.getenv(
            "SENTINEL_RECOGNITION_RECORD_SNAPSHOT_DIR",
            str(DATA_ROOT / "recognition_records"),
        ),
        recognition_record_jpeg_quality=max(
            min(_parse_int(os.getenv("SENTINEL_RECOGNITION_RECORD_JPEG_QUALITY"), 88), 100), 1
        ),
        reference_image_dir=os.getenv("SENTINEL_REFERENCE_IMAGE_DIR", str(DATA_ROOT / "reference_images")),
        enrollment_enhancement_enabled=_parse_bool(os.getenv("SENTINEL_ENROLLMENT_ENHANCEMENT_ENABLED"), True),
        liveportrait_python=os.getenv("SENTINEL_LIVEPORTRAIT_PYTHON", str(WORKSPACE_ROOT / ".external/liveportrait-env/Scripts/python.exe")),
        liveportrait_repository=os.getenv("SENTINEL_LIVEPORTRAIT_REPOSITORY", str(WORKSPACE_ROOT / ".external/LivePortrait")),
        enrollment_generation_timeout=_parse_int(os.getenv("SENTINEL_ENROLLMENT_GENERATION_TIMEOUT"), 180),
        reference_image_jpeg_quality=max(
            min(_parse_int(os.getenv("SENTINEL_REFERENCE_IMAGE_JPEG_QUALITY"), 92), 100), 1
        ),
        face_alignment_model_pack=os.getenv("SENTINEL_ALIGN_MODEL_PACK", "buffalo_sc"),
        face_alignment_enabled=_parse_bool(os.getenv("SENTINEL_ALIGN_ENABLED"), True),
        face_alignment_min_confidence=_parse_float(os.getenv("SENTINEL_ALIGN_MIN_CONFIDENCE"), 0.5),
        infer_size=_parse_int(os.getenv("SENTINEL_INFER_SIZE"), 640),
        output_width=_parse_int(os.getenv("SENTINEL_OUTPUT_W"), 1280),
        output_height=_parse_int(os.getenv("SENTINEL_OUTPUT_H"), 720),
        jpeg_quality=_parse_int(os.getenv("SENTINEL_JPEG_QUALITY"), 80),
        target_fps=_parse_int(os.getenv("SENTINEL_TARGET_FPS"), 30),
        infer_every=max(_parse_int(os.getenv("SENTINEL_INFER_EVERY"), 2), 1),
        buffer_frames=max(_parse_int(os.getenv("SENTINEL_BUFFER_FRAMES"), 8), 1),
        conf_thresh=_parse_float(os.getenv("SENTINEL_CONF_THRESH"), 0.45),
        iou_thresh=_parse_float(os.getenv("SENTINEL_IOU_THRESH"), 0.45),
        detection_nms_thresh=_parse_float(os.getenv("SENTINEL_DETECTION_NMS_THRESH"), 0.4),
        cv_num_threads=max(_parse_int(os.getenv("SENTINEL_CV_NUM_THREADS"), 4), 1),
        ort_intra_op_threads=max(_parse_int(os.getenv("SENTINEL_ORT_INTRA_OP_THREADS"), 2), 1),
        recognition_batch_size=max(_parse_int(os.getenv("SENTINEL_RECOGNITION_BATCH_SIZE"), 8), 1),
        recognition_queue_size=max(_parse_int(os.getenv("SENTINEL_RECOGNITION_QUEUE_SIZE"), 32), 1),
        encode_workers=max(_parse_int(os.getenv("SENTINEL_ENCODE_WORKERS"), 3), 1),
        enable_pipeline=_parse_bool(os.getenv("SENTINEL_ENABLE_PIPELINE"), True),
        preload_models=_parse_bool(os.getenv("SENTINEL_PRELOAD_MODELS"), True),
        log_level=os.getenv("SENTINEL_LOG_LEVEL", "INFO"),
        faiss_index_path=os.getenv("SENTINEL_FAISS_INDEX_PATH", str(DATA_ROOT / "faiss.index")),
        faiss_metadata_path=os.getenv(
            "SENTINEL_FAISS_METADATA_PATH",
            str(DATA_ROOT / "faiss.meta.json"),
        ),
        recognition_threshold=_parse_float(os.getenv("SENTINEL_RECOGNITION_THRESHOLD"), 0.53),
        recognition_top_k=max(_parse_int(os.getenv("SENTINEL_RECOGNITION_TOP_K"), 5), 1),
        recognition_unknown_cooldown_seconds=_parse_float(
            os.getenv("SENTINEL_UNKNOWN_COOLDOWN_SECONDS"),
            0.5,
        ),
        recognition_event_cooldown_seconds=_parse_float(
            os.getenv("SENTINEL_EVENT_COOLDOWN_SECONDS"),
            30.0,
        ),
        recognition_reverify_seconds=_parse_float(
            os.getenv("SENTINEL_REVERIFY_SECONDS"),
            60.0,
        ),
        recognition_min_face_size=_parse_int(
            os.getenv("SENTINEL_RECOGNITION_MIN_FACE_SIZE"), 32
        ),
        enrollment_min_face_size=_parse_int(
            os.getenv("SENTINEL_ENROLLMENT_MIN_FACE_SIZE"), 48
        ),
        recognition_min_blur_var=_parse_float(
            os.getenv("SENTINEL_RECOGNITION_MIN_BLUR_VAR"), 40.0
        ),
        recognition_collect_seconds=max(0.2, _parse_float(os.getenv("SENTINEL_RECOGNITION_COLLECT_SECONDS"), 0.8)),
        recognition_sample_seconds=max(0.15, _parse_float(os.getenv("SENTINEL_RECOGNITION_SAMPLE_SECONDS"), 0.2)),
        recognition_vote_seconds=max(1.0, _parse_float(os.getenv("SENTINEL_RECOGNITION_VOTE_SECONDS"), 2.0)),
        recognition_identity_margin=max(0.0, _parse_float(os.getenv("SENTINEL_RECOGNITION_IDENTITY_MARGIN"), 0.05)),
        live_quality_threshold=_parse_float(os.getenv("SENTINEL_LIVE_QUALITY_THRESHOLD"), 1.23059),
        live_quality_retry_seconds=max(0.2, _parse_float(os.getenv("SENTINEL_LIVE_QUALITY_RETRY_SECONDS"), 0.2)),
        face_quality_model_path=os.getenv(
            "SENTINEL_FACE_QUALITY_MODEL_PATH", str(BACKEND_ROOT / "models" / "crfiqa_s" / "32572backbone.pth")
        ),
        enrollment_quality_threshold=_parse_float(
            os.getenv("SENTINEL_ENROLLMENT_QUALITY_THRESHOLD"), 1.8852214813232422
        ),
        recognition_min_det_score=_parse_float(
            os.getenv("SENTINEL_RECOGNITION_MIN_DET_SCORE"), 0.60
        ),
        recognition_template_size=_parse_int(
            os.getenv("SENTINEL_RECOGNITION_TEMPLATE_SIZE"), 5
        ),
        recognition_max_good_misses=_parse_int(
            os.getenv("SENTINEL_RECOGNITION_MAX_GOOD_MISSES"), 4
        ),
        recognition_confirm_matches=_parse_int(
            os.getenv("SENTINEL_RECOGNITION_CONFIRM_MATCHES"), 1
        ),
        recognition_reopen_margin=_parse_float(
            os.getenv("SENTINEL_RECOGNITION_REOPEN_MARGIN"), 1.25
        ),
        recognition_shadow_mode=_parse_bool(
            os.getenv("SENTINEL_RECOGNITION_SHADOW_MODE"), True
        ),
        publish_unknown_events=_parse_bool(os.getenv("SENTINEL_PUBLISH_UNKNOWN_EVENTS"), True),
        track_stale_seconds=_parse_float(os.getenv("SENTINEL_TRACK_STALE_SECONDS"), 5.0),
        track_max_age=max(_parse_int(os.getenv("SENTINEL_TRACK_MAX_AGE"), 30), 1),
        track_n_init=max(_parse_int(os.getenv("SENTINEL_TRACK_N_INIT"), 1), 1),
        track_max_iou_distance=_parse_float(
            os.getenv("SENTINEL_TRACK_MAX_IOU_DISTANCE"),
            0.7,
        ),
        track_max_cosine_distance=_resolve_cosine_distance(track_embedder),
        track_embedder=track_embedder,
        bootstrap_cameras=_parse_bootstrap_cameras(os.getenv("SENTINEL_BOOTSTRAP_CAMERAS")),
    )
