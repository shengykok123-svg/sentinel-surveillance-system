"""
FastAPI application factory and service wiring.

``create_app()`` builds the ASGI application with all middleware and
route handlers.  ``build_services()`` instantiates every repository,
service, and infrastructure object, then attaches them to
``app.state`` for dependency injection via ``app/api/deps.py``.

The ``lifespan`` context manager handles startup (model loading,
pipeline threads) and shutdown (clean resource release).
"""

from __future__ import annotations

import asyncio
import warnings
import logging
from contextlib import asynccontextmanager, nullcontext, suppress
from app.core.backend_instance import BackendInstanceLock

import cv2
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

# ── API routers ─────────────────────────────────────────────────────
from app.api.routes_analytics import router as analytics_router
from app.api.routes_cameras import router as cameras_router
from app.api.routes_events import router as events_router
from app.api.routes_auth import router as auth_router
from app.api.routes_poi import router as poi_router
from app.api.routes_search import router as search_router
from app.api.routes_stream import router as stream_router, ws_router as stream_ws_router
from app.api.routes_system import router as system_router

# ── Core infrastructure ─────────────────────────────────────────────
from app.core.config import get_settings, DATA_ROOT
from app.services.settings_service import SettingsService
from app.api.routes_settings import router as settings_router
from app.core.database import DatabaseManager
from app.core.logging import configure_logging
from app.core.websocket_manager import WebSocketManager

# ── Data access repositories ────────────────────────────────────────
from app.repositories.analytics_repository import AnalyticsRepository
from app.repositories.camera_repository import CameraRepository
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.person_repository import PersonRepository
from app.repositories.recognition_event_repository import RecognitionEventRepository
from app.repositories.reference_image_repository import ReferenceImageRepository
from app.repositories.reference_image_access_repository import ReferenceImageAccessRepository
from app.repositories.user_repository import UserRepository

# ── Business logic services ─────────────────────────────────────────
from app.services.adaface_service import AdaFaceService
from app.services.analytics_service import AnalyticsService
from app.services.auth_service import AuthService
from app.services.camera_service import CameraService
from app.services.enrollment_service import EnrollmentService
from app.services.enrollment_enhancement_service import EnrollmentEnhancementService
from app.services.event_service import EventService
from app.services.face_alignment_service import FaceAlignmentService
from app.services.faiss_service import FaissService
from app.services.inference_service import InferenceService
from app.services.recognition_service import RecognitionService
from app.services.recognition_snapshot_service import RecognitionSnapshotService
from app.services.reference_image_service import ReferenceImageService
from app.services.resource_service import ResourceService
from app.services.tracking_service import TrackingService
from app.services.video_service import VideoService
from app.services.stream_access_service import StreamAccessService
from app.services.enrollment_job_service import EnrollmentJobService

# Suppress noisy FutureWarnings from third-party libraries (torch, etc.).
warnings.filterwarnings("ignore", category=FutureWarning)


def build_services(app: FastAPI) -> None:
    """
    Instantiate all services and attach them to ``app.state``.

    Called once during the ``lifespan`` startup phase.  The wiring
    order matters — each service receives its dependencies via
    constructor injection.
    """
    settings = get_settings()
    app.state.settings_service = SettingsService(settings, DATA_ROOT / "runtime_settings.json")
    configure_logging(settings.log_level)

    # OpenCV spawns one worker thread per core by default. Combined with ONNX
    # Runtime's intra-op pool, torch's own pool, and one reader plus one
    # pipeline thread per camera, that oversubscribes every core badly enough
    # that context switching costs more than the parallelism returns.
    cv2.setNumThreads(max(settings.cv_num_threads, 1))

    # ── Infrastructure layer ────────────────────────────────────────
    websocket_manager = WebSocketManager()
    app.state.stream_access_service = StreamAccessService()
    app.state.enrollment_job_service = EnrollmentJobService()
    database = DatabaseManager(settings)
    database.init()  # Validate / create schema

    # ── Repository layer ────────────────────────────────────────────
    analytics_repository = AnalyticsRepository(database)
    camera_repository = CameraRepository(database)
    person_repository = PersonRepository(database)
    embedding_repository = EmbeddingRepository(database)
    event_repository = RecognitionEventRepository(database)
    reference_image_repository = ReferenceImageRepository(database)
    reference_image_access_repository = ReferenceImageAccessRepository(database)
    user_repository = UserRepository(database)

    # ── Service layer ───────────────────────────────────────────────
    camera_service = CameraService(camera_repository, settings)
    camera_service.ensure_bootstrap_cameras()  # Seed default cameras from config
    auth_service = AuthService(settings, user_repository)

    analytics_service = AnalyticsService(analytics_repository, settings)
    snapshot_service = RecognitionSnapshotService(settings)
    reference_image_service = ReferenceImageService(settings)
    event_service = EventService(event_repository, websocket_manager, snapshot_service)
    inference_service = InferenceService(settings)
    adaface_service = AdaFaceService(settings)
    face_alignment_service = FaceAlignmentService(settings)
    faiss_service = FaissService(settings)
    recognition_service = RecognitionService(
        settings=settings,
        inference_service=inference_service,
        adaface_service=adaface_service,
        face_alignment_service=face_alignment_service,
        faiss_service=faiss_service,
        person_repository=person_repository,
        embedding_repository=embedding_repository,
        event_repository=event_repository,
    )
    resource_service = ResourceService()
    tracking_service = TrackingService(settings)
    video_service = VideoService(
        settings=settings,
        camera_service=camera_service,
        inference_service=inference_service,
        tracking_service=tracking_service,
        recognition_service=recognition_service,
        event_service=event_service,
        # One detector session per camera: the shared session above stays for
        # enrollment and search, but live cameras must not queue behind a
        # single lock for the GPU.
        detector_factory=lambda: InferenceService(settings),
    )
    enrollment_service = EnrollmentService(
        face_quality_service=video_service.face_quality_service,
        settings=settings,
        inference_service=inference_service,
        adaface_service=adaface_service,
        face_alignment_service=face_alignment_service,
        faiss_service=faiss_service,
        person_repository=person_repository,
        embedding_repository=embedding_repository,
        reference_image_repository=reference_image_repository,
        reference_image_service=reference_image_service,
        video_service=video_service,
    )

    enrollment_service.enhancement_service = EnrollmentEnhancementService(enrollment_service)

    # ── Expose services on app.state for FastAPI dependency injection ─
    app.state.settings = settings
    app.state.database = database
    app.state.websocket_manager = websocket_manager
    app.state.analytics_service = analytics_service
    app.state.camera_service = camera_service
    app.state.event_service = event_service
    app.state.auth_service = auth_service
    app.state.face_alignment_service = face_alignment_service
    app.state.recognition_service = recognition_service
    app.state.enrollment_service = enrollment_service
    app.state.resource_service = resource_service
    app.state.video_service = video_service
    app.state.reference_image_repository = reference_image_repository
    app.state.reference_image_access_repository = reference_image_access_repository
    app.state.reference_image_service = reference_image_service


async def _broadcast_stats(app: FastAPI) -> None:
    """Push camera FPS/counts, active tracks and host resources once a second.

    Replaces four dashboard polling loops (cameras, fps, counts, resources)
    with one message on the events WebSocket, so the browser's HTTP
    connection budget is left to on-demand requests.
    """
    manager = app.state.websocket_manager
    logger = logging.getLogger("sentinel.stats")
    while True:
        await asyncio.sleep(1)
        if not manager.has_connections():
            continue
        try:
            video_service = app.state.video_service
            payload = {
                "type": "stats",
                "cameras": video_service.list_cameras(),
                "active_tracks": [track.model_dump() for track in video_service.get_active_tracks()],
                "resources": await asyncio.to_thread(app.state.resource_service.get_resources),
            }
            await manager.broadcast_json(jsonable_encoder(payload))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Stats broadcast failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager (startup / shutdown).

    **Startup**: builds all services, attaches the event loop for
    WebSocket broadcasting, loads ML models, and starts the video
    processing pipeline threads.

    **Shutdown**: stops the video pipeline, releases ML models, and
    disposes the database connection pool.
    """
    settings = get_settings()
    guard = BackendInstanceLock(settings.faiss_index_path) if settings.enable_pipeline else nullcontext()
    with guard:
        try:
            build_services(app)
            app.state.websocket_manager.attach_loop(asyncio.get_running_loop())
            if app.state.settings.enable_pipeline:
                app.state.face_alignment_service.start()
                app.state.video_service.start()
            stats_task = asyncio.create_task(_broadcast_stats(app))
            try:
                yield
            finally:
                stats_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stats_task
        finally:
            if hasattr(app.state, "enrollment_job_service"):
                await asyncio.to_thread(app.state.enrollment_job_service.stop)
            if hasattr(app.state, "video_service"):
                app.state.video_service.stop()
            if hasattr(app.state, "enrollment_service"):
                app.state.enrollment_service.face_quality_service.stop()
            if hasattr(app.state, "face_alignment_service"):
                app.state.face_alignment_service.stop()
            if hasattr(app.state, "database"):
                app.state.database.dispose()


def create_app() -> FastAPI:
    """
    Application factory – builds and returns a fully configured FastAPI app.

    Registers CORS middleware and all API routers.  Actual service
    instantiation is deferred to the ``lifespan`` context manager
    so that async resources (event loop) are available.
    """
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    # Allow the React dashboard (possibly on a different port) to call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register API routers ────────────────────────────────────────
    app.include_router(settings_router)
    app.include_router(auth_router)       # /api/auth/*
    app.include_router(stream_router)     # /api/stream/*, /api/counts, /api/fps, /api/tracks/active
    app.include_router(stream_ws_router)  # /ws/stream/{cam_id}
    app.include_router(system_router)     # /health, /api/resources
    app.include_router(events_router)     # /api/events, /ws/events
    app.include_router(cameras_router)    # /api/cameras/*
    app.include_router(poi_router)        # /api/poi/*
    app.include_router(search_router)     # /api/search, /api/embeddings/stats
    app.include_router(analytics_router)  # /api/analytics/*

    return app
