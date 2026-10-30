from __future__ import annotations

# Standard library imports for async streaming, logging, threading,
# queues, timing, dataclasses, and timezone-aware timestamps.
import asyncio
import logging
import queue as stdlib_queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

# Third-party libraries for image/frame processing.
import cv2
import numpy as np

# Application configuration, schemas, and service dependencies.
from app.core.config import Settings
from app.models.schemas import ActiveTrackRead, CameraRead
from app.services.camera_service import CameraService
from app.services.event_service import EventService
from app.services.inference_service import DetectionResult, InferenceService
from app.services.recognition_service import RecognitionMatch, RecognitionOutcome, RecognitionService
from app.services.tracking_service import TrackingService
from app.services.video_sources import CameraSource, OpenCVVideoSource
from app.services.face_quality_service import FaceQualityService
from app.services.recognition_diagnostics import RecognitionDiagnostics
from app.services.track_recognition import CandidateWindow, IdentityResolver, candidate_score


@dataclass(slots=True)
class TrackSnapshot:
    # Lightweight read-only representation of a tracked face/person.
    # Used when returning active tracks or drawing overlays.
    track_id: int
    bbox: list[int]
    status: str
    person_id: str | None
    person_name: str | None
    person_type: str | None
    similarity: float | None
    last_seen: datetime
    last_recognition: datetime | None
    recognition_phase: str = 'collecting'
    generation: int = 0


@dataclass(slots=True)
class TrackCacheEntry:
    # Mutable internal cache entry for one active tracker.
    # This stores recognition state, timing state, and the latest bbox.
    camera_id: str
    camera_name: str
    track_id: int
    tracker_id: int
    bbox: list[int]
    frame_width: int
    frame_height: int
    status: str = "unknown"
    person_id: str | None = None
    person_name: str | None = None
    person_type: str | None = None
    similarity: float | None = None
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_recognition: datetime | None = None
    last_event_at: datetime | None = None
    last_event_status: str | None = None
    recognition_miss_streak: int = 0
    last_refresh_version: int = 0
    # True while a recognition job for this track is queued or running, so the
    # camera thread does not enqueue the same face on every cycle.
    recognition_pending: bool = False
    generation: int = 0
    recognition_phase: str = 'collecting'
    detection_continuous: bool = True
    recognition_attempts: int = 0
    best_similarity: float | None = None
    quality_checks: int = 0
    quality_rejections: int = 0
    last_quality_score: float | None = None
    candidates: CandidateWindow = field(default_factory=CandidateWindow)
    resolver: IdentityResolver = field(default_factory=IdentityResolver)


@dataclass(slots=True)
class RecognitionJob:
    """One face handed from a camera thread to the recognition worker."""

    camera_id: str
    tracker_id: int
    track_id: int
    bbox: list[int]
    detection_score: float
    # Detector keypoints, when the detection that updated this track is fresh.
    # With keypoints, alignment is a plain affine warp; without them the
    # landmark detector has to run again for this face.
    kps: np.ndarray | None
    # Reference (not a copy) to the frame the bbox was measured on. The camera
    # thread never mutates a frame after publishing it, so this is safe.
    frame: np.ndarray
    captured_at: float = field(default_factory=time.monotonic)
    generation: int = 0
    quality_score: float = 0.0
    candidates: list | None = None


@dataclass(slots=True)
class CameraRuntime:
    # Runtime state for a single camera stream.
    # This includes input/output queues, FPS counters, and thread state.
    camera: CameraRead
    source: OpenCVVideoSource
    queue: deque[bytes]
    read_queue: stdlib_queue.Queue[np.ndarray]
    stop_event: threading.Event
    # Private detector session for this camera. A single shared session
    # serialises every camera behind one lock, so three cameras queue up for
    # the GPU instead of overlapping; with one session each they run in
    # parallel. ``None`` falls back to the shared service.
    detector: InferenceService | None = None
    jpeg_lock: threading.Lock = field(default_factory=threading.Lock)
    online: bool = False
    face_count: int = 0
    output_fps: int = 0
    infer_fps: int = 0
    fps_frames: int = 0
    fps_time: float = field(default_factory=time.time)
    infer_frames: int = 0
    infer_time: float = field(default_factory=time.time)
    last_frame_at: float = 0.0
    reader_thread: threading.Thread | None = None
    worker_thread: threading.Thread | None = None
    source_fps: float | None = None
    active_consumers: int = 0
    cycle: int = 0
    replay_count: int = 0
    # Detector hand-off. The frame loop leaves at most one frame in
    # ``detect_request``; the detector thread leaves its newest result in
    # ``detect_result`` for the loop to merge on its next frame.
    detect_thread: threading.Thread | None = None
    detect_lock: threading.Lock = field(default_factory=threading.Lock)
    detect_wakeup: threading.Event = field(default_factory=threading.Event)
    detect_request: tuple[np.ndarray, int] | None = None
    detect_result: tuple[DetectionResult, np.ndarray, int] | None = None
    detect_busy: bool = False
    # Encode mailbox: one slot holding the newest frame awaiting JPEG encode.
    # The frame loop never waits on it and a newer frame replaces an older
    # unencoded one, so the viewer always gets the latest picture.
    encode_lock: threading.Lock = field(default_factory=threading.Lock)
    encode_slot: tuple[np.ndarray, list] | None = None
    encode_busy: bool = False
    encoded_sequence: int = 0
    timings_ms: dict = field(default_factory=dict)
    frame_ready: asyncio.Event | None = None
    stream_loop: asyncio.AbstractEventLoop | None = None


class VideoService:
    # Aspect ratio bounds for plausible face bboxes. Real faces range from
    # roughly 0.5 (narrow profile) to 1.5 (wide frontal with hair). Anything
    # outside [0.3, 2.0] is almost certainly a degenerate Kalman prediction
    # from an unmatched track and must not be drawn.
    _MIN_BBOX_ASPECT = 0.3
    _MAX_BBOX_ASPECT = 2.0
    _MIN_BBOX_SIDE = 12  # pixels; boxes smaller than this are noise

    def __init__(
        self,
        settings: Settings,
        camera_service: CameraService,
        inference_service: InferenceService,
        tracking_service: TrackingService,
        recognition_service: RecognitionService,
        event_service: EventService,
        detector_factory: Callable[[], InferenceService] | None = None,
    ) -> None:
        # Store injected dependencies used throughout the video pipeline.
        self.settings = settings
        self.camera_service = camera_service
        self.inference_service = inference_service
        # Builds one detector session per camera so cameras never wait on
        # each other's inference. Without a factory every camera shares
        # ``inference_service`` (kept for tests and single-camera setups).
        self._detector_factory = detector_factory
        self.tracking_service = tracking_service
        self.recognition_service = recognition_service
        self.event_service = event_service
        self.logger = logging.getLogger("sentinel.video")
        if not np.isfinite(settings.live_quality_threshold):
            raise ValueError('Live quality threshold must be finite')
        self.face_quality_service = FaceQualityService(settings)
        self.diagnostics = RecognitionDiagnostics(settings)

        # Camera runtime state, keyed by camera ID.
        self._camera_states: dict[str, CameraRuntime] = {}

        # Per-camera track cache, keyed by camera ID then tracker ID.
        self._track_cache: dict[str, dict[int, TrackCacheEntry]] = {}

        # Stable visible track IDs, maintained separately from tracker IDs.
        self._next_track_ids: dict[str, int] = {}

        # JPEG encoding is done in a small thread pool. Camera threads submit
        # and move on; they never wait for the result.
        self._encode_pool = ThreadPoolExecutor(
            max_workers=max(settings.encode_workers, 1),
            thread_name_prefix="sentinel-encode",
        )

        # Recognition (alignment + AdaFace + FAISS + event persistence) runs on
        # its own thread. Keeping it off the camera threads means a burst of
        # new faces cannot stall frame delivery.
        self._recognition_queue: stdlib_queue.Queue[RecognitionJob] = stdlib_queue.Queue(
            maxsize=max(settings.recognition_queue_size, 1)
        )
        self._recognition_thread: threading.Thread | None = None
        self._recognition_metrics: dict = {}

        # Background thread references and synchronization primitives.
        self._threads: list[threading.Thread] = []
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Pipeline lifecycle flag.
        self._started = False

        # Incrementing version used to force recognition refreshes.
        self._recognition_refresh_version = 0

    def start(self) -> None:
        # Do not start twice, and do not start if the pipeline is disabled.
        if self._started or not self.settings.enable_pipeline:
            return

        # Clear any previous stop signal before starting services.
        self._stop_event.clear()

        # OpenCV defaults to one worker thread per core. With ONNX Runtime,
        # torch, one reader plus one worker thread per camera, and the encode
        # pool all live at once, that oversubscribes every core and the
        # resulting context switching costs more than the parallelism gains.
        cv2.setNumThreads(max(self.settings.cv_num_threads, 1))

        # Start dependent services before camera processing begins.
        self.inference_service.start()
        self.recognition_service.start()
        if self.settings.preload_models:
            self.face_quality_service.start()

        # Recognition worker must be running before cameras start feeding it.
        self._recognition_thread = threading.Thread(
            target=self._recognition_worker,
            daemon=True,
            name="sentinel-recognition",
        )
        self._recognition_thread.start()
        self._threads.append(self._recognition_thread)

        # Load active cameras and start their reader + worker threads.
        self.refresh_cameras()

        # Mark pipeline as running.
        self._started = True
        self.logger.info(
            "Track-aware pipeline started for %s cameras (one worker thread each)",
            len(self._camera_states),
        )

    def stop(self) -> None:
        # Signal all loops and camera readers to stop.
        self._stop_event.set()

        with self._state_lock:
            # Snapshot runtime and camera IDs before clearing state.
            runtimes = list(self._camera_states.values())
            camera_ids = list(self._camera_states.keys())

            # Stop each camera runtime and release its source.
            for runtime in runtimes:
                runtime.stop_event.set()
                self._wake_streams(runtime)
                runtime.source.close()
                with runtime.jpeg_lock:
                    runtime.queue.clear()

            # Clear all runtime and tracking state.
            self._camera_states.clear()
            self._track_cache.clear()
            self._next_track_ids.clear()
            self._recognition_refresh_version = 0

        # Remove tracker instances for cameras that were active.
        for camera_id in camera_ids:
            self.tracking_service.remove_tracker(camera_id)

        # Drain any queued recognition work so the worker can exit promptly.
        self._drain_recognition_queue()

        if self._recognition_thread is not None and self._recognition_thread is not threading.current_thread():
            self._recognition_thread.join()

        # Stop background encoding and dependent services.
        self._encode_pool.shutdown(wait=False, cancel_futures=True)
        self.inference_service.stop()
        self.recognition_service.stop()
        self.face_quality_service.stop()
        self.diagnostics.stop()

        # Reset thread list and lifecycle state.
        self._threads.clear()
        self._recognition_thread = None
        self._started = False

    def refresh_cameras(self) -> None:
        # Get all currently active cameras and index them by camera ID.
        active_cameras = {camera.id: camera for camera in self.camera_service.list_active_cameras()}

        with self._state_lock:
            # Compare currently running cameras with desired active cameras.
            current_ids = set(self._camera_states)
            desired_ids = set(active_cameras)

            # Rebuild readers when the configured source changes.
            changed_ids = {
                camera_id for camera_id in current_ids & desired_ids
                if (self._camera_states[camera_id].camera.source_type,
                    self._camera_states[camera_id].camera.source_uri) !=
                   (active_cameras[camera_id].source_type, active_cameras[camera_id].source_uri)
            }
            # Stop disabled cameras and readers attached to old sources.
            for camera_id in (current_ids - desired_ids) | changed_ids:
                runtime = self._camera_states.pop(camera_id)
                runtime.stop_event.set()
                self._wake_streams(runtime)
                runtime.source.close()
                self.tracking_service.remove_tracker(camera_id)
                self._track_cache.pop(camera_id, None)
                self._next_track_ids.pop(camera_id, None)

            # Add new active cameras or update existing camera configuration.
            for camera_id, camera in active_cameras.items():
                runtime = self._camera_states.get(camera_id)

                if runtime is None:
                    # Create runtime state for a new camera.
                    runtime = self._build_runtime(camera)
                    self._camera_states[camera_id] = runtime
                    self._track_cache[camera_id] = {}
                    self._next_track_ids.setdefault(camera_id, 1)

                    # Decode and processing run on separate threads so that a
                    # slow inference cycle never stalls frame acquisition.
                    runtime.reader_thread = threading.Thread(
                        target=self._camera_reader,
                        args=(runtime,),
                        daemon=True,
                        name=f"reader-{camera_id}",
                    )
                    runtime.reader_thread.start()
                    self._threads.append(runtime.reader_thread)

                    # One pipeline thread per camera. OpenCV, ONNX Runtime and
                    # torch all release the GIL inside their native calls, so
                    # additional cameras genuinely run in parallel instead of
                    # extending a single serial cycle.
                    runtime.worker_thread = threading.Thread(
                        target=self._camera_worker,
                        args=(runtime,),
                        daemon=True,
                        name=f"pipeline-{camera_id}",
                    )
                    runtime.worker_thread.start()
                    self._threads.append(runtime.worker_thread)
                else:
                    # Keep runtime queues/threads, but refresh camera metadata.
                    runtime.camera = camera

    def list_cameras(self) -> list[dict]:
        # Use active tracks to calculate per-camera active track counts.
        active_tracks = self.get_active_tracks()
        track_counts: dict[str, int] = {}

        for track in active_tracks:
            track_counts[track.camera_id] = track_counts.get(track.camera_id, 0) + 1

        with self._state_lock:
            # Snapshot runtime values while holding lock, then release quickly.
            runtime_snapshot = {
                camera_id: {
                    "online": runtime.online,
                    "face_count": runtime.face_count,
                    "fps": runtime.output_fps,
                }
                for camera_id, runtime in self._camera_states.items()
            }

        cameras = []

        # Build response for all configured cameras, not only active runtime cameras.
        for camera in self.camera_service.list_cameras():
            runtime = runtime_snapshot.get(camera.id)

            cameras.append(
                {
                    "id": camera.id,
                    "name": camera.name,
                    "source_type": camera.source_type,
                    "source_uri": camera.source_uri,
                    "is_active": camera.is_active,
                    "online": runtime["online"] if runtime else False,
                    "active_tracks": track_counts.get(camera.id, 0),
                    "fps": runtime["fps"] if runtime else 0,
                    "face_count": runtime["face_count"] if runtime else 0,
                }
            )

        return cameras

    def get_counts(self) -> dict[str, int]:
        # Return current face counts per active camera.
        with self._state_lock:
            return {
                camera_id: runtime.face_count
                for camera_id, runtime in self._camera_states.items()
            }

    def get_fps(self) -> dict[str, dict[str, int]]:
        # Return output FPS and inference FPS per active camera.
        with self._state_lock:
            return {
                camera_id: {
                    "output": runtime.output_fps,
                    "infer": runtime.infer_fps,
                    "timings_ms": dict(runtime.timings_ms),
                    "recognition": dict(self._recognition_metrics),
                    "recognition_queue": self._recognition_queue.qsize(),
                }
                for camera_id, runtime in self._camera_states.items()
            }

    def get_active_tracks(self) -> list[ActiveTrackRead]:
        # Only return tracks that were seen recently enough to be considered active.
        now = datetime.now(timezone.utc)

        with self._state_lock:
            # Snapshot cache entries so filtering happens outside the lock.
            cache_snapshot = {
                camera_id: list(cache.values())
                for camera_id, cache in dict(self._track_cache).items()
            }

        results = []

        for cache in cache_snapshot.values():
            for entry in cache:
                # Skip tracks that have not been seen recently.
                if now - entry.last_seen > timedelta(seconds=self.settings.track_stale_seconds):
                    continue

                # Convert internal cache entry into public API schema.
                results.append(
                    ActiveTrackRead(
                        camera_id=entry.camera_id,
                        camera_name=entry.camera_name,
                        track_id=entry.track_id,
                        bbox=entry.bbox,
                        frame_width=entry.frame_width,
                        frame_height=entry.frame_height,
                        status=entry.status,
                        recognition_phase=entry.recognition_phase,
                        recognition_attempts=entry.recognition_attempts,
                        best_similarity=entry.best_similarity,
                        quality_checks=entry.quality_checks,
                        quality_rejections=entry.quality_rejections,
                        last_quality_score=entry.last_quality_score,
                        person_id=entry.person_id,
                        person_name=entry.person_name,
                        person_type=entry.person_type,
                        similarity=entry.similarity,
                        last_seen=entry.last_seen,
                        last_recognition=entry.last_recognition,
                    )
                )

        # Most recently seen tracks appear first.
        results.sort(key=lambda item: item.last_seen, reverse=True)
        return results

    async def generate_frames(self, cam_id: str):
        """MJPEG multipart parts for ``/api/stream/{cam_id}``."""
        async for sequence, jpeg in self.iter_encoded_frames(cam_id):
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Sequence: "
                   + str(sequence).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    async def iter_encoded_frames(self, cam_id: str):
        """Yield ``(sequence, jpeg_bytes)`` for every newly encoded frame.

        Shared by the MJPEG response and the WebSocket stream. Each consumer
        keeps its own cursor and always receives the newest frame, so a slow
        client skips frames instead of accumulating latency.
        """
        # Register this client as an active stream consumer.
        with self._state_lock:
            runtime = self._camera_states.get(cam_id)

            if runtime is None:
                raise KeyError(cam_id)

            runtime.active_consumers += 1
            if runtime.frame_ready is None:
                runtime.frame_ready = asyncio.Event()
                runtime.stream_loop = asyncio.get_running_loop()

        last_sequence = -1
        try:
            # Broadcast the latest frame. Each connection keeps its own cursor;
            # no viewer removes frames that another viewer still needs.
            while not self._stop_event.is_set() and not runtime.stop_event.is_set():
                runtime.frame_ready.clear()
                with runtime.jpeg_lock:
                    sequence = runtime.encoded_sequence
                    jpeg = runtime.queue[-1] if runtime.queue else None
                if jpeg is None or sequence == last_sequence:
                    await runtime.frame_ready.wait()
                    continue
                last_sequence = sequence
                try:
                    yield sequence, jpeg
                except (asyncio.CancelledError, GeneratorExit):
                    # Client disconnected or stream was cancelled.
                    break
        finally:
            # Decrement consumer count and clear buffered output when nobody is watching.
            with self._state_lock:
                runtime.active_consumers = max(runtime.active_consumers - 1, 0)

                if runtime.active_consumers == 0:
                    with runtime.jpeg_lock:
                        runtime.queue.clear()

    def _build_runtime(self, camera: CameraRead) -> CameraRuntime:
        # Build an OpenCV-backed video source from stored camera config.
        source = OpenCVVideoSource(
            CameraSource(
                id=camera.id,
                name=camera.name,
                source_type=camera.source_type,
                source_uri=camera.source_uri,
            )
        )

        # Create queues and stop event for this camera runtime.
        return CameraRuntime(
            camera=camera,
            source=source,
            queue=deque(maxlen=self.settings.buffer_frames * 2),
            read_queue=stdlib_queue.Queue(maxsize=2),
            stop_event=threading.Event(),
            detector=self._detector_factory() if self._detector_factory is not None else None,
        )

    def _camera_reader(self, runtime: CameraRuntime) -> None:
        # Fallback read interval when native FPS is unavailable.
        fallback_interval = 1.0 / max(self.settings.target_fps, 1)

        # Continuously read frames until global or per-camera stop is requested.
        while not self._stop_event.is_set() and not runtime.stop_event.is_set():
            started = time.perf_counter()

            # Read one frame from the camera or file source.
            ok, frame = runtime.source.read()
            runtime.online = ok and frame is not None
            if getattr(runtime.source, 'ended', False):
                # Leave queued final frames available to the processing worker.
                break

            if ok and frame is not None:
                runtime.last_frame_at = time.time()
                # Carry the replay epoch with each frame: the reader can be
                # ahead of the worker, so a shared flag would reset too early.
                queued_frame = (runtime.source.replay_count, frame) if runtime.source.config.source_type == "file" else frame

                # Cache native FPS once for file sources so playback speed is realistic.
                if runtime.source.config.source_type == "file" and runtime.source_fps is None:
                    runtime.source_fps = runtime.source.get_native_fps()

                try:
                    if runtime.source.config.source_type == "file":
                        # File sources should preserve frame order, so allow blocking briefly.
                        runtime.read_queue.put(queued_frame, timeout=0.5)
                    else:
                        # Live sources prefer latest frame; drop stale frame if queue is full.
                        if runtime.read_queue.full():
                            runtime.read_queue.get_nowait()
                        runtime.read_queue.put_nowait(frame)

                except stdlib_queue.Full:
                    if runtime.source.config.source_type == "file":
                        try:
                            # If file queue is still full, drop oldest frame and enqueue latest.
                            runtime.read_queue.get_nowait()
                            runtime.read_queue.put_nowait(queued_frame)
                        except stdlib_queue.Empty:
                            pass
                    else:
                        # Live camera frame can be safely dropped when consumer is behind.
                        pass

            # Use native file FPS when available; otherwise use configured target FPS.
            source_interval = fallback_interval
            if runtime.source.config.source_type == "file" and runtime.source_fps and runtime.source_fps > 0:
                source_interval = 1.0 / runtime.source_fps

            # Sleep only for remaining frame interval.
            wait = source_interval - (time.perf_counter() - started)
            if wait > 0:
                time.sleep(wait)

    def _unpack_source_frame(self, runtime: CameraRuntime, item):
        if not isinstance(item, tuple):
            return item
        replay_count, frame = item
        if replay_count != runtime.replay_count:
            camera_id = runtime.camera.id
            self.tracking_service.remove_tracker(camera_id)
            with self._state_lock:
                self._track_cache.pop(camera_id, None)
                # Keep stable IDs monotonic so late recognition jobs from the
                # previous replay cannot update new tracks with reused IDs.
            runtime.cycle = 0
            runtime.replay_count = replay_count
            self.logger.info("Camera %s replay=%s: reset tracking and recognition state", camera_id, replay_count)
        return frame

    def _camera_worker(self, runtime: CameraRuntime) -> None:
        """Run one camera: a detector thread feeds a fixed-cadence frame loop.

        Detection (30+ ms on this GPU) runs on its own thread so the frame
        loop only ever pays for tracking, cache sync and an encode hand-off,
        which keeps every output frame on the same beat. The detector result
        is merged into the tracker on whichever frame it lands on, so boxes
        trail the picture by at most one or two frames.
        """
        camera_id = runtime.camera.id
        detector = runtime.detector or self.inference_service
        runtime.detect_thread = threading.Thread(
            target=self._camera_detector,
            args=(runtime, detector),
            daemon=True,
            name=f"detector-{camera_id}",
        )
        runtime.detect_thread.start()
        try:
            self._camera_loop(runtime)
        except Exception:
            self.logger.exception("Camera %s pipeline stopped unexpectedly", camera_id)
            runtime.online = False
        finally:
            runtime.stop_event.set()
            runtime.detect_wakeup.set()
            runtime.detect_thread.join(timeout=5)
            if runtime.detector is not None:
                runtime.detector.stop()

    def _camera_detector(self, runtime: CameraRuntime, detector: InferenceService) -> None:
        """Detect faces on the newest frame handed over by the frame loop."""
        camera_id = runtime.camera.id
        try:
            # Session load and warm-up happen here, on the camera's own
            # thread, so cameras load in parallel and startup never waits
            # on the slowest one.
            detector.start()
        except Exception:
            self.logger.exception("Detector failed to start for camera %s", camera_id)
            return

        while not self._stop_event.is_set() and not runtime.stop_event.is_set():
            if not runtime.detect_wakeup.wait(timeout=0.1):
                continue
            runtime.detect_wakeup.clear()

            with runtime.detect_lock:
                request = runtime.detect_request
                runtime.detect_request = None
            if request is None:
                continue

            frame, cycle = request
            started = time.perf_counter()
            try:
                result = detector.detect(frame)
            except Exception:
                self.logger.exception("Detection failed for camera %s", camera_id)
                result = DetectionResult.empty()
            elapsed_ms = (time.perf_counter() - started) * 1000

            with runtime.detect_lock:
                # Only the newest result matters; an older one still waiting
                # to be merged is superseded.
                runtime.detect_result = (result, frame, cycle)
                runtime.detect_busy = False

            with self._state_lock:
                old = runtime.timings_ms.get('detection', elapsed_ms)
                runtime.timings_ms['detection'] = round(.8 * old + .2 * elapsed_ms, 2)
                runtime.infer_frames += 1
                now = time.time()
                if now - runtime.infer_time >= 1.0:
                    runtime.infer_fps = round(runtime.infer_frames / max(now - runtime.infer_time, 1e-6))
                    runtime.infer_frames = 0
                    runtime.infer_time = now

    def _frame_interval(self, runtime: CameraRuntime) -> float:
        """Seconds between output frames: the file's own rate, else target FPS."""
        if runtime.source.config.source_type == "file" and runtime.source_fps and runtime.source_fps > 0:
            return 1.0 / min(runtime.source_fps, 120.0)
        return 1.0 / max(self.settings.target_fps, 1)

    def _camera_loop(self, runtime: CameraRuntime) -> None:
        camera_id = runtime.camera.id
        infer_every = max(self.settings.infer_every, 1)
        # Frames since the last hand-off to the detector. Kept separate from
        # ``runtime.cycle`` because a file replay resets that to zero.
        since_submit = infer_every
        # Absolute schedule: each frame is due a fixed interval after the
        # previous one regardless of how long this frame took to process.
        # Sleeping "interval minus elapsed" (the old approach) let a slow
        # frame push the next one right up against it, and the browser then
        # saw two frames a few ms apart followed by a long gap.
        next_due: float | None = None

        while not self._stop_event.is_set() and not runtime.stop_event.is_set():
            try:
                frame = self._unpack_source_frame(runtime, runtime.read_queue.get(timeout=0.1))
            except stdlib_queue.Empty:
                next_due = None  # Source stalled; restart the schedule on the next frame.
                continue

            runtime.cycle += 1
            since_submit += 1
            interval = self._frame_interval(runtime)

            # Hand the frame to the detector when it is idle and a detection
            # is due. A busy detector is never queued behind: by the time it
            # finished, a newer frame would be waiting anyway.
            with runtime.detect_lock:
                if not runtime.detect_busy and since_submit >= infer_every:
                    runtime.detect_request = (frame, runtime.cycle)
                    runtime.detect_busy = True
                    since_submit = 0
                    runtime.detect_wakeup.set()
                result = runtime.detect_result
                runtime.detect_result = None
                # A result measured before a file replay reset the cycle
                # counter belongs to the previous pass; drop it rather than
                # seed the fresh tracker with boxes from the old clip.
                if result is not None and result[2] > runtime.cycle:
                    result = None

            detections = np.zeros((0, 6), dtype=np.float32)
            kpss: np.ndarray | None = None
            run_infer = result is not None
            # Recognition crops and tracker appearance features must come from
            # the frame the boxes were measured on, not the newest frame.
            detected_frame = frame
            if run_infer:
                detection, detected_frame, _ = result
                if len(detection) > 0:
                    boxes = detection.bboxes
                    # Tracker format: [x1, y1, x2, y2, conf, cls]
                    detections = np.column_stack(
                        [boxes[:, :4], boxes[:, 4], np.zeros(len(boxes), dtype=np.float32)]
                    ).astype(np.float32)
                    kpss = detection.kpss

            tracking_started = time.perf_counter()
            try:
                tracked = self.tracking_service.update_tracks(camera_id, detections, detected_frame, kpss=kpss)
            except Exception:
                self.logger.exception("Tracking failed for camera %s", camera_id)
                continue

            # Sync tracker outputs into the service-level track cache and queue
            # any faces that are due for recognition.
            sync_started = time.perf_counter()
            snapshots = self._sync_track_cache_from_tracks(
                camera_id,
                runtime.camera.name,
                detected_frame,
                tracked,
                allow_recognition=run_infer,
            )

            with self._state_lock:
                for name, elapsed in [('tracking', sync_started - tracking_started),
                                      ('track_sync', time.perf_counter() - sync_started)]:
                    old = runtime.timings_ms.get(name, elapsed * 1000)
                    runtime.timings_ms[name] = round(.8 * old + .2 * elapsed * 1000, 2)
                # Update current face count.
                runtime.face_count = len(snapshots)

                # Update output FPS counter.
                runtime.fps_frames += 1
                now = time.time()

                if now - runtime.fps_time >= 1.0:
                    runtime.output_fps = round(runtime.fps_frames / max(now - runtime.fps_time, 1e-6))
                    runtime.fps_frames = 0
                    runtime.fps_time = now

            # Publish on the beat. If the loop has fallen more than two frames
            # behind (source hiccup, GC pause) resynchronise instead of
            # bursting frames out to catch up.
            now = time.perf_counter()
            if next_due is None or now - next_due > 2 * interval:
                next_due = now
            elif next_due > now:
                time.sleep(next_due - now)
            next_due += interval

            self._submit_encode(runtime, camera_id, frame, snapshots)

    def _submit_encode(
        self,
        runtime: CameraRuntime,
        camera_id: str,
        frame: np.ndarray,
        snapshots: list[TrackSnapshot],
    ) -> None:
        """Leave the frame in the camera's encode mailbox and never wait on it.

        The mailbox holds one frame. A newer frame replaces an unencoded older
        one, so the encoder always works on the latest picture and the frame
        loop never blocks. Nothing is dropped merely because the previous
        encode is still in flight: the drain loop picks the frame up as soon
        as it finishes.
        """
        with self._state_lock:
            if runtime.active_consumers == 0:
                return

        with runtime.encode_lock:
            runtime.encode_slot = (frame, snapshots)
            if runtime.encode_busy:
                return
            runtime.encode_busy = True

        try:
            self._encode_pool.submit(self._encode_drain, runtime, camera_id)
        except RuntimeError:
            # Pool already shut down during stop().
            with runtime.encode_lock:
                runtime.encode_busy = False
                runtime.encode_slot = None

    def _encode_drain(self, runtime: CameraRuntime, camera_id: str) -> None:
        """Encode mailbox frames until the mailbox is empty."""
        while True:
            with runtime.encode_lock:
                item = runtime.encode_slot
                runtime.encode_slot = None
                if item is None:
                    runtime.encode_busy = False
                    return
            frame, snapshots = item
            try:
                # Copy here rather than on the frame loop: overlays are drawn
                # in place and recognition jobs still hold the original.
                self._annotate_and_encode(camera_id, frame.copy(), snapshots)
            except Exception:
                self.logger.exception("Encode failed for camera %s", camera_id)

    def _sync_track_cache_from_tracks(
        self,
        camera_id: str,
        camera_name: str,
        frame: np.ndarray,
        tracks: list[object],
        *,
        allow_recognition: bool,
    ) -> list[TrackSnapshot]:
        # Timestamp used for all cache updates in this sync pass.
        now = datetime.now(timezone.utc)

        # Convert tracker objects into validated observations.
        observations = self._tracks_to_observations(tracks, frame)
        sampled_at = time.monotonic()
        for observation in observations:
            observation['quality'] = candidate_score(
                frame, observation['bbox'], observation.get('kps'),
                float(observation.get('detection_score') or 0), self.settings,
            ) if allow_recognition else None
            if allow_recognition:
                self.diagnostics.capture(frame, observation['bbox'], observation.get('kps'),
                    dict(stage='basic_pass' if observation['quality'] is not None else 'basic_reject',
                         camera_id=camera_id, track_id=observation['tracker_id'],
                         detection_score=observation.get('detection_score'),candidate_score=observation['quality']))

        # Recognition jobs are collected while holding the lock and dispatched
        # afterwards, so the queue is never touched with the state lock held.
        pending_jobs: list[RecognitionJob] = []

        # Track IDs seen in this frame/update.
        seen_tracker_ids: set[int] = set()

        with self._state_lock:
            # Get or create cache for this camera.
            cache = self._track_cache.setdefault(camera_id, {})

            if allow_recognition:
                # A missed scheduled detection breaks identity continuity even
                # if DeepSORT later reuses the SAME tracker ID. Predicted boxes
                # may remain drawable, but cannot carry an unverified identity.
                # Skipped detector cycles never enter this branch.
                fresh_ids = {obs['tracker_id'] for obs in observations if obs.get('fresh_detection', True)}
                for cached_id, cached in cache.items():
                    if cached_id not in fresh_ids and cached.detection_continuous:
                        cached.detection_continuous = False
                        self._invalidate_track_identity(cached, reason='scheduled detection missed')

            for observation in observations:
                tracker_id = observation["tracker_id"]
                bbox = observation["bbox"]

                # Mark this tracker as visible in the current update.
                seen_tracker_ids.add(tracker_id)

                entry = cache.get(tracker_id)

                if entry is None:
                    # Create a new cache entry or reuse a recently lost one.
                    entry = self._reuse_or_create_track_entry(
                        camera_id=camera_id,
                        camera_name=camera_name,
                        tracker_id=tracker_id,
                        bbox=bbox,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        now=now,
                        seen_tracker_ids=seen_tracker_ids,
                    )

                # Refresh cached geometry and timing.
                entry.bbox = bbox
                entry.tracker_id = tracker_id
                entry.frame_width = frame.shape[1]
                entry.frame_height = frame.shape[0]
                entry.last_seen = now
                entry.camera_name = camera_name
                if allow_recognition and observation.get('fresh_detection', True):
                    entry.detection_continuous = True

                # Queue recognition if this track is eligible.
                if allow_recognition and self._should_recognize(entry, now):
                    if observation['quality'] is None:
                        if entry.candidates.started is None:
                            entry.recognition_phase = 'low_quality'
                        # Flush an existing window even when the latest frame is poor.
                        ready = (entry.candidates.started is not None and
                                 sampled_at-entry.candidates.started >= self.settings.recognition_collect_seconds)
                    else:
                        candidate = RecognitionJob(
                            camera_id=camera_id,
                            tracker_id=tracker_id,
                            track_id=entry.track_id,
                            bbox=list(entry.bbox),
                            detection_score=float(observation.get("detection_score") or 0.0),
                            kps=observation.get("kps"),
                            frame=frame,
                            captured_at=sampled_at,
                            generation=entry.generation,
                            quality_score=observation['quality'],
                        )
                        ready = entry.candidates.offer(candidate, observation['quality'], sampled_at, self.settings,
                                                       immediate=True)
                    if ready:
                        choices = entry.candidates.take()
                        if choices:
                            entry.recognition_pending = True
                            entry.recognition_phase = 'processing'
                            # A separate envelope avoids a self-referential candidate list.
                            envelope = RecognitionJob(
                                camera_id=camera_id, tracker_id=tracker_id, track_id=entry.track_id,
                                bbox=list(entry.bbox), detection_score=0, kps=None, frame=frame,
                                captured_at=sampled_at, generation=entry.generation, candidates=choices,
                            )
                            pending_jobs.append(envelope)

            # Remove stale cache entries that were not seen recently.
            self._prune_stale_tracks(cache, now, seen_tracker_ids)

            # Convert current visible cache entries into immutable snapshots.
            snapshots = [
                self._entry_to_snapshot(cache[tracker_id])
                for tracker_id in seen_tracker_ids
                if tracker_id in cache
            ]

        for job in pending_jobs:
            self._enqueue_recognition(job)

        # Sort by visible stable track ID for deterministic output.
        snapshots.sort(key=lambda item: item.track_id)
        return snapshots

    # ------------------------------------------------------------ recognition

    def _enqueue_recognition(self, job: RecognitionJob) -> None:
        """Queue a face for the recognition worker, dropping it if saturated."""
        try:
            self._recognition_queue.put_nowait(job)
        except stdlib_queue.Full:
            # Recognition is behind. Dropping is correct: the same track will
            # be offered again on a later cycle once its cooldown allows it.
            self._clear_recognition_pending(job.camera_id, job.tracker_id, job.track_id, job.generation)
            self.logger.debug(
                "Track %s recognition dropped: queue full camera=%s",
                job.track_id,
                job.camera_id,
            )

    def _clear_recognition_pending(self, camera_id: str, tracker_id: int, track_id: int, generation: int | None = None) -> None:
        with self._state_lock:
            entry = self._track_cache.get(camera_id, {}).get(tracker_id)
            if entry is not None and entry.track_id == track_id and (generation is None or entry.generation == generation):
                entry.recognition_pending = False

    def _drain_recognition_queue(self) -> None:
        while True:
            try:
                self._recognition_queue.get_nowait()
            except stdlib_queue.Empty:
                return

    def _recognition_worker(self) -> None:
        """Align, embed, and identify queued faces in batches.

        Batching matters a great deal here: a single AdaFace forward pass costs
        about the same as nine batched together, because the per-call overhead
        (host-to-device copy plus synchronisation) dominates the arithmetic.
        Recognising faces one at a time was previously the single largest cost
        in the pipeline.
        """
        batch_size = max(self.settings.recognition_batch_size, 1)

        while not self._stop_event.is_set():
            try:
                first = self._recognition_queue.get(timeout=0.1)
            except stdlib_queue.Empty:
                continue

            jobs = [first]

            # Opportunistically absorb everything else already queued, so faces
            # that arrived in the same cycle share one GPU round trip.
            while len(jobs) < batch_size:
                try:
                    jobs.append(self._recognition_queue.get_nowait())
                except stdlib_queue.Empty:
                    break

            try:
                self._process_recognition_batch(jobs)
            except Exception:
                self.logger.exception("Recognition batch failed for %s faces", len(jobs))
                for job in jobs:
                    self._clear_recognition_pending(job.camera_id, job.tracker_id, job.track_id, job.generation)

    def _evaluate_candidates(self, envelopes):
        """Score one candidate per track; embed only images passing the live gate."""
        prepared = []
        for envelope in envelopes:
            for job in envelope.candidates or [envelope]:
                if time.monotonic()-job.captured_at > self.settings.recognition_vote_seconds:
                    continue
                if job.kps is None:
                    continue
                aligned = self.recognition_service.face_alignment_service.align_from_keypoints(job.frame, job.kps)
                if aligned is not None:
                    prepared.append((envelope, job, aligned))
        grouped = {}
        for envelope, job, aligned in prepared:
            grouped.setdefault(id(envelope), []).append((job, aligned))
        # Pick one cheap-score winner per track; quality is a gate, not extra
        # inference over every detection or a requirement to collect two frames.
        ranking = [max(rows, key=lambda row: row[0].quality_score) for rows in grouped.values()]
        selected = []
        results = []
        size = max(1, self.settings.recognition_batch_size)
        for offset in range(0, len(ranking), size):
            chunk = ranking[offset:offset+size]
            scores = self.face_quality_service.assess_batch([row[1] for row in chunk])
            for (job, aligned), score in zip(chunk, scores, strict=True):
                job.quality_score = score
                if score < self.settings.live_quality_threshold:
                    self.diagnostics.capture(job.frame, job.bbox, job.kps,
                        dict(stage='quality_reject',camera_id=job.camera_id,track_id=job.track_id,score=score,
                             age_seconds=time.monotonic()-job.captured_at),aligned)
                    results.append((job, RecognitionOutcome(quality_score=score, quality_rejected=True)))
                else:
                    selected.append((job, aligned))
        selected.sort(key=lambda row: row[0].captured_at)
        size = max(1, self.settings.recognition_batch_size)
        for offset in range(0, len(selected), size):
            chunk = selected[offset:offset+size]
            jobs = [row[0] for row in chunk]
            outcomes = self.recognition_service.recognize_aligned_batch([row[1] for row in chunk], jobs)
            for job, outcome in zip(jobs, outcomes, strict=True):
                outcome.quality_score = job.quality_score
            for (job, aligned), outcome in zip(chunk, outcomes, strict=True):
                self.diagnostics.capture(job.frame, job.bbox, job.kps,
                    dict(stage='match' if outcome.match else 'no_match',camera_id=job.camera_id,
                         track_id=job.track_id,score=job.quality_score,candidates=outcome.candidates,
                         ambiguous=outcome.ambiguous,age_seconds=time.monotonic()-job.captured_at),aligned)
            results.extend(zip(jobs, outcomes, strict=True))
        return results

    def _process_recognition_batch(self, jobs: list[RecognitionJob]) -> None:
        batch_started = time.perf_counter()
        oldest_wait_ms = (max(0.0, time.monotonic()-min(job.captured_at for job in jobs))*1000
                          if jobs else 0.0)
        evaluations = self._evaluate_candidates(jobs)
        self._recognition_metrics = dict(batch_size=len(jobs),
            oldest_wait_ms=round(oldest_wait_ms, 2),
            evaluation_ms=round((time.perf_counter()-batch_started)*1000, 2),
            oldest_job_ms=round((time.monotonic()-min(job.captured_at for job in jobs))*1000, 2)) if jobs else {}
        recognized_at = datetime.now(timezone.utc)

        # Events are published outside the state lock because publishing writes
        # to the database; holding the lock across that would block every
        # camera thread and every API request for the duration of the insert.
        events_to_publish: list[tuple[TrackCacheEntry, RecognitionJob]] = []
        unknown_to_publish: list[tuple[TrackCacheEntry, float | None]] = []

        with self._state_lock:
            for envelope in jobs:
                cache = self._track_cache.get(envelope.camera_id, {})
                entry = cache.get(envelope.tracker_id)
                if entry is not None and entry.track_id == envelope.track_id and entry.generation == envelope.generation:
                    entry.recognition_pending = False
                    entry.last_recognition = recognized_at
            for job, outcome in evaluations:
                cache = self._track_cache.get(job.camera_id)
                entry = cache.get(job.tracker_id) if cache else None

                # Ignore recognition results for stale/replaced tracks.
                if entry is None or entry.track_id != job.track_id or entry.generation != job.generation:
                    continue

                entry.recognition_pending = False
                entry.last_recognition = recognized_at
                if outcome.quality_score is not None:
                    entry.quality_checks += 1
                    entry.last_quality_score = outcome.quality_score
                if outcome.quality_rejected:
                    entry.quality_rejections += 1
                    entry.recognition_phase = 'low_quality'
                    continue
                if (not outcome.evaluated
                        or time.monotonic()-job.captured_at > self.settings.recognition_vote_seconds
                        or job.captured_at <= entry.resolver.last_observed):
                    continue
                entry.recognition_attempts += 1
                entry.best_similarity = outcome.best_similarity
                entry.resolver.last_observed = job.captured_at
                match = outcome.match
                evidence = job
                phase = 'recognized' if match else ('ambiguous' if outcome.ambiguous else 'unknown')
                entry.recognition_phase = phase
                self.logger.info('Track %s camera=%s recognition=%s attempt=%s best_similarity=%s',
                                 entry.track_id, entry.camera_id, phase, entry.recognition_attempts,
                                 outcome.best_similarity)

                if match is not None:
                    if self._apply_match(entry, match, recognized_at):
                        # Keep the exact frame and bbox that produced this
                        # match. The track cache can move on before the event
                        # is persisted, which would misalign evidence images.
                        events_to_publish.append((self._entry_event_payload(entry), evidence))
                    entry.resolver.observations.clear()
                    continue

                if entry.status == "recognized" and entry.person_id is not None:
                    if outcome.match is not None and outcome.match.person_id == entry.person_id:
                        entry.recognition_miss_streak = 0
                        continue
                    # For previously recognized tracks, require repeated misses
                    # before clearing identity to avoid flicker.
                    entry.recognition_miss_streak += 1

                    if entry.recognition_miss_streak >= self.settings.recognition_max_good_misses:
                        self._clear_identity(entry, reason="after repeated misses")
                        entry.recognition_miss_streak = 0
                else:
                    # Unknown tracks remain unknown when no match is accepted.
                    self.logger.debug(
                        "Track %s recognition attempt rejected: best_similarity=%s bbox=%s",
                        entry.track_id,
                        f"{outcome.best_similarity:.4f}" if outcome.best_similarity is not None else "none",
                        entry.bbox,
                    )
                    entry.status = "unknown"

                    # Rejections are what give the event table a denominator,
                    # so analytics can chart a real recognition rate and the
                    # whole similarity distribution rather than only successes.
                    if phase == 'unknown' and self._mark_unknown_event(entry, outcome, recognized_at):
                        unknown_to_publish.append(
                            (self._entry_event_payload(entry), outcome.best_similarity)
                        )

        for payload, job in events_to_publish:
            try:
                self.event_service.publish_recognition_event(
                    camera_id=payload.camera_id,
                    camera_name=payload.camera_name,
                    track_id=payload.track_id,
                    status=payload.status,
                    event_type="Person recognized",
                    similarity=payload.similarity,
                    # The evidence frame belongs to this job, so its box must
                    # come from the same job as well. ``payload.bbox`` is the
                    # mutable track cache's latest box and may already refer
                    # to a later frame by the time this event is persisted.
                    bbox=job.bbox,
                    person_id=payload.person_id,
                    person_name=payload.person_name,
                    person_type=payload.person_type,
                    detection_confidence=job.detection_score,
                    snapshot_frame=job.frame,
                )
            except Exception:
                self.logger.exception("Failed to publish recognition event for track %s", payload.track_id)

        for payload, best_similarity in unknown_to_publish:
            try:
                self.event_service.publish_recognition_event(
                    camera_id=payload.camera_id,
                    camera_name=payload.camera_name,
                    track_id=payload.track_id,
                    status="unknown",
                    event_type="Unknown person detected",
                    similarity=best_similarity,
                    bbox=payload.bbox,
                )
            except Exception:
                self.logger.exception("Failed to publish unknown event for track %s", payload.track_id)

    @staticmethod
    def _entry_event_payload(entry: TrackCacheEntry) -> TrackCacheEntry:
        """Snapshot the fields an event needs, so publishing can drop the lock."""
        return TrackCacheEntry(
            camera_id=entry.camera_id,
            camera_name=entry.camera_name,
            track_id=entry.track_id,
            tracker_id=entry.tracker_id,
            bbox=list(entry.bbox),
            frame_width=entry.frame_width,
            frame_height=entry.frame_height,
            status=entry.status,
            person_id=entry.person_id,
            person_name=entry.person_name,
            person_type=entry.person_type,
            similarity=entry.similarity,
        )

    def _tracks_to_observations(self, tracks: list[object], frame: np.ndarray) -> list[dict]:
        # Convert tracker objects into normalized observation dictionaries.
        observations: list[dict] = []

        # Extract frame dimensions for bbox clamping.
        frame_h, frame_w = frame.shape[:2]

        for track in tracks:
            # Reject only deleted tracks. Accept Confirmed tracks (stable
            # identities) AND Tentative tracks that have at least one
            # detection hit - a new face should appear on screen
            # immediately, not after n_init cycles of invisible waiting.
            # The aspect-ratio and size filters below will discard any
            # degenerate Kalman predictions, so accepting tentatives
            # here does not reintroduce the phantom red ellipses.
            if getattr(track, "is_deleted", lambda: False)():
                continue

            # Read tracker lifecycle state safely, even if implementation differs.
            is_confirmed = getattr(track, "is_confirmed", lambda: False)()
            is_tentative = getattr(track, "is_tentative", lambda: False)()
            hits = int(getattr(track, "hits", 0))

            # Ignore tracks that are neither confirmed nor newly detected tentative tracks.
            if not is_confirmed and not (is_tentative and hits >= 1):
                continue

            # Allow the track to survive one full infer_every gap without
            # a fresh detection. With infer_every=2, time_since_update
            # naturally reaches 1 on non-inference cycles; rejecting at
            # > 1 would blank out every alternate frame. We cap at
            # infer_every + 1 so long-lost tracks still get culled.
            time_since_update = int(getattr(track, "time_since_update", 0))

            if time_since_update > self.settings.infer_every + 1:
                continue

            # Prefer the detection-anchored bbox when recent; otherwise
            # use the Kalman-predicted bbox so tracks stay visible on
            # skipped inference cycles. The aspect-ratio guard below
            # catches any degenerate predictions.
            raw_bbox = track.to_ltrb(orig=True) if time_since_update <= 1 else None

            if raw_bbox is None:
                raw_bbox = track.to_ltrb()

            if raw_bbox is None:
                continue

            # Clamp bbox coordinates to valid frame boundaries.
            bbox = self._clamp_bbox(raw_bbox, frame_w, frame_h)

            if bbox is None:
                continue

            # Reject degenerate boxes: too small on either side, or aspect
            # ratio far outside the face-plausible range. These are phantom
            # Kalman predictions, not faces.
            bx1, by1, bx2, by2 = bbox
            box_w = bx2 - bx1
            box_h = by2 - by1

            if box_w < self._MIN_BBOX_SIDE or box_h < self._MIN_BBOX_SIDE:
                continue

            aspect = box_w / box_h

            if aspect < self._MIN_BBOX_ASPECT or aspect > self._MAX_BBOX_ASPECT:
                continue

            # DeepSORT clears `others` when a track is not matched this cycle,
            # so recognition metadata is only available for a fresh detection.
            other = getattr(track, "others", None) if time_since_update == 0 else None
            detection_score: float | None = None
            if isinstance(other, dict):
                kps = other.get("kps")
                raw_score = other.get("detection_score")
                if raw_score is not None:
                    detection_score = float(raw_score)
            else:
                # Support tracks created before the metadata payload format.
                kps = other
                raw_score = getattr(track, "det_conf", None) if time_since_update == 0 else None
                if raw_score is not None:
                    detection_score = float(raw_score)

            if kps is not None:
                kps = np.asarray(kps, dtype=np.float32)
                if kps.shape != (5, 2):
                    kps = None

            observations.append(
                {
                    "tracker_id": int(track.track_id),
                    "bbox": bbox,
                    "kps": kps,
                    "detection_score": detection_score,
                    "fresh_detection": time_since_update == 0,
                }
            )

        return observations

    def _reuse_or_create_track_entry(
        self,
        camera_id: str,
        camera_name: str,
        tracker_id: int,
        bbox: list[int],
        frame_width: int,
        frame_height: int,
        now: datetime,
        seen_tracker_ids: set[int],
    ) -> TrackCacheEntry:
        # Get or create per-camera track cache.
        cache = self._track_cache.setdefault(camera_id, {})

        # Recent entries inside this window can be reused if geometry matches.
        reuse_window = timedelta(seconds=1.0)

        # Track the best reusable cache entry.
        best_match_key: int | None = None
        best_match_score = 0.0

        for existing_tracker_id, existing in cache.items():
            # Do not reuse an entry already seen in this sync pass.
            if existing_tracker_id in seen_tracker_ids:
                continue

            # Do not reuse entries that are too old.
            if now - existing.last_seen > reuse_window:
                continue

            # Compare old bbox and new bbox using overlap and center distance.
            overlap = self._bbox_iou(existing.bbox, bbox)
            center_distance = self._bbox_center_distance(existing.bbox, bbox)

            existing_diag = self._bbox_diagonal(existing.bbox)
            new_diag = self._bbox_diagonal(bbox)

            # Larger boxes can tolerate larger center movement.
            max_center_distance = max(existing_diag, new_diag) * 0.75 + 20.0

            # Reject if both overlap and proximity are weak.
            if overlap < 0.15 and center_distance > max_center_distance:
                continue

            # Score combines IoU and normalized center-distance similarity.
            distance_score = max(0.0, 1.0 - (center_distance / max(max_center_distance, 1.0)))
            score = overlap + (distance_score * 0.5)

            if score > best_match_score:
                best_match_score = score
                best_match_key = existing_tracker_id

        if best_match_key is not None:
            # Reassign a recently lost cache entry to the new tracker ID.
            entry = cache.pop(best_match_key)
            entry.generation += 1
            entry.candidates = CandidateWindow()
            entry.resolver = IdentityResolver()
            entry.recognition_phase = 'collecting'
            entry.recognition_attempts = 0
            entry.best_similarity = None
            entry.quality_checks = 0
            entry.quality_rejections = 0
            entry.last_quality_score = None
            self._clear_identity(entry, reason='tracker reassociated')
            entry.last_recognition = None

            # Reuse a cache slot for geometric continuity, but clear identity fields if stale:
            # geometric proximity alone is not enough to inherit identity.
            if now - entry.last_seen > timedelta(seconds=2):
                self._clear_identity(entry, reason="stale slot reused")
                entry.last_recognition = None

            # Update reused entry with new tracker and geometry.
            entry.camera_id = camera_id
            entry.camera_name = camera_name
            entry.tracker_id = tracker_id
            entry.bbox = bbox
            entry.frame_width = frame_width
            entry.frame_height = frame_height
            entry.last_seen = now
            # An in-flight job was keyed to the old tracker id and will be
            # discarded on completion, so this slot must not stay blocked.
            entry.recognition_pending = False

            cache[tracker_id] = entry
            return entry

        # No reusable entry found, so allocate a new stable visible track ID.
        next_track_id = self._next_track_ids.get(camera_id, 1)
        self._next_track_ids[camera_id] = next_track_id + 1

        # Create a fresh track cache entry.
        entry = TrackCacheEntry(
            camera_id=camera_id,
            camera_name=camera_name,
            track_id=next_track_id,
            tracker_id=tracker_id,
            bbox=bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            last_refresh_version=self._recognition_refresh_version,
        )

        cache[tracker_id] = entry
        return entry

    def _cached_track_snapshots(self, camera_id: str) -> list[TrackSnapshot]:
        # Return snapshots from the current cache for a single camera.
        with self._state_lock:
            cache = self._track_cache.setdefault(camera_id, {})

            # Convert mutable cache entries into read-only snapshots.
            snapshots = [self._entry_to_snapshot(entry) for entry in cache.values()]

        # Keep output deterministic by sorting by stable visible track ID.
        snapshots.sort(key=lambda item: item.track_id)
        return snapshots

    def request_recognition_refresh(self, *, reason: str) -> int:
        # Increment refresh version so eligible unknown tracks are rechecked.
        with self._state_lock:
            self._recognition_refresh_version += 1
            version = self._recognition_refresh_version

        # Log the reason to make refresh triggers traceable.
        self.logger.info("Recognition refresh requested version=%s reason=%s", version, reason)
        return version

    def _should_recognize(self, entry: TrackCacheEntry, now: datetime) -> bool:
        # A job for this track is already queued or running; queueing another
        # would multiply GPU work for no additional information.
        if entry.recognition_pending:
            return False

        # A face far below the recognition resolution cannot produce a usable
        # embedding: it fails alignment and falls back to an upscaled crop that
        # matches nothing. SCRFD happily reports faces down to ~15 px, so
        # without this gate the majority of recognition work is spent on faces
        # that can never be identified. Set SENTINEL_RECOGNITION_MIN_FACE_SIZE=0
        # to attempt recognition at any size.
        min_face = self.settings.recognition_min_face_size
        if min_face > 0:
            x1, y1, x2, y2 = entry.bbox
            if (x2 - x1) < min_face or (y2 - y1) < min_face:
                entry.recognition_phase = 'low_quality'
                return False

        # Force recognition for unknown tracks when refresh version has advanced.
        if entry.status != "recognized" and entry.last_refresh_version < self._recognition_refresh_version:
            entry.last_refresh_version = self._recognition_refresh_version

            self.logger.debug(
                "Track %s recognition forced by refresh version=%s bbox=%s",
                entry.track_id,
                self._recognition_refresh_version,
                entry.bbox,
            )
            return True

        # Recognized tracks are rechecked less frequently to confirm identity.
        if entry.status == "recognized" and entry.person_id is not None and entry.recognition_phase == 'recognized':
            if entry.last_recognition is None:
                self.logger.debug(
                    "Track %s recognition attempt allowed: recognized track missing last_recognition bbox=%s",
                    entry.track_id,
                    entry.bbox,
                )
                return True

            # Reverify recognized identities after the configured interval.
            elapsed = (now - entry.last_recognition).total_seconds()

            if elapsed >= self.settings.recognition_reverify_seconds:
                self.logger.debug(
                    "Track %s recognition attempt allowed: reverify elapsed %.2fs >= %.2fs bbox=%s",
                    entry.track_id,
                    elapsed,
                    self.settings.recognition_reverify_seconds,
                    entry.bbox,
                )
                return True

            # Skip recognition until reverify cooldown expires.
            return False

        # Unknown tracks should be recognized at least once.
        if entry.last_recognition is None:
            self.logger.debug(
                "Track %s recognition attempt allowed: first attempt bbox=%s",
                entry.track_id,
                entry.bbox,
            )
            return True

        # Unknown tracks are retried after a cooldown.
        elapsed = (now - entry.last_recognition).total_seconds()

        # Confirmation is part of the current attempt, not an unknown-person retry.
        if entry.recognition_phase in ('low_quality', 'processing', 'collecting'):
            return elapsed >= self.settings.live_quality_retry_seconds

        if elapsed >= self.settings.recognition_unknown_cooldown_seconds:
            self.logger.debug(
                "Track %s recognition attempt allowed: unknown cooldown elapsed %.2fs >= %.2fs bbox=%s",
                entry.track_id,
                elapsed,
                self.settings.recognition_unknown_cooldown_seconds,
                entry.bbox,
            )
            return True

        return False

    def _apply_match(self, entry: TrackCacheEntry, match: RecognitionMatch, now: datetime) -> bool:
        """Update a track's identity. Returns True when an event should be logged."""
        # Check whether this recognition result changes the track identity.
        is_new_identity = entry.person_id != match.person_id or entry.status != "recognized"

        # Store accepted recognition identity data.
        entry.status = "recognized"
        entry.person_id = match.person_id
        entry.person_name = match.person_name
        entry.person_type = match.person_type
        entry.similarity = match.similarity

        # Reset miss counter because recognition succeeded.
        entry.recognition_miss_streak = 0

        self.logger.info(
            "Track %s recognized as %s type=%s similarity=%.4f bbox=%s",
            entry.track_id,
            entry.person_name,
            entry.person_type,
            entry.similarity if entry.similarity is not None else -1.0,
            entry.bbox,
        )

        # Log/publish an event when identity is newly recognized.
        should_log = is_new_identity

        # If same identity remains visible, publish again only after cooldown.
        if not should_log and entry.last_event_at is not None:
            should_log = (
                now - entry.last_event_at
            ).total_seconds() >= self.settings.recognition_event_cooldown_seconds

        if should_log:
            # Update event timestamp to enforce cooldown. The event itself is
            # published by the caller once the state lock has been released.
            entry.last_event_at = now
            entry.last_event_status = "recognized"

        return should_log

    def _mark_unknown_event(self, entry: TrackCacheEntry, outcome: RecognitionOutcome, now: datetime) -> bool:
        """Decide whether a rejected detection should be persisted as an event.

        Returns True when the caller should publish. The publish itself is left
        to the caller so it can happen once the state lock has been released --
        writing to the database under the lock would stall every camera thread
        and every API request for the duration of the insert.

        Recording rejections is what gives the event table a denominator: without
        them every stored event is a success, so the recognition rate is always
        100% and the similarity distribution is cut off at the threshold. The
        rejected score is carried through as the event's similarity so both
        halves of that distribution can be charted.

        Publication is skipped when nothing was actually compared, and is rate
        limited per track by the same cooldown that governs recognised events,
        with the first transition into the unknown state always reported.
        """
        self.logger.debug(
            "Track %s recognition attempt rejected: best_similarity=%s bbox=%s",
            entry.track_id,
            f"{outcome.best_similarity:.4f}" if outcome.best_similarity is not None else "none",
            entry.bbox,
        )
        if not self.settings.publish_unknown_events or not outcome.evaluated:
            return False

        # A track that was previously reported as something else is news; after
        # that, fall back to the cooldown so a lingering stranger logs once per
        # window rather than once per recognition attempt.
        if entry.last_event_status == "unknown" and entry.last_event_at is not None:
            elapsed = (now - entry.last_event_at).total_seconds()
            if elapsed < self.settings.recognition_event_cooldown_seconds:
                return False

        entry.last_event_at = now
        entry.last_event_status = "unknown"
        return True

    def _clear_identity(self, entry: TrackCacheEntry, *, reason: str) -> None:
        # Remove recognition identity from a track while keeping tracking state.
        self.logger.info(
            "Track %s identity cleared %s bbox=%s previous_person=%s",
            entry.track_id,
            reason,
            entry.bbox,
            entry.person_name,
        )

        entry.status = "unknown"
        entry.person_id = None
        entry.person_name = None
        entry.person_type = None
        entry.similarity = None
        # The published identity no longer holds, so whatever this track is
        # recognised as next -- a new person or a stranger -- counts as news
        # and must not be suppressed by the previous event's cooldown.
        entry.last_event_status = None

    def _invalidate_track_identity(self, entry: TrackCacheEntry, *, reason: str) -> None:
        # Invalidate queued/in-flight evidence and delayed overlays atomically
        # with the identity; neither may restore the previous occupant's label.
        entry.generation += 1
        entry.candidates = CandidateWindow()
        entry.resolver = IdentityResolver()
        entry.recognition_pending = False
        entry.recognition_phase = 'collecting'
        entry.last_recognition = None
        entry.recognition_miss_streak = 0
        self._clear_identity(entry, reason=reason)

    def _prune_stale_tracks(
        self,
        cache: dict[int, TrackCacheEntry],
        now: datetime,
        seen_track_ids: set[int],
    ) -> None:
        # Tracks not seen within this duration are removed from the cache.
        stale_after = timedelta(seconds=self.settings.track_stale_seconds)

        for track_id in list(cache):
            entry = cache[track_id]

            # Never prune tracks observed in the current update.
            if track_id in seen_track_ids:
                continue

            # Remove tracks that have been inactive too long.
            if now - entry.last_seen > stale_after:
                self.logger.debug(
                    "Track %s recognition skipped: track inactive stale_for=%.2fs",
                    entry.track_id,
                    (now - entry.last_seen).total_seconds(),
                )
                cache.pop(track_id, None)

    @staticmethod
    def _entry_to_snapshot(entry: TrackCacheEntry) -> TrackSnapshot:
        # Convert internal mutable cache entry into an immutable snapshot.
        return TrackSnapshot(
            track_id=entry.track_id,
            bbox=entry.bbox,
            status=entry.status,
            person_id=entry.person_id,
            person_name=entry.person_name,
            person_type=entry.person_type,
            similarity=entry.similarity,
            last_seen=entry.last_seen,
            last_recognition=entry.last_recognition,
            recognition_phase=entry.recognition_phase,
            generation=entry.generation,
        )

    @staticmethod
    def _overlay_style(track: TrackSnapshot) -> tuple[tuple[int, int, int], str]:
        # A track is considered recognized only if status and person_id agree.
        is_recognized = track.status == "recognized" and track.person_id is not None

        # Watchlist people get a stronger alert label and color.
        is_watchlist = is_recognized and track.person_type == "watchlist"

        if is_watchlist:
            color = (38, 38, 220)
            name_part = track.person_name or "Watchlist"
            conf_part = f" {track.similarity:.2f}" if track.similarity is not None else ""
            return color, f"ALERT #{track.track_id} {name_part}{conf_part}"

        if is_recognized:
            color = (34, 197, 94)
            name_part = track.person_name or "Recognized"
            conf_part = f" {track.similarity:.2f}" if track.similarity is not None else ""
            return color, f"#{track.track_id} {name_part}{conf_part}"

        # Unknown tracks use the default non-alert label.
        label = {'collecting': 'Collecting', 'processing': 'Recognizing', 'low_quality': 'Waiting for clear face',
                 'tentative': 'Confirming', 'unmatched': 'No match', 'ambiguous': 'Ambiguous', 'unknown': 'Unknown'}.get(track.recognition_phase, 'Unknown')
        return (59, 130, 246), f"#{track.track_id} {label}"

    @staticmethod
    def _clamp_bbox(raw_bbox, width: int, height: int) -> list[int] | None:
        # Convert raw bbox coordinates to integers.
        x1, y1, x2, y2 = [int(round(value)) for value in raw_bbox]

        # Clamp all coordinates inside the frame.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Reject invalid or collapsed boxes.
        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    @staticmethod
    def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
        # Unpack both boxes.
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Calculate intersection rectangle.
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(inter_x2 - inter_x1, 0)
        inter_h = max(inter_y2 - inter_y1, 0)

        # No overlap.
        if inter_w == 0 or inter_h == 0:
            return 0.0

        # Compute intersection-over-union.
        intersection = inter_w * inter_h
        area_a = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)
        area_b = max(bx2 - bx1, 1) * max(by2 - by1, 1)

        return intersection / max(area_a + area_b - intersection, 1)

    @staticmethod
    def _bbox_center_distance(box_a: list[int], box_b: list[int]) -> float:
        # Unpack both boxes.
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Compute box centers.
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        # Euclidean distance between centers.
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)

    @staticmethod
    def _bbox_diagonal(box: list[int]) -> float:
        # Calculate diagonal length of a bounding box.
        x1, y1, x2, y2 = box
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)

        return float((width**2 + height**2) ** 0.5)

    def _refresh_overlay_states(self, camera_id, tracks):
        """Refresh identities at draw time, preserving this frame's geometry."""
        with self._state_lock:
            entries = {entry.track_id: entry for entry in self._track_cache.get(camera_id, {}).values()}
            refreshed = []
            for track in tracks:
                entry = entries.get(track.track_id)
                if entry is not None and entry.generation == track.generation:
                    refreshed.append(replace(track, status=entry.status, person_id=entry.person_id,
                        person_name=entry.person_name, person_type=entry.person_type,
                        similarity=entry.similarity, recognition_phase=entry.recognition_phase,
                        last_recognition=entry.last_recognition))
                elif entry is not None:
                    # Reassociation invalidates any identity from the older job.
                    refreshed.append(replace(track, status='unknown', person_id=None,
                        person_name=None, person_type=None, similarity=None, recognition_phase='collecting'))
                else:
                    refreshed.append(track)
            return refreshed

    @staticmethod
    def _wake_streams(runtime):
        if runtime.stream_loop is not None and runtime.frame_ready is not None:
            try:
                runtime.stream_loop.call_soon_threadsafe(runtime.frame_ready.set)
            except RuntimeError:
                pass  # Event loop already closed during shutdown.

    @staticmethod
    def _publish_encoded_frame(runtime, jpeg):
        with runtime.jpeg_lock:
            runtime.queue.clear()
            runtime.queue.append(jpeg)
            runtime.encoded_sequence += 1
        VideoService._wake_streams(runtime)

    def _annotate_and_encode(
        self,
        camera_id: str,
        frame: np.ndarray,
        tracks: list[TrackSnapshot],
    ) -> None:
        # Look up runtime and skip encoding if no client is consuming frames.
        encode_started = time.perf_counter()
        with self._state_lock:
            runtime = self._camera_states.get(camera_id)

            if runtime is None or runtime.active_consumers == 0:
                return

        tracks = self._refresh_overlay_states(camera_id, tracks)
        # Draw overlays for each visible track.
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            box_color, label = self._overlay_style(track)

            # Draw bounding box around the tracked face/person.
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Prepare label text dimensions.
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw label background pill above the box. If there's no room
            # above (box near top of frame), draw it inside the top of the box.
            pad_x, pad_y = 6, 4
            pill_h = text_h + baseline + pad_y * 2
            pill_w = text_w + pad_x * 2
            pill_x1 = x1
            pill_y1 = y1 - pill_h if y1 - pill_h >= 0 else y1
            pill_x2 = min(pill_x1 + pill_w, frame.shape[1] - 1)
            pill_y2 = pill_y1 + pill_h

            # Draw filled label background.
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), box_color, -1)

            # Draw label text on top of the background.
            cv2.putText(
                frame,
                label,
                (pill_x1 + pad_x, pill_y2 - pad_y - baseline // 2),
                font,
                font_scale,
                (15, 23, 42),
                font_thickness,
                cv2.LINE_AA,
            )

        # Draw a small summary badge showing number of visible tracks.
        badge = f"Faces: {len(tracks)}"
        cv2.rectangle(frame, (6, 6), (140, 34), (15, 23, 42), -1)
        cv2.putText(
            frame,
            badge,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )

        # Output dimensions are upper bounds. Upscaling an 800x600 input to
        # 1280x720 adds encoding/network work and distorts faces without adding
        # detail. Keep the source aspect ratio and downscale only when needed.
        height, width = frame.shape[:2]
        scale = min(1., self.settings.output_width / width, self.settings.output_height / height)
        output = (cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                  if scale < 1 else frame)

        # Encode resized frame as JPEG.
        ok, buffer = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])

        # Skip if encoding failed.
        if not ok:
            return

        # Append encoded frame to streaming queue.
        self._publish_encoded_frame(runtime, buffer.tobytes())
        with self._state_lock:
            elapsed = (time.perf_counter()-encode_started)*1000
            runtime.timings_ms['encode'] = round(.8*runtime.timings_ms.get('encode', elapsed)+.2*elapsed, 2)
