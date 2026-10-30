"""DeepSORT multi-object tracking wrapper for per-camera face tracking.

This module provides a thin abstraction over the ``deep_sort_realtime``
library, managing one ``DeepSort`` tracker instance per camera.  It handles:

- Lazy creation of trackers with tuned hyper-parameters (max age, n_init,
  IOU/cosine distance thresholds, appearance descriptor).
- Conversion of detections ``[x1, y1, x2, y2, conf, cls]`` into the
  ``([x, y, w, h], conf, class)`` tuples expected by DeepSORT.
- Computing the appearance descriptor used for association.  The library's
  built-in MobileNetV2 re-identification network costs roughly 15 ms per frame
  for nine faces -- more than the detector, the Kalman/assignment step and the
  JPEG encoder combined -- so the default is a colour histogram that costs
  microseconds.  Faces detected at 15 Hz or more have strong frame-to-frame
  IOU continuity, which is what actually drives association here.
- Carrying the detector's 5-point keypoints through to the matched track via
  DeepSORT's ``others`` channel, so the recognition stage can align faces with
  a plain affine warp instead of re-running a landmark detector.
- Confirming tracks at birth when ``n_init <= 1``, so a new face is drawable
  immediately and survives the skipped inference cycles that ``infer_every``
  introduces.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.core.config import Settings


# Hue/saturation bin counts for the cheap appearance descriptor.
_HIST_BINS = (8, 8)
_HIST_RANGES = (0, 180, 0, 256)

# Side length the face crop is reduced to before histogramming. Small enough to
# be nearly free, large enough to be stable.
_HIST_CROP_SIDE = 32

# Descriptor width DeepSORT stores per track. The histogram descriptor is
# 8 * 8 = 64 values; "none" mode emits a constant vector of this width.
_HIST_DIM = _HIST_BINS[0] * _HIST_BINS[1]


def _build_track_class():
    """Build a Track subclass that is Confirmed from birth when n_init <= 1.

    deep_sort_realtime always creates tracks Tentative and only promotes them
    inside ``Track.update()``, i.e. when a second detection matches. With
    ``infer_every > 1`` the very next cycle carries no detections, so
    ``mark_missed()`` deletes the tentative track before it is ever promoted
    and new faces never become drawable.

    Promoting the track after ``update_tracks()`` returns -- the previous
    approach -- is not safe: ``Tracker.update()`` registers appearance features
    in ``metric.samples`` for confirmed tracks only, and it has already done so
    by the time control returns. A track promoted afterwards is confirmed but
    absent from ``samples``, so the next frame that carries detections raises
    ``KeyError`` inside the matching cascade. That went unnoticed only because
    at ``infer_every=2`` the intervening empty-detection frame skips the
    appearance metric while still running ``partial_fit``, registering the
    track just in time. At ``infer_every=1`` it raised on nearly every frame.

    Confirming inside ``__init__`` instead means the track is already confirmed
    when ``partial_fit`` runs at the end of that same update, so its features
    are registered and the metric stays consistent.
    """
    from deep_sort_realtime.deep_sort.track import Track, TrackState

    class ImmediateConfirmTrack(Track):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self._n_init <= 1:
                self.state = TrackState.Confirmed

    return ImmediateConfirmTrack


# TrackingService class definition: Manages per-camera DeepSORT tracker instances for multi-object face tracking.
class TrackingService:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger("sentinel.tracking")
        # Per-camera tracker instances, keyed by camera_id
        self._trackers: dict[str, object] = {}
        self._embedder_mode = (settings.track_embedder or "histogram").strip().lower()
        if self._embedder_mode not in {"histogram", "mobilenet", "none"}:
            self.logger.warning(
                "Unknown track_embedder %r; falling back to 'histogram'",
                self._embedder_mode,
            )
            self._embedder_mode = "histogram"

    @property
    def uses_builtin_embedder(self) -> bool:
        """True when DeepSORT computes appearance features itself."""
        return self._embedder_mode == "mobilenet"

# Ensure the tracker for a given camera ID exists and return it, creating a new one if necessary.
    def ensure_tracker(self, camera_id: str) -> object:
        tracker = self._trackers.get(camera_id)
        if tracker is not None:
            return tracker

        # Deferred import to avoid loading heavy dependencies at module level
        from deep_sort_realtime.deepsort_tracker import DeepSort

        # embedder=None tells DeepSORT that we will supply features ourselves,
        # and crucially stops it from loading the MobileNet weights at all.
        embedder = "mobilenet" if self.uses_builtin_embedder else None

        tracker = DeepSort(
            max_age=self.settings.track_max_age,
            n_init=self.settings.track_n_init,
            max_iou_distance=self.settings.track_max_iou_distance,
            max_cosine_distance=self.settings.track_max_cosine_distance,
            embedder=embedder,
            embedder_gpu=True,
            nn_budget=100,
            override_track_class=_build_track_class(),
        )
        self._trackers[camera_id] = tracker
        self.logger.info(
            "Tracker created for %s (embedder=%s, max_cosine_distance=%.2f)",
            camera_id,
            self._embedder_mode,
            self.settings.track_max_cosine_distance,
        )
        return tracker

# Remove and discard the tracker for a camera, typically called when a camera is deactivated or removed to prevent memory leaks.
    def remove_tracker(self, camera_id: str) -> None:
        self._trackers.pop(camera_id, None)

# Filter out tracks DeepSORT has marked for deletion.
    def _normalize_tracks(self, tracks: list[object]) -> list[object]:
        # Confirmation is handled at track construction (see _build_track_class);
        # mutating track.state from out here corrupts the appearance metric.
        return [track for track in tracks if not track.is_deleted()]

    def _appearance_features(self, frame: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        """Build one appearance descriptor per detection.

        ``histogram`` mode uses a normalised hue/saturation histogram of the
        face crop: robust enough to separate different people in the same
        frame, and cheap enough to be invisible in a profile.  ``none`` mode
        emits a constant vector so association falls back to Kalman + IOU.
        """
        if self._embedder_mode == "none":
            features = np.zeros((len(boxes), _HIST_DIM), dtype=np.float32)
            features[:, 0] = 1.0
            return features

        frame_h, frame_w = frame.shape[:2]
        features = np.zeros((len(boxes), _HIST_DIM), dtype=np.float32)

        for index, (x, y, w, h) in enumerate(boxes):
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x + w), frame_w)
            y2 = min(int(y + h), frame_h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                # Degenerate crop; leave a zero descriptor, which the cosine
                # metric treats as maximally distant from everything.
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (_HIST_CROP_SIDE, _HIST_CROP_SIDE), interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, list(_HIST_BINS), list(_HIST_RANGES))
            vector = histogram.reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-6:
                features[index] = vector / norm

        return features

# Update the tracker for a given camera with new detections and return the list of active tracks.
# Detections are expected as [x1, y1, x2, y2, conf, cls] and are converted to DeepSORT's format internally.
# An empty detection list is still passed to DeepSORT to ensure consistent Kalman prediction and lifecycle management.
    def update_tracks(
        self,
        camera_id: str,
        detections: np.ndarray,
        frame: np.ndarray,
        kpss: np.ndarray | None = None,
    ) -> list[object]:

        tracker = self.ensure_tracker(camera_id)

        if detections is None or detections.size == 0:
            # Use DeepSORT's public API with an empty detection list so
            # Kalman predict + lifecycle bookkeeping stay internally
            # consistent. Calling tracker.tracker.predict() directly bumps
            # time_since_update on every track but skips the state
            # transitions DeepSORT expects after predict, which causes
            # tentative tracks to get mis-counted and die before reaching
            # confirmation - that is why bounding boxes stopped appearing.
            return self._normalize_tracks(tracker.update_tracks([], frame=frame, embeds=np.zeros((0, _HIST_DIM), np.float32)))

        # Convert corner-format [x1, y1, x2, y2] to DeepSORT's [x, y, w, h].
        # Degenerate boxes are dropped here rather than inside DeepSORT: the
        # library filters raw_detections for positive width/height but does
        # not filter the parallel embeds/others lists, which would silently
        # misalign features and keypoints with their detections.
        formatted: list[tuple[list[float], float, str]] = []
        boxes: list[list[float]] = []
        others: list[np.ndarray | None] = []

        for index, row in enumerate(detections):
            x1, y1, x2, y2, conf = (float(value) for value in row[:5])
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            ltwh = [x1, y1, width, height]
            formatted.append((ltwh, conf, "face"))
            boxes.append(ltwh)
            others.append(np.asarray(kpss[index], dtype=np.float32) if kpss is not None else None)

        if not formatted:
            return self._normalize_tracks(tracker.update_tracks([], frame=frame, embeds=np.zeros((0, _HIST_DIM), np.float32)))

        if self.uses_builtin_embedder:
            # Let DeepSORT run its own MobileNet embedder over the crops.
            tracks = tracker.update_tracks(formatted, frame=frame, others=others)
        else:
            tracks = tracker.update_tracks(
                formatted,
                frame=frame,
                embeds=self._appearance_features(frame, boxes),
                others=others,
            )

        return self._normalize_tracks(tracks)
