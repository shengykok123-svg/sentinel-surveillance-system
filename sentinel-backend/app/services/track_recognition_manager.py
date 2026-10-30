"""Per-track recognition state machine.

Instead of recognising a track once on its first frame, this manager gates
recognition on cheap frame quality checks, pools embeddings into a
quality-weighted template, retries across frames until a confident match
or K good-quality misses, guards against DeepSORT ID-switches, and can
re-open a written-off track when a much better frame arrives or after a
recognition refresh (new enrollment).

The manager is designed to be driven from ``VideoService._sync_track_cache_from_tracks``
one call per confirmed track per frame via ``process()``, with ``sync_active()``
called once at the end of each camera cycle to drop retired tracks.

Thread safety: a ``threading.Lock`` protects ``self._tracks`` since
``refresh()`` can be called from API threads while ``process()`` runs
from the pipeline thread.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import cv2
import numpy as np

from app.core.config import Settings
from app.services.recognition_service import RecognitionService


class TrackState(str, Enum):
    PENDING = "pending"
    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"


@dataclass
class Decision:
    """What the video loop should do with this track this frame."""

    state: TrackState
    person_id: str | None = None
    person_name: str | None = None
    person_type: str | None = None
    similarity: float | None = None
    embedding_id: str | None = None
    det_score: float = 0.0
    embedded: bool = False
    log_event: bool = False
    event_kind: str | None = None
    message: str = ""


@dataclass
class _Track:
    embeddings: list[np.ndarray] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    state: TrackState = TrackState.PENDING
    good_miss_count: int = 0
    best_quality: float = 0.0
    person_id: str | None = None
    person_name: str | None = None
    person_type: str | None = None
    embedding_id: str | None = None
    similarity: float | None = None
    pending_person_id: str | None = None
    match_streak: int = 0

    def template(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        embs = np.stack(self.embeddings)
        w = np.asarray(self.weights, dtype=np.float32)
        w = w / (w.sum() + 1e-9)
        pooled = (embs * w[:, None]).sum(axis=0)
        return pooled / (np.linalg.norm(pooled) + 1e-9)


class TrackRecognitionManager:
    _MIN_TEMPLATE_CONSISTENCY = 0.55

    def __init__(
        self,
        recognition_service: RecognitionService,
        settings: Settings,
    ) -> None:
        self._rs = recognition_service
        self._settings = settings
        self._tracks: dict[tuple[str, int], _Track] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger("sentinel.track_manager")

    # ------------------------------------------------------------------ #
    # Main entry point — call once per confirmed track per frame.        #
    # ------------------------------------------------------------------ #
    def process(
        self,
        camera_id: str,
        track_id: int,
        frame: np.ndarray,
        bbox: Sequence[float],
        det_score: float,
        *,
        kps: np.ndarray | None = None,
    ) -> Decision:
        key = (camera_id, track_id)
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                track = _Track()
                self._tracks[key] = track

        # Already recognised -> reuse label, no compute
        if track.state is TrackState.RECOGNIZED:
            return self._decide(track, det_score=det_score, msg="already recognised")

        # Cheap quality gate (no embedding yet)
        passed, score = self._assess(frame, bbox, det_score)
        prev_best = track.best_quality
        track.best_quality = max(prev_best, score)

        # Unknown -> only re-open on a MUCH better frame
        if track.state is TrackState.UNKNOWN:
            if passed and score >= prev_best * self._settings.recognition_reopen_margin:
                self._reopen(track)
            else:
                return self._decide(track, det_score=det_score, msg="unknown (kept)")

        # PENDING path
        if not passed:
            return self._decide(track, det_score=det_score, msg="low quality, waiting")

        # Good frame -> extract embedding
        try:
            embedding, raw_quality = self._rs.extract_embedding_from_frame(frame, bbox, kps=kps)
        except (ValueError, Exception) as exc:
            self._logger.debug(
                "Track (%s, %s) embedding extraction failed: %s", camera_id, track_id, exc
            )
            return self._decide(track, det_score=det_score, msg="embedding error, pending")

        embedding = np.asarray(embedding, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        weight = float(raw_quality)
        self._update_template(track, embedding, weight)

        # Match the pooled template against FAISS
        template = track.template()
        results = self._rs.faiss_service.search(template, self._settings.recognition_top_k)
        best = results[0] if results else None
        sim = float(best["similarity"]) if best else -1.0

        if best is not None and sim >= self._settings.recognition_threshold:
            return self._on_match(track, best, sim, det_score)
        return self._on_miss(track, det_score)

    # ------------------------------------------------------------------ #
    # State transitions                                                   #
    # ------------------------------------------------------------------ #
    def _on_match(self, track: _Track, best: dict, sim: float, det_score: float) -> Decision:
        pid = best["person_id"]
        if track.pending_person_id == pid:
            track.match_streak += 1
        else:
            track.pending_person_id, track.match_streak = pid, 1

        if track.match_streak >= self._settings.recognition_confirm_matches:
            track.state = TrackState.RECOGNIZED
            track.person_id = pid
            track.person_name = best.get("person_name")
            track.person_type = best.get("person_type")
            track.embedding_id = best.get("embedding_id")
            track.similarity = sim
            return self._decide(
                track,
                det_score=det_score,
                embedded=True,
                log_event=True,
                event_kind="recognized",
                msg=f"recognised {track.person_name} (sim={sim:.3f})",
            )
        return self._decide(
            track,
            det_score=det_score,
            embedded=True,
            msg=f"provisional {best.get('person_name')} "
            f"({track.match_streak}/{self._settings.recognition_confirm_matches})",
        )

    def _on_miss(self, track: _Track, det_score: float) -> Decision:
        track.good_miss_count += 1
        track.pending_person_id, track.match_streak = None, 0
        if track.good_miss_count >= self._settings.recognition_max_good_misses:
            track.state = TrackState.UNKNOWN
            return self._decide(
                track,
                det_score=det_score,
                embedded=True,
                log_event=True,
                event_kind="unknown",
                msg="confirmed unknown (exhausted good misses)",
            )
        return self._decide(
            track,
            det_score=det_score,
            embedded=True,
            msg=f"no match ({track.good_miss_count}/"
            f"{self._settings.recognition_max_good_misses} good misses)",
        )

    def _reopen(self, track: _Track) -> None:
        track.state = TrackState.PENDING
        track.good_miss_count = 0
        track.match_streak = 0
        track.pending_person_id = None

    # ------------------------------------------------------------------ #
    # Template management (quality-weighted pooling + ID-switch guard)    #
    # ------------------------------------------------------------------ #
    def _update_template(self, track: _Track, emb: np.ndarray, weight: float) -> None:
        current = track.template()
        if current is not None:
            if float(np.dot(emb, current)) < self._MIN_TEMPLATE_CONSISTENCY:
                track.embeddings, track.weights = [emb], [weight]
                track.good_miss_count = 0
                track.match_streak = 0
                track.pending_person_id = None
                return
        track.embeddings.append(emb)
        track.weights.append(weight)
        if len(track.embeddings) > self._settings.recognition_template_size:
            order = np.argsort(track.weights)[::-1][: self._settings.recognition_template_size]
            track.embeddings = [track.embeddings[i] for i in order]
            track.weights = [track.weights[i] for i in order]

    # ------------------------------------------------------------------ #
    # Quality gate (cheap, pre-embedding)                                 #
    # ------------------------------------------------------------------ #
    def _assess(
        self, frame: np.ndarray, bbox: Sequence[float], det_score: float
    ) -> tuple[bool, float]:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        face_h = max(0, y2 - y1)
        blur = self._blur_variance(frame, (x1, y1, x2, y2))
        passed = (
            face_h >= self._settings.recognition_min_face_size
            and det_score >= self._settings.recognition_min_det_score
            and blur >= self._settings.recognition_min_blur_var
        )
        size_s = float(np.clip(face_h / 120.0, 0.0, 1.0))
        blur_s = float(np.clip(blur / 150.0, 0.0, 1.0))
        det_s = float(np.clip(det_score, 0.0, 1.0))
        score = 0.4 * blur_s + 0.3 * size_s + 0.3 * det_s
        return passed, score

    @staticmethod
    def _blur_variance(frame: np.ndarray, box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ------------------------------------------------------------------ #
    # Lifecycle hooks                                                     #
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Re-open every UNKNOWN track so newly enrolled people get re-checked."""
        with self._lock:
            for track in self._tracks.values():
                if track.state is TrackState.UNKNOWN:
                    self._reopen(track)

    def sync_active(self, camera_id: str, active_ids: Sequence[int]) -> None:
        """Drop tracks the tracker no longer reports for this camera."""
        active = set(active_ids)
        with self._lock:
            stale = [k for k in self._tracks if k[0] == camera_id and k[1] not in active]
            for k in stale:
                self._tracks.pop(k, None)

    def clear(self) -> None:
        """Drop all tracked state (for VideoService.stop())."""
        with self._lock:
            self._tracks.clear()

    # ------------------------------------------------------------------ #
    # Helper                                                              #
    # ------------------------------------------------------------------ #
    def _decide(
        self,
        track: _Track,
        *,
        det_score: float = 0.0,
        embedded: bool = False,
        log_event: bool = False,
        event_kind: str | None = None,
        msg: str = "",
    ) -> Decision:
        return Decision(
            state=track.state,
            person_id=track.person_id,
            person_name=track.person_name,
            person_type=track.person_type,
            similarity=track.similarity,
            embedding_id=track.embedding_id,
            det_score=det_score,
            embedded=embedded,
            log_event=log_event,
            event_kind=event_kind,
            message=msg,
        )
