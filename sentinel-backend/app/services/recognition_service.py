"""High-level face recognition logic -- search, person CRUD, and live recognition.

This module orchestrates the full recognition pipeline:

1. **Live recognition** -- given a video frame and a bounding box, align the
   face, extract an embedding via AdaFace, query the FAISS index, and return
   a match if the similarity exceeds the configured threshold.
2. **Image search** -- accept an uploaded image, detect the largest face,
   embed it, and return the top-k nearest enrolled persons enriched with
   their most recent recognition events.
3. **Person CRUD** -- list, update, and soft-delete enrolled persons while
   keeping the FAISS index in sync.

The service depends on ``InferenceService`` (detection), ``FaceAlignmentService``
(landmark alignment), ``AdaFaceService`` (embedding extraction), ``FaissService``
(vector search), and the person/embedding/event database repositories.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

import cv2
import numpy as np

from app.core.config import Settings
from app.models.schemas import ActiveTrackRead, EmbeddingStats, PersonRead, PersonUpdate, SearchMatch, SearchResponse
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.person_repository import PersonRepository, _UNSET
from app.repositories.recognition_event_repository import RecognitionEventRepository
from app.services.adaface_service import AdaFaceService
from app.services.face_alignment_service import FaceAlignmentService
from app.services.faiss_service import FaissService, IndexedEmbedding
from app.services.inference_service import DetectionResult, InferenceService


@dataclass(slots=True)

# Result of a successful face recognition against the FAISS index.
class RecognitionMatch:

    person_id: str
    person_name: str
    person_type: str | None
    similarity: float
    embedding_id: str


@dataclass(slots=True)

# Outcome of a live recognition attempt, including candidates that were rejected.
class RecognitionOutcome:
    """Full result of one recognition attempt against the FAISS index.

    ``match`` is populated only when the best candidate cleared
    ``recognition_threshold``.  ``best_similarity`` carries the score of the
    best candidate whenever the index was actually queried -- the rejected
    case included -- so analytics can chart the whole similarity distribution
    instead of only the accepted half.  ``evaluated`` stays False when no
    comparison could be made at all (for example the face could not be
    embedded), which must not be reported as an unknown person.
    """

    match: RecognitionMatch | None = None
    best_similarity: float | None = None
    evaluated: bool = False
    ambiguous: bool = False
    quality_score: float | None = None
    quality_rejected: bool = False
    candidates: list[dict] = field(default_factory=list)


# The main service class for orchestrating face recognition operations, including live recognition during video processing, image-based search, and person record management.
class RecognitionService:

    def __init__(
        self,
        settings: Settings,
        inference_service: InferenceService,
        adaface_service: AdaFaceService,
        face_alignment_service: FaceAlignmentService,
        faiss_service: FaissService,
        person_repository: PersonRepository,
        embedding_repository: EmbeddingRepository,
        event_repository: RecognitionEventRepository,
    ) -> None:

        self.settings = settings
        self.inference_service = inference_service
        self.adaface_service = adaface_service
        self.face_alignment_service = face_alignment_service
        self.faiss_service = faiss_service
        self.person_repository = person_repository
        self.embedding_repository = embedding_repository
        self.event_repository = event_repository
        self.logger = logging.getLogger("sentinel.recognition")

# Start dependent services and load or rebuild the FAISS index. 
    def start(self) -> None:
        
        self.adaface_service.start()
        self.face_alignment_service.start()
        # Attempt to load persisted index; rebuild if missing or schema is outdated
        if not self.faiss_service.load_from_disk() or not self.faiss_service.has_person_type_metadata():
            self._refresh_faiss_index()

# Stop dependent services (alignment and embedding).
    def stop(self) -> None:
        self.face_alignment_service.stop()
        self.adaface_service.stop()

    @staticmethod
    # Helper method to decode raw uploaded image bytes into a BGR NumPy array for processing by OpenCV and the detection model.
    def _decode_upload(contents: bytes) -> np.ndarray:
        data = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image")
        return frame

# Rebuild the FAISS index from all embeddings currently in the database.
    def _refresh_faiss_index(self) -> None:
        records = []
        for entity in self.embedding_repository.list_all():
            # Skip orphaned embeddings whose person was soft-deleted
            if entity.person is None:
                continue
            records.append(
                IndexedEmbedding(
                    embedding_id=str(entity.id),
                    person_id=str(entity.person_id),
                    person_name=entity.person.full_name,
                    person_type=entity.person.person_type,
                    vector=np.asarray(entity.embedding_vector, dtype=np.float32),
                )
            )
        self.faiss_service.rebuild(records)

# Detect faces in a frame and return the largest one, aligned if possible.
    def _detect_primary_face(self, frame: np.ndarray) -> tuple[np.ndarray, list[int], bool]:

        det_result = self.inference_service.detect_faces(frame)
        if det_result.bboxes.shape[0] == 0:
            raise ValueError("No face detected in the provided image")

        # Select the detection with the largest bounding-box area
        areas = (det_result.bboxes[:, 2] - det_result.bboxes[:, 0]) * (det_result.bboxes[:, 3] - det_result.bboxes[:, 1])
        best_idx = int(np.argmax(areas))
        bbox = det_result.bboxes[best_idx, :4].astype(np.int32).tolist()
        kps = det_result.kpss[best_idx]

        # Try keypoint-based alignment first (no re-detection needed)
        aligned = self.face_alignment_service.align_from_keypoints(frame, kps)
        if aligned is not None:
            return aligned, bbox, True

        # Fallback: landmark-based alignment via re-detection
        aligned = self.face_alignment_service.align(frame, bbox)
        if aligned is not None:
            return aligned, bbox, True

        # Fall back to a simple crop when alignment fails
        self.logger.warning("Alignment failed for search bbox=%s; falling back to legacy crop", bbox)
        crop, crop_bbox = self.adaface_service.crop_face(frame, bbox)
        if crop.size == 0:
            raise ValueError("Detected face crop is empty")
        return crop, crop_bbox, False

# Extract a face embedding from a video frame given a bounding box.
    def extract_embedding_from_frame(self, frame: np.ndarray, bbox: list[int], *, kps: np.ndarray | None = None) -> tuple[np.ndarray, float]:

        aligned = self._align_for_embedding(frame, bbox, kps)
        if aligned is not None:
            embedding, quality = self.adaface_service.extract_embedding_aligned(aligned)
        else:
            # Fall back to simple crop when landmark detection fails
            embedding, quality = self._embed_unaligned(frame, bbox)
        if embedding is None or np.asarray(embedding).size == 0:
            raise ValueError("No embedding returned for detected face")
        return embedding, quality

# Attempt to recognize a face in a video frame during live processing.

    def recognize(
        self,
        frame: np.ndarray,
        bbox: list[int],
        *,
        track_id: int | None = None,
        kps: np.ndarray | None = None,
    ) -> RecognitionOutcome:

        self.logger.debug("Track %s recognition attempt started bbox=%s", track_id, bbox)
        try:
            embedding, _ = self.extract_embedding_from_frame(frame, bbox, kps=kps)
        except ValueError as exc:
            self.logger.debug("Track %s recognition skipped: %s bbox=%s", track_id, exc, bbox)
            raise
        # Query FAISS for the nearest enrolled embeddings
        results = self.faiss_service.search(embedding, self.settings.recognition_top_k)
        return self._match_from_results(results, bbox=bbox, track_id=track_id)

    def recognize_batch(
        self,
        requests: list[tuple[np.ndarray, list[int], np.ndarray | None, int | None]],
    ) -> list[RecognitionOutcome]:
        """Recognise several faces in one batched AdaFace pass and FAISS query.

        Each request is ``(frame, bbox, kps, track_id)``. When ``kps`` is
        supplied the face is aligned with a plain affine warp; otherwise the
        landmark detector runs as a fallback, which costs an extra ONNX
        inference per face.

        Faces that cannot be aligned or embedded are skipped and reported as
        an unevaluated ``RecognitionOutcome`` in the corresponding output slot,
        so the result list always lines up positionally with ``requests`` and
        pipeline failures stay out of the unknown statistics.
        """
        if not requests:
            return []

        vectors: list[np.ndarray | None] = [None] * len(requests)

        aligned_faces: list[np.ndarray] = []
        # Maps a row in aligned_faces back to its index in requests.
        source_indices: list[int] = []

        for index, (frame, bbox, kps, track_id) in enumerate(requests):
            try:
                aligned = self._align_for_embedding(frame, bbox, kps)
            except Exception:
                self.logger.exception("Track %s alignment failed bbox=%s", track_id, bbox)
                continue

            if aligned is not None:
                aligned_faces.append(aligned)
                source_indices.append(index)
                continue

            # Alignment unavailable for this face: fall back to the unaligned
            # square crop, one face at a time. This is the rare path -- the
            # detector supplies keypoints for everything it finds.
            try:
                vectors[index], _quality = self._embed_unaligned(frame, bbox)
            except ValueError as exc:
                self.logger.debug("Track %s recognition skipped: %s bbox=%s", track_id, exc, bbox)

        if aligned_faces:
            # One GPU round trip for every alignable face in this cycle.
            batch_vectors, _norms = self.adaface_service.extract_embeddings_aligned(aligned_faces)
            for row, index in enumerate(source_indices):
                vectors[index] = batch_vectors[row]

        embedded = [(index, vector) for index, vector in enumerate(vectors) if vector is not None]
        results: list[RecognitionOutcome] = [RecognitionOutcome() for _ in requests]
        if not embedded:
            return results

        # One FAISS call for every face in this cycle.
        candidate_lists = self.faiss_service.search_batch(
            [vector for _index, vector in embedded],
            self.settings.recognition_top_k,
        )

        for (index, _vector), candidates in zip(embedded, candidate_lists, strict=False):
            _frame, bbox, _kps, track_id = requests[index]
            results[index] = self._match_from_results(candidates, bbox=bbox, track_id=track_id)

        return results

    def recognize_aligned_batch(self, faces: list[np.ndarray], requests: list) -> list[RecognitionOutcome]:
        """Strict live path: only quality-approved aligned images enter AdaFace."""
        vectors, _ = self.adaface_service.extract_embeddings_aligned(faces)
        candidates = self.faiss_service.search_batch(
            vectors, max(2, self.settings.recognition_top_k), distinct_people=True,
        )
        outcomes = []
        for rows, job in zip(candidates, requests, strict=True):
            outcome = self._match_from_results(rows, bbox=job.bbox, track_id=job.track_id)
            outcome.candidates = rows[:2]
            if (outcome.match is not None and len(rows) > 1
                    and rows[0]['similarity'] - rows[1]['similarity'] < self.settings.recognition_identity_margin):
                outcome.match = None
                outcome.ambiguous = True
            outcomes.append(outcome)
        return outcomes

    def _align_for_embedding(
        self,
        frame: np.ndarray,
        bbox: list[int],
        kps: np.ndarray | None,
    ) -> np.ndarray | None:
        """Produce a 112x112 aligned face, preferring detector keypoints.

        Keypoint alignment is a pure affine warp (~0.1 ms). The bbox fallback
        re-runs a whole landmark detection network on a crop (~2.7 ms), so it
        should only trigger when the detector did not supply keypoints.

        Returns ``None`` when the face cannot be aligned at all, leaving the
        caller to decide whether an unaligned crop is acceptable.
        """
        if kps is not None and np.asarray(kps).shape == (5, 2):
            aligned = self.face_alignment_service.align_from_keypoints(frame, np.asarray(kps))
            if aligned is not None:
                return aligned

        return self.face_alignment_service.align(frame, bbox)

    def _embed_unaligned(self, frame: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, float]:
        """Embed a face from a plain margin crop when alignment is unavailable."""
        self.logger.warning("Alignment failed for live recognition bbox=%s; falling back to legacy crop", bbox)
        crop, _ = self.adaface_service.crop_face(frame, bbox)
        if crop.size == 0:
            raise ValueError("Detected face crop is empty")
        return self.adaface_service.extract_embedding(crop)

    def _match_from_results(
        self,
        results: list[dict],
        *,
        bbox: list[int],
        track_id: int | None,
    ) -> RecognitionMatch | None:
        """Apply the acceptance threshold to a ranked FAISS candidate list."""
        if not results:
            # The comparison ran but the index held nothing to compare against.
            # That is still a genuine unknown, just without a similarity score.
            self.logger.debug("Track %s recognition candidate: none bbox=%s", track_id, bbox)
            return RecognitionOutcome(evaluated=True)
        # Take the best (highest similarity) candidate
        best = results[0]
        similarity = float(best["similarity"])
        accepted = similarity >= self.settings.recognition_threshold
        self.logger.debug(
            "Track %s recognition candidate: %s similarity=%.4f threshold=%.4f accepted=%s bbox=%s",
            track_id,
            best["person_name"],
            similarity,
            self.settings.recognition_threshold,
            str(accepted).lower(),
            bbox,
        )
        if not accepted:
            # Keep the rejected score: it is the other half of the similarity
            # distribution and the only evidence of a near-miss.
            return RecognitionOutcome(best_similarity=similarity, evaluated=True)
        return RecognitionOutcome(
            match=RecognitionMatch(
                person_id=best["person_id"],
                person_name=best["person_name"],
                person_type=best.get("person_type"),
                similarity=similarity,
                embedding_id=best["embedding_id"],
            ),
            best_similarity=similarity,
            evaluated=True,
        )

# Search the enrolled face database using an uploaded query image, returning the top-k most similar persons along with their most recent recognition events. 
    def search_image(self, image_bytes: bytes, top_k: int) -> SearchResponse:

        self.start()
        frame = self._decode_upload(image_bytes)
        crop, bbox, is_aligned = self._detect_primary_face(frame)
        # Time the embedding extraction and FAISS search together
        t0 = time.perf_counter()
        if is_aligned:
            embedding, _ = self.adaface_service.extract_embedding_aligned(crop)
        else:
            embedding, _ = self.adaface_service.extract_embedding(crop)
        # One row per enrolled person, at that person's closest template. A
        # person is enrolled with many templates, so a plain top-k search can
        # fill the whole ranked list with the same identity.
        matches = self.faiss_service.search_batch([embedding], top_k, distinct_people=True)[0]
        search_ms = (time.perf_counter() - t0) * 1000.0

        # Enrich each match with the latest recognition event data (camera, timestamp)
        latest_events = self.event_repository.latest_for_person(
            person_ids=[match["person_id"] for match in matches]
        )
        results = []
        for match in matches:
            event = latest_events.get(match["person_id"])
            results.append(
                SearchMatch(
                    person_id=match["person_id"],
                    name=match["person_name"],
                    similarity=match["similarity"],
                    embedding_id=match["embedding_id"],
                    camera=event.camera_name if event else None,
                    timestamp=event.timestamp if event else None,
                )
            )
        return SearchResponse(query_bbox=bbox, results=results, search_ms=round(search_ms, 3))

# List all enrolled persons with their embedding counts, latest recognition event, and active/inactive status based on live camera tracks.
    def list_people(self, *, active_tracks: list[ActiveTrackRead] | None = None) -> list[PersonRead]:
        latest_events = self.event_repository.latest_for_person()
        live_tracks = self._build_live_track_lookup(active_tracks or [])
        return [
            self._build_person_read(
                person=person,
                embedding_count=embedding_count,
                last_event=latest_events.get(str(person.id)),
                live_track=live_tracks.get(str(person.id)),
            )
            for person, embedding_count in self.person_repository.list_with_embedding_counts()
        ]

# Update a person's fields and rebuild the FAISS index.
    def update_person(
        self,
        person_id: str,
        payload: PersonUpdate,
        *,
        active_tracks: list[ActiveTrackRead] | None = None,
    ) -> PersonRead | None:
        
        changes = payload.model_dump(exclude_unset=True)
        entity = self.person_repository.update_person(
            person_id,
            full_name=changes.get("name", _UNSET),
            person_type=changes.get("person_type", _UNSET),
            notes=changes.get("notes", _UNSET),
            is_active=changes.get("is_active", _UNSET),
            embedding_only_mode=changes.get("embedding_only_mode", _UNSET),
            retention_expires_at=changes.get("retention_expires_at", _UNSET),
        )
        if entity is None:
            return None
        # Rebuild FAISS to pick up name/type changes in the index metadata
        self._refresh_faiss_index()
        record = self.person_repository.get_with_embedding_count(person_id)
        if record is None:
            return None
        person, embedding_count = record
        latest_event = self.event_repository.latest_for_person(person_ids=[person.id]).get(str(person.id))
        live_track = self._build_live_track_lookup(active_tracks or []).get(str(person.id))
        return self._build_person_read(
            person=person,
            embedding_count=embedding_count,
            last_event=latest_event,
            live_track=live_track,
        )

# Soft-delete a person and rebuild the FAISS index to remove their embeddings from search results.
    def delete_person(self, person_id: str) -> bool:
        
        deleted = self.person_repository.delete(person_id)
        if deleted:
            # Remove the deleted person's embeddings from the FAISS index
            self._refresh_faiss_index()
        return deleted

# Return aggregate statistics about the FAISS index and embedding store.
    def get_embedding_stats(self) -> EmbeddingStats:
        
        total_vectors, total_persons = self.embedding_repository.stats()
        return EmbeddingStats(**self.faiss_service.stats(total_persons=total_persons))

    @staticmethod
    # Helper method to build a metadata dict from a person entity's type and notes fields, merging any JSON-parsable notes into the metadata for richer search results.
    def _person_metadata(person) -> dict[str, Any]:
        metadata = {"person_type": person.person_type}
        if person.notes:
            try:
                parsed = json.loads(person.notes)
                if isinstance(parsed, dict):
                    # Merge parsed JSON keys directly into the metadata dict
                    metadata.update(parsed)
                else:
                    metadata["notes"] = person.notes
            except json.JSONDecodeError:
                # Notes are plain text, not JSON
                metadata["notes"] = person.notes
        return metadata

    @staticmethod
    # Helper method to build a lookup of active live tracks by person_id, used for determining which enrolled persons are currently visible in camera feeds.
    def _build_live_track_lookup(active_tracks: list[ActiveTrackRead]) -> dict[str, ActiveTrackRead]:

        live_tracks: dict[str, ActiveTrackRead] = {}
        for track in active_tracks:
            # Only consider tracks that have been successfully recognized
            if track.status != "recognized" or not track.person_id:
                continue
            current = live_tracks.get(track.person_id)
            # Keep the most recently seen track for each person
            if current is None or track.last_seen > current.last_seen:
                live_tracks[track.person_id] = track
        return live_tracks

# Helper method to construct a PersonRead schema object with active/inactive status based on the presence of a live track or recent recognition event, 
# used for enriching person listings with real-time status information.
    def _build_person_read(self, *, person, embedding_count: int, last_event, live_track: ActiveTrackRead | None = None) -> PersonRead:
        
        status = "inactive"
        last_seen_camera = None
        last_seen_time = None
        if live_track is not None:
            # Person is currently visible in a live camera feed
            status = "active"
            last_seen_camera = live_track.camera_name
            last_seen_time = live_track.last_seen
        elif last_event is not None:
            # Person was seen historically but is not currently on screen
            timestamp = last_event.timestamp
            # Ensure the timestamp is timezone-aware for consistent serialisation
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            last_seen_camera = last_event.camera_name
            last_seen_time = timestamp
        return PersonRead(
            id=str(person.id),
            name=person.full_name,
            person_type=person.person_type,
            notes=person.notes,
            is_active=person.is_active,
            embedding_only_mode=person.embedding_only_mode,
            enrolled_at=person.enrolled_at,
            retention_expires_at=person.retention_expires_at,
            metadata=self._person_metadata(person),
            total_embeddings=embedding_count,
            last_seen_camera=last_seen_camera,
            last_seen_time=last_seen_time,
            status=status,
        )
