"""
Person enrollment service orchestrating face detection, embedding extraction,
database persistence, and FAISS index management.

This module handles the complete enrollment workflow:
    1. Decode and validate uploaded face images.
    2. Require exactly one clear face in each image.
    3. Attempt 5-point landmark alignment; fall back to margin-based cropping.
    4. Extract 512-d AdaFace embeddings from aligned/cropped faces.
    5. Reject duplicate embeddings (cosine similarity >= 0.999).
    6. Persist the person profile and embedding records in the database.
    7. Rebuild the FAISS nearest-neighbour index for real-time recognition.
    8. Notify the video service to refresh its recognition pipeline.

Key constants:
    - ``_DUPLICATE_EMBEDDING_SIMILARITY``: Cosine similarity threshold (0.999)
      above which two embeddings are considered duplicates of the same face pose.
    - ``_NO_FACE_DETECTED_REASON``: Error message when no face is found.
    - ``_LANDMARK_ALIGNMENT_FAILED_REASON``: Error message when landmark
      extraction fails.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.core.config import Settings
from app.models.entities import validate_embedding_vector
from app.models.schemas import EnrollmentResponse
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.person_repository import PersonRepository, normalize_person_type
from app.services.adaface_service import AdaFaceService
from app.services.face_alignment_service import FaceAlignmentService
from app.services.faiss_service import FaissService, IndexedEmbedding
from app.services.inference_service import DetectionResult, InferenceService
from app.repositories.reference_image_repository import ReferenceImageRepository
from app.services.reference_image_service import ReferenceImageService
from app.services.face_quality_service import FaceQualityService

if TYPE_CHECKING:
    from app.services.video_service import VideoService


# Cosine similarity threshold above which two embeddings are treated as duplicates.
# 0.999 is extremely high, meaning only near-identical face poses are rejected.
_DUPLICATE_EMBEDDING_SIMILARITY = 0.999

# Standard rejection reason strings used in error construction
_NO_FACE_DETECTED_REASON = "No face detected in the provided image"
_MULTIPLE_FACES_REASON = "Exactly one face must appear in the provided image"
_LANDMARK_ALIGNMENT_FAILED_REASON = "could not align face landmarks"


@dataclass(slots=True)
# Represents the raw image data uploaded for enrollment, including the original filename and binary contents.
class EnrollmentImagePayload:
    filename: str
    contents: bytes


# Represents a successfully extracted embedding from an enrollment image, including the source filename, bounding box, embedding vector, and optional quality score.
@dataclass(slots=True)
class ExtractedEmbedding:
    filename: str
    bbox: list[int]
    vector: np.ndarray
    quality_score: float | None
    frame: np.ndarray

# Represents an embedding record prepared for FAISS indexing, including person metadata and the embedding vector.
class EnrollmentValidationError(ValueError):

# Initialise the validation error.
    def __init__(
        self,
        message: str,
        *,
        code: str = "enrollment_validation_failed",
        rejected_images: list[str] | None = None,
    ) -> None:

        super().__init__(message)
        self.code = code
        self.rejected_images = rejected_images or []


# Represents an embedding record prepared for FAISS indexing, including person metadata and the embedding vector.
def serialized_enrollment(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._enrollment_lock:
            return method(self, *args, **kwargs)
    return wrapped


class EnrollmentService:

# Initialise the enrollment service with all required dependencies.
    def __init__(
        self,
        settings: Settings,
        inference_service: InferenceService,
        adaface_service: AdaFaceService,
        face_alignment_service: FaceAlignmentService,
        faiss_service: FaissService,
        person_repository: PersonRepository,
        embedding_repository: EmbeddingRepository,
        reference_image_repository: ReferenceImageRepository | None = None,
        reference_image_service: ReferenceImageService | None = None,
        video_service: VideoService | None = None,
        face_quality_service: FaceQualityService | None = None,
    ) -> None:
        self.settings = settings
        self.inference_service = inference_service
        self.adaface_service = adaface_service
        self.face_alignment_service = face_alignment_service
        self.faiss_service = faiss_service
        self.person_repository = person_repository
        self.embedding_repository = embedding_repository
        self.reference_image_repository = reference_image_repository
        self.reference_image_service = reference_image_service
        self.video_service = video_service
        self.face_quality_service = face_quality_service or FaceQualityService(settings)
        self.logger = logging.getLogger("sentinel.enrollment")
        self._enrollment_lock = threading.RLock()
        self.enhancement_service = None

# Start the enrollment service by ensuring all dependent services are ready and the FAISS index is loaded or rebuilt.
    def start(self) -> None:
        self.adaface_service.start()
        self.face_alignment_service.start()
        # Attempt to load persisted FAISS index; rebuild if missing or incomplete
        if not self.faiss_service.load_from_disk() or not self.faiss_service.has_person_type_metadata():
            self.index_embeddings_faiss()

# Create a person profile in the database without embedding processing.
    def create_person_profile(
        self,
        *,
        full_name: str,
        person_type: str,
        is_active: bool,
        embedding_only_mode: bool,
        retention_expires_at: datetime | None,
        notes: str | dict[str, Any] | list[Any] | None,
        enrolled_by_user_id: str,
    ):
       
        return self.person_repository.create_person(
            full_name=full_name,
            person_type=person_type,
            notes=self._normalize_notes(notes),
            is_active=is_active,
            embedding_only_mode=embedding_only_mode,
            enrolled_by_user_id=enrolled_by_user_id,
            retention_expires_at=self._normalize_datetime(retention_expires_at),
        )

# Process uploaded images to extract face embeddings, handling decoding, detection, alignment, extraction, and duplicate rejection.
    def process_images_and_extract_embeddings(
        self,
        images: list[EnrollmentImagePayload],
        existing_vectors: list[np.ndarray] | None = None,
    ) -> tuple[list[ExtractedEmbedding], list[str]]:
        
        extracted: list[ExtractedEmbedding] = []
        rejected_images: list[str] = []

        for index, image in enumerate(images, start=1):
            # Use original filename or generate a fallback name
            filename = image.filename or f"image-{index}.jpg"
            try:
                frame = self._decode_upload(image.contents)
                crop, bbox, quality_score = self._assess_face(frame)
                is_aligned = True
                if quality_score < self.settings.enrollment_quality_threshold:
                    raise ValueError("Face image quality is too low; please upload a clearer, unobstructed face image")
                # Use the aligned-specific extraction if alignment succeeded
                if is_aligned:
                    embedding, _ = self.adaface_service.extract_embedding_aligned(crop)
                else:
                    embedding, _ = self.adaface_service.extract_embedding(crop)
                embedding = self._normalize_embedding_for_storage(embedding)
                # Check for near-duplicate embeddings within this enrollment batch
                if self._is_duplicate_embedding(embedding, extracted, existing_vectors or []):
                    rejected_images.append(f"{filename}: duplicate embedding")
                    continue
                extracted.append(
                    ExtractedEmbedding(
                        filename=filename,
                        bbox=bbox,
                        vector=embedding,
                        quality_score=quality_score,
                        frame=frame,
                    )
                )
            except ValueError as exc:
                self.logger.warning("Rejected enrollment image %s: %s", filename, exc)
                rejected_images.append(f"{filename}: {exc}")

        # If no images produced valid embeddings, raise a structured error
        if not extracted:
            raise self._build_failed_enrollment_error(rejected_images)

        return extracted, rejected_images

    def _assess_face(self, frame):
        crop, bbox, _, aligned = self._detect_primary_face(frame)
        if not aligned:
            raise ValueError(_LANDMARK_ALIGNMENT_FAILED_REASON)
        return crop, bbox, self.face_quality_service.assess(crop)

    def assess_image_quality(self, image: EnrollmentImagePayload) -> dict:
        """Preview exactly the registration quality check without embeddings or persistence."""
        result = dict(filename=image.filename, model="CR-FIQA(S)",
                      threshold=self.settings.enrollment_quality_threshold, score=None)
        try:
            frame = self._decode_upload(image.contents)
            _, _, score = self._assess_face(frame)
        except ValueError as exc:
            return {**result, "status": "invalid_image", "message": str(exc)}
        passed = score >= self.settings.enrollment_quality_threshold
        return {**result, "score": score, "status": "passed" if passed else "low_quality",
                "message": "Meets quality threshold" if passed else "Below quality threshold; try another photo"}

    @serialized_enrollment
    def add_reference_images(
        self,
        *,
        person_id: str,
        images: list[EnrollmentImagePayload],
    ) -> dict[str, Any]:
        """Append validated image embeddings to an existing POI profile."""
        person = self.person_repository.get_person_by_id(person_id)
        if person is None:
            raise ValueError("POI profile not found")
        if not images:
            raise ValueError("At least one image is required")

        self.start()
        existing_embeddings = self.embedding_repository.list_for_person(person_id)
        existing_vectors = [
            self._normalize_embedding_for_storage(entity.embedding_vector)
            for entity in existing_embeddings
        ]
        extracted, rejected_images = self.process_images_and_extract_embeddings(images, existing_vectors)

        references = {}
        for source, item in enumerate(extracted):
            embedding = self.embedding_repository.save_embedding(
                person_id,
                self._normalize_embedding_for_storage(item.vector).tolist(),
                source_type="reference_upload",
                quality_score=item.quality_score,
                expires_at=person.retention_expires_at,
            )
            references[source] = self._retain_reference_image(person_id, embedding.id, item)

        enhancement = self._enhance(person_id, extracted, references, retain=True,
                                    expires_at=person.retention_expires_at, existing=existing_embeddings)

        # Reference additions can be searched only once all new vectors have
        # been persisted, so one rebuild is enough for the entire upload.
        self.index_embeddings_faiss()
        if self.video_service is not None:
            self.video_service.request_recognition_refresh(reason=f"reference-images:{person_id}")

        # The operator explicitly chose viewable references, so the profile
        # must no longer advertise embedding-only storage.
        if person.embedding_only_mode:
            self.person_repository.update_person(person_id, embedding_only_mode=False)

        return {
            "person_id": person_id,
            "embeddings_added": len(extracted) + enhancement.get("embeddings_added", 0),
            "original_images_added": len(extracted),
            "enhancement": enhancement,
            "rejected_images": rejected_images,
        }

# Rebuild the FAISS nearest-neighbour index from all stored embeddings
    def index_embeddings_faiss(self) -> None:

        records: list[IndexedEmbedding] = []
        for entity in self.embedding_repository.list_all():
            # Skip orphaned embeddings without a linked person
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

# Main entry point: enroll a new person with face images.
    @serialized_enrollment
    def enroll_person(
        self,
        *,
        full_name: str,
        person_type: str,
        is_active: bool,
        embedding_only_mode: bool,
        retention_expires_at: datetime | None,
        notes: str | dict[str, Any] | list[Any] | None,
        images: list[EnrollmentImagePayload],
        enrolled_by_user_id: str | None,
    ) -> EnrollmentResponse:
        
        # Input validation
        normalized_name = full_name.strip()
        if not normalized_name:
            raise ValueError("full_name is required")
        if not images:
            raise ValueError("At least one image is required for enrollment")
        if not enrolled_by_user_id:
            raise ValueError("An authenticated user is required for enrollment")
        normalized_type = normalize_person_type(person_type)

        # Ensure all dependent services are ready
        self.start()

        # Phase 1: Extract embeddings from uploaded images
        extracted_embeddings, rejected_images = self.process_images_and_extract_embeddings(images)
        normalized_expiry = self._normalize_datetime(retention_expires_at)

        # Phase 2: Create the person profile in the database
        person = self.create_person_profile(
            full_name=normalized_name,
            person_type=normalized_type,
            is_active=is_active,
            embedding_only_mode=embedding_only_mode,
            retention_expires_at=normalized_expiry,
            notes=notes,
            enrolled_by_user_id=enrolled_by_user_id,
        )

        # Phase 3: Persist each extracted embedding linked to the new person
        references = {}
        for source, extracted in enumerate(extracted_embeddings):
            embedding = self.embedding_repository.save_embedding(
                str(person.id),
                self._normalize_embedding_for_storage(extracted.vector).tolist(),
                source_type="manual_upload",
                quality_score=extracted.quality_score,
                expires_at=normalized_expiry,
            )
            if not embedding_only_mode:
                references[source] = self._retain_reference_image(str(person.id), embedding.id, extracted)

        enhancement = self._enhance(str(person.id), extracted_embeddings, references,
                                    retain=not embedding_only_mode, expires_at=normalized_expiry)

        # Phase 4: Rebuild FAISS index to include newly added embeddings
        self.index_embeddings_faiss()

        # Phase 5: Notify video service to refresh recognition pipeline
        if self.video_service is not None:
            refresh_version = self.video_service.request_recognition_refresh(
                reason=f"enrollment:{person.id}"
            )
            self.logger.info(
                "Requested recognition refresh version=%s after enrolling %s",
                refresh_version,
                person.full_name,
            )
        self.logger.info(
            "Enrolled person %s with %s embeddings (%s rejected)",
            person.full_name,
            len(extracted_embeddings),
            len(rejected_images),
        )
        return EnrollmentResponse(
            person_id=str(person.id),
            full_name=person.full_name,
            embeddings_count=len(extracted_embeddings) + enhancement.get("embeddings_added", 0),
            enhancement=enhancement,
            status="success",
            rejected_images=rejected_images,
        )

    @staticmethod
    # Helper method to Decode raw binary image data into a BGR NumPy array.
    def _decode_upload(contents: bytes) -> np.ndarray:
        
        data = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image")
        return frame

# Detect exactly one high-quality face before alignment and embedding extraction.
    def _detect_primary_face(self, frame: np.ndarray) -> tuple[np.ndarray, list[int], float | None, bool]:

        det_result = self.inference_service.detect_faces(frame)
        if det_result.bboxes.shape[0] == 0:
            raise ValueError(_NO_FACE_DETECTED_REASON)
        if det_result.bboxes.shape[0] != 1:
            raise ValueError(_MULTIPLE_FACES_REASON)

        best_idx = 0
        bbox = det_result.bboxes[best_idx, :4].astype(np.int32).tolist()
        score = float(det_result.bboxes[best_idx, 4])
        face_width = max(bbox[2] - bbox[0], 0)
        face_height = max(bbox[3] - bbox[1], 0)
        if min(face_width, face_height) < self.settings.enrollment_min_face_size:
            raise ValueError(f"Face is too small (minimum {self.settings.enrollment_min_face_size}px)")
        if score < self.settings.recognition_min_det_score:
            raise ValueError("Face detection confidence is too low")
        face = frame[max(bbox[1], 0):min(bbox[3], frame.shape[0]), max(bbox[0], 0):min(bbox[2], frame.shape[1])]
        if face.size == 0:
            raise ValueError(_NO_FACE_DETECTED_REASON)
        kps = det_result.kpss[best_idx]

        # Try keypoint-based alignment first (no re-detection needed)
        aligned = self.face_alignment_service.align_from_keypoints(frame, kps)
        if aligned is not None:
            return aligned, bbox, score, True

        # Fallback: landmark-based alignment via re-detection
        aligned = self.face_alignment_service.align(frame, bbox)
        if aligned is not None:
            return aligned, bbox, score, True

        # Alignment failed -- check if it is mandatory
        if self.settings.face_alignment_enabled:
            self.logger.warning("Alignment failed for enrollment bbox=%s", bbox)
            raise ValueError(_LANDMARK_ALIGNMENT_FAILED_REASON)

        # Fall back to margin-based square crop (alignment not required)
        crop, crop_bbox = self.adaface_service.crop_face(frame, bbox)
        if crop.size == 0:
            raise ValueError("Detected face crop is empty")
        return crop, crop_bbox, score, False

    @staticmethod
    # Helper method to Ensure a datetime value is timezone-aware in UTC.
    def _normalize_datetime(value: datetime | None) -> datetime | None:
       
        if value is None:
            return None
        # Attach UTC timezone if the datetime is naive
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        # Convert offset-aware datetimes to UTC
        return value.astimezone(timezone.utc)

    @staticmethod
    # Helper method to Normalize notes input to a string suitable for database storage.
    def _normalize_notes(notes: str | dict[str, Any] | list[Any] | None) -> str | None:
        
        if notes is None:
            return None
        if isinstance(notes, str):
            value = notes.strip()
            if not value:
                return None
            # Attempt to parse as JSON to validate and re-serialise
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                # Not JSON -- return the plain string as-is
                return value
            return json.dumps(parsed)
        # dict or list: serialise to JSON
        return json.dumps(notes)

    @staticmethod
    # Helper method to Convert an embedding from various formats to a validated float32 NumPy array.
    def _normalize_embedding_for_storage(embedding) -> np.ndarray:
        
        # Guard against accidentally receiving serialised string data
        if isinstance(embedding, (str, bytes, bytearray)):
            raise ValueError("Embedding extraction returned invalid type: string")

        # Handle PyTorch tensors (duck-typed to avoid importing torch)
        if hasattr(embedding, "detach") and hasattr(embedding, "cpu") and hasattr(embedding, "numpy"):
            candidate = embedding.detach().cpu().numpy()
        # Handle NumPy arrays and similar array-like objects
        elif hasattr(embedding, "tolist") and not isinstance(embedding, (list, tuple)):
            candidate = embedding.tolist()
        else:
            candidate = embedding

        # Validate dimensions and values, then convert to float32
        return np.asarray(validate_embedding_vector(candidate), dtype=np.float32)

    @staticmethod
    # Helper method to Check if a candidate embedding is a near-duplicate of any already-extracted embedding in this batch.
    def _is_duplicate_embedding(
        candidate: np.ndarray,
        extracted: list[ExtractedEmbedding],
        existing_vectors: list[np.ndarray],
    ) -> bool:
        
        vector = candidate.astype(np.float32)
        for item in extracted:
            # Dot product of unit vectors equals cosine similarity
            similarity = float(np.dot(vector, item.vector.astype(np.float32)))
            if similarity >= _DUPLICATE_EMBEDDING_SIMILARITY:
                return True
        for item in existing_vectors:
            similarity = float(np.dot(vector, item.astype(np.float32)))
            if similarity >= _DUPLICATE_EMBEDDING_SIMILARITY:
                return True
        return False

    def _enhance(self, person_id, originals, references, *, retain, expires_at, existing=()):
        if self.enhancement_service is None or not self.settings.enrollment_enhancement_enabled:
            return {"status": "disabled", "embeddings_added": 0, "generated_images": 0}
        try:
            variants, warnings = self.enhancement_service.prepare(originals, existing)
            count = self.enhancement_service.persist(person_id, variants, references,
                retain=retain, expires_at=expires_at)
            return dict(status="partial" if warnings else "complete", embeddings_added=count,
                        generated_images=len(variants), retained_images=len(variants) if retain else 0,
                        warnings=warnings, recipe="enrollment-v1")
        except Exception:
            self.logger.exception("Enhancement failed; original enrollment retained")
            return dict(status="failed", embeddings_added=0, generated_images=0,
                        warnings=["Enhancement could not be completed. Original images remain enrolled; check backend logs."])

    @serialized_enrollment
    def generate_reference_images(self, person_id: str):
        """Generate missing descendants for retained originals without duplicating uploads."""
        person = self.person_repository.get_person_by_id(person_id)
        if person is None:
            raise ValueError("POI profile not found")
        rows = self.reference_image_repository.list_for_person(person_id)
        completed = {row.original_reference_id for row in rows if row.image_kind != "original"}
        pending = [row for row in rows if row.image_kind == "original" and row.id not in completed]
        if not pending:
            return dict(status="nothing_to_generate", embeddings_added=0, generated_images=0, warnings=[])
        self.start()
        existing = self.embedding_repository.list_for_person(person_id)
        vectors = {str(item.id): item for item in existing}
        originals, references, rejected = [], {}, []
        for row in pending:
            try:
                relative = self.reference_image_repository.storage_path_for(person_id, row.id)
                path = self.reference_image_service.resolve(relative) if relative else None
                if path is None or row.embedding_id not in vectors:
                    raise ValueError("Stored source is unavailable")
                frame = self._decode_upload(path.read_bytes())
                _, bbox, score = self._assess_face(frame)
                if score < self.settings.enrollment_quality_threshold:
                    raise ValueError("Stored source is below the registration quality threshold")
                references[len(originals)] = row
                originals.append(ExtractedEmbedding(row.original_filename, bbox,
                    self._normalize_embedding_for_storage(vectors[row.embedding_id].embedding_vector), score, frame))
            except ValueError as exc:
                rejected.append(f"{row.original_filename}: {exc}")
        if not originals:
            raise self._build_failed_enrollment_error(rejected)
        result = self._enhance(person_id, originals, references, retain=True,
                               expires_at=person.retention_expires_at, existing=existing)
        result.setdefault("warnings", []).extend(rejected)
        self.index_embeddings_faiss()
        if self.video_service is not None:
            self.video_service.request_recognition_refresh(reason=f"generated-references:{person_id}")
        return result

    def _retain_reference_image(self, person_id: str, embedding_id, item: ExtractedEmbedding):
        if self.reference_image_repository is None or self.reference_image_service is None:
            return
        try:
            stored = self.reference_image_service.save(item.frame, person_id)
            return self.reference_image_repository.add(
                person_id=person_id,
                embedding_id=str(embedding_id),
                storage_path=stored.path,
                original_filename=item.filename,
                width=stored.width,
                height=stored.height,
            )
        except Exception:
            self.logger.exception("Embedding saved but reference image could not be retained: %s", item.filename)

    @staticmethod
    # Helper method to Extract the reason portion from a "filename: reason" rejection string.
    def _extract_rejection_reason(rejected_image: str) -> str:
        
        if ": " not in rejected_image:
            return rejected_image
        return rejected_image.split(": ", 1)[1]

    @classmethod
# Helper method to Construct a user-friendly error when all enrollment images fail, analysing the rejection reasons to produce specific error codes and messages.
    def _build_failed_enrollment_error(
        cls,
        rejected_images: list[str],
    ) -> EnrollmentValidationError:
        
        reasons = [cls._extract_rejection_reason(item) for item in rejected_images]
        # Determine singular vs plural label for the error message
        reference_label = "the reference image" if len(rejected_images) == 1 else "the uploaded reference images"

        # All images failed because no face was detected
        if reasons and all(reason == _NO_FACE_DETECTED_REASON for reason in reasons):
            return EnrollmentValidationError(
                f"Profile was not created because no face was detected in {reference_label}.",
                code="face_not_detected",
                rejected_images=rejected_images,
            )

        # All images failed because landmark alignment could not be performed
        if reasons and all(reason == _LANDMARK_ALIGNMENT_FAILED_REASON for reason in reasons):
            return EnrollmentValidationError(
                f"Profile was not created because facial landmarks could not be extracted from {reference_label}.",
                code="landmark_extraction_failed",
                rejected_images=rejected_images,
            )

        # Mixed failure reasons -- use a generic error message
        return EnrollmentValidationError(
            "Profile was not created because none of the uploaded reference images produced a usable face embedding.",
            code="profile_creation_failed",
            rejected_images=rejected_images,
        )
