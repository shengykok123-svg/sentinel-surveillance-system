"""Bounded enrollment augmentation for the local demo/consenting participants.

Pose templates use the evaluated 75/25 original anchor. Environment templates
remain independent, the arm with the best results in all four stress tests.
Generation failures leave original enrollment usable and are reported explicitly.
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import subprocess
import tempfile
import uuid

import cv2
import numpy as np
import psutil

from app.models.entities import FaceEmbeddingEntity, GeneratedReferenceImageEntity, PersonEntity, validate_embedding_vector
from app.services.environment_augmentation import simulate_environment, stable_seed
from app.services.multiview_enrollment_service import anchored_templates

CONDITIONS = {
    "lowlight_moderate": dict(exposure=.45, noise_sigma=2.),
    "lowlight_severe": dict(exposure=.20, noise_sigma=4.),
    "lowres_moderate": dict(face_width=32),
    "lowres_severe": dict(face_width=20),
}


@dataclass
class Variant:
    source: int
    frame: np.ndarray
    kind: str
    generation: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    vector: np.ndarray | None = None


class EnrollmentEnhancementService:
    def __init__(self, enrollment):
        self.enrollment = enrollment
        self.settings = enrollment.settings
        self.logger = logging.getLogger("sentinel.enhancement")

    def _views(self, originals):
        python = Path(self.settings.liveportrait_python)
        repo = Path(self.settings.liveportrait_repository)
        if not python.is_file() or not (repo / "src/live_portrait_wrapper.py").is_file():
            raise RuntimeError("LivePortrait runtime or model repository is unavailable")
        with tempfile.TemporaryDirectory(prefix="sentinel-views-") as directory:
            folder = Path(directory)
            inputs = []
            for i, item in enumerate(originals):
                det = self.enrollment.inference_service.detect_faces(item.frame)
                if len(det.bboxes) != 1:
                    raise ValueError("Source face could not be located for view generation")
                name = f"source_{i}.png"
                if not cv2.imwrite(str(folder / name), item.frame):
                    raise RuntimeError("Unable to stage source image")
                inputs.append(dict(source=i, file=name, landmarks=det.kpss[0].tolist()))
            (folder / "inputs.json").write_text(json.dumps(inputs))
            process = subprocess.Popen(
                [str(python), str(Path(__file__).with_name("enrollment_view_worker.py")), str(folder), str(repo)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                _, stderr = process.communicate(timeout=self.settings.enrollment_generation_timeout)
            except subprocess.TimeoutExpired:
                # Windows venv launchers spawn another Python process. Stop the
                # entire worker tree so it cannot keep using the GPU/temp files.
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.communicate()
                psutil.wait_procs(children, timeout=5)
                raise RuntimeError("Angle generation timed out") from None
            if process.returncode:
                self.logger.error("View worker failed: %s", stderr[-2000:])
                raise RuntimeError("LivePortrait generation failed; check backend logs")
            results = []
            for row in json.loads((folder / "outputs.json").read_text()):
                frame = cv2.imread(str(folder / row["file"]))
                if frame is None:
                    raise RuntimeError("Generated view is unreadable")
                results.append((row, frame))
            if len(results) != len(originals) * 7:
                raise RuntimeError("View generation returned an incomplete batch")
            return results

    def _assess(self, variant, original):
        e = self.enrollment
        try:
            if variant.kind == "view":
                crop, _, quality = e._assess_face(variant.frame)
                variant.generation["quality_score"] = quality
                if quality < self.settings.enrollment_quality_threshold:
                    raise ValueError("Below registration quality threshold")
            else:
                det = e.inference_service.detect_faces(variant.frame)
                if len(det.bboxes) != 1:
                    raise ValueError("Exactly one detected face is required")
                if float(det.bboxes[0, 4]) < self.settings.conf_thresh:
                    raise ValueError("Detection confidence is too low")
                crop = e.face_alignment_service.align_from_keypoints(variant.frame, det.kpss[0])
                if crop is None:
                    raise ValueError("Face alignment failed")
            vector, _ = e.adaface_service.extract_embedding_aligned(crop)
            vector = np.asarray(validate_embedding_vector(vector), np.float32)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm < 1e-6:
                raise ValueError("Invalid embedding norm")
            vector /= norm
            cosine = float(vector @ original.vector)
            variant.generation["original_cosine"] = cosine
            if cosine < .5:
                raise ValueError("Identity consistency below 0.50")
            variant.vector = vector
            variant.generation["status"] = "eligible"
        except ValueError as exc:
            variant.generation.update(status="rejected", reason=str(exc))

    def prepare(self, originals, existing=()):
        variants, warnings = [], []
        try:
            for info, frame in self._views(originals):
                info = dict(info)
                source = info.pop("source")
                info.pop("file", None)
                view = Variant(source, frame, "view", dict(info, recipe="enrollment-v1"))
                self._assess(view, originals[source])
                variants.append(view)
        except Exception as exc:
            self.logger.exception("View enhancement unavailable")
            warnings.append(f"Angle generation unavailable: {exc}")

        # Generate previews at every angle, even when the parent is ineligible.
        # Ineligible parents cannot contribute recognition templates.
        sources = [(i, item.frame, None, True, item.bbox) for i, item in enumerate(originals)]
        for view in list(variants):
            det = self.enrollment.inference_service.detect_faces(view.frame)
            bbox = det.bboxes[0, :4] if len(det.bboxes) == 1 else [0, 0, view.frame.shape[1], view.frame.shape[0]]
            eligible = view.vector is not None and view.generation["target_yaw"] != 0
            sources.append((view.source, view.frame, view, eligible, bbox))
        for source, frame, parent, eligible, bbox in sources:
            for condition, parameters in CONDITIONS.items():
                params = dict(parameters)
                target = params.pop("face_width", None)
                if target is not None:
                    params["scale"] = min(1., target / max(float(bbox[2] - bbox[0]), 1.))
                yaw = parent.generation["target_yaw"] if parent else 0
                seed = stable_seed(f"enrollment-v1|{source}|{originals[source].filename}|{yaw}|{condition}")
                image = simulate_environment(frame, seed=seed, **params)
                variant = Variant(source, image, "environment", dict(
                    condition=condition, parameters=params, seed=seed, target_yaw=yaw,
                    recipe="enrollment-v1", status="rejected", reason="Parent view is not eligible",
                ), parent_id=parent.id if parent else None)
                if eligible:
                    self._assess(variant, originals[source])
                variants.append(variant)

        used = Counter(entity.source_type for entity in existing)
        priorities = {-30: 0, 30: 1, 0: 2, -15: 3, 15: 4, -45: 5, 45: 6}
        for condition in CONDITIONS:
            candidates = sorted((v for v in variants if v.kind == "environment"
                                 and v.generation["condition"] == condition and v.vector is not None),
                                key=lambda v: (priorities[int(v.generation["target_yaw"])], v.source))
            remaining = max(0, 4 - used[f"environment_{condition}"])
            for i, variant in enumerate(candidates):
                if i >= remaining:
                    variant.vector = None
                    variant.generation.update(status="not_selected", reason="Per-person condition budget reached")
        for variant in variants:
            if variant.kind == "view" and variant.vector is not None:
                if variant.generation["target_yaw"] == 0:
                    variant.vector = None
                    variant.generation.update(status="not_selected", reason="Zero-yaw preview; original template retained")
                else:
                    variant.vector = anchored_templates(originals[variant.source].vector, [variant.vector])[-1]
                    variant.generation["template_mode"] = "original_anchor_75_25"
            elif variant.vector is not None:
                variant.generation["template_mode"] = "independent"
            if variant.vector is not None:
                variant.generation.update(status="selected", reason=None)
        return variants, warnings

    def persist(self, person_id, variants, references, *, retain, expires_at):
        """Commit generated metadata and vectors together; clean staged files on rollback."""
        e = self.enrollment
        paths, count = [], 0
        try:
            with e.embedding_repository.database.session() as session:
                person = session.get(PersonEntity, uuid.UUID(person_id))
                if person is None or person.deleted_at is not None:
                    raise ValueError("POI was deleted during generation")
                for variant in variants:
                    embedding_id = None
                    if variant.vector is not None:
                        embedding_id = uuid.uuid4()
                        source_type = "generated_view" if variant.kind == "view" else f'environment_{variant.generation["condition"]}'
                        session.add(FaceEmbeddingEntity(id=embedding_id, person_id=person.id,
                            embedding_vector=variant.vector.tolist(), source_type=source_type,
                            quality_score=variant.generation.get("quality_score"), is_primary=False, expires_at=expires_at))
                        session.flush()
                        count += 1
                    if retain:
                        reference = references.get(variant.source)
                        if reference is None:
                            raise RuntimeError("Original reference could not be retained")
                        stored = e.reference_image_service.save(variant.frame, person_id)
                        paths.append(stored.path)
                        label = (f'Yaw {variant.generation["target_yaw"]:+g}°' if variant.kind == "view"
                                 else variant.generation["condition"].replace("_", " "))
                        session.add(GeneratedReferenceImageEntity(id=uuid.UUID(variant.id), person_id=person.id,
                            original_reference_id=uuid.UUID(reference.id),
                            parent_id=uuid.UUID(variant.parent_id) if variant.parent_id else None,
                            embedding_id=embedding_id, storage_path=stored.path, original_filename=label,
                            width=stored.width, height=stored.height, image_kind=variant.kind, generation=variant.generation))
                        session.flush()
            return count
        except Exception:
            for path in paths:
                e.reference_image_service.delete(path)
            raise
