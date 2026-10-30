"""Bounded candidate selection and temporal identity decisions for live tracks."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CandidateWindow:
    started: float | None = None
    last_sample: float = float('-inf')
    candidates: list = field(default_factory=list)

    def offer(self, candidate, score: float, now: float, settings, *, immediate: bool = False) -> bool:
        if self.started is None:
            self.started = now
        if now - self.last_sample >= settings.recognition_sample_seconds:
            self.last_sample = now
            self.candidates.append((score, candidate))
            self.candidates.sort(key=lambda item: item[0], reverse=True)
            del self.candidates[3:]
        return bool(self.candidates) and (immediate or len(self.candidates) >= 2
                or now - self.started >= settings.recognition_collect_seconds)

    def take(self):
        items = [item for _, item in self.candidates]
        self.candidates.clear()
        self.started = None
        return items


def candidate_score(frame, bbox, kps, detection_score, settings):
    """Validate model inputs; leave blur, pose and exposure quality to CR-FIQA.

    Detection confidence ranks valid candidates before worker quality scoring.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    if (not np.isfinite(bbox).all() or x2 <= x1 or y2 <= y1
            or min(x2-x1, y2-y1) < settings.recognition_min_face_size
            or x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h
            or not np.isfinite(detection_score)
            or detection_score < settings.recognition_min_det_score):
        return None
    if kps is None or np.asarray(kps).shape != (5, 2) or not np.isfinite(kps).all():
        return None
    points = np.asarray(kps)
    if not ((points[:, 0] >= x1).all() and (points[:, 0] <= x2).all()
            and (points[:, 1] >= y1).all() and (points[:, 1] <= y2).all()):
        return None
    return float(detection_score)


@dataclass
class IdentityResolver:
    observations: deque = field(default_factory=lambda: deque(maxlen=3))
    last_observed: float = float('-inf')

    def observe(self, outcome, job, now, settings):
        """Return (phase, supporting job, match); never count one image twice."""
        self.observations = deque(
            (row for row in self.observations if now-row[0] <= settings.recognition_vote_seconds),
            maxlen=3,
        )
        stamp = job.captured_at
        if (not outcome.evaluated or now-stamp > settings.recognition_vote_seconds
                or stamp <= self.last_observed):
            return 'collecting', None, None
        self.last_observed = stamp
        self.observations.append((stamp, outcome, job))
        votes = {}
        for _, result, evidence in self.observations:
            if result.match is not None:
                votes.setdefault(result.match.person_id, []).append((result.match, evidence))
        for rows in votes.values():
            if len(rows) >= 2:
                # Evidence must support this identity and prefer the highest CR-FIQA score.
                match, evidence = max(rows, key=lambda row: row[1].quality_score)
                return 'recognized', evidence, match
        if len(votes) > 1 or any(getattr(row[1], 'ambiguous', False) for row in self.observations):
            return 'ambiguous', None, None
        if votes:
            return 'tentative', None, None
        if len(self.observations) >= 2:
            return 'unknown', None, None
        return 'unmatched', None, None
