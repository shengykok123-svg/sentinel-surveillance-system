"""Opt-in, bounded evidence capture. Disk I/O never blocks camera/recognition workers."""
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


class RecognitionDiagnostics:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path(settings.recognition_record_snapshot_dir).parent / 'diagnostics'
        self.toggle = self.root / 'ENABLED'
        self.session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:6]
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='recognition-evidence')
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(4)
        self.count = 0
        self.last = {}

    def capture(self, frame, bbox, kps, metadata, aligned=None):
        if not self.toggle.is_file():
            return
        key = (metadata.get('camera_id'), metadata.get('track_id'), metadata['stage'])
        with self.lock:
            now = time.monotonic()
            if self.count >= 120 or now-self.last.get(key, -100) < 2 or not self.slots.acquire(False):
                return
            self.count += 1
            number = self.count
            self.last[key] = now
        try:
            self.pool.submit(self._write, number, frame, list(bbox), kps, dict(metadata), aligned)
        except RuntimeError:
            self.slots.release()

    def _write(self, number, frame, bbox, kps, metadata, aligned):
        try:
            path = self.root / self.session / f'{number:04d}'
            path.mkdir(parents=True, exist_ok=False)
            x1,y1,x2,y2 = bbox
            crop = frame[max(y1,0):max(y2,0),max(x1,0):max(x2,0)].copy()
            if crop.size:
                gray=cv2.cvtColor(cv2.resize(crop,(112,112)),cv2.COLOR_BGR2GRAY)
                metadata.update(face_width=x2-x1,face_height=y2-y1,
                                required_min_side=self.settings.recognition_min_face_size,
                                raw_blur_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                                blur_gate_enabled=False)
                cv2.imwrite(str(path/'face.jpg'), crop)
                if kps is not None:
                    for x,y in np.asarray(kps).reshape(-1,2):
                        if np.isfinite([x,y]).all():
                            cv2.circle(crop,(int(x-x1),int(y-y1)),2,(0,255,0),-1)
                cv2.imwrite(str(path/'landmarks.jpg'), crop)
            if aligned is not None:
                cv2.imwrite(str(path/'aligned.png'), aligned)
            metadata.update(timestamp=datetime.now(timezone.utc).isoformat(),bbox=bbox,
                            frame_shape=list(frame.shape),quality_threshold=self.settings.live_quality_threshold,
                            match_threshold=self.settings.recognition_threshold,
                            identity_margin=self.settings.recognition_identity_margin)
            (path/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        except Exception:
            logging.getLogger('sentinel.diagnostics').exception('Evidence capture failed')
        finally:
            self.slots.release()

    def stop(self):
        self.pool.shutdown(wait=True)
