"""Bounded, owner-scoped enrollment jobs for the single-process backend."""
from concurrent.futures import ThreadPoolExecutor
import copy
import logging
import threading
import time

from app.services.enrollment_service import EnrollmentValidationError


class EnrollmentJobService:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='enrollment')

    def submit(self, job_id, owner, operation, **kwargs):
        with self._lock:
            now = time.time()
            self._jobs = {key: job for key, job in self._jobs.items()
                          if job['finished_at'] is None or now - job['finished_at'] < 3600}
            existing = self._jobs.get(job_id)
            if existing:
                if existing['owner'] != owner:
                    raise ValueError('Job ID is unavailable')
                return self._public(existing)
            pending = [job for job in self._jobs.values() if job['finished_at'] is None]
            if any(job['owner'] == owner for job in pending):
                raise ValueError('An enrollment is already in progress for this operator')
            if len(pending) >= 8 or len(self._jobs) >= 256:
                raise ValueError('Enrollment queue is full; retry later')
            job = dict(id=job_id, owner=owner, status='queued', created_at=now,
                       finished_at=None, result=None, error=None)
            self._jobs[job_id] = job
            self._worker.submit(self._run, job_id, operation, kwargs)
            return self._public(job)

    def _run(self, job_id, operation, kwargs):
        with self._lock:
            self._jobs[job_id]['status'] = 'running'
        result, error = None, None
        try:
            result = operation(**kwargs).model_dump(mode='json')
        except EnrollmentValidationError as exc:
            error = dict(code=exc.code, message=str(exc), rejected_images=exc.rejected_images)
        except ValueError as exc:
            error = dict(code='validation_failed', message=str(exc))
        except Exception:
            logging.getLogger('sentinel.enrollment.jobs').exception('Enrollment job failed: %s', job_id)
            error = dict(code='enrollment_interrupted', message=
                         'Enrollment could not finish. Check the person list before retrying; some data may have been saved.')
        with self._lock:
            self._jobs[job_id].update(status='failed' if error else 'completed',
                                      result=result, error=error, finished_at=time.time())

    @staticmethod
    def _public(job):
        value = {key: item for key, item in job.items() if key != 'owner'}
        value['elapsed_seconds'] = round((job['finished_at'] or time.time()) - job['created_at'], 1)
        return copy.deepcopy(value)

    def get(self, job_id, owner):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job['owner'] != owner:
                return None
            return self._public(job)

    def stop(self):
        self._worker.shutdown(wait=True)
