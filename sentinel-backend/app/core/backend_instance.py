"""OS-owned lock preventing two live pipelines from sharing one FAISS index."""
import os
from pathlib import Path


class BackendInstanceLock:
    def __init__(self, index_path):
        self.path = Path(index_path).with_suffix('.runtime.lock')
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+b')
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError('Another Sentinel video backend is already using this index. '
                               'Stop the existing backend before starting another; use one Uvicorn worker.') from exc
        self.file = handle
        return self

    def __exit__(self, *args):
        if self.file is not None:
            # Closing (including on process exit) releases the OS lock. The
            # persistent empty marker is harmless and is never a stale lock.
            self.file.close()
            self.file = None
