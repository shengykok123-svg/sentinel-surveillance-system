"""
Sentinel ISS backend application package.

Exposes ``create_app()`` as the single public entry point so that
``main.py`` (uvicorn target) and test fixtures can import it uniformly.
"""

import os

# Three camera workers already run concurrently. Small tracker matrices and
# CPU tensor preparation should not each recruit a full native thread pool.
# Apply defaults before NumPy/SciPy/PyTorch import; respect explicit overrides.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from .main import create_app

__all__ = ["create_app"]
