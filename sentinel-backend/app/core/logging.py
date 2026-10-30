"""
Logging configuration for the Sentinel ISS backend.

Sets up Python's built-in ``logging`` with a uniform format including
timestamp, level, and logger name.  Called once during startup from
``app/main.py`` with the level specified in ``Settings.log_level``.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a standard format and the given level.

    Parameters
    ----------
    level : str
        Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).  Defaults to
        ``INFO`` if the string doesn't map to a known level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
