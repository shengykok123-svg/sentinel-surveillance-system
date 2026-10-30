"""
Database module – SQLAlchemy engine, session factory, and schema validation.

Provides ``DatabaseManager``, the single point of contact for all database
operations.  It creates the engine from the configured ``database_url``,
exposes a context-managed ``session()`` helper with automatic
commit / rollback, and validates that PostgreSQL has all required tables
on startup (SQLite auto-creates them instead).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import Settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base that all ORM entities inherit from."""
    pass


class DatabaseManager:
    """
    Manages the SQLAlchemy engine and session lifecycle.

    Parameters
    ----------
    settings : Settings
        Application settings containing ``database_url`` and ``database_echo``.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # SQLite requires check_same_thread=False for multi-threaded access.
        connect_args: dict[str, object] = {}
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        # Create the SQLAlchemy engine with connection pooling.
        # pool_pre_ping ensures stale connections are recycled.
        self.engine: Engine = create_engine(
            settings.database_url,
            echo=settings.database_echo,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        # Session factory – autoflush and autocommit are disabled so that
        # commits are explicit inside the ``session()`` context manager.
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
        )

    def init(self) -> None:
        """
        Initialise the database.

        - **SQLite**: automatically creates all tables from ORM metadata.
        - **PostgreSQL**: validates that every required table already exists;
          raises ``RuntimeError`` if any are missing (schema is managed externally).
        """
        from app.models import entities  # noqa: F401  — import to register models

        if self.settings.database_url.startswith("sqlite"):
            Base.metadata.create_all(self.engine)
            self._migrate_sqlite_event_snapshot_columns()
            self._migrate_reference_generation()
            return

        # For Postgres, verify the schema was migrated correctly.
        inspector = inspect(self.engine)
        existing = set(inspector.get_table_names())
        required = {
            "cameras",
            "detection_events",
            "enrolled_persons",
            "face_embeddings",
            "roles",
            "track_detection_events",
            "tracks",
            "user_credentials",
            "user_roles",
            "users",
        }
        missing = required - existing
        if missing:
            raise RuntimeError(
                "The configured PostgreSQL schema is missing required tables: "
                + ", ".join(sorted(missing))
            )
        self._migrate_postgres_event_snapshot_columns()
        self._migrate_postgres_quality_scores()
        self._migrate_postgres_embedding_sources()
        entities.FaceReferenceImageEntity.__table__.create(self.engine, checkfirst=True)
        entities.ReferenceImageAccessLogEntity.__table__.create(self.engine, checkfirst=True)
        entities.GeneratedReferenceImageEntity.__table__.create(self.engine, checkfirst=True)
        self._migrate_reference_generation()

    def _migrate_reference_generation(self) -> None:
        columns = {c["name"] for c in inspect(self.engine).get_columns("reference_image_access_logs")}
        if "generated_image_id" not in columns:
            with self.engine.begin() as connection:
                connection.execute(text("ALTER TABLE reference_image_access_logs ADD COLUMN generated_image_id VARCHAR(36)"))

    def _migrate_postgres_embedding_sources(self) -> None:
        """Extend the existing allowed origins without discarding legacy values."""
        checks = {c['name']: c['sqltext'] for c in inspect(self.engine).get_check_constraints('face_embeddings')}
        old = checks.get('face_embeddings_source_type_check')
        sources = ('reference_upload', 'generated_view', 'environment_lowlight_moderate',
                   'environment_lowlight_severe', 'environment_lowres_moderate', 'environment_lowres_severe')
        if not old or all(f"'{source}'" in old for source in sources):
            return
        allowed = ', '.join(f"'{source}'" for source in sources)
        with self.engine.begin() as connection:
            connection.execute(text('ALTER TABLE face_embeddings DROP CONSTRAINT face_embeddings_source_type_check'))
            connection.execute(text(f'ALTER TABLE face_embeddings ADD CONSTRAINT face_embeddings_source_type_check CHECK (({old}) OR source_type IN ({allowed}))'))

    def _migrate_sqlite_event_snapshot_columns(self) -> None:
        """Add record-image columns for local databases created before this feature."""
        required_columns = {
            "snapshot_path": "VARCHAR(512)",
            "snapshot_width": "INTEGER",
            "snapshot_height": "INTEGER",
        }
        with self.engine.begin() as connection:
            existing = {row[1] for row in connection.execute(text("PRAGMA table_info(detection_events)"))}
            for name, definition in required_columns.items():
                if name not in existing:
                    connection.execute(text(f"ALTER TABLE detection_events ADD COLUMN {name} {definition}"))

    def _migrate_postgres_quality_scores(self) -> None:
        """Legacy probability bounds cannot store raw CR-FIQA scores."""
        checks = {item['name']: item['sqltext'] for item in inspect(self.engine).get_check_constraints('face_embeddings')}
        legacy = checks.get('face_embeddings_quality_score_check', '')
        if 'quality_score <= 1.0' not in legacy or 'quality_score >= 0.0' not in legacy:
            return
        with self.engine.begin() as connection:
            connection.execute(text('ALTER TABLE face_embeddings DROP CONSTRAINT face_embeddings_quality_score_check'))
            connection.execute(text("ALTER TABLE face_embeddings ADD CONSTRAINT face_embeddings_quality_score_check CHECK (quality_score > '-Infinity'::double precision AND quality_score < 'Infinity'::double precision)"))

    def _migrate_postgres_event_snapshot_columns(self) -> None:
        """Apply the additive record-image schema update to existing deployments."""
        statements = (
            "ALTER TABLE detection_events ADD COLUMN IF NOT EXISTS snapshot_path VARCHAR(512)",
            "ALTER TABLE detection_events ADD COLUMN IF NOT EXISTS snapshot_width INTEGER",
            "ALTER TABLE detection_events ADD COLUMN IF NOT EXISTS snapshot_height INTEGER",
        )
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """
        Provide a transactional database session via a context manager.

        Auto-commits on clean exit, rolls back on exception, and always
        closes the session afterwards.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Shut down the connection pool and release all resources."""
        self.engine.dispose()
