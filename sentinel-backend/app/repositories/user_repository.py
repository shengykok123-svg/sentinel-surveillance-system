"""
User repository module for Sentinel.

Provides data-access operations for user accounts and authentication
data, including looking up users by identifier or ID, creating the
initial bootstrap administrator, and recording login outcomes.

User entities are never hard-deleted; a ``deleted_at`` timestamp marks
removal.  Role assignments are filtered to exclude expired entries
when building the ``StoredUser`` data-transfer object.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, or_, select
from sqlalchemy.orm import joinedload

from app.core.database import DatabaseManager
from app.models.entities import RoleEntity, UserCredentialEntity, UserEntity, UserRoleEntity


def utcnow() -> datetime:
    """Return the current UTC datetime as a timezone-aware object.

    Returns:
        A ``datetime`` representing "now" in UTC.
    """
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class StoredUser:
    """Read-only data-transfer object representing a persisted user.

    This dataclass is returned by all public ``UserRepository`` query
    methods.  It decouples callers from the SQLAlchemy ORM entity and
    includes the password hash and resolved role names needed for
    authentication and authorisation logic.

    Attributes:
        id: UUID of the user (as a string).
        username: Unique login name.
        email: Optional email address.
        is_active: Whether the account is enabled.
        is_locked: Whether the account is currently locked out.
        account_locked_reason: Human-readable reason for the lock,
            or ``None``.
        failed_login_count: Consecutive failed login attempts.
        locked_until: Datetime until which the account remains
            locked, or ``None``.
        last_login_at: Datetime of the most recent successful login,
            or ``None``.
        deleted_at: Soft-deletion timestamp, or ``None`` if active.
        password_hash: The stored password hash string, or ``None``
            if no credential record exists.
        roles: List of role name strings currently assigned to the
            user (expired role assignments are excluded).
    """
    id: str
    username: str
    email: str | None
    is_active: bool
    is_locked: bool
    account_locked_reason: str | None
    failed_login_count: int
    locked_until: datetime | None
    last_login_at: datetime | None
    deleted_at: datetime | None
    password_hash: str | None
    roles: list[str]


class UserRepository:
    """Repository handling persistence operations for user accounts.

    All public methods open their own database sessions via the
    injected ``DatabaseManager``.
    """

    def __init__(self, database: DatabaseManager) -> None:
        """Initialise the repository with a database manager.

        Args:
            database: The ``DatabaseManager`` instance used to obtain
                database sessions.
        """
        self.database = database

    def count_users(self) -> int:
        """Return the number of non-deleted user accounts.

        Returns:
            An integer count of active (non-soft-deleted) user records.
        """
        with self.database.session() as session:
            stmt = select(func.count()).select_from(UserEntity).where(UserEntity.deleted_at.is_(None))
            return int(session.scalar(stmt) or 0)

    def get_by_identifier(self, identifier: str) -> StoredUser | None:
        """Look up a user by username **or** email address.

        The comparison is case-insensitive.  Credential and role
        relationships are eagerly loaded to avoid lazy-loading outside
        the session.

        Args:
            identifier: A username or email address to search for.

        Returns:
            A ``StoredUser`` if a matching non-deleted account is
            found; otherwise ``None``.
        """
        # Normalise to lower-case for case-insensitive comparison
        normalized = identifier.strip().lower()
        with self.database.session() as session:
            stmt = (
                select(UserEntity)
                .options(
                    joinedload(UserEntity.credential),
                    joinedload(UserEntity.user_roles).joinedload(UserRoleEntity.role),
                )
                .where(
                    UserEntity.deleted_at.is_(None),
                    or_(
                        func.lower(UserEntity.username) == normalized,
                        # coalesce prevents NULL emails from breaking
                        # the comparison
                        func.lower(func.coalesce(UserEntity.email, "")) == normalized,
                    ),
                )
            )
            entity = session.execute(stmt).scalars().unique().first()
            return self._to_stored_user(entity)

    def get_by_id(self, user_id: str) -> StoredUser | None:
        """Fetch a user by their primary-key UUID.

        Credential and role relationships are eagerly loaded.

        Args:
            user_id: UUID string of the user to retrieve.

        Returns:
            A ``StoredUser`` if the user exists and has not been
            soft-deleted; otherwise ``None``.
        """
        with self.database.session() as session:
            stmt = (
                select(UserEntity)
                .options(
                    joinedload(UserEntity.credential),
                    joinedload(UserEntity.user_roles).joinedload(UserRoleEntity.role),
                )
                .where(UserEntity.id == uuid.UUID(str(user_id)), UserEntity.deleted_at.is_(None))
            )
            entity = session.execute(stmt).scalars().unique().first()
            return self._to_stored_user(entity)

    def create_bootstrap_admin(
        self,
        identifier: str,
        password_hash: str,
        email: str | None = None,
    ) -> StoredUser:
        """Create the very first administrator account (bootstrap only).

        This method is intended for first-time system setup.  It will
        **refuse** to create a user if any non-deleted users already
        exist, ensuring the bootstrap path cannot be re-used after
        initial provisioning.

        If *identifier* contains an ``@`` sign it is treated as an
        email address: the local part becomes the username and the full
        address is stored as the email.

        Args:
            identifier: Username or email address for the new admin.
            password_hash: Pre-hashed password string.
            email: Optional explicit email; if ``None`` and
                *identifier* looks like an email, it is used.

        Returns:
            A ``StoredUser`` representing the newly created admin.

        Raises:
            ValueError: If *identifier* is blank, if users already
                exist, or if a duplicate username/email is detected.
        """
        normalized_identifier = identifier.strip()
        if not normalized_identifier:
            raise ValueError("Identifier is required")

        # If the identifier looks like an email, extract the username
        # portion before the '@' sign.
        username = normalized_identifier if "@" not in normalized_identifier else normalized_identifier.split("@", 1)[0]
        resolved_email = email.strip() if email else None
        if "@" in normalized_identifier and resolved_email is None:
            # Use the full identifier as email when no explicit email
            # was provided and the identifier is an email address.
            resolved_email = normalized_identifier

        with self.database.session() as session:
            # Guard: bootstrap is only allowed when the user table is empty
            existing_users = session.scalar(
                select(func.count()).select_from(UserEntity).where(UserEntity.deleted_at.is_(None))
            )
            if existing_users:
                raise ValueError("Bootstrap admin is only available when no users exist")

            # Check for duplicate username or email (even among
            # soft-deleted records) to prevent conflicts.
            duplicate_checks = [func.lower(UserEntity.username) == username.lower()]
            if resolved_email:
                duplicate_checks.append(func.lower(func.coalesce(UserEntity.email, "")) == resolved_email.lower())
            duplicate_stmt = select(UserEntity).where(or_(*duplicate_checks))
            if session.scalar(duplicate_stmt) is not None:
                raise ValueError("A user with that identifier already exists")

            # Create the user entity
            user = UserEntity(
                username=username,
                email=resolved_email,
                is_active=True,
                is_locked=False,
                failed_login_count=0,
            )
            session.add(user)
            session.flush()

            # Create the credential record with the hashed password
            credential = UserCredentialEntity(
                user_id=user.id,
                password_hash=password_hash,
                hash_algorithm="pbkdf2_sha256",
            )
            session.add(credential)

            # Assign the highest-priority available role to the admin
            role = self._resolve_bootstrap_role(session)
            if role is not None:
                session.add(
                    UserRoleEntity(
                        user_id=user.id,
                        role_id=role.id,
                        # The admin assigns the role to themselves
                        assigned_by=user.id,
                    )
                )

            session.flush()
            session.refresh(user)
            roles = [role.name] if role is not None else []
            return StoredUser(
                id=str(user.id),
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                is_locked=user.is_locked,
                account_locked_reason=user.account_locked_reason,
                failed_login_count=user.failed_login_count,
                locked_until=user.locked_until,
                last_login_at=user.last_login_at,
                deleted_at=user.deleted_at,
                password_hash=password_hash,
                roles=roles,
            )

    def change_password(self, user_id: str, password_hash: str) -> None:
        with self.database.session() as session:
            credential = session.scalar(select(UserCredentialEntity).where(
                UserCredentialEntity.user_id == uuid.UUID(str(user_id))))
            if credential is None:
                raise ValueError("Account credential no longer exists")
            credential.password_hash = password_hash
            credential.hash_algorithm = "pbkdf2_sha256"
            credential.password_changed_at = utcnow()
            credential.must_change_password = False

    def record_successful_login(self, user_id: str) -> None:
        """Update user state after a successful authentication.

        Resets the failed-login counter, clears any account lock, and
        records the current UTC time as the last login timestamp.

        Args:
            user_id: UUID string of the user who logged in.
        """
        with self.database.session() as session:
            user = session.get(UserEntity, uuid.UUID(str(user_id)))
            if user is None:
                return
            user.last_login_at = utcnow()
            user.failed_login_count = 0
            # Clear any existing lockout state
            user.is_locked = False
            user.locked_until = None
            user.account_locked_reason = None
            session.add(user)

    def record_failed_login(self, user_id: str) -> None:
        """Increment the failed-login counter after an unsuccessful attempt.

        The counter is incremented by one.  Upstream business logic
        (not in this repository) is responsible for deciding when the
        count warrants locking the account.

        Args:
            user_id: UUID string of the user whose login failed.
        """
        with self.database.session() as session:
            user = session.get(UserEntity, uuid.UUID(str(user_id)))
            if user is None:
                return
            user.failed_login_count = int(user.failed_login_count or 0) + 1
            session.add(user)

    @staticmethod
    def _resolve_bootstrap_role(session) -> RoleEntity | None:
        """Select the most appropriate role to assign to the bootstrap admin.

        Roles are checked in preference order: ``admin`` > ``operator``
        > ``viewer``.  If none of those exist, the first available role
        in the database is used as a fallback.  If no roles are defined
        at all, ``None`` is returned.

        Args:
            session: An active SQLAlchemy session.

        Returns:
            The chosen ``RoleEntity``, or ``None`` if the roles table
            is empty.
        """
        preferred = ("admin", "operator", "viewer")
        roles = session.scalars(select(RoleEntity)).all()
        # Index all roles by lower-cased name for quick lookup
        by_name = {role.name.lower(): role for role in roles}
        for name in preferred:
            if name in by_name:
                return by_name[name]
        # Fallback: return the first role found, or None
        return roles[0] if roles else None

    @staticmethod
    def _to_stored_user(entity: UserEntity | None) -> StoredUser | None:
        """Convert an ORM ``UserEntity`` into a ``StoredUser`` dataclass.

        Expired role assignments (those with a non-null ``expires_at``)
        are filtered out so that only currently valid roles appear in
        the returned object.

        Args:
            entity: The ORM entity to convert, or ``None``.

        Returns:
            A ``StoredUser`` instance, or ``None`` if *entity* is
            ``None``.
        """
        if entity is None:
            return None
        # Include only roles that have no expiry set (i.e. still valid)
        roles = [
            user_role.role.name
            for user_role in entity.user_roles
            if user_role.role is not None and user_role.expires_at is None
        ]
        password_hash = entity.credential.password_hash if entity.credential is not None else None
        return StoredUser(
            id=str(entity.id),
            username=entity.username,
            email=entity.email,
            is_active=entity.is_active,
            is_locked=entity.is_locked,
            account_locked_reason=entity.account_locked_reason,
            failed_login_count=entity.failed_login_count,
            locked_until=entity.locked_until,
            last_login_at=entity.last_login_at,
            deleted_at=entity.deleted_at,
            password_hash=password_hash,
            roles=roles,
        )
