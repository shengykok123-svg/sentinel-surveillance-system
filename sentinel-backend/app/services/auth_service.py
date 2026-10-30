"""
Authentication service providing business logic for login, bootstrap, and
token verification.

This module implements the core authentication flows for the Sentinel
application:
    - **Bootstrap**: First-time admin account creation on a fresh install.
    - **Login**: Credential validation with account lockout handling, issuing JWTs.
    - **Token verification**: Decoding JWTs and returning authenticated user context.

Custom exception hierarchy:
    - ``AuthError``: Base authentication error.
    - ``BootstrapRequiredError``: No users exist; the client must bootstrap first.
    - ``BootstrapUnavailableError``: Bootstrap is disabled because users already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.core.config import Settings
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.models.auth_schemas import (
    BootstrapAdminResponse,
    LoginResponse,
    SetupStateResponse,
    UserSummary,
)
from app.repositories.user_repository import StoredUser, UserRepository

# ── Exception definitions ─────────────────────────────────────────────
class AuthError(Exception):
    pass

# Custom exceptions for specific authentication error cases, allowing the API layer to return appropriate HTTP responses and guide the client on next steps (e.g., redirecting to bootstrap flow).
class BootstrapRequiredError(AuthError):
    pass

# Raised when a bootstrap attempt is made but users already exist, preventing unauthorised admin creation after initial setup.
class BootstrapUnavailableError(AuthError):
    pass


@dataclass(slots=True)
# Represents the authenticated user's context extracted from a validated JWT, used in request handling to enforce permissions and access control.
class AuthenticatedUser:
    id: str
    username: str
    email: str | None
    roles: list[str]
    is_active: bool

# The AuthService class encapsulates all authentication-related business logic, coordinating between the user repository for data access and the security module for password hashing and JWT handling.
class AuthService:
# Initialization requires application settings (for JWT configuration) and a user repository instance for database interactions related to user records.
    def __init__(self, settings: Settings, user_repository: UserRepository) -> None:
        self.settings = settings
        self.user_repository = user_repository

# Check the bootstrap state to determine if the system is in a fresh state requiring admin account creation.
    def get_setup_state(self) -> SetupStateResponse:
        user_count = self.user_repository.count_users()
        return SetupStateResponse(needs_bootstrap=user_count == 0, user_count=user_count)

# Create the first admin user on a fresh installation, enforcing that this can only be done once and requires no existing users.
    def bootstrap_admin(
        self,
        identifier: str,
        password: str,
        email: str | None = None,
    ) -> BootstrapAdminResponse:
        
        if self.user_repository.count_users() > 0:
            raise BootstrapUnavailableError("Bootstrap admin is disabled because users already exist")
        # Hash the plaintext password for secure storage
        password_hash = hash_password(password)
        user = self.user_repository.create_bootstrap_admin(
            identifier=identifier,
            password_hash=password_hash,
            email=email,
        )
        return BootstrapAdminResponse(
            message="Bootstrap admin created successfully",
            user=self._to_user_summary(user),
        )

# Authenticate a user with their credentials and issue a JWT, with comprehensive checks for account status and lockout conditions.
    def login(self, identifier: str, password: str) -> LoginResponse:
    
        # Guard: require at least one user to exist before allowing login
        if self.user_repository.count_users() == 0:
            raise BootstrapRequiredError("No users exist yet. Create the first admin account first.")

        user = self.user_repository.get_by_identifier(identifier)
        if user is None or user.password_hash is None:
            raise AuthError("Invalid credentials")

        # Check if account has been deactivated or soft-deleted
        if not user.is_active or user.deleted_at is not None:
            raise AuthError("This account is inactive")

        # Check if account is currently locked (with optional expiry)
        if user.is_locked and (user.locked_until is None or user.locked_until > datetime.now(timezone.utc)):
            reason = user.account_locked_reason or "This account is locked"
            raise AuthError(reason)

        # Verify password against stored bcrypt hash
        if not verify_password(password, user.password_hash):
            # Record the failed attempt for lockout threshold tracking
            self.user_repository.record_failed_login(user.id)
            raise AuthError("Invalid credentials")

        # Successful authentication: record login and refresh user data
        self.user_repository.record_successful_login(user.id)
        # Re-fetch user to get updated last_login timestamp
        current = self.user_repository.get_by_id(user.id) or user
        # Create JWT with user identity and role claims
        access_token = create_access_token(
            self.settings,
            subject=current.id,
            additional_claims={
                "username": current.username,
                "roles": current.roles,
            },
        )
        return LoginResponse(
            access_token=access_token,
            user=self._to_user_summary(current),
        )

# Decode a JWT and return the authenticated user's context, verifying the token's validity and ensuring the user still exists and is active.
    def get_current_user(self, token: str) -> AuthenticatedUser:
        
        payload = decode_access_token(self.settings, token)
        if payload.get("scope") is not None:
            raise AuthError("A login access token is required")
        # Look up the user by the "sub" (subject) claim in the JWT
        user = self.user_repository.get_by_id(str(payload["sub"]))
        if user is None:
            raise AuthError("Authenticated user no longer exists")
        if not user.is_active or user.deleted_at is not None:
            raise AuthError("Authenticated user is inactive")
        return AuthenticatedUser(
            id=user.id,
            username=user.username,
            email=user.email,
            roles=user.roles,
            is_active=user.is_active,
        )

    def change_password(self, current_user, current_password, new_password):
        user = self.user_repository.get_by_id(current_user.id)
        if user is None or not user.password_hash or not verify_password(current_password, user.password_hash):
            raise AuthError("Current password is incorrect")
        if current_password == new_password:
            raise AuthError("Choose a password different from your current password")
        self.user_repository.change_password(user.id, hash_password(new_password))

    def create_sensitive_access_token(self, current_user: AuthenticatedUser, password: str) -> tuple[str, datetime]:
        """Re-authenticate an administrator and issue a five-minute image-view token."""
        if "admin" not in {role.lower() for role in current_user.roles}:
            raise AuthError("Administrator role is required for sensitive image access")

        user = self.user_repository.get_by_id(current_user.id)
        if user is None or not user.is_active or user.deleted_at is not None or not user.password_hash:
            raise AuthError("Sensitive re-authentication failed")
        if not verify_password(password, user.password_hash):
            raise AuthError("Sensitive re-authentication failed")

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.settings.sensitive_access_minutes)
        token = create_access_token(
            self.settings,
            subject=user.id,
            additional_claims={"scope": "reference_images", "reauthenticated": True},
            expires_minutes=self.settings.sensitive_access_minutes,
        )
        return token, expires_at

    @staticmethod
    # Helper method to convert a StoredUser repository object to a UserSummary API schema, used in responses to avoid exposing sensitive fields and to provide a consistent user representation in the API layer.
    def _to_user_summary(user: StoredUser) -> UserSummary:
        return UserSummary(
            id=user.id,
            username=user.username,
            email=user.email,
            roles=user.roles,
            is_active=user.is_active,
        )
