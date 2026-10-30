"""
Security utilities – password hashing and JWT token management.

Uses PBKDF2-SHA256 for password hashing (via passlib) and HS256-signed
JWTs (via python-jose) for stateless authentication.  All functions
are pure / stateless and receive ``Settings`` where needed so they
remain easy to test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import Settings

# Password hashing context – PBKDF2-SHA256 with automatic upgrade of
# deprecated schemes.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# FastAPI dependency that extracts the ``Authorization: Bearer <token>``
# header.  ``auto_error=False`` lets the route decide how to handle
# missing tokens instead of raising 403 automatically.
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Return a PBKDF2-SHA256 hash of *password*."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check *plain_password* against a previously stored hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    settings: Settings,
    subject: str,
    additional_claims: dict[str, Any] | None = None,
    expires_minutes: int | None = None,
) -> str:
    """
    Mint a new JWT access token.

    Parameters
    ----------
    settings : Settings
        Provides the signing key, algorithm, and expiry duration.
    subject : str
        The ``sub`` claim – typically the user's username or ID.
    additional_claims : dict, optional
        Extra claims merged into the JWT payload (e.g. roles).

    Returns
    -------
    str
        The encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,           # Who the token represents
        "iat": int(now.timestamp()),        # Issued-at timestamp
        "exp": int(expires_at.timestamp()), # Expiry timestamp
    }
    if additional_claims:
        payload.update(additional_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(settings: Settings, token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Raises ``ValueError`` if the token is expired, tampered with, or
    missing a ``sub`` claim.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid or expired access token") from exc
    if not payload.get("sub"):
        raise ValueError("Access token is missing a subject")
    return payload


def extract_bearer_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    """
    Pull the raw token string from FastAPI's ``HTTPAuthorizationCredentials``.

    Raises ``ValueError`` when the header is absent or not a Bearer token.
    """
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise ValueError("Missing bearer token")
    return credentials.credentials
