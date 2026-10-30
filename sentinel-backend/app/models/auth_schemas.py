"""
Pydantic schemas for authentication and user management endpoints.

Covers login / logout, first-time admin bootstrap, and setup state
queries.  Separated from ``schemas.py`` to keep auth concerns isolated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UserSummary(BaseModel):
    """Lightweight user representation returned inside auth responses."""
    id: str                                                # User UUID
    username: str                                          # Login handle
    email: str | None = None                               # Optional email address
    roles: list[str] = Field(default_factory=list)         # Role names (e.g. ["admin"])
    is_active: bool = True                                 # Whether the account is enabled


class LoginRequest(BaseModel):
    """POST body for ``/auth/login``."""
    identifier: str                                        # Username or email
    password: str                                          # Plaintext password (over HTTPS)


class LoginResponse(BaseModel):
    """Successful login response containing the JWT."""
    access_token: str                                      # Signed JWT token
    token_type: Literal["bearer"] = "bearer"               # Always "bearer"
    user: UserSummary                                      # Authenticated user info


class SensitiveAccessRequest(BaseModel):
    """Password re-entry required before viewing retained reference images."""
    password: str = Field(min_length=1)


class SensitiveAccessResponse(BaseModel):
    """Short-lived, scope-limited credential for sensitive reference images."""
    sensitive_access_token: str
    expires_at: datetime


class LogoutResponse(BaseModel):
    """Confirmation message after logout."""
    message: str


class BootstrapAdminRequest(BaseModel):
    """POST body for creating the first admin user on a fresh install."""
    identifier: str                                        # Desired username
    password: str                                          # Desired password
    email: str | None = None                               # Optional email


class BootstrapAdminResponse(BaseModel):
    """Response after the initial admin account is created."""
    message: str
    user: UserSummary                                      # Newly created admin


class SetupStateResponse(BaseModel):
    """Tells the frontend whether the system needs first-time setup."""
    needs_bootstrap: bool                                  # True if no users exist yet
    user_count: int                                        # Total user count in the database
