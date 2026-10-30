"""
Authentication API routes for the Sentinel backend.

Router prefix: ``/api/auth``

This module provides endpoints for the full authentication lifecycle:

* **GET  /api/auth/setup-state**    -- Check whether the system still needs
  first-time admin bootstrap.
* **POST /api/auth/login**          -- Authenticate with an identifier
  (username or email) and password; returns a JWT access token.
* **GET  /api/auth/me**             -- Retrieve the profile of the currently
  authenticated user (requires a valid Bearer token).
* **POST /api/auth/logout**         -- Log out the current user (requires a
  valid Bearer token).  Currently a no-op on the server side since JWTs are
  stateless, but signals intent to the client.
* **POST /api/auth/bootstrap-admin** -- Create the very first admin account.
  Only available when no users exist in the database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_auth_service, get_current_user
from app.models.auth_schemas import (
    BootstrapAdminRequest,
    BootstrapAdminResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    SensitiveAccessRequest,
    SensitiveAccessResponse,
    SetupStateResponse,
    UserSummary,
)
from app.services.auth_service import (
    AuthService,
    AuthenticatedUser,
    AuthError,
    BootstrapRequiredError,
    BootstrapUnavailableError,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/setup-state", response_model=SetupStateResponse)
async def get_setup_state(auth_service: AuthService = Depends(get_auth_service)):
    """Check whether the system requires initial admin bootstrap.

    **HTTP Method:** GET
    **Path:** /api/auth/setup-state
    **Auth required:** No

    Returns a :class:`SetupStateResponse` indicating whether the system has
    been bootstrapped (i.e. at least one user exists).  The frontend uses
    this on first load to decide whether to show the setup wizard or the
    login form.

    Returns:
        SetupStateResponse: Contains a boolean flag such as
            ``needs_bootstrap``.
    """
    return auth_service.get_setup_state()


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, auth_service: AuthService = Depends(get_auth_service)):
    """Authenticate a user and return a JWT access token.

    **HTTP Method:** POST
    **Path:** /api/auth/login
    **Auth required:** No

    The client sends an ``identifier`` (username **or** email) together with
    a ``password``.  On success the response includes a signed JWT.

    Args:
        payload: Login credentials containing ``identifier`` and ``password``.

    Returns:
        LoginResponse: Contains the JWT ``access_token`` and token type.

    Raises:
        HTTPException 409: If no users exist yet and bootstrap is required.
            The detail body includes ``{"needs_bootstrap": true}``.
        HTTPException 401: If the credentials are invalid.
    """
    try:
        return auth_service.login(payload.identifier, payload.password)
    except BootstrapRequiredError as exc:
        # No users exist -- the client should redirect to bootstrap flow
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "needs_bootstrap": True},
        ) from exc
    except AuthError as exc:
        # Bad credentials (wrong password, unknown user, inactive account, etc.)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.get("/me", response_model=UserSummary)
async def get_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    """Return the profile of the currently authenticated user.

    **HTTP Method:** GET
    **Path:** /api/auth/me
    **Auth required:** Yes (Bearer token)

    Uses the :func:`get_current_user` dependency to validate the JWT and
    extract the user.  The response is a trimmed-down :class:`UserSummary`
    containing only the fields the frontend needs for display/navigation.

    Returns:
        UserSummary: Public-facing user profile (id, username, email,
            roles, is_active).

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    return UserSummary(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        roles=current_user.roles,
        is_active=current_user.is_active,
    )


@router.post("/sensitive-access", response_model=SensitiveAccessResponse)
async def create_sensitive_access(
    payload: SensitiveAccessRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Re-enter an admin password to obtain a five-minute reference-image token."""
    try:
        token, expires_at = auth_service.create_sensitive_access_token(current_user, payload.password)
        return SensitiveAccessResponse(sensitive_access_token=token, expires_at=expires_at)
    except AuthError as exc:
        status_code = status.HTTP_403_FORBIDDEN if "Administrator role" in str(exc) else status.HTTP_401_UNAUTHORIZED
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/logout", response_model=LogoutResponse)
async def logout(_current_user: AuthenticatedUser = Depends(get_current_user)):
    """Log out the current user.

    **HTTP Method:** POST
    **Path:** /api/auth/logout
    **Auth required:** Yes (Bearer token)

    Because authentication is JWT-based (stateless), no server-side session
    is destroyed.  The endpoint exists so the client can confirm the token
    was still valid at the time of logout and clear local storage.

    Returns:
        LogoutResponse: A confirmation message.

    Raises:
        HTTPException 401: If the Bearer token is missing or invalid.
    """
    return LogoutResponse(message="Logged out successfully")


@router.post(
    "/bootstrap-admin",
    response_model=BootstrapAdminResponse,
    status_code=status.HTTP_201_CREATED,
)
async def bootstrap_admin(
    payload: BootstrapAdminRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Create the very first administrator account during system setup.

    **HTTP Method:** POST
    **Path:** /api/auth/bootstrap-admin
    **Auth required:** No (but only callable when zero users exist)

    This endpoint is a one-shot operation: once an admin has been created,
    subsequent calls will return HTTP 409.  It is guarded by the service
    layer, not by a Bearer token, because no tokens can exist yet.

    Args:
        payload: Contains ``identifier`` (username), ``password``, and
            optionally ``email`` for the new admin account.

    Returns:
        BootstrapAdminResponse: Confirmation that the admin was created,
            along with initial user details.

    Raises:
        HTTPException 409: If at least one user already exists (bootstrap
            is no longer available).
        HTTPException 400: If the payload fails validation (e.g. password
            too short, identifier already taken).
    """
    try:
        return auth_service.bootstrap_admin(
            identifier=payload.identifier,
            password=payload.password,
            email=payload.email,
        )
    except BootstrapUnavailableError as exc:
        # At least one user already exists; bootstrap window has closed
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        # Input validation failure (e.g. weak password)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
