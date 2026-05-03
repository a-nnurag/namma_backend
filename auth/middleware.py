"""
FastAPI middleware and dependencies for JWT authentication.

Provides:
  get_current_user(token)          — any authenticated user
  get_registered_user(token)       — requires is_registered=True
  get_admin_user(token)            — requires role officer/super_admin
  RequestContextMiddleware         — injects request_id + user_id into log context
"""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.jwt import validate_token_with_cache
from cache.redis_client import get_session
from core.exceptions import (
    JWTExpiredError,
    JWTInvalidError,
    NotRegisteredError,
    UserInactiveError,
)
from core.logging import get_logger, set_request_context
from db.crud import get_user_by_id
from db.session import get_db

log = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)


def _http_401(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _http_403(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict:
    if credentials is None:
        raise _http_401("Authorization header missing")

    try:
        payload = await validate_token_with_cache(credentials.credentials)
    except JWTExpiredError:
        raise _http_401("Token has expired")
    except JWTInvalidError:
        raise _http_401("Invalid token")

    # Check Redis for is_active (catches FRAUD deactivation without re-login)
    user_id = payload.get("sub", "")
    cached = await get_session(user_id)
    if cached and not cached.get("is_active", True):
        raise _http_403("Account deactivated")

    set_request_context(request_id=str(uuid.uuid4()), user_id=user_id)
    return payload


async def get_registered_user(
    payload: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Require full registration (face + Aadhaar complete)."""
    if not payload.get("is_registered"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration not complete. Complete face and Aadhaar verification first.",
        )
    return payload


async def get_admin_user(
    payload: Annotated[dict, Depends(get_current_user)],
) -> dict:
    role = payload.get("role", "")
    if role not in ("officer", "super_admin"):
        raise _http_403("Admin access required")
    return payload


async def get_super_admin_user(
    payload: Annotated[dict, Depends(get_admin_user)],
) -> dict:
    if payload.get("role") != "super_admin":
        raise _http_403("Super admin access required")
    return payload
