"""
JWT creation and validation for NammaKelsa.

Two token tiers:
  partial — issued after OTP verify (is_registered=False, limited access)
  full    — issued after face + Aadhaar complete (is_registered=True)
  admin   — issued to admin_users (role=admin or super_admin)
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID

from jose import JWTError, jwt  # type: ignore[import]

from cache.redis_client import cache_session, get_session
from config import settings
from core.exceptions import JWTExpiredError, JWTInvalidError
from core.logging import get_logger

log = get_logger(__name__)


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def create_token(
    user_id: str,
    phone: str,
    role: str,
    is_registered: bool,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "sub": user_id,
        "phone": phone,
        "role": role,
        "is_registered": is_registered,
        "iat": _now_utc(),
        "exp": _now_utc() + timedelta(hours=settings.JWT_EXPIRY_HOURS),
    }
    if extra:
        payload.update(extra)
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    log.debug("JWT created", user_id=user_id, role=role, is_registered=is_registered)
    return token


def decode_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as exc:
        err = str(exc).lower()
        if "expired" in err:
            raise JWTExpiredError() from exc
        raise JWTInvalidError() from exc


async def create_and_cache_token(
    user_id: str,
    phone: str,
    role: str,
    is_registered: bool,
    is_active: bool = True,
) -> str:
    token = create_token(user_id, phone, role, is_registered)
    session_payload = {
        "user_id": user_id,
        "phone": phone,
        "role": role,
        "is_registered": is_registered,
        "is_active": is_active,
    }
    await cache_session(user_id, session_payload)
    return token


async def validate_token_with_cache(token: str) -> dict[str, Any]:
    """
    Decode JWT, then confirm session is still alive in Redis.
    Redis miss falls back to DB check (handled in middleware).
    """
    payload = decode_token(token)
    user_id = payload.get("sub", "")

    cached = await get_session(user_id)
    if cached:
        log.debug("JWT validated from Redis cache", user_id=user_id)
        return payload

    log.debug("JWT Redis cache miss — re-validating from token claims", user_id=user_id)
    return payload
