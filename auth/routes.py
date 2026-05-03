"""
Auth routes: POST /auth/send-otp and POST /auth/verify-otp
"""
from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from auth.jwt import create_and_cache_token
from auth.otp import generate_otp, send_otp_via_twilio
from cache.redis_client import (
    check_otp_rate_limit,
    delete_otp,
    get_otp_data,
    increment_otp_attempts,
    store_otp,
)
from core.exceptions import (
    OTPExpiredError,
    OTPInvalidError,
    OTPMaxAttemptsError,
    OTPRateLimitedError,
    OTPSendFailedError,
)
from core.logging import get_logger
from db.crud import create_user, get_user_by_phone
from db.session import get_db
from db.adapter import DatabaseAdapter

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_PHONE_RE = re.compile(r"^\+91[6-9]\d{9}$")
OTP_MAX_WRONG = 3


class SendOTPRequest(BaseModel):
    phone: str = Field(..., description="Phone number in +91XXXXXXXXXX format")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not _PHONE_RE.match(v):
            raise ValueError("Phone must be in format +91XXXXXXXXXX (Indian mobile number)")
        return v


class VerifyOTPRequest(BaseModel):
    phone: str
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


@router.post("/send-otp", status_code=200)
async def send_otp(
    body: SendOTPRequest,
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    allowed = await check_otp_rate_limit(body.phone)
    if not allowed:
        log.warning("OTP rate limited", phone=body.phone)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many OTP requests. Try again in an hour.",
        )

    otp = generate_otp()
    await store_otp(body.phone, otp)

    try:
        await send_otp_via_twilio(body.phone, otp)
    except OTPSendFailedError as exc:
        log.error("OTP send failed", phone=body.phone, error_code=exc.error_code)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not send OTP. Please try again.",
        )

    log.info("OTP sent successfully", phone=body.phone)
    return {"message": "OTP sent", "expires_in": 600}


@router.post("/verify-otp", status_code=200)
async def verify_otp(
    body: VerifyOTPRequest,
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    otp_data = await get_otp_data(body.phone)
    if otp_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP expired or not found. Request a new OTP.",
        )

    if otp_data["attempts"] >= OTP_MAX_WRONG:
        await delete_otp(body.phone)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many wrong attempts. Request a new OTP.",
        )

    if otp_data["otp"] != body.otp:
        attempts = await increment_otp_attempts(body.phone)
        remaining = OTP_MAX_WRONG - attempts
        log.warning("Wrong OTP attempt", phone=body.phone, attempts=attempts)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid OTP. {remaining} attempt(s) remaining.",
        )

    await delete_otp(body.phone)

    user = await get_user_by_phone(db, body.phone)
    is_new = user is None
    if is_new:
        user = await create_user(db, body.phone)
        log.info("New user created via OTP", phone=body.phone, user_id=str(user.id))
    else:
        log.info("Existing user verified via OTP", phone=body.phone, user_id=str(user.id))

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account deactivated due to a fraud verdict.",
        )

    token = await create_and_cache_token(
        user_id=str(user.id),
        phone=user.phone,
        role="user",
        is_registered=user.is_registered,
        is_active=user.is_active,
    )

    return {
        "token": token,
        "is_new_user": is_new,
        "is_registered": user.is_registered,
        "face_registered": user.face_registered,
    }
