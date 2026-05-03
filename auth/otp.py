"""
OTP send and verify logic via Twilio SMS.
"""
from __future__ import annotations

import random
import string

from config import settings
from core.exceptions import OTPSendFailedError
from core.logging import get_logger

log = get_logger(__name__)


def generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


async def send_otp_via_twilio(phone: str, otp: str) -> None:
    """
    Send OTP SMS via Twilio. Raises OTPSendFailedError on any failure.
    In mock mode (no SID configured), logs the OTP instead.
    """
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        log.warning(
            "Twilio credentials not configured — OTP logged for dev",
            phone=phone,
            otp=otp,
        )
        return

    try:
        from twilio.rest import Client  # type: ignore[import]

        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Your NammaKelsa OTP is {otp}. Valid for 10 minutes. Do not share.",
            from_=settings.TWILIO_FROM_NUMBER,
            to=phone,
        )
        log.info(
            "OTP SMS sent via Twilio",
            phone=phone,
            twilio_sid=message.sid,
        )
    except Exception as exc:
        log.error("Twilio send_otp failed", phone=phone, exc_info=True)
        raise OTPSendFailedError(str(exc)) from exc
