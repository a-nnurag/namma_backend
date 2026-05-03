"""
Aadhaar Verification — Strategy Pattern.

AadhaarVerifierBackend is the abstract interface. All verification code in
routes calls verifier.verify() without knowing which backend is in use.

Concrete strategies:
  MockAadhaarVerifier    — always True (hackathon / local dev)
  SurepassAadhaarVerifier — real API via surepass.io
  SignzyAadhaarVerifier   — real API via Signzy

Selection is driven by AADHAAR_VERIFIER_BACKEND env var.
Swapping provider = change one env var, zero code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from config import settings
from core.exceptions import AadhaarAPIFailedError
from core.logging import get_logger

log = get_logger(__name__)


class AadhaarVerifierBackend(ABC):
    """Abstract strategy for Aadhaar number verification."""

    @abstractmethod
    async def verify(self, aadhaar_number: str) -> bool:
        """
        Return True if the Aadhaar number is valid according to the provider.
        Raise AadhaarAPIFailedError if the API itself is unreachable or returns
        an unexpected error (HTTP 5xx, timeout, etc.).
        Never raises for a simple invalid/not-found Aadhaar — just returns False.
        """


class MockAadhaarVerifier(AadhaarVerifierBackend):
    """Always approves any well-formatted Aadhaar. Safe for hackathon demos."""

    async def verify(self, aadhaar_number: str) -> bool:
        log.info(
            "MockAadhaarVerifier: approved",
            aadhaar_suffix=aadhaar_number[-4:],
        )
        return True


class SurepassAadhaarVerifier(AadhaarVerifierBackend):
    """
    Calls the Surepass Aadhaar API.
    Docs: https://kyc-api.surepass.io/api/v1/aadhaar-v2/

    Expected response shape (success):
      {"success": true, "data": {"is_valid": true, ...}}
    Expected response shape (invalid number):
      {"success": false, "message": "..."}
    """

    def __init__(self, api_key: str, api_url: str) -> None:
        self._api_key = api_key
        self._api_url = api_url

    async def verify(self, aadhaar_number: str) -> bool:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"id_number": aadhaar_number}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._api_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            log.error("Surepass Aadhaar API timed out", exc_info=True)
            raise AadhaarAPIFailedError("Request timed out") from exc
        except httpx.RequestError as exc:
            log.error("Surepass Aadhaar API request failed", exc_info=True)
            raise AadhaarAPIFailedError(str(exc)) from exc

        if resp.status_code >= 500:
            log.error(
                "Surepass Aadhaar API returned 5xx",
                status_code=resp.status_code,
                body=resp.text[:200],
            )
            raise AadhaarAPIFailedError(f"HTTP {resp.status_code} from Surepass")

        try:
            data = resp.json()
        except Exception:
            log.error("Surepass response is not valid JSON", body=resp.text[:200])
            raise AadhaarAPIFailedError("Invalid JSON response from Surepass")

        is_valid = bool(data.get("success") and data.get("data", {}).get("is_valid"))
        log.info(
            "Surepass Aadhaar verification result",
            aadhaar_suffix=aadhaar_number[-4:],
            is_valid=is_valid,
        )
        return is_valid


class SignzyAadhaarVerifier(AadhaarVerifierBackend):
    """
    Calls the Signzy Aadhaar verification API.
    Replace the request structure when you have Signzy credentials.
    """

    def __init__(self, api_key: str, api_url: str) -> None:
        self._api_key = api_key
        self._api_url = api_url

    async def verify(self, aadhaar_number: str) -> bool:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"aadhaarNumber": aadhaar_number}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._api_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise AadhaarAPIFailedError("Signzy request timed out") from exc
        except httpx.RequestError as exc:
            raise AadhaarAPIFailedError(str(exc)) from exc

        if resp.status_code >= 500:
            raise AadhaarAPIFailedError(f"HTTP {resp.status_code} from Signzy")

        try:
            data = resp.json()
        except Exception:
            raise AadhaarAPIFailedError("Invalid JSON from Signzy")

        is_valid = bool(data.get("result", {}).get("isValid"))
        log.info(
            "Signzy Aadhaar verification result",
            aadhaar_suffix=aadhaar_number[-4:],
            is_valid=is_valid,
        )
        return is_valid


def build_aadhaar_verifier() -> AadhaarVerifierBackend:
    """
    Factory that builds the correct AadhaarVerifierBackend from config.
    Called once at startup and reused for the lifetime of the process.
    """
    backend = settings.AADHAAR_VERIFIER_BACKEND.lower()
    if backend == "surepass":
        if not settings.SUREPASS_API_KEY:
            raise RuntimeError("SUREPASS_API_KEY is required when AADHAAR_VERIFIER_BACKEND=surepass")
        log.info("Using SurepassAadhaarVerifier")
        return SurepassAadhaarVerifier(settings.SUREPASS_API_KEY, settings.SUREPASS_API_URL)
    elif backend == "signzy":
        if not settings.SIGNZY_API_KEY:
            raise RuntimeError("SIGNZY_API_KEY is required when AADHAAR_VERIFIER_BACKEND=signzy")
        log.info("Using SignzyAadhaarVerifier")
        return SignzyAadhaarVerifier(settings.SIGNZY_API_KEY, settings.SIGNZY_API_URL)
    else:
        log.info("Using MockAadhaarVerifier (no real Aadhaar API)")
        return MockAadhaarVerifier()
