"""
Custom exception hierarchy for NammaKelsa backend.

Every exception carries an ErrorCode for structured logging and API responses.
HTTP status codes are NOT embedded here — they are decided at the route layer.
"""
from __future__ import annotations
from typing import Any

from .error_codes import ErrorCode


class NammaKelsaError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.detail = detail or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message!r})"


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthError(NammaKelsaError):
    pass


class OTPRateLimitedError(AuthError):
    def __init__(self, phone: str) -> None:
        super().__init__(
            f"Too many OTP requests for {phone}. Try again in an hour.",
            ErrorCode.OTP_RATE_LIMITED,
            {"phone": phone},
        )


class OTPInvalidError(AuthError):
    def __init__(self, attempts_left: int) -> None:
        super().__init__(
            f"Invalid OTP. {attempts_left} attempt(s) remaining.",
            ErrorCode.OTP_INVALID,
            {"attempts_left": attempts_left},
        )


class OTPExpiredError(AuthError):
    def __init__(self) -> None:
        super().__init__("OTP has expired. Request a new one.", ErrorCode.OTP_EXPIRED)


class OTPMaxAttemptsError(AuthError):
    def __init__(self) -> None:
        super().__init__(
            "Too many wrong OTP attempts. Request a new OTP.",
            ErrorCode.OTP_MAX_ATTEMPTS,
        )


class OTPSendFailedError(AuthError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"Failed to send OTP via Twilio. {reason}",
            ErrorCode.OTP_SEND_FAILED,
            {"reason": reason},
        )


class JWTInvalidError(AuthError):
    def __init__(self) -> None:
        super().__init__("Invalid or malformed JWT token.", ErrorCode.JWT_INVALID)


class JWTExpiredError(AuthError):
    def __init__(self) -> None:
        super().__init__("JWT token has expired.", ErrorCode.JWT_EXPIRED)


class NotRegisteredError(AuthError):
    def __init__(self, step: str = "") -> None:
        super().__init__(
            f"Registration not complete. Pending step: {step}",
            ErrorCode.NOT_REGISTERED,
            {"pending_step": step},
        )


class UserInactiveError(AuthError):
    def __init__(self) -> None:
        super().__init__(
            "Account is deactivated due to a FRAUD verdict.",
            ErrorCode.USER_INACTIVE,
        )


# ── Registration ──────────────────────────────────────────────────────────────

class RegistrationError(NammaKelsaError):
    pass


class FaceAlreadyRegisteredError(RegistrationError):
    def __init__(self, reason: str = "hash match") -> None:
        super().__init__(
            "This face is already registered to an account.",
            ErrorCode.FACE_ALREADY_REGISTERED,
            {"reason": reason},
        )


class FaceNotDetectedError(RegistrationError):
    def __init__(self) -> None:
        super().__init__(
            "No clear face detected in the video. Please re-record in good lighting.",
            ErrorCode.FACE_NOT_DETECTED,
        )


class FaceVideoTooLargeError(RegistrationError):
    def __init__(self, size_mb: float) -> None:
        super().__init__(
            f"Face video is {size_mb:.1f}MB. Maximum is 10MB.",
            ErrorCode.FACE_VIDEO_TOO_LARGE,
            {"size_mb": size_mb},
        )


class FaceVideoInvalidFormatError(RegistrationError):
    def __init__(self, fmt: str) -> None:
        super().__init__(
            f"Invalid video format: {fmt}. Allowed: mp4, mov, webm.",
            ErrorCode.FACE_VIDEO_INVALID_FORMAT,
            {"format": fmt},
        )


class FaceEmbeddingFailedError(RegistrationError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"Failed to extract face embedding. {reason}",
            ErrorCode.FACE_EMBEDDING_FAILED,
            {"reason": reason},
        )


class AadhaarInvalidFormatError(RegistrationError):
    def __init__(self) -> None:
        super().__init__(
            "Aadhaar number must be exactly 12 digits and not all the same digit.",
            ErrorCode.AADHAAR_INVALID_FORMAT,
        )


class AadhaarAlreadyRegisteredError(RegistrationError):
    def __init__(self) -> None:
        super().__init__(
            "This Aadhaar number is already linked to another account.",
            ErrorCode.AADHAAR_ALREADY_REGISTERED,
        )


class AadhaarAPIFailedError(RegistrationError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            "Aadhaar verification service is unavailable. Try again later.",
            ErrorCode.AADHAAR_API_FAILED,
            {"reason": reason},
        )


class AadhaarVerificationFailedError(RegistrationError):
    def __init__(self) -> None:
        super().__init__(
            "Aadhaar number could not be verified. Please check and try again.",
            ErrorCode.AADHAAR_VERIFICATION_FAILED,
        )


class FaceNotYetRegisteredError(RegistrationError):
    def __init__(self) -> None:
        super().__init__(
            "Face must be registered before Aadhaar verification.",
            ErrorCode.FACE_NOT_YET_REGISTERED,
        )


# ── Skill / Application ───────────────────────────────────────────────────────

class SkillError(NammaKelsaError):
    pass


class SkillNotFoundError(SkillError):
    def __init__(self, skill_id: str = "") -> None:
        super().__init__(
            f"Skill not found: {skill_id}",
            ErrorCode.SKILL_NOT_FOUND,
            {"skill_id": skill_id},
        )


class MaxAttemptsReachedError(SkillError):
    def __init__(self, skill_name: str) -> None:
        super().__init__(
            f"Maximum 3 attempts reached for skill: {skill_name}",
            ErrorCode.MAX_ATTEMPTS_REACHED,
            {"skill_name": skill_name},
        )


class InCooldownError(SkillError):
    def __init__(self, skill_name: str, cooldown_until: str) -> None:
        super().__init__(
            f"In cooldown for {skill_name} until {cooldown_until}.",
            ErrorCode.IN_COOLDOWN,
            {"skill_name": skill_name, "cooldown_until": cooldown_until},
        )


class ApplicationNotFoundError(SkillError):
    def __init__(self, application_id: str = "") -> None:
        super().__init__(
            f"Application not found: {application_id}",
            ErrorCode.APPLICATION_NOT_FOUND,
            {"application_id": application_id},
        )


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentError(NammaKelsaError):
    pass


class UploadNotFoundError(DocumentError):
    def __init__(self, upload_id: str) -> None:
        super().__init__(
            f"Upload session not found: {upload_id}",
            ErrorCode.UPLOAD_NOT_FOUND,
            {"upload_id": upload_id},
        )


class UploadAlreadyCompleteError(DocumentError):
    def __init__(self, upload_id: str) -> None:
        super().__init__(
            f"Upload already completed: {upload_id}",
            ErrorCode.UPLOAD_ALREADY_COMPLETE,
            {"upload_id": upload_id},
        )


class ChunksMissingError(DocumentError):
    def __init__(self, missing: list[int]) -> None:
        super().__init__(
            f"Missing chunks: {missing}. Re-send missing chunks.",
            ErrorCode.CHUNKS_MISSING,
            {"missing_chunks": missing},
        )


class FileTooLargeError(DocumentError):
    def __init__(self, size_mb: float, max_mb: int) -> None:
        super().__init__(
            f"File is {size_mb:.1f}MB. Maximum allowed is {max_mb}MB.",
            ErrorCode.FILE_TOO_LARGE,
            {"size_mb": size_mb, "max_mb": max_mb},
        )


class InvalidFileFormatError(DocumentError):
    def __init__(self, fmt: str, allowed: list[str]) -> None:
        super().__init__(
            f"Invalid file format: {fmt}. Allowed: {', '.join(allowed)}",
            ErrorCode.INVALID_FILE_FORMAT,
            {"format": fmt, "allowed": allowed},
        )


# ── Interview ─────────────────────────────────────────────────────────────────

class InterviewError(NammaKelsaError):
    pass


class SessionNotFoundError(InterviewError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Interview session not found: {session_id}",
            ErrorCode.SESSION_NOT_FOUND,
            {"session_id": session_id},
        )


class SessionAlreadyEndedError(InterviewError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Interview session already ended: {session_id}",
            ErrorCode.SESSION_ALREADY_ENDED,
            {"session_id": session_id},
        )


class NoActiveApplicationError(InterviewError):
    def __init__(self) -> None:
        super().__init__(
            "No active application in INTERVIEW_PENDING status.",
            ErrorCode.NO_ACTIVE_APPLICATION,
        )


# ── Infrastructure ────────────────────────────────────────────────────────────

class KafkaError(NammaKelsaError):
    pass


class KafkaSendFailedError(KafkaError):
    def __init__(self, topic: str, reason: str = "") -> None:
        super().__init__(
            f"Failed to send message to Kafka topic '{topic}'. {reason}",
            ErrorCode.KAFKA_SEND_FAILED,
            {"topic": topic, "reason": reason},
        )


class KafkaNotConnectedError(KafkaError):
    def __init__(self) -> None:
        super().__init__(
            "Kafka producer is not connected.",
            ErrorCode.KAFKA_NOT_CONNECTED,
        )


class DatabaseError(NammaKelsaError):
    pass


class DBConnectionError(DatabaseError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"Database connection failed. {reason}",
            ErrorCode.DB_CONNECTION_FAILED,
            {"reason": reason},
        )


class DBQueryError(DatabaseError):
    def __init__(self, operation: str, reason: str = "") -> None:
        super().__init__(
            f"Database query failed during '{operation}'. {reason}",
            ErrorCode.DB_QUERY_FAILED,
            {"operation": operation, "reason": reason},
        )


class DBIntegrityError(DatabaseError):
    def __init__(self, constraint: str = "") -> None:
        super().__init__(
            f"Database integrity constraint violated: {constraint}",
            ErrorCode.DB_INTEGRITY_ERROR,
            {"constraint": constraint},
        )


class CacheError(NammaKelsaError):
    pass


class RedisConnectionError(CacheError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"Redis connection failed. {reason}",
            ErrorCode.REDIS_CONNECTION_FAILED,
            {"reason": reason},
        )


class MLServiceError(NammaKelsaError):
    pass


class MLServiceUnreachableError(MLServiceError):
    def __init__(self, url: str = "") -> None:
        super().__init__(
            f"ML service is unreachable at {url}.",
            ErrorCode.ML_SERVICE_UNREACHABLE,
            {"url": url},
        )


class MLFaceCheckFailedError(MLServiceError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"ML face check failed. {reason}",
            ErrorCode.ML_FACE_CHECK_FAILED,
            {"reason": reason},
        )
