"""
Structured JSON logging for NammaKelsa backend.

Every log line includes:
  - timestamp (ISO 8601 UTC)
  - level
  - logger name
  - message
  - error_code (when applicable)
  - user_id / phone (when available via context var)
  - request_id (injected by middleware)
  - extra fields passed to the log call

Usage:
    from core.logging import get_logger
    log = get_logger(__name__)

    log.info("OTP sent", phone=phone, user_id=user_id)
    log.error("Kafka send failed", topic=topic, error_code=ErrorCode.KAFKA_SEND_FAILED)
"""
import json
import logging
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from .error_codes import ErrorCode


# Context variables injected by middleware — available anywhere in the call stack
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
_user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")


def set_request_context(request_id: str, user_id: str = "") -> None:
    _request_id_ctx.set(request_id)
    _user_id_ctx.set(user_id)


def get_request_id() -> str:
    return _request_id_ctx.get()


def get_context_user_id() -> str:
    return _user_id_ctx.get()


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = _request_id_ctx.get()
        if request_id:
            log_entry["request_id"] = request_id

        user_id = _user_id_ctx.get()
        if user_id:
            log_entry["user_id"] = user_id

        # Merge extra fields attached via log.info("msg", extra={"key": "val"})
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                log_entry[key] = value

        if record.exc_info:
            log_entry["traceback"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_entry["traceback"] = record.exc_text

        return json.dumps(log_entry, default=str)


def _configure_root_logger(level: str) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class _BoundLogger:
    """Thin wrapper that pre-attaches keyword fields to every log call."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        error_code: ErrorCode | None = kwargs.pop("error_code", None)
        if error_code is not None:
            kwargs["error_code"] = error_code.value if isinstance(error_code, ErrorCode) else error_code
        exc_info = kwargs.pop("exc_info", False)
        self._logger.log(level, msg, extra=kwargs, exc_info=exc_info)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, **kwargs)


def get_logger(name: str) -> _BoundLogger:
    return _BoundLogger(logging.getLogger(name))


def init_logging(level: str = "INFO") -> None:
    _configure_root_logger(level)
