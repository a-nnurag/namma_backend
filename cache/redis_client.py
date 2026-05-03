"""
Redis client and key helpers for NammaKelsa backend.

All Redis key patterns are defined as constants here — no raw key strings
in business logic. TTLs are explicit on every operation.

Key namespaces:
  otp:{phone}                  — OTP value + wrong-attempt count
  ratelimit:otp:{phone}        — OTP send rate limit counter
  session:{user_id}            — JWT session cache
  ml_status:{application_id}  — ML pipeline status cache (from poller)
  cooldown:{user_id}:{skill_id}— Cooldown timestamp for fast skill selection check
  upload:{upload_id}           — Chunked upload state

Pub/Sub channels (Redis pub/sub is global — not DB-scoped):
  ml_result:{application_id}  — ML service publishes terminal status here
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import redis.asyncio as aioredis

from config import settings
from core.exceptions import RedisConnectionError
from core.logging import get_logger

log = get_logger(__name__)

# TTLs (seconds)
OTP_TTL = 600             # 10 min
OTP_RATE_LIMIT_TTL = 3600 # 1 hour
SESSION_TTL = 86400       # 24 hours
ML_STATUS_TTL = 60        # 1 min (refreshed by poller every 30s)
UPLOAD_TTL = 86400        # 24 hours (resume window)

_redis: aioredis.Redis | None = None


def _key_otp(phone: str) -> str:
    return f"otp:{phone}"

def _key_ratelimit(phone: str) -> str:
    return f"ratelimit:otp:{phone}"

def _key_session(user_id: str) -> str:
    return f"session:{user_id}"

def _key_ml_status(application_id: str) -> str:
    return f"ml_status:{application_id}"

def _key_cooldown(user_id: str, skill_id: str) -> str:
    return f"cooldown:{user_id}:{skill_id}"

def _key_upload(upload_id: str) -> str:
    return f"upload:{upload_id}"


async def init_redis() -> None:
    global _redis
    try:
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await _redis.ping()
        log.info("Redis connection established", db=settings.REDIS_DB)
    except Exception as exc:
        log.critical("Redis connection failed at startup", exc_info=True)
        raise RedisConnectionError(str(exc)) from exc


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        log.info("Redis connection closed")


def _get_client() -> aioredis.Redis:
    if _redis is None:
        raise RedisConnectionError("Redis client not initialized. Call init_redis() first.")
    return _redis


# ── OTP ───────────────────────────────────────────────────────────────────────

async def store_otp(phone: str, otp: str) -> None:
    r = _get_client()
    payload = json.dumps({"otp": otp, "attempts": 0})
    await r.set(_key_otp(phone), payload, ex=OTP_TTL)
    log.info("OTP stored in Redis", phone=phone)


async def get_otp_data(phone: str) -> dict | None:
    r = _get_client()
    raw = await r.get(_key_otp(phone))
    if raw is None:
        return None
    return json.loads(raw)


async def increment_otp_attempts(phone: str) -> int:
    """Increment wrong OTP attempt counter and return new count."""
    r = _get_client()
    key = _key_otp(phone)
    raw = await r.get(key)
    if raw is None:
        return 0
    data = json.loads(raw)
    data["attempts"] += 1
    ttl = await r.ttl(key)
    await r.set(key, json.dumps(data), ex=max(ttl, 1))
    return data["attempts"]


async def delete_otp(phone: str) -> None:
    r = _get_client()
    await r.delete(_key_otp(phone))
    log.debug("OTP deleted from Redis", phone=phone)


async def check_otp_rate_limit(phone: str, max_sends: int = 3) -> bool:
    """Return True if allowed to send OTP, False if rate-limited."""
    r = _get_client()
    key = _key_ratelimit(phone)
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, OTP_RATE_LIMIT_TTL)
    log.debug("OTP rate limit check", phone=phone, count=count, max=max_sends)
    return count <= max_sends


# ── Session ───────────────────────────────────────────────────────────────────

async def cache_session(user_id: str, payload: dict) -> None:
    r = _get_client()
    await r.set(_key_session(user_id), json.dumps(payload), ex=SESSION_TTL)
    log.debug("Session cached", user_id=user_id)


async def get_session(user_id: str) -> dict | None:
    r = _get_client()
    raw = await r.get(_key_session(user_id))
    if raw is None:
        return None
    return json.loads(raw)


async def delete_session(user_id: str) -> None:
    r = _get_client()
    await r.delete(_key_session(user_id))
    log.info("Session deleted from cache", user_id=user_id)


async def refresh_session_ttl(user_id: str) -> None:
    r = _get_client()
    await r.expire(_key_session(user_id), SESSION_TTL)


# ── ML Status ─────────────────────────────────────────────────────────────────

async def set_ml_status(application_id: str, payload: dict) -> None:
    r = _get_client()
    await r.set(_key_ml_status(application_id), json.dumps(payload), ex=ML_STATUS_TTL)


async def get_ml_status(application_id: str) -> dict | None:
    r = _get_client()
    raw = await r.get(_key_ml_status(application_id))
    if raw is None:
        return None
    return json.loads(raw)


async def delete_ml_status(application_id: str) -> None:
    r = _get_client()
    await r.delete(_key_ml_status(application_id))


# ── ML Result Pub/Sub ─────────────────────────────────────────────────────────

async def subscribe_ml_result(
    application_id: str,
    timeout: float = 660.0,
) -> "AsyncGenerator[dict, None]":
    """
    Async generator that yields one dict when the ML service publishes
    a terminal status for this application, then stops.

    Sends a SSE heartbeat comment every 20 s to keep proxy connections alive.
    Times out after `timeout` seconds (default 11 min).
    """
    from typing import AsyncGenerator  # local import avoids circular at module level

    pubsub = _get_client().pubsub()
    channel = f"ml_result:{application_id}"
    await pubsub.subscribe(channel)

    deadline = asyncio.get_event_loop().time() + timeout
    listen_iter = pubsub.listen().__aiter__()

    try:
        while asyncio.get_event_loop().time() < deadline:
            try:
                message = await asyncio.wait_for(listen_iter.__anext__(), timeout=20.0)
            except asyncio.TimeoutError:
                # Yield a sentinel so the SSE generator can emit a heartbeat
                yield {"_heartbeat": True}
                continue
            except StopAsyncIteration:
                break

            if message.get("type") == "message":
                yield json.loads(message["data"])
                return
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()


# ── Cooldown ──────────────────────────────────────────────────────────────────

async def set_cooldown(user_id: str, skill_id: str, cooldown_until_iso: str, ttl_seconds: int) -> None:
    r = _get_client()
    await r.set(_key_cooldown(user_id, skill_id), cooldown_until_iso, ex=ttl_seconds)
    log.debug(
        "Cooldown cached",
        user_id=user_id,
        skill_id=skill_id,
        cooldown_until=cooldown_until_iso,
    )


async def get_cooldown(user_id: str, skill_id: str) -> str | None:
    r = _get_client()
    return await r.get(_key_cooldown(user_id, skill_id))


# ── Chunked Upload ────────────────────────────────────────────────────────────

async def init_upload_session(upload_id: str, data: dict) -> None:
    r = _get_client()
    await r.set(_key_upload(upload_id), json.dumps(data), ex=UPLOAD_TTL)
    log.info("Upload session initialized", upload_id=upload_id)


async def get_upload_session(upload_id: str) -> dict | None:
    r = _get_client()
    raw = await r.get(_key_upload(upload_id))
    if raw is None:
        return None
    return json.loads(raw)


async def update_upload_session(upload_id: str, data: dict) -> None:
    r = _get_client()
    key = _key_upload(upload_id)
    ttl = await r.ttl(key)
    await r.set(key, json.dumps(data), ex=max(ttl, 1))


async def delete_upload_session(upload_id: str) -> None:
    r = _get_client()
    await r.delete(_key_upload(upload_id))
    log.info("Upload session deleted", upload_id=upload_id)
