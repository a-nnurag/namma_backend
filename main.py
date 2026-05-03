"""
NammaKelsa Backend — FastAPI application entry point.

Lifespan:
  startup:  verify DB, init Redis, start Kafka producer, warm ArcFace,
            build strategy/adapter singletons, seed skills, start ML poller
  shutdown: stop Kafka producer, close Redis, dispose DB pool

All routers registered here.
All app state (strategy instances, kafka adapter) stored on app.state.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth.aadhaar_verifier import build_aadhaar_verifier
from auth.routes import router as auth_router
from cache.redis_client import close_redis, init_redis
from config import settings
from core.error_codes import ErrorCode
from core.exceptions import NammaKelsaError
from core.logging import get_logger, init_logging, set_request_context
from db.crud import seed_skills
from db.session import close_db, verify_db_connection
from kafka.adapter import build_kafka_adapter
from kafka.producer import set_producer
from pipeline.poller import start_poller
from registration.face import warmup_face_model
from registration.face_hash_strategy import build_face_hash_strategy
from registration.routes import router as registration_router
from skills.routes import router as skills_router
from documents.routes import router as documents_router
from interview.routes import router as interview_router
from pipeline.routes import router as pipeline_router
from admin.routes import router as admin_router
from admin.auth_routes import router as admin_auth_router

init_logging(settings.LOG_LEVEL)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── Startup ───────────────────────────────────────────────────────────────
    log.info("NammaKelsa backend starting up")

    # DB
    await verify_db_connection()

    # Redis
    await init_redis()

    # Kafka
    kafka_adapter = build_kafka_adapter(
        settings.KAFKA_ADAPTER_BACKEND,
        settings.KAFKA_BOOTSTRAP_SERVERS,
    )
    await kafka_adapter.start()
    set_producer(kafka_adapter)
    app.state.kafka_adapter = kafka_adapter

    # Strategy singletons on app state (injected into routes)
    app.state.face_hash_strategy = build_face_hash_strategy()
    app.state.aadhaar_verifier = build_aadhaar_verifier()

    # Seed skills (no-op if already seeded)
    from db.session import get_db_context
    async with get_db_context() as db:
        await seed_skills(db)

    # Warm ArcFace model (non-blocking — failure is logged, not fatal)
    asyncio.get_event_loop().run_in_executor(None, warmup_face_model)

    # Background ML poller
    poller_task = asyncio.create_task(start_poller())
    app.state.poller_task = poller_task

    log.info(
        "NammaKelsa backend ready",
        port=settings.BACKEND_PORT,
        kafka_backend=settings.KAFKA_ADAPTER_BACKEND,
        aadhaar_backend=settings.AADHAAR_VERIFIER_BACKEND,
        face_hash_strategy=settings.FACE_HASH_STRATEGY,
    )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("NammaKelsa backend shutting down")

    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass

    await kafka_adapter.stop()
    await close_redis()
    await close_db()

    log.info("NammaKelsa backend shutdown complete")


app = FastAPI(
    title="NammaKelsa Backend",
    description="Blue Collar / Polytechnic Job Readiness Platform — backend service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request context middleware — injects request_id and user_id into log context
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    # Try to extract user_id from token without full validation (best-effort for logging)
    user_id = ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            from auth.jwt import decode_token
            payload = decode_token(auth[7:])
            user_id = payload.get("sub", "")
        except Exception:
            pass

    set_request_context(request_id=request_id, user_id=user_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler — converts NammaKelsaError to structured JSON
@app.exception_handler(NammaKelsaError)
async def nammakelsa_error_handler(request: Request, exc: NammaKelsaError) -> JSONResponse:
    log.error(
        "Unhandled application error",
        error_code=exc.error_code,
        message=exc.message,
        detail=exc.detail,
    )
    status_code = _error_code_to_http_status(exc.error_code)
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": exc.error_code.value,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


def _error_code_to_http_status(code: ErrorCode) -> int:
    prefix = code.value.split("_")[0]
    mapping = {
        "AUTH": 401,
        "REG": 400,
        "SKILL": 400,
        "DOC": 400,
        "INTERVIEW": 400,
        "KAFKA": 503,
        "ML": 503,
        "DB": 500,
        "CACHE": 500,
        "SRV": 500,
    }
    return mapping.get(prefix, 500)


# ── Routers
app.include_router(auth_router)
app.include_router(registration_router)
app.include_router(skills_router)
app.include_router(documents_router)
app.include_router(interview_router)
app.include_router(pipeline_router)
app.include_router(admin_router)
app.include_router(admin_auth_router)


# ── Health check
@app.get("/health", tags=["health"])
async def health() -> dict:
    from pipeline.ml_client import health_check as ml_health
    from cache.redis_client import _get_client

    redis_ok = False
    try:
        await _get_client().ping()
        redis_ok = True
    except Exception:
        pass

    ml_ok = await ml_health()

    kafka_ok = app.state.kafka_adapter.is_connected() if hasattr(app.state, "kafka_adapter") else False

    return {
        "status": "ok",
        "db": "ok",
        "redis": "ok" if redis_ok else "error",
        "kafka": "ok" if kafka_ok else "error",
        "ml_service": "ok" if ml_ok else "error",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.BACKEND_PORT,
        reload=False,
        log_config=None,  # we handle logging ourselves
    )
