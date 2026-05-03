"""
Document routes:
  POST /documents/degree              — upload degree image (sync ML check)
  POST /documents/workexp/init        — init chunked video upload
  POST /documents/workexp/chunk       — send one chunk
  GET  /documents/workexp/status/{id} — check upload progress
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from auth.middleware import get_registered_user
from core.exceptions import (
    ChunksMissingError,
    FileTooLargeError,
    InvalidFileFormatError,
    UploadAlreadyCompleteError,
    UploadNotFoundError,
)
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import create_document, get_application_by_id
from db.session import get_db
from documents.chunked_upload import (
    get_upload_status,
    init_upload,
    receive_chunk,
)
from documents.degree import check_degree_with_ml
from pipeline.kafka_producer import send_video_to_kafka

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_DEGREE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/degree", status_code=200)
async def upload_degree(
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
    application_id: str = Form(...),
    image: UploadFile = File(...),
) -> dict:
    from pathlib import Path

    ext = Path(image.filename or "").suffix.lower()
    if ext not in _ALLOWED_DEGREE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {ext}. Allowed: jpg, jpeg, png, webp",
        )

    user_id = payload["sub"]
    app = await get_application_by_id(db, application_id)
    if app is None or str(app.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Application not found")

    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Degree image must be under 10MB")

    try:
        result = await check_degree_with_ml(user_id, image_bytes, image.filename or "degree.jpg")
    except Exception as exc:
        log.error("Degree ML check failed", user_id=user_id, exc_info=True)
        raise HTTPException(status_code=503, detail="Degree check service unavailable")

    # Save file to disk
    from pathlib import Path
    import uuid

    save_path = Path(f"/tmp/backend/degree/{user_id}_{uuid.uuid4()}{ext}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(image_bytes)

    doc = await create_document(db, user_id, application_id, "DEGREE", str(save_path))

    log.info("Degree uploaded and checked", user_id=user_id, document_id=str(doc.id))
    return {
        "document_id": str(doc.id),
        "is_valid_doc": result.get("is_valid_doc"),
        "extracted_degree": result.get("extracted_degree"),
        "extracted_institution": result.get("extracted_institution"),
    }


class InitUploadRequest(BaseModel):
    application_id: str
    total_chunks: int
    file_size_bytes: int
    file_name: str


@router.post("/workexp/init", status_code=200)
async def init_workexp_upload(
    body: InitUploadRequest,
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
) -> dict:
    try:
        upload_id = await init_upload(
            body.application_id,
            body.total_chunks,
            body.file_size_bytes,
            body.file_name,
        )
    except (FileTooLargeError, InvalidFileFormatError) as exc:
        raise HTTPException(status_code=400, detail=exc.message)

    return {
        "upload_id": upload_id,
        "chunk_size_bytes": 65536,
        "resume_from": 0,
    }


@router.post("/workexp/chunk", status_code=200)
async def upload_workexp_chunk(
    payload: Annotated[dict, Depends(get_registered_user)],
    db: Annotated[DatabaseAdapter, Depends(get_db)],
    upload_id: str = Form(...),
    chunk_seq: int = Form(...),
    is_last: bool = Form(False),
    file: UploadFile = File(...),
) -> dict:
    chunk_bytes = await file.read()
    user_id = payload["sub"]

    try:
        result = await receive_chunk(upload_id, chunk_seq, chunk_bytes, is_last)
    except UploadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
    except UploadAlreadyCompleteError as exc:
        raise HTTPException(status_code=409, detail=exc.message)
    except ChunksMissingError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": exc.message, "missing_chunks": exc.detail["missing_chunks"]},
        )
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=exc.message)

    # If assembly is complete, send to Kafka and record document
    if result.get("status") == "uploaded":
        session = await __import__(
            "cache.redis_client", fromlist=["get_upload_session"]
        ).get_upload_session(upload_id)
        if session:
            assembled_path = result.get("assembled_path", "")
            try:
                await send_video_to_kafka(
                    candidate_id=user_id,
                    topic="candidate.workexp.video",
                    video_path=assembled_path,
                )
                log.info("Workexp video sent to Kafka", user_id=user_id, upload_id=upload_id)
            except Exception:
                log.error("Workexp Kafka send failed", user_id=user_id, exc_info=True)

            # Save document record
            if session.get("application_id"):
                try:
                    doc = await create_document(
                        db, user_id, session["application_id"], "WORKEXP_VIDEO", assembled_path
                    )
                    result["document_id"] = str(doc.id)
                except Exception:
                    log.error("Document record creation failed", user_id=user_id, exc_info=True)

    return result


@router.get("/workexp/status/{upload_id}", status_code=200)
async def workexp_upload_status(
    upload_id: str,
    payload: Annotated[dict, Depends(get_registered_user)],
) -> dict:
    try:
        return await get_upload_status(upload_id)
    except UploadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
