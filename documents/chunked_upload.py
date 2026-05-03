"""
Chunked workexp video upload — reassembly and Kafka dispatch.

Frontend sends 64KB chunks over HTTP (low-bandwidth friendly).
Backend reassembles to full file, then re-chunks to 256KB for Kafka.
Upload state (received chunks, status) is tracked in Redis.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cache.redis_client import (
    delete_upload_session,
    get_upload_session,
    init_upload_session,
    update_upload_session,
)
from config import settings
from core.exceptions import (
    ChunksMissingError,
    FileTooLargeError,
    InvalidFileFormatError,
    UploadAlreadyCompleteError,
    UploadNotFoundError,
)
from core.logging import get_logger

log = get_logger(__name__)

_ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi"}
MAX_VIDEO_BYTES = settings.MAX_WORKEXP_VIDEO_SIZE_MB * 1024 * 1024


def _upload_dir(upload_id: str) -> Path:
    base = Path(settings.LOCAL_STORAGE_DIR) / "workexp" / upload_id
    base.mkdir(parents=True, exist_ok=True)
    return base


async def init_upload(
    application_id: str,
    total_chunks: int,
    file_size_bytes: int,
    file_name: str,
) -> str:
    ext = Path(file_name).suffix.lower()
    if ext not in _ALLOWED_VIDEO_EXTS:
        raise InvalidFileFormatError(ext, list(_ALLOWED_VIDEO_EXTS))

    if file_size_bytes > MAX_VIDEO_BYTES:
        raise FileTooLargeError(
            file_size_bytes / (1024 * 1024), settings.MAX_WORKEXP_VIDEO_SIZE_MB
        )

    upload_id = str(uuid.uuid4())
    data = {
        "upload_id": upload_id,
        "application_id": application_id,
        "total_chunks": total_chunks,
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "received": [],
        "status": "in_progress",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    await init_upload_session(upload_id, data)
    _upload_dir(upload_id)  # create directory now
    log.info(
        "Upload session initialized",
        upload_id=upload_id,
        application_id=application_id,
        total_chunks=total_chunks,
        file_size_bytes=file_size_bytes,
    )
    return upload_id


async def receive_chunk(
    upload_id: str,
    chunk_seq: int,
    chunk_bytes: bytes,
    is_last: bool,
) -> dict:
    session = await get_upload_session(upload_id)
    if session is None:
        raise UploadNotFoundError(upload_id)

    if session["status"] == "complete":
        raise UploadAlreadyCompleteError(upload_id)

    # Idempotent — skip duplicate chunks
    if chunk_seq in session["received"]:
        log.debug("Duplicate chunk ignored", upload_id=upload_id, chunk_seq=chunk_seq)
        return {"chunk_seq": chunk_seq, "status": "duplicate", "next_expected": chunk_seq + 1}

    # Write chunk to disk
    chunk_path = _upload_dir(upload_id) / f"{chunk_seq:05d}.bin"
    chunk_path.write_bytes(chunk_bytes)

    session["received"].append(chunk_seq)
    await update_upload_session(upload_id, session)

    log.debug(
        "Chunk received",
        upload_id=upload_id,
        chunk_seq=chunk_seq,
        bytes=len(chunk_bytes),
        is_last=is_last,
    )

    if not is_last:
        return {
            "chunk_seq": chunk_seq,
            "status": "received",
            "next_expected": chunk_seq + 1,
        }

    # is_last=True — attempt assembly
    return await _assemble(upload_id, session)


async def _assemble(upload_id: str, session: dict) -> dict:
    total = session["total_chunks"]
    received = sorted(session["received"])
    expected = list(range(total))

    missing = [c for c in expected if c not in received]
    if missing:
        log.warning(
            "Assembly attempted with missing chunks",
            upload_id=upload_id,
            missing=missing,
        )
        raise ChunksMissingError(missing)

    upload_path = _upload_dir(upload_id)
    assembled_path = upload_path / "assembled.mp4"

    total_bytes = 0
    with assembled_path.open("wb") as out:
        for seq in expected:
            chunk_data = (upload_path / f"{seq:05d}.bin").read_bytes()
            out.write(chunk_data)
            total_bytes += len(chunk_data)

    if total_bytes > MAX_VIDEO_BYTES:
        assembled_path.unlink(missing_ok=True)
        log.warning(
            "Assembled file exceeds max size",
            upload_id=upload_id,
            bytes=total_bytes,
        )
        raise FileTooLargeError(total_bytes / (1024 * 1024), settings.MAX_WORKEXP_VIDEO_SIZE_MB)

    # Delete individual chunk files
    for seq in expected:
        (upload_path / f"{seq:05d}.bin").unlink(missing_ok=True)

    session["status"] = "complete"
    session["assembled_path"] = str(assembled_path)
    await update_upload_session(upload_id, session)

    log.info(
        "Workexp video assembled",
        upload_id=upload_id,
        total_bytes=total_bytes,
        assembled_path=str(assembled_path),
    )
    return {
        "status": "uploaded",
        "assembled_path": str(assembled_path),
        "total_bytes": total_bytes,
    }


async def get_upload_status(upload_id: str) -> dict:
    session = await get_upload_session(upload_id)
    if session is None:
        raise UploadNotFoundError(upload_id)

    total = session["total_chunks"]
    received = sorted(session["received"])
    missing = [c for c in range(total) if c not in received]

    return {
        "upload_id": upload_id,
        "status": session["status"],
        "total_chunks": total,
        "received_chunks": received,
        "missing": missing,
    }
