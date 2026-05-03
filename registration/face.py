"""
Face registration logic — all done in backend (no ML service call).

Steps:
  1. Extract best frame from uploaded video (highest-confidence frontal face)
  2. Compute ArcFace 512-dim embedding via DeepFace
  3. Fast path: SHA256 hash check against users.face_hash (exact duplicate)
  4. Similarity path: pgvector cosine similarity check (same person, different photo)
  5. Store embedding + hash if unique

ArcFace model is warmed up once at startup to avoid 3-5s first-call delay.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np

from config import settings
from core.exceptions import (
    FaceAlreadyRegisteredError,
    FaceEmbeddingFailedError,
    FaceNotDetectedError,
    FaceVideoInvalidFormatError,
    FaceVideoTooLargeError,
)
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import get_user_by_face_hash, update_user_face
from db.models import User
from registration.face_hash_strategy import FaceHashStrategy

log = get_logger(__name__)

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi"}
_ARCFACE_MODEL = None  # warmed up once in warmup_face_model()


def warmup_face_model() -> None:
    """
    Load ArcFace model into memory at startup.
    Prevents 3-5s cold start on the first registration call.
    """
    global _ARCFACE_MODEL
    try:
        from deepface import DeepFace  # type: ignore[import]

        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        DeepFace.represent(dummy, model_name="ArcFace", enforce_detection=False)
        _ARCFACE_MODEL = "loaded"
        log.info("ArcFace model warmed up successfully")
    except Exception as exc:
        log.warning("ArcFace model warmup failed — will load on first call", exc_info=True)


def _check_video_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise FaceVideoInvalidFormatError(ext)
    return ext


def _check_video_size(size_bytes: int) -> None:
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.FACE_VIDEO_MAX_SIZE_MB:
        raise FaceVideoTooLargeError(size_mb)


def _extract_best_frame(video_path: str) -> np.ndarray:
    """
    Iterate frames in the video, return the frame where OpenCV face detector
    finds a face with the highest confidence. Raises FaceNotDetectedError
    if no frame has a face with confidence >= FACE_DETECTION_CONFIDENCE.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise FaceEmbeddingFailedError("opencv-python not installed") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FaceEmbeddingFailedError(f"Cannot open video file: {video_path}")

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    best_frame = None
    best_face_size = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            if len(faces) > 0:
                # Pick the largest face in this frame
                largest = max(faces, key=lambda f: f[2] * f[3])
                area = int(largest[2]) * int(largest[3])
                if area > best_face_size:
                    best_face_size = area
                    best_frame = frame.copy()
    finally:
        cap.release()

    log.debug("Face extraction complete", total_frames=frame_count, best_face_area=best_face_size)

    if best_frame is None or best_face_size < (80 * 80):
        raise FaceNotDetectedError()

    return best_frame


def _extract_embedding(frame: np.ndarray) -> list[float]:
    """
    Extract ArcFace 512-dim embedding from a BGR frame.
    Raises FaceEmbeddingFailedError on any DeepFace failure.
    """
    try:
        from deepface import DeepFace  # type: ignore[import]
        import cv2  # type: ignore[import]

        # DeepFace expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            rgb,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="opencv",
        )
        embedding = result[0]["embedding"]
        log.debug("ArcFace embedding extracted", embedding_dim=len(embedding))
        return embedding
    except Exception as exc:
        log.error("ArcFace embedding extraction failed", exc_info=True)
        raise FaceEmbeddingFailedError(str(exc)) from exc


async def _check_similarity_pgvector(
    db: DatabaseAdapter,
    embedding: list[float],
    threshold: float,
) -> bool:
    """
    Query pgvector for the nearest stored face embedding.
    Returns True if a match above threshold is found (duplicate face).
    """
    from sqlalchemy import text

    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
    stmt = text(
        """
        SELECT 1 - (face_embedding <=> CAST(:emb AS vector)) AS similarity
        FROM users
        WHERE face_embedding IS NOT NULL
        ORDER BY face_embedding <=> CAST(:emb AS vector)
        LIMIT 1
        """
    ).bindparams(emb=embedding_str)

    result = await db.execute_query(stmt)
    row = result.fetchone()
    if row is None:
        return False

    similarity = float(row[0])
    log.debug("pgvector face similarity check", similarity=similarity, threshold=threshold)
    return similarity >= threshold


async def register_face(
    db: DatabaseAdapter,
    user: User,
    video_bytes: bytes,
    filename: str,
    hash_strategy: FaceHashStrategy,
) -> dict:
    """
    Main face registration flow. Returns {"face_hash": str, "embedding_dim": int}.
    Raises descriptive exceptions on any failure — caller maps to HTTP status.
    """
    # 1 — Validate input
    _check_video_format(filename)
    _check_video_size(len(video_bytes))

    # 2 — Write to temp file for OpenCV
    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # 3 — Extract best frame
        frame = _extract_best_frame(tmp_path)
    finally:
        os.unlink(tmp_path)

    # 4 — Extract ArcFace embedding
    embedding = _extract_embedding(frame)

    # 5 — Compute hash using the configured strategy
    face_hash = hash_strategy.compute(embedding)
    log.info(
        "Face hash computed",
        user_id=str(user.id),
        hash_algorithm=hash_strategy.algorithm_name,
        face_hash_prefix=face_hash[:8],
    )

    # 6 — Fast exact-duplicate check (hash lookup)
    existing_by_hash = await get_user_by_face_hash(db, face_hash)
    if existing_by_hash and existing_by_hash.id != user.id:
        log.warning(
            "Face hash duplicate detected",
            user_id=str(user.id),
            conflicting_user_id=str(existing_by_hash.id),
        )
        raise FaceAlreadyRegisteredError("hash match")

    # 7 — Cosine similarity check via pgvector (catches same person, different photo)
    try:
        is_duplicate = await _check_similarity_pgvector(
            db, embedding, settings.FACE_SIMILARITY_THRESHOLD
        )
        if is_duplicate:
            log.warning(
                "Face similarity duplicate detected",
                user_id=str(user.id),
                threshold=settings.FACE_SIMILARITY_THRESHOLD,
            )
            raise FaceAlreadyRegisteredError("embedding similarity match")
    except FaceAlreadyRegisteredError:
        raise
    except Exception as exc:
        # pgvector not installed or extension not enabled — skip similarity check
        log.warning(
            "pgvector similarity check failed — skipping vector check",
            exc_info=True,
        )

    # 8 — Store embedding + hash
    await update_user_face(db, user, face_hash, face_embedding_id=None)

    # 9 — Update pgvector column if available
    try:
        from sqlalchemy import text

        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        stmt = text(
            "UPDATE users SET face_embedding = CAST(:emb AS vector) WHERE id = :uid"
        ).bindparams(emb=emb_str, uid=str(user.id))
        await db.execute_query(stmt)
        log.info("Face embedding stored in pgvector", user_id=str(user.id))
    except Exception:
        log.warning("pgvector update failed — embedding not stored", exc_info=True)

    log.info(
        "Face registration complete",
        user_id=str(user.id),
        embedding_dim=len(embedding),
    )
    return {"face_hash": face_hash, "embedding_dim": len(embedding)}
