"""
High-level Kafka send functions — re-chunks media and produces to correct topics.

Sending order (as per plan):
  1. candidate.workexp.video  (if present)
  2. candidate.interview.audio
  3. candidate.interview.video
  4. candidate.meta            ← triggers ML pipeline, MUST be last
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from kafka.producer import get_producer

if TYPE_CHECKING:
    from interview.media_buffer import _AudioBuffer, _VideoBuffer

log = get_logger(__name__)

KAFKA_CHUNK_BYTES = settings.KAFKA_CHUNK_SIZE_BYTES  # 256KB

TOPIC_WORKEXP_VIDEO = "candidate.workexp.video"
TOPIC_INTERVIEW_AUDIO = "candidate.interview.audio"
TOPIC_INTERVIEW_VIDEO = "candidate.interview.video"
TOPIC_META = "candidate.meta"


def _chunk_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


async def _produce_chunks(
    candidate_id: str,
    topic: str,
    data: bytes,
) -> None:
    producer = get_producer()
    chunks = _chunk_bytes(data, KAFKA_CHUNK_BYTES)
    total = len(chunks)

    log.info(
        "Producing Kafka chunks",
        topic=topic,
        candidate_id=candidate_id,
        total_chunks=total,
        total_bytes=len(data),
    )

    for seq, chunk in enumerate(chunks):
        payload = {
            "candidate_id": candidate_id,
            "chunk_seq": seq,
            "is_last": seq == total - 1,
            "data": base64.b64encode(chunk).decode("ascii"),
        }
        await producer.send_json(topic, key=candidate_id, payload=payload)

    log.info(
        "Kafka chunks produced",
        topic=topic,
        candidate_id=candidate_id,
        chunks_sent=total,
    )


async def send_video_to_kafka(
    candidate_id: str,
    topic: str,
    video_path: str,
) -> None:
    """Send a video file from disk to a Kafka topic in 256KB chunks."""
    path = Path(video_path)
    if not path.exists():
        log.error("Video file not found for Kafka send", path=video_path)
        return

    video_bytes = path.read_bytes()
    await _produce_chunks(candidate_id, topic, video_bytes)


async def send_interview_media_to_kafka(
    candidate_id: str,
    session_id: str,
    application_id: str,
    skill_name: str,
    audio_buffer: "_AudioBuffer",
    video_buffer: "_VideoBuffer",
    has_degree: bool = False,
    has_workexp: bool = False,
    language: str = "kn",
) -> None:
    """
    After interview ends: encode and send audio + video to Kafka, then send meta.
    Order matters: meta must be LAST.
    """
    # 1. Interview audio
    if audio_buffer.total_bytes() > 0:
        try:
            wav_bytes = audio_buffer.to_wav_bytes()
            await _produce_chunks(candidate_id, TOPIC_INTERVIEW_AUDIO, wav_bytes)
        except Exception:
            log.error("Failed to send interview audio to Kafka", candidate_id=candidate_id, exc_info=True)

    # 2. Interview video
    if video_buffer.frame_count() > 0:
        try:
            mp4_bytes = video_buffer.to_mp4_bytes()
            await _produce_chunks(candidate_id, TOPIC_INTERVIEW_VIDEO, mp4_bytes)
        except Exception:
            log.error("Failed to send interview video to Kafka", candidate_id=candidate_id, exc_info=True)

    # 3. Meta — triggers ML pipeline
    meta = {
        "candidate_id":   candidate_id,
        "has_workexp":    has_workexp,
        "has_degree":     has_degree,
        "claimed_role":   skill_name,
        "session_id":     session_id,
        "application_id": application_id,
        "language":       language,   # kn | hi | en | te | ta
    }
    try:
        producer = get_producer()
        await producer.send_json(TOPIC_META, key=candidate_id, payload=meta)
        log.info(
            "Meta message sent to Kafka — ML pipeline triggered",
            candidate_id=candidate_id,
            application_id=application_id,
        )
    except Exception:
        log.error("Failed to send meta message to Kafka", candidate_id=candidate_id, exc_info=True)
        raise
