"""
In-memory buffer for interview audio and video frames.

Audio: raw PCM chunks (bytes) buffered per session
Video: JPEG frame bytes buffered per session

After interview ends:
  - PCM chunks concatenated → WAV file
  - JPEG frames → MP4 via cv2.VideoWriter

Both are held in memory during the ~90-120s interview (~6MB total per session).
"""
from __future__ import annotations

import io
import struct
import wave
from collections import defaultdict
from typing import NamedTuple

from core.logging import get_logger

log = get_logger(__name__)


class _AudioBuffer:
    def __init__(self) -> None:
        self._chunks: list[tuple[int, bytes]] = []  # (seq, pcm_bytes)

    def add(self, seq: int, pcm: bytes) -> None:
        self._chunks.append((seq, pcm))

    def to_wav_bytes(self, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
        sorted_chunks = sorted(self._chunks, key=lambda t: t[0])
        raw_pcm = b"".join(data for _, data in sorted_chunks)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_pcm)
        return buf.getvalue()

    def total_bytes(self) -> int:
        return sum(len(data) for _, data in self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


class _VideoBuffer:
    def __init__(self) -> None:
        self._frames: list[tuple[int, bytes]] = []  # (seq, jpeg_bytes)

    def add(self, seq: int, jpeg: bytes) -> None:
        self._frames.append((seq, jpeg))

    def to_mp4_bytes(self, fps: int = 1, width: int = 640, height: int = 480) -> bytes:
        try:
            import cv2  # type: ignore[import]
            import numpy as np
            import tempfile
            import os

            sorted_frames = sorted(self._frames, key=lambda t: t[0])

            tmp_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

            for _, jpeg_bytes in sorted_frames:
                buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Resize to target dimensions if needed
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)

            writer.release()
            mp4_bytes = open(tmp_path, "rb").read()
            os.unlink(tmp_path)
            return mp4_bytes
        except ImportError:
            log.warning("opencv-python not available — returning raw JPEG concatenation")
            return b"".join(data for _, data in sorted(self._frames, key=lambda t: t[0]))

    def total_bytes(self) -> int:
        return sum(len(data) for _, data in self._frames)

    def frame_count(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()


class SessionBuffer:
    """Holds audio and video buffers for a single interview session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.audio = _AudioBuffer()
        self.video = _VideoBuffer()

    def memory_usage_bytes(self) -> int:
        return self.audio.total_bytes() + self.video.total_bytes()

    def clear(self) -> None:
        self.audio.clear()
        self.video.clear()
        log.debug("Session buffer cleared", session_id=self.session_id)


# Global registry: session_id → SessionBuffer
_buffers: dict[str, SessionBuffer] = {}


def get_or_create_buffer(session_id: str) -> SessionBuffer:
    if session_id not in _buffers:
        _buffers[session_id] = SessionBuffer(session_id)
        log.debug("Session buffer created", session_id=session_id)
    return _buffers[session_id]


def release_buffer(session_id: str) -> None:
    buf = _buffers.pop(session_id, None)
    if buf:
        buf.clear()
        log.debug("Session buffer released", session_id=session_id)
