"""
Face Hash Strategy Pattern for NammaKelsa backend.

FaceHashStrategy is the abstract interface. The concrete strategy is selected
at startup via FACE_HASH_STRATEGY config and used throughout the backend.

Why a strategy here? The hash is stored in DB and compared on every registration.
If we need to migrate to a faster or more collision-resistant algorithm, swapping
the strategy is a single config change.

Concrete strategies:
  SHA256FaceHashStrategy  — default, 64-char hex, strong collision resistance
  Blake2FaceHashStrategy  — faster than SHA256, equally secure for this purpose
  MD5FaceHashStrategy     — NOT recommended for production, provided for testing only
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from config import settings
from core.logging import get_logger

log = get_logger(__name__)


class FaceHashStrategy(ABC):
    """
    Abstract strategy for hashing ArcFace embeddings.

    Input:  embedding — list of 512 floats produced by ArcFace
    Output: hex string stored in users.face_hash (used for exact-match fast path)
    """

    @abstractmethod
    def compute(self, embedding: list[float]) -> str:
        """Compute hash from a 512-dim float embedding list."""

    @abstractmethod
    def compute_from_bytes(self, embedding_bytes: bytes) -> str:
        """Compute hash directly from the raw float32 bytes."""

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Human-readable name, used in logs."""


class SHA256FaceHashStrategy(FaceHashStrategy):
    """SHA-256 of the float32 binary representation. Default strategy."""

    def compute(self, embedding: list[float]) -> str:
        raw = np.array(embedding, dtype=np.float32).tobytes()
        return self.compute_from_bytes(raw)

    def compute_from_bytes(self, embedding_bytes: bytes) -> str:
        return hashlib.sha256(embedding_bytes).hexdigest()

    @property
    def algorithm_name(self) -> str:
        return "sha256"


class Blake2FaceHashStrategy(FaceHashStrategy):
    """
    BLAKE2b with 32-byte digest (64 hex chars, same length as SHA-256).
    Faster than SHA-256 on modern CPUs. Equally collision-resistant for this use.
    """

    def compute(self, embedding: list[float]) -> str:
        raw = np.array(embedding, dtype=np.float32).tobytes()
        return self.compute_from_bytes(raw)

    def compute_from_bytes(self, embedding_bytes: bytes) -> str:
        return hashlib.blake2b(embedding_bytes, digest_size=32).hexdigest()

    @property
    def algorithm_name(self) -> str:
        return "blake2"


class MD5FaceHashStrategy(FaceHashStrategy):
    """
    MD5 — fast but weak. Only suitable for dev/testing environments.
    Do NOT use in production: MD5 has known collision vulnerabilities.
    """

    def compute(self, embedding: list[float]) -> str:
        raw = np.array(embedding, dtype=np.float32).tobytes()
        return self.compute_from_bytes(raw)

    def compute_from_bytes(self, embedding_bytes: bytes) -> str:
        return hashlib.md5(embedding_bytes).hexdigest()  # nosec: intentional dev-only

    @property
    def algorithm_name(self) -> str:
        return "md5"


def build_face_hash_strategy() -> FaceHashStrategy:
    """
    Factory: build FaceHashStrategy from config.

    FACE_HASH_STRATEGY=sha256  → SHA256FaceHashStrategy  (default)
    FACE_HASH_STRATEGY=blake2  → Blake2FaceHashStrategy
    FACE_HASH_STRATEGY=md5     → MD5FaceHashStrategy (dev only)
    """
    name = settings.FACE_HASH_STRATEGY.lower()
    if name == "blake2":
        log.info("Using Blake2FaceHashStrategy")
        return Blake2FaceHashStrategy()
    elif name == "md5":
        log.warning(
            "MD5FaceHashStrategy in use — NOT recommended for production",
            strategy=name,
        )
        return MD5FaceHashStrategy()
    else:
        log.info("Using SHA256FaceHashStrategy (default)")
        return SHA256FaceHashStrategy()
