"""
Kafka Producer Adapter pattern for NammaKelsa backend.

KafkaProducerAdapter is the abstract interface — no business logic depends on
AIOKafka internals directly. This makes the Kafka layer testable (MockKafkaAdapter)
and swappable (e.g. Confluent Kafka adapter) without touching any business code.

Concrete implementations:
  AIOKafkaProducerAdapter  — production, backed by aiokafka
  MockKafkaProducerAdapter — testing / local dev without Kafka running
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from core.exceptions import KafkaNotConnectedError, KafkaSendFailedError
from core.logging import get_logger

log = get_logger(__name__)


class KafkaProducerAdapter(ABC):
    """Abstract Kafka producer interface. All Kafka sends go through this."""

    @abstractmethod
    async def start(self) -> None:
        """Connect to Kafka brokers. Called once at app startup."""

    @abstractmethod
    async def stop(self) -> None:
        """Flush and disconnect. Called on app shutdown."""

    @abstractmethod
    async def send_bytes(self, topic: str, key: bytes, value: bytes) -> None:
        """Send a raw bytes message to a Kafka topic."""

    @abstractmethod
    async def send_json(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        """Serialize payload to JSON and send to topic."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the producer is ready to send."""


class AIOKafkaProducerAdapter(KafkaProducerAdapter):
    """
    Production Kafka adapter backed by aiokafka.

    A single AIOKafkaProducer is held as a singleton inside this adapter.
    start() / stop() are called by FastAPI lifespan events.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer: Any = None  # aiokafka.AIOKafkaProducer
        self._connected = False

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer  # type: ignore[import]

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=None,  # we handle serialization ourselves
                compression_type="gzip",
                request_timeout_ms=30_000,
                retry_backoff_ms=200,
            )
            await self._producer.start()
            self._connected = True
            log.info(
                "Kafka producer connected",
                bootstrap_servers=self._bootstrap_servers,
            )
        except Exception as exc:
            log.error(
                "Kafka producer failed to connect",
                bootstrap_servers=self._bootstrap_servers,
                exc_info=True,
            )
            raise KafkaSendFailedError("__startup__", str(exc)) from exc

    async def stop(self) -> None:
        if self._producer:
            try:
                await self._producer.stop()
                log.info("Kafka producer disconnected cleanly")
            except Exception:
                log.warning("Kafka producer stop raised an error", exc_info=True)
            finally:
                self._connected = False
                self._producer = None

    async def send_bytes(self, topic: str, key: bytes, value: bytes) -> None:
        if not self._connected or self._producer is None:
            raise KafkaNotConnectedError()
        try:
            await self._producer.send_and_wait(topic, value=value, key=key)
            log.debug(
                "Kafka message sent",
                topic=topic,
                key=key.decode("utf-8", errors="replace"),
                bytes_sent=len(value),
            )
        except Exception as exc:
            log.error(
                "Kafka send_bytes failed",
                topic=topic,
                exc_info=True,
            )
            raise KafkaSendFailedError(topic, str(exc)) from exc

    async def send_json(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        value = json.dumps(payload).encode("utf-8")
        await self.send_bytes(topic, key.encode("utf-8"), value)

    def is_connected(self) -> bool:
        return self._connected


class MockKafkaProducerAdapter(KafkaProducerAdapter):
    """
    In-memory Kafka adapter for testing and local dev.

    Messages are stored in self.messages[topic] so tests can assert on them.
    start() / stop() are no-ops.
    """

    def __init__(self) -> None:
        self.messages: dict[str, list[dict]] = defaultdict(list)
        self._connected = False

    async def start(self) -> None:
        self._connected = True
        log.info("MockKafkaProducerAdapter started (no-op)")

    async def stop(self) -> None:
        self._connected = False
        log.info("MockKafkaProducerAdapter stopped")

    async def send_bytes(self, topic: str, key: bytes, value: bytes) -> None:
        self.messages[topic].append({"key": key, "value": value})
        log.debug(
            "MockKafka message stored",
            topic=topic,
            key=key.decode("utf-8", errors="replace"),
            bytes=len(value),
        )

    async def send_json(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        self.messages[topic].append({"key": key, "payload": payload})
        log.debug("MockKafka JSON message stored", topic=topic, key=key)

    def is_connected(self) -> bool:
        return self._connected

    def get_messages(self, topic: str) -> list[dict]:
        return self.messages.get(topic, [])

    def clear(self) -> None:
        self.messages.clear()


def build_kafka_adapter(backend: str, bootstrap_servers: str) -> KafkaProducerAdapter:
    """
    Factory that builds a KafkaProducerAdapter based on the config value.

    backend = "aiokafka" → AIOKafkaProducerAdapter
    backend = "mock"     → MockKafkaProducerAdapter
    """
    if backend == "mock":
        log.info("Using MockKafkaProducerAdapter (no real Kafka connection)")
        return MockKafkaProducerAdapter()
    log.info("Using AIOKafkaProducerAdapter", bootstrap_servers=bootstrap_servers)
    return AIOKafkaProducerAdapter(bootstrap_servers)
