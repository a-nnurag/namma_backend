"""
Kafka producer singleton for the backend.

The adapter instance is initialized once in main.py lifespan and stored here
as a module-level variable so any module can import and use it directly.
"""
from kafka.adapter import KafkaProducerAdapter

# Populated in main.py lifespan startup
_producer: KafkaProducerAdapter | None = None


def set_producer(adapter: KafkaProducerAdapter) -> None:
    global _producer
    _producer = adapter


def get_producer() -> KafkaProducerAdapter:
    if _producer is None:
        raise RuntimeError("Kafka producer not initialized. Call set_producer() first.")
    return _producer
