from typing import Iterator, Optional
import json

from confluent_kafka import Consumer, KafkaError

from .base_stream import BaseStream
from .schema import WarehouseLog


class KafkaStream(BaseStream):
    """
    Kafka-backed ingestion stream for warehouse logs.

    Consumes messages from a Kafka topic and converts them into
    WarehouseLog objects for downstream processing.
    """

    def __init__(self, config: dict, topic: str) -> None:
        self._config = config
        self._topic = topic
        self._consumer: Optional[Consumer] = None
        self._running = False

    def connect(self) -> None:
        """
        Initialize Kafka consumer and subscribe to topic.
        """
        self._consumer = Consumer(self._config)
        self._consumer.subscribe([self._topic])
        self._running = True

    def stream(self) -> Iterator[WarehouseLog]:
        """
        Continuously poll Kafka and yield WarehouseLog objects.
        """
        if self._consumer is None:
            raise RuntimeError("KafkaStream must be connected before streaming.")

        while self._running:
            msg = self._consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                # Handle Kafka-level errors safely
                if msg.error().code() == KafkaError._PARTITION_EOF:  # type: ignore
                    continue  # normal condition
                else:
                    # Log and continue instead of crashing
                    print(f"[Kafka Error] {msg.error()}")
                    continue

            log = self._parse_message(msg.value())

            if log is not None:
                yield log

    def close(self) -> None:
        """
        Close Kafka consumer cleanly.
        """
        self._running = False

        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception as e:
                print(f"[Kafka Close Error] {e}")

    # -----------------------------
    # Internal helpers
    # -----------------------------

    @staticmethod
    def _parse_message(raw_bytes: bytes) -> Optional[WarehouseLog]:
        """
        Safely parse Kafka message into WarehouseLog.
        Returns None if message is invalid.
        """
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
            return WarehouseLog.from_dict(payload)

        except json.JSONDecodeError:
            print("[Parse Error] Invalid JSON message")
            return None

        except (ValueError, KeyError) as e:
            print(f"[Schema Error] {e}")
            return None

        except Exception as e:
            print(f"[Unexpected Error] {e}")
            return None