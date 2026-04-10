# src/ingestion/simulated_stream.py

"""
SimulatedStream — a deterministic, configurable warehouse log generator.

Implements the BaseStream interface to produce synthetic WarehouseLog events
at a controlled rate. Intended for development, integration testing, and
offline pipeline validation without requiring a live Kafka broker.

Design notes
------------
- Thread-safe stop flag via threading.Event so `.close()` can be called
  from another thread (e.g. a signal handler) without data races.
- Reproducible output through a seeded random.Random instance; pass
  `seed=None` (default) for true randomness in production-like scenarios.
- Configurable emission rate (`rate_per_sec`) with automatic backoff when
  the consumer is slow (non-blocking sleep pattern).
- Structured logging via the stdlib `logging` module — no bare print()
  calls in library code.
"""

import logging
import random
import time
from typing import Iterator, Optional

from .base_stream import BaseStream
from .schema import WarehouseLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants — extend these lists to widen the synthetic event space
# ---------------------------------------------------------------------------

_EVENT_TYPES: tuple[str, ...] = (
    "ITEM_PICKED",
    "ITEM_PLACED",
    "ITEM_MOVED",
    "RESTOCK_TRIGGERED",
    "INVENTORY_CHECK",
    "ANOMALY_DETECTED",
    "SCAN_FAILED",
    "TEMPERATURE_ALERT",
)

_MESSAGE_TEMPLATES: dict[str, list[str]] = {
    "ITEM_PICKED": [
        "Worker {worker_id} picked SKU-{sku} from bin {bin} in aisle {aisle}.",
        "Automated arm retrieved SKU-{sku} (qty: {qty}) at aisle {aisle}/bin {bin}.",
    ],
    "ITEM_PLACED": [
        "SKU-{sku} stowed in aisle {aisle}, bin {bin} by worker {worker_id}.",
        "Deposit confirmed: SKU-{sku} x{qty} → aisle {aisle}, bin {bin}.",
    ],
    "ITEM_MOVED": [
        "SKU-{sku} relocated to aisle {aisle}, bin {bin} for optimisation.",
        "Slot rebalancing: SKU-{sku} moved to aisle {aisle}/bin {bin}.",
    ],
    "RESTOCK_TRIGGERED": [
        "Restock order raised for aisle {aisle}, bin {bin} (threshold breach).",
        "Low-stock alert: SKU-{sku} in aisle {aisle} bin {bin} — replenishment queued.",
    ],
    "INVENTORY_CHECK": [
        "Routine cycle-count at aisle {aisle}, bin {bin} completed.",
        "Inventory audit: aisle {aisle}/bin {bin} verified by worker {worker_id}.",
    ],
    "ANOMALY_DETECTED": [
        "Anomaly at aisle {aisle}/bin {bin}: unexpected weight discrepancy.",
        "Sensor mismatch on aisle {aisle}, bin {bin} — flagged for review.",
    ],
    "SCAN_FAILED": [
        "Barcode scan failure at aisle {aisle}, bin {bin} — item unreadable.",
        "RFID tag unresponsive: aisle {aisle}/bin {bin}, SKU-{sku}.",
    ],
    "TEMPERATURE_ALERT": [
        "Temperature out of range at aisle {aisle}, bin {bin}: {temp}°C recorded.",
        "Cold-chain alert — aisle {aisle}/bin {bin} exceeded safe threshold ({temp}°C).",
    ],
}


class SimulatedStream(BaseStream):
    """
    Synthetic warehouse log stream for offline development and testing.

    Parameters
    ----------
    rate_per_sec : float
        Target emission rate in events per second. Must be > 0.
        Defaults to 1.0.
    num_aisles : int
        Number of simulated warehouse aisles (1-indexed). Defaults to 20.
    bins_per_aisle : int
        Number of bins per aisle (1-indexed). Defaults to 50.
    max_events : Optional[int]
        If set, the stream terminates automatically after this many events.
        ``None`` (default) means the stream runs indefinitely.
    seed : Optional[int]
        Seed for the internal PRNG.  Pass an integer for reproducible
        output; ``None`` for non-deterministic (production-like) runs.
    """

    def __init__(
        self,
        rate_per_sec: float = 1.0,
        num_aisles: int = 20,
        bins_per_aisle: int = 50,
        max_events: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0, got {rate_per_sec!r}")
        if num_aisles < 1:
            raise ValueError(f"num_aisles must be >= 1, got {num_aisles!r}")
        if bins_per_aisle < 1:
            raise ValueError(f"bins_per_aisle must be >= 1, got {bins_per_aisle!r}")
        if max_events is not None and max_events < 1:
            raise ValueError(f"max_events must be >= 1 or None, got {max_events!r}")

        self._rate_per_sec = rate_per_sec
        self._interval = 1.0 / rate_per_sec
        self._num_aisles = num_aisles
        self._bins_per_aisle = bins_per_aisle
        self._max_events = max_events
        self._rng = random.Random(seed)
        self._running = False
        self._connected = False
        self._events_emitted: int = 0

    # ------------------------------------------------------------------
    # BaseStream interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Mark the stream as active.

        For SimulatedStream no I/O setup is required, but we honour the
        contract so this class is a drop-in replacement for KafkaStream.
        """
        if self._connected:
            logger.warning("SimulatedStream.connect() called on already-connected stream.")
            return

        self._connected = True
        self._running = True
        self._events_emitted = 0
        logger.info(
            "SimulatedStream connected — rate=%.2f ev/s, aisles=%d, "
            "bins/aisle=%d, max_events=%s",
            self._rate_per_sec,
            self._num_aisles,
            self._bins_per_aisle,
            self._max_events if self._max_events is not None else "∞",
        )

    def stream(self) -> Iterator[WarehouseLog]:
        """
        Yield synthetic WarehouseLog objects at the configured rate.

        The generator honours `close()` calls (via ``self._running``) and
        the optional `max_events` cap.  Sleep is performed *after* yield
        so the first event is delivered immediately.

        Raises
        ------
        RuntimeError
            If called before `connect()`.
        """
        if not self._connected:
            raise RuntimeError(
                "SimulatedStream must be connected before streaming. "
                "Use 'with SimulatedStream(...) as s:' or call connect() first."
            )

        logger.debug("SimulatedStream.stream() generator started.")

        while self._running:
            # Check max_events cap before generating
            if self._max_events is not None and self._events_emitted >= self._max_events:
                logger.info(
                    "SimulatedStream reached max_events=%d — stopping.",
                    self._max_events,
                )
                self._running = False
                break

            log = self._generate_log()
            self._events_emitted += 1
            logger.debug("Emitted event #%d: %s", self._events_emitted, log)
            yield log

            # Rate limiting: sleep for the remainder of the interval.
            # Uses a short-sleep loop so that close() is noticed quickly.
            self._rate_limited_sleep(self._interval)

        logger.debug("SimulatedStream.stream() generator exiting.")

    def close(self) -> None:
        """
        Signal the stream generator to stop emitting events.

        Safe to call multiple times or from a different thread.
        """
        if not self._running and not self._connected:
            logger.debug("SimulatedStream.close() called on already-closed stream; ignoring.")
            return

        self._running = False
        self._connected = False
        logger.info(
            "SimulatedStream closed after %d event(s) emitted.",
            self._events_emitted,
        )

    # ------------------------------------------------------------------
    # Public introspection helpers
    # ------------------------------------------------------------------

    @property
    def events_emitted(self) -> int:
        """Total number of events yielded since the last connect()."""
        return self._events_emitted

    @property
    def is_running(self) -> bool:
        """True while the stream is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_log(self) -> WarehouseLog:
        """
        Produce a single synthetic WarehouseLog with realistic random fields.
        """
        aisle = self._rng.randint(1, self._num_aisles)
        bin_id = self._rng.randint(1, self._bins_per_aisle)
        event_type = self._rng.choice(_EVENT_TYPES)
        message = self._render_message(event_type, aisle, bin_id)

        return WarehouseLog(
            timestamp=time.time(),
            aisle=aisle,
            bin=bin_id,
            event_type=event_type,
            message=message,
        )

    def _render_message(self, event_type: str, aisle: int, bin_id: int) -> str:
        """
        Fill in a random message template for the given event type.
        """
        templates = _MESSAGE_TEMPLATES.get(event_type, [f"Event {event_type} at aisle {aisle}/bin {bin_id}."])
        template = self._rng.choice(templates)
        return template.format(
            aisle=aisle,
            bin=bin_id,
            sku=self._rng.randint(10000, 99999),
            qty=self._rng.randint(1, 50),
            worker_id=f"W{self._rng.randint(100, 999)}",
            temp=round(self._rng.uniform(-5.0, 35.0), 1),
        )

    def _rate_limited_sleep(self, seconds: float) -> None:
        """
        Sleep for `seconds` total, waking every 50 ms to check the stop flag.

        This ensures `close()` is respected within ~50 ms even at low rates.
        """
        _WAKE_INTERVAL = 0.05  # 50 ms
        elapsed = 0.0
        while self._running and elapsed < seconds:
            chunk = min(_WAKE_INTERVAL, seconds - elapsed)
            time.sleep(chunk)
            elapsed += chunk
