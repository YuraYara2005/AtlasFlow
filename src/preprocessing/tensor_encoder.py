# src/preprocessing/tensor_encoder.py

"""
TensorEncoder — converts a window of :class:`WarehouseLog` objects into a
:class:`torch.Tensor` suitable for downstream model inference.

Design notes
------------
- Feature set is intentionally sparse: time_delta, aisle, bin, event_type_id.
  All four fields are cast to ``torch.float32`` so the output can be fed
  directly into any PyTorch model without further casting.
- ``event_type`` is mapped to a stable integer via a module-level constant
  (``EVENT_TYPE_MAP``).  The mapping is deterministic and documented so that
  model checkpoints are reproducible across restarts.
- No batching: this encoder handles a single window (one sequence) at a time,
  consistent with the ``SlidingWindowManager`` contract.
- No state is mutated after construction — the class is effectively immutable
  once instantiated, making it safe to share across threads.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.ingestion.schema import WarehouseLog

# ---------------------------------------------------------------------------
# Event-type vocabulary
# ---------------------------------------------------------------------------
# Mirrors the canonical tuple defined in ``src/ingestion/simulated_stream.py``.
# Keys are added here as an explicit, ordered mapping so that the integer IDs
# are frozen and never shift silently when new event types are appended.
#
# Convention: IDs are 1-based so that 0 is available as a dedicated
# <PAD> / <UNKNOWN> sentinel for future use.

EVENT_TYPE_MAP: Dict[str, int] = {
    "ITEM_PICKED":        1,
    "ITEM_PLACED":        2,
    "ITEM_MOVED":         3,
    "RESTOCK_TRIGGERED":  4,
    "INVENTORY_CHECK":    5,
    "ANOMALY_DETECTED":   6,
    "SCAN_FAILED":        7,
    "TEMPERATURE_ALERT":  8,
}

_UNKNOWN_EVENT_ID: int = 0  # sentinel for unseen event types


class TensorEncoder:
    """
    Encodes a sequence of :class:`WarehouseLog` objects into a 2-D
    ``torch.float32`` tensor of shape ``(sequence_length, feature_dim)``.

    Feature layout (column order)
    ------------------------------
    0 : ``time_delta``     — seconds elapsed since the first log in the window
    1 : ``aisle``          — integer aisle index
    2 : ``bin``            — integer bin index
    3 : ``event_type_id``  — integer from :data:`EVENT_TYPE_MAP` (0 = unknown)

    Parameters
    ----------
    event_type_map : dict[str, int], optional
        Override the default :data:`EVENT_TYPE_MAP`.  Useful in tests or when
        the vocabulary needs to be extended without touching module-level state.

    Examples
    --------
    >>> encoder = TensorEncoder()
    >>> tensor = encoder.encode(window)   # window: List[WarehouseLog]
    >>> tensor.shape
    torch.Size([32, 4])
    >>> tensor.dtype
    torch.float32
    """

    #: Number of features produced per log entry.
    FEATURE_DIM: int = 4

    __slots__ = ("_event_type_map",)

    def __init__(
        self,
        event_type_map: Dict[str, int] | None = None,
    ) -> None:
        self._event_type_map: Dict[str, int] = (
            event_type_map if event_type_map is not None else EVENT_TYPE_MAP
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, window: List[WarehouseLog]) -> torch.Tensor:
        """
        Convert a window of logs into a ``float32`` tensor.

        Parameters
        ----------
        window : list[WarehouseLog]
            An ordered, non-empty sequence of warehouse log entries.
            Typically, the snapshot returned by
            :meth:`~src.preprocessing.window_manager.SlidingWindowManager.add`.

        Returns
        -------
        torch.Tensor
            Shape ``(len(window), FEATURE_DIM)`` with ``dtype=torch.float32``.

        Raises
        ------
        ValueError
            If *window* is empty.
        """
        if not window:
            raise ValueError("window must contain at least one WarehouseLog.")

        # Anchor all timestamps to the first entry so that feature 0 stays
        # small (seconds, not epoch values) and gradient-friendly.
        baseline_time: float = window[0].timestamp
        rows = [self._encode_single(log, baseline_time) for log in window]
        # Stack into (sequence_length, FEATURE_DIM) — one syscall, no copies.
        return torch.tensor(rows, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_single(self, log: WarehouseLog, baseline_time: float) -> List[float]:
        """
        Extract and encode the four features from a single log entry.

        Using ``time_delta`` (seconds since the window's first event) rather
        than the raw Unix timestamp keeps feature 0 in a small, stable numeric
        range and prevents large epoch values from dominating gradient updates.

        Unknown ``event_type`` strings are mapped to ``_UNKNOWN_EVENT_ID``
        rather than raising, so the pipeline degrades gracefully if a new
        event type reaches the encoder before the vocabulary is updated.

        Parameters
        ----------
        log : WarehouseLog
            A single warehouse event.
        baseline_time : float
            Unix timestamp of the first log in the window (``window[0].timestamp``).
            Subtracted from ``log.timestamp`` to produce a relative time delta.

        Returns
        -------
        list[float]
            Four-element list ``[time_delta, aisle, bin, event_type_id]``
            where ``time_delta = log.timestamp - baseline_time`` (seconds).
        """
        time_delta: float = log.timestamp - baseline_time
        event_id = self._event_type_map.get(log.event_type, _UNKNOWN_EVENT_ID)
        return [
            time_delta,
            float(log.aisle),
            float(log.bin),
            float(event_id),
        ]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"feature_dim={self.FEATURE_DIM}, "
            f"vocab_size={len(self._event_type_map)})"
        )
