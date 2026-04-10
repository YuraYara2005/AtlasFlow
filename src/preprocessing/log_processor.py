# src/preprocessing/log_processor.py

"""
LogProcessor — the single entry-point for the preprocessing pipeline.

Responsibilities
----------------
- Accept one :class:`~src.ingestion.schema.WarehouseLog` at a time.
- Accumulate logs in a fixed-size sliding window.
- Produce a ``torch.Tensor`` the moment the window is full; return ``None``
  while the window is still warming up.

This class is intentionally a *thin orchestrator*: all windowing logic lives
in :class:`~src.preprocessing.window_manager.SlidingWindowManager` and all
encoding logic lives in :class:`~src.preprocessing.tensor_encoder.TensorEncoder`.
No business logic is duplicated here.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.ingestion.schema import WarehouseLog
from src.preprocessing.tensor_encoder import TensorEncoder
from src.preprocessing.window_manager import SlidingWindowManager


class LogProcessor:
    """
    Stateful, single-log-at-a-time preprocessing pipeline.

    Internally wires together a :class:`SlidingWindowManager` (buffering) and
    a :class:`TensorEncoder` (feature extraction) so that callers interact
    with a single, minimal interface.

    Parameters
    ----------
    window_size : int
        Number of consecutive :class:`WarehouseLog` entries to collect before
        emitting a tensor.  Passed directly to :class:`SlidingWindowManager`.
    encoder : TensorEncoder, optional
        Pre-configured encoder instance.  If omitted a default
        :class:`TensorEncoder` is constructed automatically, which covers the
        standard warehouse event vocabulary.  Inject a custom encoder in tests
        or when the event-type vocabulary has been extended.

    Examples
    --------
    >>> processor = LogProcessor(window_size=32)
    >>> for log in stream:
    ...     tensor = processor.process(log)
    ...     if tensor is not None:
    ...         model(tensor)   # shape: (32, 4), dtype: torch.float32
    """

    __slots__ = ("_window_manager", "_encoder")

    def __init__(
        self,
        window_size: int,
        encoder: TensorEncoder | None = None,
    ) -> None:
        self._window_manager = SlidingWindowManager(window_size=window_size)
        self._encoder: TensorEncoder = encoder if encoder is not None else TensorEncoder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, log: WarehouseLog) -> Optional[torch.Tensor]:
        """
        Ingest a single log entry and return a tensor when the window is full.

        Parameters
        ----------
        log : WarehouseLog
            The incoming warehouse event to process.

        Returns
        -------
        torch.Tensor or None
            A ``float32`` tensor of shape ``(window_size, feature_dim)``
            once the internal window has accumulated enough logs;
            ``None`` while the window is still warming up.

        Notes
        -----
        After the window is full every subsequent call *also* returns a tensor
        (the window slides: oldest log is evicted, newest is appended).
        """
        window = self._window_manager.add(log)
        if window is None:
            return None
        return self._encoder.encode(window)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"window_size={self._window_manager._window.maxlen}, "
            f"encoder={self._encoder!r})"
        )
