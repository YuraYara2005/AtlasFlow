# src/preprocessing/window_manager.py

from collections import deque
from typing import List, Optional

from src.ingestion.schema import WarehouseLog


class SlidingWindowManager:
    """
    Maintains a fixed-size sliding window of :class:`WarehouseLog` objects.

    Internally backed by :class:`collections.deque` with a ``maxlen`` so that
    every ``add`` operation — including the implicit eviction of the oldest
    entry when the window is full — is **O(1)** in both time and space.

    Parameters
    ----------
    window_size : int
        Maximum number of log entries to retain at any point in time.
        Must be a positive integer.

    Raises
    ------
    ValueError
        If ``window_size`` is not a positive integer.

    Example
    -------
    >>> manager = SlidingWindowManager(window_size=3)
    >>> manager.add(log1)   # None — window not yet full
    >>> manager.add(log2)   # None — window not yet full
    >>> window = manager.add(log3)   # returns list of 3 logs
    """

    __slots__ = ("_window",)

    def __init__(self, window_size: int) -> None:
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(
                f"window_size must be a positive integer, got {window_size!r}"
            )
        self._window: deque[WarehouseLog] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, log: WarehouseLog) -> Optional[List[WarehouseLog]]:
        """
        Append *log* to the sliding window.

        When the window reaches its configured capacity a **snapshot** of the
        current window is returned as an ordered list (oldest → newest).
        Before the window is full ``None`` is returned so callers can cheaply
        skip processing.

        Parameters
        ----------
        log : WarehouseLog
            The incoming warehouse event to append.

        Returns
        -------
        list[WarehouseLog] or None
            A list of ``window_size`` logs when the window is full;
            ``None`` otherwise.

        Notes
        -----
        * The deque's ``maxlen`` guarantees the oldest entry is evicted
          automatically — no manual bookkeeping required.
        * Converting the deque to a list is O(n) with respect to window size,
          not stream length, so the overall per-event cost remains O(1)
          relative to the unbounded input stream.
        """
        self._window.append(log)
        if len(self._window) == self._window.maxlen:
            return list(self._window)
        return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current number of entries in the window."""
        return len(self._window)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"window_size={self._window.maxlen}, "
            f"current_size={len(self._window)})"
        )
