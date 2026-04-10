# src/ingestion/base_stream.py

from abc import ABC, abstractmethod
from typing import Iterator
from .schema import WarehouseLog


class BaseStream(ABC):
    """
    Abstract interface for all warehouse log ingestion streams.

    This class defines a consistent contract for different data sources
    such as Kafka streams and simulated generators.
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Initialize connection or resources required for streaming.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> Iterator[WarehouseLog]:
        """
        Yield WarehouseLog objects continuously from the data source.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release resources and close the stream.
        """
        raise NotImplementedError

    def __enter__(self) -> "BaseStream":
        """
        Enable usage with context manager (with statement).
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Ensure resources are properly cleaned up.
        """
        self.close()