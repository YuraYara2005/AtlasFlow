# src/ingestion/schema.py

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class WarehouseLog:
    """
    Immutable domain model representing a warehouse event log.
    """

    timestamp: float
    aisle: int
    bin: int
    event_type: str
    message: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarehouseLog":
        """
        Safely construct a WarehouseLog from a dictionary.
        Raises ValueError with clear message if invalid.
        """
        try:
            timestamp = float(data["timestamp"])
            aisle = int(data["aisle"])
            bin_id = int(data["bin"])
            event_type = str(data["event_type"])
            message = str(data["message"])
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid field type: {e}") from e

        if aisle < 0:
            raise ValueError("Aisle must be non-negative")

        if bin_id < 0:
            raise ValueError("Bin must be non-negative")

        if not event_type:
            raise ValueError("event_type cannot be empty")

        return cls(
            timestamp=timestamp,
            aisle=aisle,
            bin=bin_id,
            event_type=event_type,
            message=message,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize WarehouseLog to dictionary.
        """
        return {
            "timestamp": self.timestamp,
            "aisle": self.aisle,
            "bin": self.bin,
            "event_type": self.event_type,
            "message": self.message,
        }