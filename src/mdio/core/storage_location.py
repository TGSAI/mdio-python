"""StorageLocation class for managing local and cloud storage locations."""
from dataclasses import dataclass, field
from pathlib import Path

from pyparsing import Any


@dataclass
class StorageLocation:
    """A class to represent a local or cloud storage location for SEG-Y or MDIO files."""

    uri: str
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def storage_type(self) -> str:
        """Determine the storage type based on the URI scheme."""
        if self.uri.startswith("file://"):
            return "file"
        if self.uri.startswith("s3://"):
            return "cloud:s3"
        if self.uri.startswith("gs://"):
            return "cloud:gs"
        # Default to file storage type if no specific type is detected
        return "file"

    def exists(self) -> bool:
        """Check if the storage location exists."""
        if self.storage_type == "file":
            return Path(self.uri).exists()
        if self.storage_type.startswith("cloud:"):
            err = "Existence check for cloud storage is not implemented yet."
            raise NotImplementedError(err)
        err = f"Unsupported storage type: {self.storage_type}"
        raise ValueError(err)
