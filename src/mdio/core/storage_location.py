"""StorageLocation class for managing local and cloud storage locations."""

from pathlib import Path
from typing import Any

import fsspec


# TODO(Dmitriy Repin): Reuse fsspec functions for some methods we implemented here
# https://github.com/TGSAI/mdio-python/issues/597
class StorageLocation:
    """A class to represent a local or cloud storage location for SEG-Y or MDIO files.

    This class abstracts the storage location, allowing for both local file paths and
    cloud storage URIs (e.g., S3, GCS). It uses fsspec to check existence and manage options.
    Note, we do not want to make it a dataclass because we want the uri and the options to
    be read-only immutable properties.

        uri: The URI of the storage location (e.g., '/path/to/file', 'file:///path/to/file',
                's3://bucket/path', 'gs://bucket/path').
        options: Optional dictionary of options for the cloud, such as credentials.

    """

    def __init__(self, uri: str = "", options: dict[str, Any] = None):
        self._uri = uri
        self._options = options or {}
        self._fs = None

        if uri.startswith(("s3://", "gs://")):
            return

        if uri.startswith(("http://", "https://")):
            return

        if uri.startswith("file://"):
            self._uri = self._uri.removeprefix("file://")
        # For local paths, ensure they are absolute and resolved
        self._uri = str(Path(self._uri).resolve())
        return

    @property
    def uri(self) -> str:
        """Get the URI (read-only)."""
        return self._uri

    @property
    def options(self) -> dict[str, Any]:
        """Get the options (read-only)."""
        # Return a copy to prevent external modification
        return self._options.copy()

    @property
    def _filesystem(self) -> fsspec.AbstractFileSystem:
        """Get the fsspec filesystem instance for this storage location."""
        if self._fs is None:
            self._fs = fsspec.filesystem(self._protocol, **self._options)
        return self._fs

    @property
    def _path(self) -> str:
        """Extract the path portion from the URI."""
        if "://" in self._uri:
            return self._uri.split("://", 1)[1]
        return self._uri  # For local paths without file:// prefix

    @property
    def _protocol(self) -> str:
        """Extract the protocol/scheme from the URI."""
        if "://" in self._uri:
            return self._uri.split("://", 1)[0]
        return "file"  # Default to file protocol

    def exists(self) -> bool:
        """Check if the storage location exists using fsspec."""
        try:
            return self._filesystem.exists(self._path)
        except Exception as e:
            # Log the error and return False for safety
            # In a production environment, you might want to use proper logging
            print(f"Error checking existence of {self._uri}: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the storage location."""
        return self._uri

    def __repr__(self) -> str:
        """Developer representation of the storage location."""
        return f"StorageLocation(uri='{self._uri}', options={self._options})"
