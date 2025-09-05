"""Unit tests for StorageLocation class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from mdio.core.storage_location import StorageLocation


class TestStorageLocation:
    """Test cases for StorageLocation class."""

    @patch("fsspec.filesystem")
    def test_exists(self, mock_filesystem: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        """Test the exists() method of StorageLocation."""
        # Test exists() returns True when file exists.
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_filesystem.return_value = mock_fs
        location = StorageLocation("/test/existing/file")
        result = location.exists()
        assert result is True
        mock_fs.exists.assert_called_once()

        # Test exists() returns False when file does not exist.
        mock_fs = Mock()
        mock_fs.exists.return_value = False
        mock_filesystem.return_value = mock_fs
        location = StorageLocation("/test/nonexistent/file")
        result = location.exists()
        assert result is False
        mock_fs.exists.assert_called_once()

        # Test exists() handles exceptions gracefully.
        mock_fs = Mock()
        mock_fs.exists.side_effect = Exception("Connection failed")
        mock_filesystem.return_value = mock_fs
        location = StorageLocation("s3://bucket/file")
        result = location.exists()
        assert result is False
        captured = capsys.readouterr()
        assert "Error checking existence of s3://bucket/file: Connection failed" in captured.out

    def test_representations(self) -> None:
        """Test string and developer representations of StorageLocation."""
        # Test string representation of StorageLocation.
        location = StorageLocation("/test/path")
        assert str(location) == str(Path("/test/path"))

        # Test developer representation of StorageLocation.

        uri = "s3://my-bucket/file.segy"
        options = {"region": "us-west-2"}
        location = StorageLocation(uri=uri, options=options)
        expected = "StorageLocation(uri='s3://my-bucket/file.segy', options={'region': 'us-west-2'})"
        assert repr(location) == expected

    def test_from_path(self) -> None:
        """Test from_path class method."""
        # Test with string path.
        path_str = "/home/user/data.segy"
        location = StorageLocation(path_str)
        # Should resolve to absolute path
        expected_path = str(Path(path_str).resolve())
        assert location.uri == expected_path
        assert location.options == {}

        # Test with path uri path.
        location = StorageLocation(f"file://{path_str}")
        # Should resolve to absolute path
        expected_path = str(Path(path_str).resolve())
        assert location.uri == expected_path
        assert location.options == {}

        # Test with real local file operations.
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"test content")
        try:
            # Test with real local file
            location = StorageLocation(str(temp_path))
            # Should exist
            assert location.exists() is True
            # Should have correct URI
            assert location.uri == str(temp_path.resolve())
        finally:
            # Clean up
            temp_path.unlink()
            # Now should not exist
            assert location.exists() is False

    def test_from_cloud(self) -> None:
        """Test class for cloud storage URIs."""
        # Test from_s3 without options.
        s3_uri = "s3://bucket/file"
        location = StorageLocation(s3_uri)
        assert location.uri == s3_uri
        assert location.options == {}

        # Test from_s3 with valid S3 URI.
        s3_uri = "s3://my-bucket/path/to/file.segy"
        options = {"region": "us-west-2", "aws_access_key_id": "key123"}
        location = StorageLocation(s3_uri, options=options)
        assert location.uri == s3_uri
        assert location.options == options

    def test_options_immutability(self) -> None:
        """Test that options property returns a defensive copy."""
        original_options = {"region": "us-east-1", "timeout": 30}
        location = StorageLocation(uri="s3://bucket/file", options=original_options)

        # Get options through property
        returned_options = location.options

        # Verify it's equal to original
        assert returned_options == original_options

        # Modify the returned dict
        returned_options["new_key"] = "new_value"
        returned_options["timeout"] = 60

        # Original should be unchanged
        assert location.options == original_options
        assert "new_key" not in location.options
        assert location.options["timeout"] == 30
