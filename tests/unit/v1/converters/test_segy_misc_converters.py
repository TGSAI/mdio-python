"""Unit tests for the type converter module."""

from datetime import datetime
import pytest

from segy.standards import SegyStandard
from mdio.converters.segy_to_mdio_v1 import get_segy_standard_version


def test_get_segy_standard_version() -> None:
    """Test the _get_segy_standard_version function."""

    # Test with a valid version number
    version_number = 1.0
    segy_standard_version = get_segy_standard_version(version_number)
    assert isinstance(segy_standard_version, SegyStandard)
    assert segy_standard_version == 1.0

    # Test with a valid version name
    version_name = "REV21"
    segy_standard_version = get_segy_standard_version(version_name)
    assert isinstance(segy_standard_version, SegyStandard)
    assert segy_standard_version == 2.1

    # Test with an non-existent version number
    invalid_version_number = 234
    with pytest.raises(ValueError, match="Invalid SEG-Y standard version '234'."):
        get_segy_standard_version(invalid_version_number)
        
    # Test with an invalid version name (e.g., datetime)
    invalid_version_name = "2025-08-01 17:09:58.651543"
    with pytest.raises(ValueError, match="Invalid SEG-Y standard version '2025-08-01 17:09:58.651543'."):
        get_segy_standard_version(invalid_version_name)

   # Test with an non-existent version number
    invalid_version_number = "REV999"
    with pytest.raises(ValueError, match="Invalid SEG-Y standard version 'REV999'."):
        get_segy_standard_version(invalid_version_number)

    # Test with an invalid version name
    invalid_version_name = "2025-08-01 17:09:58.651543"
    with pytest.raises(ValueError, match="Invalid SEG-Y standard version '2025-08-01 17:09:58.651543'."):
        get_segy_standard_version(invalid_version_name)
