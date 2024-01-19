"""Test configuration before everything runs."""


from pathlib import Path
from urllib.request import urlretrieve

import pytest


@pytest.fixture(scope="session")
def fake_segy_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the fake SEG-Y files we are going to create."""
    return tmp_path_factory.mktemp(r"fake_segy")


@pytest.fixture(scope="session")
def segy_input(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download teapot dome dataset for testing."""
    url = "http://s3.amazonaws.com/teapot/filt_mig.sgy"
    tmp_dir = tmp_path_factory.mktemp("segy")
    tmp_file = tmp_dir / "teapot.segy"
    urlretrieve(url, tmp_file)  # noqa: S310

    return tmp_file


@pytest.fixture(scope="session")
def zarr_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"mdio")


@pytest.fixture(scope="session")
def zarr_tmp2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"mdio2")


@pytest.fixture(scope="session")
def segy_export_ibm_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the round-trip IBM SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    return tmp_dir / "teapot_roundtrip_ibm.segy"


@pytest.fixture(scope="session")
def segy_export_ieee_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the round-trip IEEE SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    return tmp_dir / "teapot_roundtrip_ieee.segy"
