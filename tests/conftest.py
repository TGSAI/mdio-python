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
