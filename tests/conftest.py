"""Test configuration before everything runs."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from shutil import copyfile
from urllib.request import urlretrieve

import pytest

SODA_LAKE_SHOT_URL = "https://gdr-data-lake.s3.us-west-2.amazonaws.com/soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY"

# Suppress Dask's chunk balancing warning
warnings.filterwarnings(
    "ignore",
    message="Could not balance chunks to be equal",
    category=UserWarning,
    module="dask.array.rechunk",
)


@pytest.fixture(scope="session")
def fake_segy_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the fake SEG-Y files we are going to create."""
    return tmp_path_factory.mktemp(r"fake_segy")


@pytest.fixture(scope="session")
def segy_input_uri() -> str:
    """URL or local path to the Soda Lake shot SEG-Y used in integration tests."""
    return os.environ.get("MDIO_TEST_SEGY_URI", SODA_LAKE_SHOT_URL)


@pytest.fixture(scope="session")
def segy_input(segy_input_uri: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fetch the Soda Lake shot SEG-Y for testing."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    tmp_file = tmp_dir / "soda_lake.segy"
    source = Path(segy_input_uri)
    if source.is_file():
        copyfile(source, tmp_file)
    else:
        urlretrieve(segy_input_uri, tmp_file)  # noqa: S310
    return tmp_file


@pytest.fixture(scope="module")
def zarr_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"mdio")


@pytest.fixture(scope="module")
def zarr_tmp2(tmp_path_factory: pytest.TempPathFactory) -> Path:  # pragma: no cover - used by disabled test
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"mdio2")


@pytest.fixture(scope="session")
def segy_export_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the round-trip IBM SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    return tmp_dir / "soda_lake_roundtrip.segy"
