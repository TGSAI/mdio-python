"""Test configuration before everything runs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
    """Path to dome dataset for cloud testing."""
    return "http://s3.amazonaws.com/teapot/filt_mig.sgy"


@pytest.fixture(scope="session")
def segy_input(segy_input_uri: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download teapot dome dataset for testing."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    tmp_file = tmp_dir / "teapot.segy"
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
    return tmp_dir / "teapot_roundtrip.segy"
