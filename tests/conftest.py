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

@pytest.fixture(scope="session")
def teapot_mdio_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"teapot.mdio")


@pytest.fixture(scope="module")
def mdio_4d_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"tmp_4d.mdio")


@pytest.fixture(scope="module")
def zarr_tmp2(tmp_path_factory: pytest.TempPathFactory) -> Path:  # pragma: no cover - used by disabled test
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"mdio2")


@pytest.fixture(scope="session")
def segy_export_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the round-trip IBM SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    return tmp_dir / "teapot_roundtrip.segy"


@pytest.fixture(scope="class")
def empty_mdio_with_headers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for empty MDIO testing."""
    path = tmp_path_factory.mktemp(r"empty_with_headers.mdio")
    return path


# @pytest.fixture(scope="session")
# def tmp_path_factory() -> pytest.TempPathFactory:
#     """Custom tmp_path_factory implementation for local debugging."""
#     from pathlib import Path  # noqa: PLC0415

#     class DebugTempPathFactory:
#         def __init__(self) -> None:
#             pass

#         def mktemp(self, basename: str, numbered: bool = True) -> Path:
#             _ = numbered
#             path = self.getbasetemp() / basename
#             path.mkdir(parents=True, exist_ok=True)
#             return path

#         def getbasetemp(self) -> Path:
#             return Path("tmp")

#     return DebugTempPathFactory()
