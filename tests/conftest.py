"""Test configuration before everything runs."""


from os import path
from urllib.request import urlretrieve

import pytest


@pytest.fixture(scope="session")
def segy_input(tmp_path_factory):
    """Download teapot dome dataset for testing."""
    url = "http://s3.amazonaws.com/teapot/filt_mig.sgy"
    tmp_dir = tmp_path_factory.mktemp("segy")
    tmp_file = path.join(tmp_dir, "teapot.segy")
    urlretrieve(url, tmp_file)  # noqa: S310

    return tmp_file


@pytest.fixture(scope="session")
def zarr_tmp(tmp_path_factory):
    """Make a temp file for the output MDIO."""
    tmp_file = tmp_path_factory.mktemp(r"mdio")
    return tmp_file


@pytest.fixture(scope="session")
def segy_export_ibm_tmp(tmp_path_factory):
    """Make a temp file for the round-trip IBM SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    tmp_file = path.join(tmp_dir, "teapot_roundtrip_ibm.segy")

    return tmp_file


@pytest.fixture(scope="session")
def segy_export_ieee_tmp(tmp_path_factory):
    """Make a temp file for the round-trip IEEE SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy")
    tmp_file = path.join(tmp_dir, "teapot_roundtrip_ieee.segy")

    return tmp_file
