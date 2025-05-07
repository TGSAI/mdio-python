"""Test cases for the __main__ module."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from mdio import __main__


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.mark.dependency
def test_main_succeeds(runner: CliRunner, segy_input: Path, zarr_tmp: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["segy", "import", str(segy_input), str(zarr_tmp)]
    cli_args.extend(["--header-locations", "181,185"])
    cli_args.extend(["--header-names", "inline,crossline"])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_cloud(runner: CliRunner, segy_input_uri: str, zarr_tmp: Path) -> None:
    """It exits with a status code of zero."""
    os.environ["MDIO__IMPORT__CLOUD_NATIVE"] = "true"
    cli_args = ["segy", "import", segy_input_uri, str(zarr_tmp)]
    cli_args.extend(["--header-locations", "181,185"])
    cli_args.extend(["--header-names", "inline,crossline"])
    cli_args.extend(["--overwrite"])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_info_succeeds(runner: CliRunner, zarr_tmp: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["info"]
    cli_args.extend([str(zarr_tmp)])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_copy(runner: CliRunner, zarr_tmp: Path, zarr_tmp2: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["copy", str(zarr_tmp), str(zarr_tmp2), "-headers", "-traces"]

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


def test_cli_version(runner: CliRunner) -> None:
    """Check if version prints without error."""
    cli_args = ["--version"]
    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0
