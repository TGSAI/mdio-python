"""Test cases for the __main__ module."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from mdio import __main__


@pytest.fixture()
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.fixture(scope="module")
def mock_zarr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"test.mdio")


@pytest.fixture(scope="module")
def mock_zarr_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the output MDIO."""
    return tmp_path_factory.mktemp(r"test_copy.mdio")


@pytest.mark.dependency()
def test_main_succeeds(runner: CliRunner, segy_input: str, mock_zarr: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["segy", "import", str(segy_input), str(mock_zarr)]
    cli_args.extend(["-loc", "181,185"])
    cli_args.extend(["-names", "inline,crossline"])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_info_succeeds(runner: CliRunner, mock_zarr: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["info"]
    cli_args.extend([str(mock_zarr)])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_copy_succeeds(
    runner: CliRunner, mock_zarr: Path, mock_zarr_copy: Path
) -> None:
    """It exits with a status code of zero."""
    cli_args = ["copy", str(mock_zarr), str(mock_zarr_copy)]

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_cli_version(runner: CliRunner) -> None:
    """Check if version prints without error."""
    cli_args = ["--version"]
    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0
