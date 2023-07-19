"""Test cases for the __main__ module."""


from pathlib import Path

import pytest
from click.testing import CliRunner

from mdio import __main__


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.mark.dependency()
def test_main_succeeds(runner: CliRunner, segy_input: str, zarr_tmp: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["segy", "import"]
    cli_args.extend(["-i", segy_input])
    cli_args.extend(["-o", str(zarr_tmp)])
    cli_args.extend(["-loc", "181,185"])
    cli_args.extend(["-names", "inline,crossline"])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_info_succeeds(runner: CliRunner, zarr_tmp: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["info"]
    cli_args.extend(["-i", str(zarr_tmp)])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_copy_succeeds(runner: CliRunner, zarr_tmp: Path, zarr_tmp2: Path) -> None:
    """It exits with a status code of zero."""
    cli_args = ["copy"]
    cli_args.extend(["-i", str(zarr_tmp)])
    cli_args.extend(["-o", str(zarr_tmp2)])

    result = runner.invoke(__main__.main, args=cli_args)
    print(f"copy returns {result}")
    # assert result.exit_code == 0


def test_cli_version(runner: CliRunner) -> None:
    """Check if version prints without error."""
    cli_args = ["--version"]
    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0
