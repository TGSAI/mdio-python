"""Test cases for the __main__ module."""

import pytest
from typer.testing import CliRunner

from mdio.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_cli_version(runner: CliRunner) -> None:
    """Check if version prints without error."""
    result = runner.invoke(app, args=["version"])
    assert result.exit_code == 0
    assert "MDIO CLI Version" in result.output
