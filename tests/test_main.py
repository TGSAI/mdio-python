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
    cli_args = ["segy", "import", str(segy_input), str(zarr_tmp), "PostStack3DTime"]
    cli_args.extend(["--header-locations", "17,13,81,85"])
    cli_args.extend(["--header-names", "inline,crossline,cdp_x,cdp_y"])
    cli_args.extend(["--overwrite"])

    result = runner.invoke(__main__.main, args=cli_args)
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_main_succeeds"])
def test_main_cloud(runner: CliRunner, segy_input_uri: str, zarr_tmp: Path) -> None:
    """It exits with a status code of zero."""
    os.environ["MDIO__IMPORT__CLOUD_NATIVE"] = "true"
    cli_args = ["segy", "import", segy_input_uri, str(zarr_tmp), "PostStack3DTime"]
    cli_args.extend(["--header-locations", "17,13,81,85"])
    cli_args.extend(["--header-names", "inline,crossline,cdp_x,cdp_y"])
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
    exp_output = """\
<xarray.Dataset> Size: 392MB
Dimensions:     (inline: 345, crossline: 188, time: 1501)
Coordinates:
    cdp_x       (inline, crossline) float64 519kB dask.array<chunksize=(345, 188), meta=np.ndarray>
    cdp_y       (inline, crossline) float64 519kB dask.array<chunksize=(345, 188), meta=np.ndarray>
  * crossline   (crossline) int32 752B 1 2 3 4 5 6 7 ... 183 184 185 186 187 188
  * inline      (inline) int32 1kB 1 2 3 4 5 6 7 ... 339 340 341 342 343 344 345
  * time        (time) int32 6kB 0 2 4 6 8 10 ... 2990 2992 2994 2996 2998 3000
Data variables:
    amplitude   (inline, crossline, time) float32 389MB dask.array<chunksize=(128, 128, 128), meta=np.ndarray>
    headers     (inline, crossline) [('inline', '<i4'), ('crossline', '<i4'), ('cdp_x', '<i4'), ('cdp_y', '<i4')] 1MB dask.array<chunksize=(128, 128), meta=np.ndarray>
    trace_mask  (inline, crossline) bool 65kB dask.array<chunksize=(345, 188), meta=np.ndarray>
Attributes:
    apiVersion:  1.0.0a1
    createdOn:   2025-09-03 04:07:20.025836+00:00
    name:        PostStack3DTime
    attributes:  {'surveyDimensionality': '3D', 'ensembleType': 'line', 'proc...
"""  # noqa: E501
    # Remove the 'createdOn:' line that changes at every test run
    expected = [item for item in exp_output.splitlines() if not item.startswith("    createdOn:")]
    actual = [item for item in result.output.splitlines() if not item.startswith("    createdOn:")]
    assert actual == expected


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
