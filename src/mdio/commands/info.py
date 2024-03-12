"""MDIO Dataset information command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from click import STRING
from click import Choice
from click import argument
from click import command
from click import option


if TYPE_CHECKING:
    from mdio import MDIOReader
    from mdio.core import Grid


@command(name="info")
@argument("mdio-path", type=STRING)
@option(
    "-access",
    "--access-pattern",
    required=False,
    default="012",
    help="Access pattern of the file",
    type=STRING,
    show_default=True,
)
@option(
    "-format",
    "--output-format",
    required=False,
    default="pretty",
    help="Output format. Pretty console or JSON.",
    type=Choice(["pretty", "json"]),
    show_default=True,
    show_choices=True,
)
def info(
    mdio_path: str,
    output_format: str,
    access_pattern: str,
) -> None:
    """Provide information on a MDIO dataset.

    By default, this returns human-readable information about the grid and stats for
    the dataset. If output-format is set to json then a json is returned to
    facilitate parsing.
    """
    from mdio import MDIOReader

    reader = MDIOReader(
        mdio_path,
        access_pattern=access_pattern,
        return_metadata=True,
    )

    grid_dict = parse_grid(reader.grid)
    stats_dict = cast_stats(reader.stats)
    access_pattern_dict = parse_access_patterns(reader)

    mdio_info = {
        "path": mdio_path,
        "stats": stats_dict,
        "grid": grid_dict,
        "access_patterns": access_pattern_dict,
    }

    if output_format == "pretty":
        pretty_print(mdio_info)

    if output_format == "json":
        json_print(mdio_info)


def cast_stats(stats_dict: dict[str, Any]) -> dict[str, float]:
    """Normalize all floats to JSON serializable floats."""
    return {k: float(v) for k, v in stats_dict.items()}


def parse_grid(grid: Grid) -> dict[str, dict[str, int | str]]:
    """Extract grid information per dimension."""
    grid_dict = {}
    for dim_name in grid.dim_names:
        dim = grid.select_dim(dim_name)
        min_ = str(dim.coords[0])
        max_ = str(dim.coords[-1])
        size = str(dim.coords.shape[0])
        grid_dict[dim_name] = {"name": dim_name, "min": min_, "max": max_, "size": size}
    return grid_dict


def parse_access_patterns(reader: MDIOReader) -> dict[str, Any]:
    """Extract access patterns and their info."""
    access_pattern_dict = {}
    for name, array in reader._data_group.items():
        pattern = name.replace("chunked_", "")
        chunks = str(array.chunks)
        format_ = str(array.dtype)
        compressor = str(array.compressor)
        access_pattern_dict[pattern] = {
            "chunks": chunks,
            "format": format_,
            "compressor": compressor,
        }

    return access_pattern_dict


def json_print(mdio_info: dict[str, Any]) -> None:
    """Convert MDIO Info to JSON and pretty print."""
    from json import dumps as json_dumps

    from rich import print

    print(json_dumps(mdio_info, indent=2))


def pretty_print(mdio_info: dict[str, Any]) -> None:
    """Print pretty MDIO Info table to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    grid_table = Table(show_edge=False)
    grid_table.add_column("Dimension", justify="right", style="cyan", no_wrap=True)
    grid_table.add_column("Min", justify="left", style="magenta")
    grid_table.add_column("Max", justify="left", style="magenta")
    grid_table.add_column("Size", justify="left", style="green")

    for _, axis_dict in mdio_info["grid"].items():
        name, min_, max_, size = axis_dict.values()
        grid_table.add_row(name, min_, max_, size)

    stat_table = Table(show_edge=False)
    stat_table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
    stat_table.add_column("Value", justify="left", style="magenta")

    for stat, value in mdio_info["stats"].items():
        stat_table.add_row(stat, f"{value:.4f}")

    access_patter_table = Table(show_edge=False)
    access_patter_table.add_column(
        "Pattern", justify="right", style="cyan", no_wrap=True
    )
    access_patter_table.add_column("Chunks", justify="left", style="magenta")
    access_patter_table.add_column("Format", justify="left", style="magenta")
    access_patter_table.add_column("Compressor", justify="left", style="magenta")

    for name, pattern_info in mdio_info["access_patterns"].items():
        chunks, format_, compressor = pattern_info.values()
        access_patter_table.add_row(name, chunks, format_, compressor)

    master_table = Table(title=f"File Information for {mdio_info['path']}")
    master_table.add_column("Grid", justify="center")
    master_table.add_column("Statistics", justify="center")
    master_table.add_column("Access Patterns", justify="center")
    master_table.add_row(grid_table, stat_table, access_patter_table)

    console.print(master_table)


cli = info
