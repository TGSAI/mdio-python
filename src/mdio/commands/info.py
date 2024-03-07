"""MDIO Dataset information command."""


from json import dumps as json_dumps

from click import Choice
from click import argument
from click import command
from click import option
from rich import print
from rich.console import Console
from rich.table import Table

from mdio import MDIOReader


@command(name="info")
@argument("mdio-path", type=str)
@option(
    "-access",
    "--access-pattern",
    required=False,
    default="012",
    help="Access pattern of the file",
    type=str,
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
    reader = MDIOReader(
        mdio_path,
        access_pattern=access_pattern,
        return_metadata=True,
    )

    mdio_dict = {}
    mdio_dict["grid"] = {}

    for dim_name in reader.grid.dim_names:
        dim = reader.grid.select_dim(dim_name)
        min_ = str(dim.coords[0])
        max_ = str(dim.coords[-1])
        size = str(dim.coords.shape[0])
        axis_dict = {"name": dim_name, "min": min_, "max": max_, "size": size}
        mdio_dict["grid"][dim_name] = axis_dict

    if output_format == "pretty":
        console = Console()

        grid_table = Table(show_edge=False)
        grid_table.add_column("Dimension", justify="right", style="cyan", no_wrap=True)
        grid_table.add_column("Min", justify="left", style="magenta")
        grid_table.add_column("Max", justify="left", style="magenta")
        grid_table.add_column("Size", justify="left", style="green")

        for _, axis_dict in mdio_dict["grid"].items():
            name, min_, max_, size = axis_dict.values()
            grid_table.add_row(name, min_, max_, size)

        stat_table = Table(show_edge=False)
        stat_table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
        stat_table.add_column("Value", justify="left", style="magenta")

        for stat, value in reader.stats.items():
            stat_table.add_row(stat, f"{value:.4f}")

        master_table = Table(title=f"File Information for {mdio_path}")
        master_table.add_column("MDIO Grid", justify="center")
        master_table.add_column("MDIO Statistics", justify="center")
        master_table.add_row(grid_table, stat_table)

        console.print(master_table)

    if output_format == "json":
        stats_cast = {k: float(v) for k, v in reader.stats.items()}
        mdio_dict["stats"] = stats_cast

        print(json_dumps(mdio_dict, indent=2))


cli = info
