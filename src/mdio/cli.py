"""Entrypoint to the MDIO command line interface (CLI)."""

import typer

from mdio.commands import segy
from mdio.commands import version

app = typer.Typer(no_args_is_help=True)
app.add_typer(segy.app, name="segy")
app.add_typer(version.app)
