"""Version command."""

import typer

from mdio import __version__

app = typer.Typer()


@app.command()
def version() -> None:
    """Print the version of the CLI."""
    print(f"MDIO CLI Version {__version__}")
