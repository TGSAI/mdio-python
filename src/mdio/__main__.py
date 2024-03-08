"""Command-line interface."""


from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Callable

import click


KNOWN_MODULES = [
    "segy.py",
    "copy.py",
    "info.py",
]


class MyCLI(click.MultiCommand):
    """CLI generator via plugin design pattern.

    This class dynamically loads command modules from the specified
    `plugin_folder`. If the command us another CLI group, the command
    module must define a `cli = click.Group(...)` and subsequent
    commands must be added to this CLI. If it is a single utility it
    must have a variable named `cli` for the command to be exposed.

    Args:
    - plugin_folder: Path to the directory containing command modules.
    """

    def __init__(self, plugin_folder: Path, *args, **kwargs):
        """Initializer function."""
        super().__init__(*args, **kwargs)
        self.plugin_folder = plugin_folder
        self.known_modules = KNOWN_MODULES

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List commands available under `commands` module."""
        rv = []
        for filename in self.plugin_folder.iterdir():
            is_known = filename.name in self.known_modules
            is_python = filename.suffix == ".py"
            if is_known and is_python:
                rv.append(filename.stem)
        rv.sort()
        return rv

    def get_command(self, ctx: click.Context, name: str) -> Callable | None:
        """Get command implementation from `commands` module."""
        try:
            filepath = self.plugin_folder / f"{name}.py"
            if filepath.name not in self.known_modules:
                click.echo(f"Command {name} is not safe to execute.")
                return None

            module_name = f"mdio.commands.{name}"
            spec = importlib.util.spec_from_file_location(module_name, str(filepath))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.cli
        except Exception as e:
            click.echo(f"Error loading command {name}: {e}")
            return None


def get_package_version(package_name: str, default: str = "unknown") -> str:
    """Safely fetch the package version, providing a default if not found."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return default


@click.command(cls=MyCLI, plugin_folder=Path(__file__).parent / "commands")
@click.version_option(get_package_version("multidimio"))
def main() -> None:
    """Welcome to MDIO!

    MDIO is an open source, cloud-native, and scalable storage engine
    for various types of energy data.

    MDIO supports importing or exporting various data containers,
    hence we allow plugins as subcommands.

    From this main command, we can see the MDIO version.
    """
