"""Command-line interface."""


from __future__ import annotations

import os
from typing import Callable

import click
import click_params

import mdio


plugin_folder = os.path.join(os.path.dirname(__file__), "commands")


class MyCLI(click.MultiCommand):
    """CLI generator via plugin design pattern.

    This class will generate the CLI submodules based on what is included
    in the `commands` folder in `mdio` module. The CLI components are
    implemented under `commands`, and this class will parse them and serve
    them as user requests it.

    Serving happens by evaluating the Python function using `eval`.

    The original implementation in the `click` reference using `eval` is
    dangerous because it allows arbitrary code execution. However, we fix
    that by providing a global namespace to `eval` with ONLY allowed modules
    and by explicitly NOT allowing any python builtin functions (like import),
    and also by providing an empty local namespace to be filled with CLI
    functions. Check the `get_command` method for implementation.

    References:
        Original implementation:
        https://click.palletsprojects.com/en/8.1.x/commands/#custom-multi-commands

        Safe version of eval:
        http://lybniz2.sourceforge.net/safeeval.html
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List commands available under `commands` module."""
        rv = []
        for filename in os.listdir(plugin_folder):
            if filename.endswith(".py") and filename != "__init__.py":
                rv.append(filename[:-3])
        rv.sort()

        return rv

    def get_command(self, ctx: click.Context, name: str) -> dict[Callable]:
        """Get command implementation from `commands` module."""
        global_ns = {
            "__builtins__": None,
            "SystemError": SystemError,
            "click": click,
            "click_params": click_params,
            "mdio": mdio,
        }
        local_ns = {}

        fn = os.path.join(plugin_folder, name + ".py")
        with open(fn) as f:
            code = compile(f.read(), fn, "exec")
            eval(code, global_ns, local_ns)  # noqa: S307

        return local_ns["cli"]


@click.command(cls=MyCLI)
@click.version_option(mdio.__version__)
def main() -> None:
    """Welcome to MDIO!

    MDIO is an open source, cloud-native, and scalable storage engine
    for various types of energy data.

    MDIO supports importing or exporting various data containers,
    hence we allow plugins as subcommands.

    From this main command, we can see the MDIO version.
    """


if __name__ == "__main__":
    main(prog_name="mdio")  # pragma: no cover
