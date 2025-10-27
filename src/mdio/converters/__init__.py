"""MDIO Data conversion API."""

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from mdio.converters.mdio import mdio_to_segy
    from mdio.converters.segy import segy_to_mdio

__all__ = ["mdio_to_segy", "segy_to_mdio"]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - required for dynamic attribute access
    """Lazy import for converters to avoid circular imports."""
    if name == "mdio_to_segy":
        from mdio.converters.mdio import (  # noqa: PLC0415 - intentionally inside the function to avoid circular imports
            mdio_to_segy,
        )

        return mdio_to_segy

    if name == "segy_to_mdio":
        from mdio.converters.segy import (  # noqa: PLC0415 - intentionally inside the function to avoid circular imports
            segy_to_mdio,
        )

        return segy_to_mdio

    err = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(err)
