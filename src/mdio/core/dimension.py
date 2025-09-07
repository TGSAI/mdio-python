"""Dimension (grid) abstraction and serializers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mdio.exceptions import ShapeError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(eq=False, order=False, slots=True)
class Dimension:
    """Dimension class.

    Dimension has a name and coordinates associated with it. The Dimension coordinates can only
    be a vector.

    Args:
        coords: Vector of coordinates.
        name: Name of the dimension.

    Attributes:
        coords: Vector of coordinates.
        name: Name of the dimension.
    """

    coords: list | tuple | NDArray | range
    name: str

    def __post_init__(self) -> None:
        """Post process and validation."""
        self.coords = np.asarray(self.coords)
        if self.coords.ndim != 1:
            msg = "Dimensions can only have vector coordinates"
            raise ShapeError(msg, ("# Dim", "Expected"), (self.coords.ndim, 1))

    @property
    def size(self) -> int:
        """Size of the dimension."""
        return len(self.coords)

    def __len__(self) -> int:
        """Length magic."""
        return self.size

    def __getitem__(self, item: int | slice | list[int]) -> NDArray:
        """Gets a specific coordinate value by index."""
        return self.coords[item]

    def __setitem__(self, key: int, value: NDArray) -> None:
        """Sets a specific coordinate value by index."""
        self.coords[key] = value

    def __hash__(self) -> int:
        """Hashing magic."""
        return hash(tuple(self.coords) + (self.name,))

    def __eq__(self, other: Dimension) -> bool:
        """Compares if the dimension has same properties."""
        if not isinstance(other, Dimension):
            other_type = type(other).__name__
            msg = f"Can't compare Dimension with {other_type}"
            raise TypeError(msg)

        return hash(self) == hash(other)

    def min(self) -> NDArray[float]:
        """Get minimum value of dimension."""
        return np.min(self.coords)

    def max(self) -> NDArray[float]:
        """Get maximum value of dimension."""
        return np.max(self.coords)
