"""Dimension (grid) abstraction and serializers."""


from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mdio.core.serialization import Serializer
from mdio.exceptions import ShapeError


# TODO: once min Python >3.10, remove slots attribute and
#  add `slots=True` to dataclass decorator and also add
#  `kw_only=True` to enforce keyword only initialization.
@dataclass(eq=False, order=False)
class Dimension:
    """Dimension class.

    Dimension has a name and coordinates associated with it.
    The Dimension coordinates can only be a vector.

    Args:
        coords: Vector of coordinates.
        name: Name of the dimension.
        non_binned: Whether the dimension is binned or not.
    """

    __slots__ = ("coords", "name", "non_binned")

    coords: list | tuple | NDArray | range
    name: str
    non_binned: bool

    def __init__(
        self,
        coords: list | tuple | NDArray | range,
        name: str,
        non_binned: bool = False,
    ):
        """Constructor."""
        self.coords = np.asarray(coords)
        if self.coords.ndim != 1:
            raise ShapeError(
                "Dimensions can only have vector coordinates",
                ("# Dim", "Expected"),
                (self.coords.ndim, 1),
            )
        self.name = name
        self.non_binned = non_binned

    def __post_init__(self):
        """Post process and validation."""
        self.coords = np.asarray(self.coords)
        if self.coords.ndim != 1:
            raise ShapeError(
                "Dimensions can only have vector coordinates",
                ("# Dim", "Expected"),
                (self.coords.ndim, 1),
            )

    @property
    def size(self) -> int:
        """Size of the dimension."""
        return len(self.coords)

    def to_dict(self) -> dict[str, Any]:
        """Convert dimension to dictionary."""
        return dict(
            name=self.name, coords=self.coords.tolist(), non_binned=self.non_binned
        )

    @classmethod
    def from_dict(cls, other: dict[str, Any]) -> Dimension:
        """Make dimension from dictionary."""
        return Dimension(**other)

    def __len__(self) -> int:
        """Length magic."""
        return self.size

    def __getitem__(self, item: int | slice | list[int]) -> NDArray:
        """Gets a specific coordinate value by index."""
        return self.coords[item]

    def __setitem__(self, key: int, value: Any) -> None:
        """Sets a specific coordinate value by index."""
        self.coords[key] = value

    def __hash__(self) -> int:
        """Hashing magic."""
        return hash(tuple(self.coords) + (self.name,) + (self.non_binned,))

    def __eq__(self, other: Dimension) -> bool:
        """Compares if the dimension has same properties."""
        if not isinstance(other, Dimension):
            other_type = type(other).__name__
            raise TypeError(f"Can't compare Dimension with {other_type}")

        return hash(self) == hash(other)

    def min(self) -> NDArray[np.float]:
        """Get minimum value of dimension."""
        return np.min(self.coords)

    def max(self) -> NDArray[np.float]:
        """Get maximum value of dimension."""
        return np.max(self.coords)

    def serialize(self, stream_format: str) -> str:
        """Serialize the dimension into buffer."""
        serializer = DimensionSerializer(stream_format)
        return serializer.serialize(self)

    @classmethod
    def deserialize(cls, stream: str, stream_format: str) -> Dimension:
        """Deserialize buffer into Dimension."""
        serializer = DimensionSerializer(stream_format)
        return serializer.deserialize(stream)


class DimensionSerializer(Serializer):
    """Serializer implementation for Dimension."""

    def serialize(self, dimension: Dimension) -> str:
        """Serialize Dimension into buffer."""
        payload = dict(
            name=dimension.name,
            length=len(dimension),
            coords=dimension.coords.tolist(),
            non_binned=dimension.non_binned,
        )
        return self.serialize_func(payload)

    def deserialize(self, stream: str) -> Dimension:
        """Deserialize buffer into Dimension."""
        signature = inspect.signature(Dimension)

        payload = self.deserialize_func(stream)
        payload = self.validate_payload(payload, signature)

        return Dimension(**payload)
