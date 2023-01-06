"""SEG-Y header abstractions."""


from collections import abc
from dataclasses import dataclass
from dataclasses import field

import numpy as np

from mdio.segy.byte_utils import OrderedType


@dataclass
class Header(OrderedType):
    """Header data type class.

    Attributes:
        type: Type definition.
        endian: Endianness of the data type.
        name: Name of the header field.
        offset: Byte offset of header field.
    """

    name: str
    offset: int


@dataclass
class HeaderGroup(abc.MutableSequence):
    """Group of headers. Such as: SEG-Y binary header or trace header.

    Args:
        name: Name of header group.
        offset: Byte offset of the group.
        itemsize: Expected length of the header group.
        headers: List of header objects. Default is an empty list.

    Attributes:
        dtype: Structured `numpy` data type.

    Methods:
    append: Append header to the list of headers.
    insert: Insert new header to given index.
    byteswap: Swaps endianness in place.
    """

    name: str
    offset: int
    itemsize: int

    headers: list[Header] = field(default_factory=list)

    def append(self, header: Header) -> None:
        """Append a new header."""
        self.headers.append(header)

    def insert(self, index: int, header) -> None:
        """Insert a new header to given index."""
        self.headers.insert(index, header)

    def __getitem__(self, item) -> Header:
        """Get a specific header by index."""
        return self.headers.__getitem__(item)

    def __setitem__(self, key, value) -> None:
        """Set a specific header by index."""
        self.headers.__setitem__(key, value)

    def __len__(self) -> int:
        """Size of struct in bytes."""
        return self.dtype.itemsize

    def __delitem__(self, key) -> None:
        """Delete header by index."""
        self.headers.__delitem__(key)

    @property
    def dtype(self) -> np.dtype:
        """Return Numpy dtype of the struct."""
        names = []
        formats = []
        offsets = []
        for header in self.headers:
            names.append(header.name)
            formats.append(header.dtype)
            offsets.append(header.offset)

        # TODO: Add strict=True and remove noqa when minimum Python is 3.10
        headers_sort = sorted(zip(offsets, names, formats))  # noqa: B905
        offsets, names, formats = zip(*headers_sort)  # noqa: B905

        dtype_dict = dict(
            names=names,
            formats=formats,
            offsets=offsets,
            itemsize=self.itemsize,
        )

        return np.dtype(dtype_dict)

    def byteswap(self):
        """Swap endianness in place."""
        [header.byteswap() for header in self.headers]
