"""Custom exceptions for SEG-Y."""


from mdio.exceptions import MDIOError


class InvalidSEGYFileError(MDIOError):
    """Raised when there is an IOError from segyio."""

    pass
