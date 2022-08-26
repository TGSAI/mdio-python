"""Core exceptions for MDIO."""


from mdio.exceptions import MDIOError


class MDIOAlreadyExistsError(MDIOError):
    """Raised when MDIO file already exists."""

    pass


class MDIONotFoundError(MDIOError):
    """Raised when MDIO file doesn't exist."""

    pass
