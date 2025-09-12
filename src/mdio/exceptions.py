"""Custom exceptions related to MDIO functionality."""

from __future__ import annotations


class MDIOError(Exception):
    """Base exceptions class."""


class ShapeError(MDIOError):
    """Raised when shapes of two or more things don't match.

    Args:
        message: Message to show with the exception.
        names: Names of the variables for the `message`.
        shapes: Shapes of the variables for the `message`.
    """

    def __init__(
        self,
        message: str,
        names: tuple[str, str] | None = None,
        shapes: tuple[int, int] | None = None,
    ):
        if names is not None and shapes is not None:
            shape_dict = zip(names, shapes, strict=True)
            extras = [f"{name}: {shape}" for name, shape in shape_dict]
            extras = " <> ".join(extras)

            message = f"{message} - {extras}"

        super().__init__(message)


class WrongTypeError(MDIOError):
    """Raised when types of two or things don't match.

    Args:
        message: Message to show with the exception.
        name: String form of variable's type for the `message`.
        expected: String form of expected type for the `message`.
    """

    def __init__(self, message: str, name: str = None, expected: str = None):
        if name is not None and expected is not None:
            extras = f"Got: {name} Expected: {expected}"
            message = f"{message} - {extras}"

        super().__init__(message)


class InvalidMDIOError(MDIOError):
    """Raised when an invalid MDIO file is encountered."""


class MDIOAlreadyExistsError(MDIOError):
    """Raised when MDIO file already exists."""


class MDIONotFoundError(MDIOError):
    """Raised when MDIO file doesn't exist."""


class MDIOMissingVariableError(MDIOError):
    """Raised when a variable is missing from the MDIO dataset."""
