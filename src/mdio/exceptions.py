"""Custom exceptions related to MDIO functionality."""


from __future__ import annotations


class MDIOError(Exception):
    """Base exceptions class."""


class ShapeError(MDIOError):
    """Raised when shapes of two or more things don't match."""

    def __init__(
        self,
        message: str,
        names: tuple[str, str] | None = None,
        shapes: tuple[int, int] | None = None,
    ):
        """Construct shape error.

        Args:
            message: Message to show with the exception.
            names: Names of the variables for the `message`.
            shapes: Shapes of the variables for the `message`.
        """
        if names is not None and shapes is not None:
            # TODO: Add strict=True and remove noqa when minimum Python is 3.10
            shape_dict = zip(names, shapes)  # noqa: B905
            extras = [f"{name}: {shape}" for name, shape in shape_dict]
            extras = " <> ".join(extras)

            message = " - ".join([message, extras])

        super().__init__(message)


class WrongTypeError(MDIOError):
    """Raised when types of two or things don't match."""

    def __init__(self, message, name=None, expected=None):
        """Construct type error.

        Args:
            message: Message to show with the exception.
            name: String form of variable's type for the `message`.
            expected: String form of expected type for the `message`.
        """
        if name is not None and expected is not None:
            extras = f"Got: {name} Expected: {expected}"

            message = " - ".join([message, extras])

        super().__init__(message)
