"""Custom exceptions for SEG-Y."""

from mdio.exceptions import MDIOError


class InvalidSEGYFileError(MDIOError):
    """Raised when there is an IOError from segyio."""


class GridOverrideInputError(MDIOError):
    """Raised when grid override parameters are not correct."""


class GridOverrideUnknownError(GridOverrideInputError):
    """Raised with custom message when grid override parameter is unknown."""

    def __init__(self, command_name: str) -> None:
        self.command_name = command_name
        self.message = f"Unknown grid override: {command_name}"
        super().__init__(self.message)


class GridOverrideKeysError(GridOverrideInputError):
    """Raised custom message when grid override is not compatible with required keys."""

    def __init__(self, command_name: str, required_keys: set) -> None:
        self.command_name = command_name
        self.required_keys = required_keys
        self.message = f"{command_name} can only be used with {required_keys} keys."
        super().__init__(self.message)


class GridOverrideMissingParameterError(GridOverrideInputError):
    """Raised with custom message when grid override parameters are not correct."""

    def __init__(self, command_name: str, missing_parameter: str) -> None:
        self.command_name = command_name
        self.missing_parameter = missing_parameter
        self.message = f"{command_name} requires {missing_parameter} parameter."
        super().__init__(self.message)


class GridOverrideIncompatibleError(GridOverrideInputError):
    """Raised with custom message when two grid overrides are incompatible."""

    def __init__(self, first_command: str, second_command: str) -> None:
        self.first_command = first_command
        self.second_command = second_command
        self.message = f"{first_command} can't be used together with {second_command}."
        super().__init__(self.message)
