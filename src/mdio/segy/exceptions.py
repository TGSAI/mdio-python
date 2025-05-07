"""Custom exceptions for SEG-Y."""

from mdio.exceptions import MDIOError


class GridOverrideInputError(MDIOError):
    """Raised when grid override parameters are not correct."""


class GridOverrideUnknownError(GridOverrideInputError):
    """Raised when grid override parameter is unknown.

    Args:
        command_name: Name of the unknown grid override parameter.
    """

    def __init__(self, command_name: str):
        self.command_name = command_name
        self.message = f"Unknown grid override: {command_name}"
        super().__init__(self.message)


class GridOverrideKeysError(GridOverrideInputError):
    """Raised when grid override is not compatible with required keys.

    Args:
        command_name: Name of the grid override command.
        required_keys: Set of required keys for the grid override.
    """

    def __init__(self, command_name: str, required_keys: set[str]):
        self.command_name = command_name
        self.required_keys = required_keys
        self.message = f"{command_name} can only be used with {required_keys} keys."
        super().__init__(self.message)


class GridOverrideMissingParameterError(GridOverrideInputError):
    """Raised when grid override parameters are not correct.

    Args:
        command_name: Name of the grid override command.
        missing_parameter: Set of missing parameters required by the command.
    """

    def __init__(self, command_name: str, missing_parameter: set[str]):
        self.command_name = command_name
        self.missing_parameter = missing_parameter
        self.message = f"{command_name} requires {missing_parameter} parameter."
        super().__init__(self.message)


class GridOverrideIncompatibleError(GridOverrideInputError):
    """Raised when two grid overrides are incompatible.

    Args:
        first_command: Name of the first grid override command.
        second_command: Name of the second grid override command.
    """

    def __init__(self, first_command: str, second_command: str):
        self.first_command = first_command
        self.second_command = second_command
        self.message = f"{first_command} can't be used together with {second_command}."
        super().__init__(self.message)
