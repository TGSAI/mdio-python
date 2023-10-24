"""Custom exceptions for SEG-Y."""


from mdio.exceptions import MDIOError


class InvalidSEGYFileError(MDIOError):
    """Raised when there is an IOError from segyio."""


class GridOverrideInputError(MDIOError):
    """Raised when grid override parameters are not correct."""


class GridOverrideUnknownError(GridOverrideInputError):
    """Raised when grid override parameter is unknown."""

    def __init__(self, command_name):
        """Initialize with custom message."""
        self.command_name = command_name
        self.message = f"Unknown grid override: {command_name}"
        super().__init__(self.message)


class GridOverrideKeysError(GridOverrideInputError):
    """Raised when grid override is not compatible with required keys."""

    def __init__(self, command_name, required_keys):
        """Initialize with custom message."""
        self.command_name = command_name
        self.required_keys = required_keys
        self.message = f"{command_name} can only be used with {required_keys} keys."
        super().__init__(self.message)


class GridOverrideMissingParameterError(GridOverrideInputError):
    """Raised when grid override parameters are not correct."""

    def __init__(self, command_name, missing_parameter):
        """Initialize with custom message."""
        self.command_name = command_name
        self.missing_parameter = missing_parameter
        self.message = f"{command_name} requires {missing_parameter} parameter."
        super().__init__(self.message)


class GridOverrideIncompatibleError(GridOverrideInputError):
    """Raised when two grid overrides are incompatible."""

    def __init__(self, first_command, second_command):
        """Initialize with custom message."""
        self.first_command = first_command
        self.second_command = second_command
        self.message = f"{first_command} can't be used together with {second_command}."
        super().__init__(self.message)
