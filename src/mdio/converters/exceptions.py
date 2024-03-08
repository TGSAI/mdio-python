"""Custom exceptions for MDIO converters."""


class EnvironmentFormatError(Exception):
    """Raised when environment variable is of the wrong format."""

    def __init__(self, name, format, msg: str = ""):
        """Initialize error."""
        self.message = (
            f"Environment variable: {name} not of expected format: {format}. "
        )
        self.message += f"\n{msg}" if msg else ""
        super().__init__(self.message)


class GridTraceCountError(Exception):
    """Raised when grid trace counts don't match the SEG-Y trace count."""

    def __init__(self, grid_traces, segy_traces):
        """Initialize error."""
        self.message = (
            f"{grid_traces} != {segy_traces}"
            f"Scanned grid trace count ({grid_traces}) doesn't "
            f"match SEG-Y file ({segy_traces}). Either indexing "
            f"parameters are wrong (not unique), or SEG-Y file has "
            f"duplicate traces."
        )

        super().__init__(self.message)


class GridTraceSparsityError(Exception):
    """Raised when mdio grid will be sparsely populated from SEG-Y traces."""

    def __init__(self, shape, num_traces, msg: str = ""):
        """Initialize error."""
        self.message = (
            f"Grid shape: {shape} but SEG-Y tracecount: {num_traces}. "
            "This grid is very sparse and most likely user error with indexing."
        )
        self.message += f"\n{msg}" if msg else ""
        super().__init__(self.message)
