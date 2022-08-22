"""Custom exceptions for MDIO converters."""


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
