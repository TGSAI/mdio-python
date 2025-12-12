"""Header Analysis System for MDIO Ingestion.

This module provides efficient header extraction from SEG-Y files based on
schema requirements. Only required headers are scanned, reducing memory usage
and processing time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from mdio.segy.parsers import parse_headers

if TYPE_CHECKING:
    from segy.arrays import HeaderArray

    from mdio.ingestion.schema_resolver import ResolvedSchema
    from mdio.segy.file import SegyFileArguments


class HeaderRequirements(BaseModel):
    """Specification of which headers need to be extracted.

    Attributes:
        required_fields: Header fields that must be present
        optional_fields: Header fields that are nice to have but not required
    """

    required_fields: set[str]
    optional_fields: set[str] = set()

    def all_fields(self) -> set[str]:
        """Get all fields (required + optional)."""
        return self.required_fields | self.optional_fields


class HeaderAnalyzer:
    """Analyzes and extracts headers from SEG-Y files.

    This class provides efficient header extraction by only reading
    the headers required for a specific schema.
    """

    @staticmethod
    def requirements_from_schema(schema: ResolvedSchema) -> HeaderRequirements:
        """Determine header requirements from a resolved schema.

        Args:
            schema: Resolved dataset schema

        Returns:
            HeaderRequirements specifying which fields to extract
        """
        return HeaderRequirements(required_fields=schema.required_header_fields())

    def analyze(
        self,
        segy_file_kwargs: SegyFileArguments,
        requirements: HeaderRequirements,
        num_traces: int,
        block_size: int = 10000,
        progress_bar: bool = True,
    ) -> HeaderArray:
        """Extract headers from SEG-Y file based on requirements.

        Args:
            segy_file_kwargs: SEG-Y file arguments
            requirements: Specification of which headers to extract
            num_traces: Total number of traces in the file
            block_size: Number of traces to read per block
            progress_bar: Whether to show progress bar

        Returns:
            HeaderArray with extracted headers
        """
        # Convert set to tuple for parse_headers
        subset = tuple(requirements.all_fields())

        return parse_headers(
            segy_file_kwargs=segy_file_kwargs,
            num_traces=num_traces,
            subset=subset,
            block_size=block_size,
            progress_bar=progress_bar,
        )
