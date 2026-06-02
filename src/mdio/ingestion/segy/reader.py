"""SEG-Y index header reader and grid dimension builder."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mdio.core.dimension import Dimension
from mdio.ingestion.segy.index_strategies import IndexStrategyRegistry
from mdio.segy.parsers import parse_headers

if TYPE_CHECKING:
    import numpy as np

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.ingestion.schema import ResolvedSchema
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.file import SegyFileInfo
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def read_index_headers(  # noqa: PLR0913
    segy_file_kwargs: SegyFileArguments,
    file_info: SegyFileInfo,
    schema: ResolvedSchema,
    grid_overrides: GridOverrides | None,
    synthesize_dims: tuple[str, ...],
    template: AbstractDatasetTemplate | None = None,
) -> tuple[np.ndarray, list[Dimension]]:
    """Parse SEG-Y headers, apply index strategy transformations, and build dimensions.

    Args:
        segy_file_kwargs: Arguments for opening the SEG-Y file.
        file_info: Metadata info of the SEG-Y file.
        schema: Final resolved MDIO schema.
        grid_overrides: Grid override parameters if any.
        synthesize_dims: Dimensions to synthesize if missing from headers.
        template: Optional dataset template for specialized strategy resolution.

    Returns:
        tuple:
            - np.ndarray: The transformed and indexed trace headers.
            - list[Dimension]: The completed list of spatial + vertical Dimensions.
    """
    # 1. Determine subset of header fields to parse
    spec = segy_file_kwargs.get("spec")
    spec_fields = {field.name for field in spec.trace.header.fields} if spec else set()

    # Drop any synthesized or missing dimensions/coordinates that aren't in the physical file spec
    subset = tuple(f for f in schema.required_header_fields() if f in spec_fields)

    # 2. Parse headers
    parsed_headers = parse_headers(
        segy_file_kwargs=segy_file_kwargs,
        num_traces=file_info.num_traces,
        subset=subset,
    )

    # 3. Apply Index Strategy
    strategy = IndexStrategyRegistry().create_strategy(
        grid_overrides=grid_overrides,
        synthesize_dims=synthesize_dims,
        template=template,
    )
    logger.info("Using index strategy: %s", strategy.name)

    indexed_headers = strategy.transform_headers(parsed_headers)

    # 4. Compute spatial dimensions
    dim_names = tuple(d.name for d in schema.dimensions if d.is_spatial)
    dimensions = strategy.compute_dimensions(indexed_headers, dim_names)

    # 5. Append vertical dimension
    sample_labels = file_info.sample_labels / 1000  # normalize
    if all(sample_labels.astype("int64") == sample_labels):
        sample_labels = sample_labels.astype("int64")

    vertical_dim_name = schema.dimensions[-1].name
    dimensions.append(Dimension(coords=sample_labels, name=vertical_dim_name))

    return indexed_headers, dimensions
