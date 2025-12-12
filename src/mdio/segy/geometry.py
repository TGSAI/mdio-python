"""SEG-Y geometry configuration models and enumerations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class GridOverrides(BaseModel):
    """Type-safe configuration for grid override operations during SEG-Y ingestion.

    Grid overrides allow automatic handling of non-standard SEG-Y geometries by applying
    transformations to trace headers and indexing during ingestion.

    Attributes:
        auto_channel_wrap: Automatically determine streamer acquisition type and wrap channel
            indices. Used when channels are numbered sequentially across cables (Type B) instead
            of restarting at 1 for each cable (Type A).
        auto_shot_wrap: Automatically determine multi-gun acquisition type and create shot_index
            from shot_point. Used when shot points are numbered uniquely across guns instead of
            restarting for each gun.
        non_binned: Index traces in a single axis without spatial binning. Requires 'chunksize'
            parameter to specify the chunk size for the trace dimension. Optionally use
            'replace_dims' to specify which dimensions to collapse into the trace dimension.
        has_duplicates: Handle datasets with duplicate trace indices by adding a trace dimension.
            Similar to NonBinned but uses fixed chunksize of 1.
        chunksize: Chunk size for the trace dimension when using NonBinned override. Only used
            if non_binned=True. Must be positive if provided.
        replace_dims: List of dimension names to replace with the trace dimension when using
            NonBinned override. If not provided, all spatial dimensions except the first are
            collapsed into trace. Only used if non_binned=True.
        extra_params: Additional parameters for future grid override extensions. Allows passing
            arbitrary key-value pairs without breaking the API.

    Examples:
        >>> # Create with snake_case or aliases (both work!)
        >>> overrides = GridOverrides(auto_channel_wrap=True, chunksize=64)
        >>> overrides = GridOverrides(AutoChannelWrap=True, chunksize=64)
        >>>
        >>> # Non-binned with specific dimensions to replace
        >>> overrides = GridOverrides(non_binned=True, chunksize=128, replace_dims=["cable", "channel"])
        >>>
        >>> # From dict - Pydantic handles both formats automatically
        >>> overrides = GridOverrides.model_validate({"AutoChannelWrap": True})
        >>> overrides = GridOverrides.model_validate({"auto_channel_wrap": True})
        >>>
        >>> # To dict with legacy keys (for metadata/serialization)
        >>> config = overrides.model_dump(by_alias=True, exclude_defaults=True)
        >>> # {"AutoChannelWrap": True, "chunksize": 64}
        >>>
        >>> # Or modern keys
        >>> config = overrides.model_dump(exclude_defaults=True)
        >>> # {"auto_channel_wrap": True, "chunksize": 64}
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    auto_channel_wrap: bool = Field(
        default=False, alias="AutoChannelWrap", description="Enable automatic channel wrapping"
    )
    auto_shot_wrap: bool = Field(default=False, alias="AutoShotWrap", description="Enable automatic shot wrapping")
    non_binned: bool = Field(default=False, alias="NonBinned", description="Use non-binned indexing")
    has_duplicates: bool = Field(default=False, alias="HasDuplicates", description="Handle duplicate indices")
    chunksize: int | None = Field(default=None, description="Chunk size for trace dimension", gt=0)
    replace_dims: list[str] | None = Field(
        default=None, description="Dimension names to replace with trace dimension for NonBinned"
    )
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    @field_validator("chunksize")
    @classmethod
    def validate_chunksize(cls, v: int | None) -> int | None:
        """Validate that chunksize is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("chunksize must be positive")
        return v

    def __bool__(self) -> bool:
        """Return True if any override is enabled."""
        return self.auto_channel_wrap or self.auto_shot_wrap or self.non_binned or self.has_duplicates
