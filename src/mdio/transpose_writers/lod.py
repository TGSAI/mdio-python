"""Level of Detail (LoD) views for MDIO datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray
    from xarray import Dataset
    from xarray import Variable


def from_variable(
    data: DataArray | Dataset | Variable,
    reduction_factor: int | dict[str, int],
    method: str = "mean",
) -> DataArray | Dataset | Variable:
    """Create a Level of Detail view by downsampling the data.

    Args:
        data: The input data to downsample.
        reduction_factor: Reduction factor for each dimension. Can be a single
            integer (applied to all spatial dimensions) or a dict mapping
            dimension names to reduction factors.
        method: Downsampling method. Options: 'mean', 'max', 'min', 'median'.

    Returns:
        A downsampled copy of the input data.

    Raises:
        NotImplementedError: If method is not supported.
    """
    msg = "Level of Detail operations are not yet implemented"
    raise NotImplementedError(msg)
