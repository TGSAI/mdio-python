"""Conversion from Numpy to MDIO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mdio.api.accessor import MDIOWriter
from mdio.converters.segy import get_compressor
from mdio.core.dimension import Dimension
from mdio.core.factory import MDIOCreateConfig
from mdio.core.factory import MDIOVariableConfig
from mdio.core.factory import create_empty
from mdio.core.grid import Grid


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import DTypeLike
    from numpy.typing import NDArray


def numpy_to_mdio(
    array: NDArray,
    mdio_path_or_buffer: str,
    chunksize: tuple[int, ...],
    index_names: list[str] | None = None,
    index_coords: dict[str, NDArray] | None = None,
    header_dtype: DTypeLike | None = None,
    lossless: bool = True,
    compression_tolerance: float = 0.01,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
):
    """Conversion from NumPy array to MDIO format.

    This module provides functionality to convert a NumPy array into the MDIO
    format. The conversion process organizes the input array into a multidimensional
    tensor with specified indexing and compression options.

    Args:
        array: Input NumPy array to be converted to MDIO format.
        mdio_path_or_buffer: Output path for the MDIO file, either local or
            cloud-based (e.g., with `s3://`, `gcs://`, or `abfs://` protocols).
        chunksize: Tuple specifying the chunk sizes for each dimension of the
            array. It must match the number of dimensions in the input array.
        index_names: List of names for the index dimensions. If not provided,
            defaults to `dim_0`, `dim_1`, ..., with the last dimension named
            `sample`.
        index_coords: Dictionary mapping dimension names to their coordinate
            arrays. If not provided, defaults to sequential integers (0 to size-1)
            for each dimension.
        header_dtype: Data type for trace headers, if applicable. Defaults to None.
        lossless: If True, uses lossless Blosc compression with zstandard.
            If False, uses ZFP lossy compression (requires `zfpy` library).
        compression_tolerance: Tolerance for ZFP compression in lossy mode.
            Ignored if `lossless=True`. Default is 0.01, providing ~70% size
            reduction.
        storage_options: Dictionary of storage options for the MDIO output file
            (e.g., cloud credentials). Defaults to None (anonymous access).
        overwrite: If True, overwrites existing MDIO file at the specified path.

    Raises:
        ValueError: If the length of `chunksize` does not match the number of
            dimensions in the input array.
        ValueError: If an element of `index_names` is not included in the
            `index_coords` dictionary.
        ValueError: If any coordinate array in `index_coords` has a size that
            does not match the corresponding array dimension.


    Examples:
        To convert a 3D NumPy array to MDIO format locally with default chunking:

        >>> import numpy as np
        >>> from mdio.converters import numpy_to_mdio
        >>>
        >>> array = np.random.rand(100, 200, 300)
        >>> numpy_to_mdio(
        ...     array=array,
        ...     mdio_path_or_buffer="output/file.mdio",
        ...     chunksize=(64, 64, 64),
        ...     index_names=["inline", "crossline", "sample"],
        ... )

        For a cloud-based output on AWS S3 with custom coordinates:

        >>> coords = {
        ...     "inline": np.arange(0, 100, 2),
        ...     "crossline": np.arange(0, 200, 4),
        ...     "sample": np.linspace(0, 0.3, 300),
        ... }
        >>> numpy_to_mdio(
        ...     array=array,
        ...     mdio_path_or_buffer="s3://bucket/file.mdio",
        ...     chunksize=(32, 32, 128),
        ...     index_names=["inline", "crossline", "sample"],
        ...     index_coords=coords,
        ...     lossless=False,
        ...     compression_tolerance=0.01,
        ... )

        To convert a 2D array with default indexing and lossless compression:

        >>> array_2d = np.random.rand(500, 1000)
        >>> numpy_to_mdio(
        ...     array=array_2d,
        ...     mdio_path_or_buffer="output/file_2d.mdio",
        ...     chunksize=(512, 512),
        ... )
    """
    storage_options = storage_options or {}

    if len(chunksize) != array.ndim:
        message = (
            f"Length of chunks={len(chunksize)} must be ",
            f"equal to array dimensions={array.ndim}",
        )
        raise ValueError(message)

    if index_names is None:
        index_names = index_names or [f"dim_{i}" for i in range(array.ndim - 1)]
        index_names.append("sample")

    if index_coords is None:
        index_coords = {}
        for name, size in zip(index_names, array.shape, strict=True):
            index_coords[name] = np.arange(size)
    else:
        for name, size in zip(index_names, array.shape, strict=True):
            if name not in index_coords:
                message = f"Index name {name} not found in index_coords"
                raise ValueError(message)

            if index_coords[name].size != size:
                message = (
                    f"Size of index_coords[{name}]={index_coords[name].size} "
                    f"does not match array dimension={size}"
                )
                raise ValueError(message)

    suffix = [dim_chunks if dim_chunks > 0 else None for dim_chunks in chunksize]
    suffix = [str(idx) for idx, value in enumerate(suffix) if value is not None]
    suffix = "".join(suffix)

    compressors = get_compressor(lossless, compression_tolerance)
    mdio_var = MDIOVariableConfig(
        name=f"chunked_{suffix}",
        dtype=str(array.dtype),
        chunks=chunksize,
        compressors=compressors,
        header_dtype=header_dtype,
    )

    dims = [Dimension(name=name, coords=index_coords[name]) for name in index_names]
    create_conf = MDIOCreateConfig(
        path=mdio_path_or_buffer,
        grid=Grid(dims),
        variables=[mdio_var],
    )
    create_empty(create_conf, overwrite, storage_options)

    writer = MDIOWriter(mdio_path_or_buffer, suffix, storage_options)
    writer[:] = array
    writer.stats = {
        "mean": array.mean().item(),
        "std": array.std().item(),
        "rms": np.sqrt((array**2).sum() / array.size).item(),
        "min": array.min().item(),
        "max": array.max().item(),
    }
