import numpy as np

from mdio.core.dimension import Dimension

def demultiple_fast(structured: np.ndarray, dims: list[Dimension], field_name: str) -> np.ndarray:
    """
    Optimized structured array demultiplexer.

    This implementation selects the min value from the specified field values for each combination of the given dimensions.
    Fast, but only ChatGPT can grok this from the first glance. :-)

    Args:
        structured: structured numpy array with named fields.
        dims: list of Dimension objects, where every dimension is among the structured array's fields.
        field_name: field name to demultiple.

    Returns:
        np.ndarray: ND array of minimum values for each coordinate combination.
        int: Maximum difference between the min and max values in the field.

    Examples:
        - structured.dtype: 
          [('inline', np.int32), ('crossline', np.int32), ('offset', np.int32), ('azimuth', np.int32), ('cdp_x', np.int32), ('cdp_y', np.int32)]
        - dims:
          [Dimension(structured['inline']), Dimension(structured['crossline'])]
        - field_name: 
          'cdp_x'
    """
    # Build value-to-index maps for each dimension (vectorized)
    dim_dicts = [np.searchsorted(dim.coords, structured[dim.name]) for dim in dims]
    # Stack indices for advanced indexing
    indices = np.stack(dim_dicts, axis=-1)

    shape = tuple(len(dim) for dim in dims)
    # Use numpy's ravel_multi_index for flat indexing
    flat_indices = np.ravel_multi_index(indices.T, shape)
    # Prepare array for minimum reduction
    min_vals = np.full(np.prod(shape), np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(min_vals, flat_indices, structured[field_name])
    max_vals = np.full(np.prod(shape), np.iinfo(np.int32).min, dtype=np.int32)
    np.maximum.at(max_vals, flat_indices, structured[field_name])
    # Reshape to ND
    min_value = min_vals.reshape(shape)
    max_value = max_vals.reshape(shape)
    return min_value, np.max(np.abs(max_value - min_value)).item()

def demultiple_min_slow(struct_data: np.ndarray, dims: list[Dimension], coord_name: str) -> np.ndarray:
    """
    Structured array demultiplexer.

    Slow, but easier to understand the implementation.
    """
    shape = tuple(len(dim) for dim in dims)
    first = np.full(shape, np.iinfo(np.int32).max, dtype=np.int32)
    # Build index lookup for each dimension
    dim_dicts = [{value: idx for idx, value in enumerate(dim)} for dim in dims]
    for row in struct_data:
        indices = tuple(dim_dicts[d][row[dims[d].name]] for d in range(len(dims)))
        first[indices] = min(first[indices], row[coord_name])
    return first

def demultiple_min_slowest(struct_data: np.ndarray, inlines: np.ndarray, crosslines: np.ndarray) -> np.ndarray:
    """
    Simple unoptimized structured array demultiplexer.

    Slow, but easiest to understand the implementation.
    """
    first = np.full((len(inlines), len(crosslines)), np.iinfo(np.int32).max, dtype=np.int32)
    il_dict = {value: index for index, value in enumerate(inlines)}
    xl_dict = {value: index for index, value in enumerate(crosslines)}
    for rec in struct_data:
        i = il_dict[rec['inline']]
        j = xl_dict[rec['crossline']]
        first[i, j] = min(first[i, j], rec['cdp_x'])
    return first