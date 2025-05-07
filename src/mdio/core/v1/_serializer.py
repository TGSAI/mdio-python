"""Internal serialization module for MDIO v1 datasets.

This module contains internal implementation details for serializing MDIO schema models
to Zarr storage. This API is not considered stable and may change without notice.
"""

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from numcodecs import Blosc as NumcodecsBlosc

from mdio.core.v1._overloads import mdio
from mdio.schema.compressors import ZFP
from mdio.schema.compressors import Blosc
from mdio.schema.dimension import NamedDimension
from mdio.schema.dtype import ScalarType
from mdio.schema.dtype import StructuredType
from mdio.schema.metadata import UserAttributes
from mdio.schema.v1.dataset import Dataset as MDIODataset
from mdio.schema.v1.dataset import DatasetMetadata
from mdio.schema.v1.units import AllUnits
from mdio.schema.v1.variable import Coordinate
from mdio.schema.v1.variable import Variable
from mdio.schema.v1.variable import VariableMetadata


try:
    import zfpy as zfpy_base  # Base library
    from numcodecs import ZFPY  # Codec
except ImportError:
    print(f"Tried to import zfpy and numcodecs zfpy but failed because {ImportError}")
    zfpy_base = None
    ZFPY = None


def make_named_dimension(name: str, size: int) -> NamedDimension:
    """Create a NamedDimension with the given name and size."""
    return NamedDimension(name=name, size=size)


def make_coordinate(
    name: str,
    dimensions: List[NamedDimension | str],
    data_type: ScalarType | StructuredType,
    long_name: str = None,
    metadata: Optional[List[AllUnits | UserAttributes]] = None,
) -> Coordinate:
    """Create a Coordinate with the given name, dimensions, data_type, and metadata."""
    return Coordinate(
        name=name,
        long_name=long_name,
        dimensions=dimensions,
        data_type=data_type,
        metadata=metadata,
    )


def make_variable(  # noqa: C901
    name: str,
    dimensions: List[NamedDimension | str],
    data_type: ScalarType | StructuredType,
    long_name: str = None,
    compressor: Blosc | ZFP | None = None,
    coordinates: Optional[List[Coordinate | str]] = None,
    metadata: Optional[
        List[AllUnits | UserAttributes] | Dict[str, Any] | VariableMetadata
    ] = None,
) -> Variable:
    """Create a Variable with the given parameters.

    Args:
        name: Name of the variable
        dimensions: List of dimensions
        data_type: Data type of the variable
        long_name: Optional long name
        compressor: Optional compressor
        coordinates: Optional list of coordinates
        metadata: Optional metadata

    Returns:
        Variable: A Variable instance with the specified parameters.

    Raises:
        TypeError: If the metadata type is not supported.
    """
    # Convert metadata to VariableMetadata if needed
    var_metadata = None
    if metadata:
        if isinstance(metadata, list):
            # Convert list of metadata to dict
            metadata_dict = {}
            for md in metadata:
                if isinstance(md, AllUnits):
                    # For units_v1, if it's a single element, use it directly
                    if isinstance(md.units_v1, list) and len(md.units_v1) == 1:
                        metadata_dict["units_v1"] = md.units_v1[0]
                    else:
                        metadata_dict["units_v1"] = md.units_v1
                elif isinstance(md, UserAttributes):
                    # For attributes, if it's a single element, use it directly
                    attrs = md.model_dump(by_alias=True)
                    if isinstance(attrs, list) and len(attrs) == 1:
                        metadata_dict["attributes"] = attrs[0]
                    else:
                        metadata_dict["attributes"] = attrs
            var_metadata = VariableMetadata(**metadata_dict)
        elif isinstance(metadata, dict):
            # Convert camelCase keys to snake_case for VariableMetadata
            converted_dict = {}
            for key, value in metadata.items():
                if key == "unitsV1":
                    # For units_v1, if it's a single element array, use the element directly
                    if isinstance(value, list) and len(value) == 1:
                        converted_dict["units_v1"] = value[0]
                    else:
                        converted_dict["units_v1"] = value
                else:
                    converted_dict[key] = value
            var_metadata = VariableMetadata(**converted_dict)
        elif isinstance(metadata, VariableMetadata):
            var_metadata = metadata
        else:
            raise TypeError(f"Unsupported metadata type: {type(metadata)}")

    # Create the variable with all attributes explicitly set
    return Variable(
        name=name,
        long_name=long_name,
        dimensions=dimensions,
        data_type=data_type,
        compressor=compressor,
        coordinates=coordinates,
        metadata=var_metadata,
    )


def make_dataset_metadata(
    name: str,
    api_version: str,
    created_on: datetime,
    attributes: Optional[Dict[str, Any]] = None,
) -> DatasetMetadata:
    """Create a DatasetMetadata with name, api_version, created_on, and optional attributes."""
    return DatasetMetadata(
        name=name,
        api_version=api_version,
        created_on=created_on,
        attributes=attributes,
    )


def make_dataset(
    variables: List[Variable],
    metadata: DatasetMetadata,
) -> MDIODataset:
    """Create a Dataset with the given variables and metadata."""
    return MDIODataset(
        variables=variables,
        metadata=metadata,
    )


def _convert_compressor(
    model: Blosc | ZFP | None,
) -> NumcodecsBlosc | ZFPY | None:
    if isinstance(model, Blosc):
        return NumcodecsBlosc(
            cname=model.algorithm.value,
            clevel=model.level,
            shuffle=model.shuffle.value,
            blocksize=model.blocksize if model.blocksize > 0 else 0,
        )
    elif isinstance(model, ZFP):
        if zfpy_base is None or ZFPY is None:
            raise ImportError("zfpy and numcodecs are required to use ZFP compression")
        return ZFPY(
            mode=model.mode.value,
            tolerance=model.tolerance,
            rate=model.rate,
            precision=model.precision,
        )
    elif model is None:
        return None
    else:
        raise TypeError(f"Unsupported compressor model: {type(model)}")


def _construct_mdio_dataset(mdio_ds: MDIODataset) -> mdio.Dataset:  # noqa: C901
    """Build an MDIO dataset with correct dimensions and dtypes.

    This internal function constructs the underlying data structure for an MDIO dataset,
    handling dimension mapping, data types, and metadata organization.

    Args:
        mdio_ds: The source MDIO dataset to construct from.

    Returns:
        The constructed dataset with proper MDIO structure and metadata.

    Raises:
        TypeError: If an unsupported data type is encountered.
    """
    # Collect dimension sizes
    dims: dict[str, int] = {}
    for var in mdio_ds.variables:
        for d in var.dimensions:
            if isinstance(d, NamedDimension):
                dims[d.name] = d.size

    # Build data variables
    data_vars: dict[str, mdio.DataArray] = {}
    for var in mdio_ds.variables:
        dim_names = [
            d.name if isinstance(d, NamedDimension) else d for d in var.dimensions
        ]
        shape = tuple(dims[name] for name in dim_names)
        dt = var.data_type
        if isinstance(dt, ScalarType):
            dtype = np.dtype(dt.value)
        elif isinstance(dt, StructuredType):
            dtype = np.dtype([(f.name, f.format.value) for f in dt.fields])
        else:
            raise TypeError(f"Unsupported data_type: {dt}")
        arr = np.zeros(shape, dtype=dtype)
        data_array = mdio.DataArray(arr, dims=dim_names)
        data_array.encoding["fill_value"] = 0.0

        # Set long_name if present
        if var.long_name is not None:
            data_array.attrs["long_name"] = var.long_name

        # Set coordinates if present, excluding dimension names
        if var.coordinates is not None:
            dim_set = set(dim_names)
            coord_names = [
                c.name if isinstance(c, Coordinate) else c
                for c in var.coordinates
                if (c.name if isinstance(c, Coordinate) else c) not in dim_set
            ]
            if coord_names:
                data_array.attrs["coordinates"] = " ".join(coord_names)

        # Attach variable metadata into DataArray attributes
        if var.metadata is not None:
            md = var.metadata.model_dump(
                by_alias=True,
                exclude_none=True,
                exclude={"chunk_grid"},
            )
            for key, value in md.items():
                if isinstance(value, list) and len(value) == 1:
                    md[key] = value[0]
            data_array.attrs.update(md)
        data_vars[var.name] = data_array

    ds = mdio.Dataset(data_vars)
    # Attach dataset metadata
    ds.attrs["apiVersion"] = mdio_ds.metadata.api_version
    ds.attrs["createdOn"] = str(mdio_ds.metadata.created_on)
    ds.attrs["name"] = mdio_ds.metadata.name
    if mdio_ds.metadata.attributes:
        ds.attrs["attributes"] = mdio_ds.metadata.attributes
    return ds
