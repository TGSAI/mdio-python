"""Conversion from SEG-Y to MDIO v1 format."""

import importlib
import importlib.util
import sys
from collections.abc import Sequence
from typing import Any

import fsspec
from segy.schema import HeaderField
from segy.schema import SegySpec
from segy.standards import get_segy_standard

from mdio.converters.segy_to_mdio_v1 import StorageLocation, segy_to_mdio_v1
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.templates.template_registry import TemplateRegistry


def customize_segy_specs(
    segy_spec: SegySpec,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
) -> SegySpec:
    """Customize SEG-Y specifications with user-defined index fields."""
    if not index_bytes:
        # No customization
        return segy_spec

    index_names = index_names or [f"dim_{i}" for i in range(len(index_bytes))]
    index_types = index_types or ["int32"] * len(index_bytes)

    if not (len(index_names) == len(index_bytes) == len(index_types)):
        raise ValueError("All index fields must have the same length.")

    # Index the dataset using a spec that interprets the user provided index headers.
    index_fields = []
    for name, byte, format_ in zip(index_names, index_bytes, index_types, strict=True):
        index_fields.append(HeaderField(name=name, byte=byte, format=format_))

    custom_spec = segy_spec.customize(trace_header_fields=index_fields)
    return custom_spec


def get_segy_specs(segy_spec: str) -> SegySpec:
    try:
        segy_spec = get_segy_standard(segy_spec)
    except:
        err = f"SEG-Y spec '{segy_spec}' is not registered."
        raise ValueError(err)
    return segy_spec



def get_registered_mdio_template(mdio_template: str) -> AbstractDatasetTemplate:
    """Get an MDIO template.
    If a custom template is loaded, we will register it.
    """

    if not TemplateRegistry().is_registered(mdio_template):
        err = f"MDIO template '{mdio_template}' is not registered."
        raise ValueError(err)
    
    return TemplateRegistry().get(mdio_template)


def segy_to_mdio_v1_custom(
    input: StorageLocation,
    output: StorageLocation,
    segy_spec: str,
    mdio_template: str,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
    overwrite: bool = False,
):
    """A function that converts a SEG-Y file to an MDIO v1 file.

    This function takes in various variations of input parameters and normalizes
    them, performs necessary customizations before calling segy_2_mdio() to
    perform the conversion from SEG-Y and MDIO v1 formats.
    """
    # Retrieve the SEG-Y specifications either from a registry or a storage location
    segy_spec = get_segy_specs(segy_spec)
    # Customize the SEG-Y specs, if customizations are provided
    segy_spec = customize_segy_specs(
        segy_spec, index_bytes=index_bytes, index_names=index_names, index_types=index_types
    )
    # Retrieve MDIO template either from a registry or a storage location
    mdio_template = get_registered_mdio_template(mdio_template)
    # Proceed with the conversion
    segy_to_mdio_v1(
        input=input,
        output=output,
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        overwrite=overwrite,
    )
