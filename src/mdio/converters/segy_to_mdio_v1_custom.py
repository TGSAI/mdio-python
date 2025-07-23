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

from mdio.converters.segy_to_mdio_v1 import segy_to_mdio_v1
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.templates.template_registry import TemplateRegistry


class StorageLocation:
    def __init__(self, uri: str, options: dict[str, Any] | None = None):
        self.uri = uri
        self.options = options


class DynamicallyLoadedModule:
    def __init__(self, module_name: str, module_path: str | StorageLocation):
        self.name = module_name
        self.path = module_path


def customize_segy_specs(
    segy_spec: SegySpec,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
) -> SegySpec:
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


def get_segy_specs(segy_spec: str | StorageLocation) -> SegySpec:
    """Get a SEG-Y specification."""
    if isinstance(segy_spec, StorageLocation):
        # TODO: Use storage_options for cloud access, e.g.:
        # with fsspec.open('s3://mybucket/myfile', 'wb', s3={"profile": "writer"}) as f:
        # with fsspec.open('gs://mybucket/myfile', 'wb', gcs={'project': 'my-project'}) as f:
        with fsspec.open(segy_spec.uri, mode="rb") as fp:
            data = fp.read()
            segy_specs = SegySpec(data)
        return segy_specs
    # segy_spec is a name of a registered SEG-Y standard
    try:
        segy_spec = get_segy_standard(segy_spec)
    except:
        err = f"SEG-Y spec '{segy_spec}' is not registered."
        raise ValueError(err)
    return segy_spec


def _load_mdio_dataset_seismic_custom_template(
    module: DynamicallyLoadedModule, domain: str
) -> AbstractDatasetTemplate:
    """Dynamically load an MDIO custom dataset template class SeismicCustomTemplate.

    The template is dynamically loaded from a Python module in a file.
    It is intended to use with the CLI
    """
    if isinstance(module.path, StorageLocation):
        # # For example:
        # storage_client = google.cloud.storage.Client()
        # # Get the bucket and blob
        # bucket = storage_client.bucket(bucket_name)  # Refers to the bucket by name
        # blob = bucket.blob(blob_name)  # Creates a blob object
        # # Create a temporary file to store the downloaded module
        # with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        #     temp_filepath = temp_file.name
        #     blob.download_to_filename(temp_filepath)  # Downloads the file to the temporary location
        #     print(f"Downloaded module to: {temp_filepath}")
        err = "Loading from a StorageLocation is not yet implemented."
        raise RuntimeError(err)
    # Create a module specification from the file location
    spec = importlib.util.spec_from_file_location(module.name, module.path)
    # Create a new module object
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules (optional, but good practice for proper import behavior)
    sys.modules[module.name] = module
    # Execute the module's code
    spec.loader.exec_module(module)
    # Get the class from the module
    SeismicCustomTemplate = module.SeismicCustomTemplate
    # Create an instance of the class
    instance = SeismicCustomTemplate(domain="depth")
    return instance


def get_registered_mdio_template(
    mdio_template: str | DynamicallyLoadedModule,
) -> AbstractDatasetTemplate:
    """Get an MDIO template.

    If a custom template is loaded, we will register it.
    """
    if isinstance(mdio_template, DynamicallyLoadedModule):
        # TODO: Use storage_options for cloud access, e.g.:
        # with fsspec.open('s3://mybucket/myfile', 'wb', s3={"profile": "writer"}) as f:
        # with fsspec.open('s3://mybucket/myfile', 'wb', gcs={'project': 'my-project'}) as f:
        # Importing a module that loads a SeismicCustomTemplate template class dynamically
        template = _load_mdio_dataset_seismic_custom_template(module=mdio_template, domain="depth")
        TemplateRegistry().register(template)
        return template
    # This looks like a name of a registered SEG-Y standard
    if not TemplateRegistry().is_registered(mdio_template):
        err = f"MDIO template '{mdio_template}' is not registered."
        raise ValueError(err)
    return mdio_template


def segy_to_mdio_v1_custom(
    input: StorageLocation,
    output: StorageLocation,
    segy_spec: str | StorageLocation,
    mdio_template: str | StorageLocation,
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
