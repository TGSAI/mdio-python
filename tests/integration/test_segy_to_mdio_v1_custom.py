import xarray as xr

from collections.abc import Sequence

from mdio.converters import segy_to_mdio_v1_custom
from mdio.converters.segy_to_mdio_v1 import StorageLocation
from mdio.schemas.v1.templates.template_registry import TemplateRegistry
from segy.standards import get_segy_standard

from mdio.converters.segy_to_mdio_v1_custom import segy_to_mdio_v1_custom

def test_customize_segy_specs() -> None:
    # def customize_segy_specs(
    #     segy_spec: SegySpec,
    #     index_bytes: Sequence[int] | None = None,
    #     index_names: Sequence[str] | None = None,
    #     index_types: Sequence[str] | None = None,
    # ) -> SegySpec:
    pass


def test_get_segy_specs() -> None:
    # def get_segy_specs(segy_spec: str | StorageLocation) -> SegySpec:
    pass


def test__load_mdio_dataset_seismic_custom_template() -> None:
    # def _load_mdio_dataset_seismic_custom_template(
    #     module: DynamicallyLoadedModule, domain: str
    # ) -> AbstractDatasetTemplate:
    pass


def test_get_registered_mdio_template() -> None:
    # def get_registered_mdio_template(
    #     mdio_template: str | DynamicallyLoadedModule,
    # ) -> AbstractDatasetTemplate:
    pass


def test_segy_to_mdio_v1_custom() -> None:
    """Test the custom SEG-Y to MDIO conversion."""
    pref_path = "/DATA/Teapot/filt_mig"
    mdio_path = f"{pref_path}_custom_v1.mdio"

    index_bytes = (181, 185)
    index_names = ("inline", "crossline")
    index_types = ("int32", "int32")

    segy_to_mdio_v1_custom(
        input= StorageLocation(f"{pref_path}.segy"),
        output= StorageLocation(mdio_path),
        segy_spec= float("1.0"),
        mdio_template= "PostStack3DTime",
        index_bytes=index_bytes,
        index_names=index_names,
        index_types=index_types,
        overwrite=True
    )

    # Load Xarray dataset from the MDIO file
    dataset = xr.open_dataset(mdio_path, engine="zarr")
    pass
