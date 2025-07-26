from segy.standards import get_segy_standard

from mdio.converters.segy_to_mdio_v1 import segy_to_mdio_v1
from mdio.converters.segy_to_mdio_v1_custom import StorageLocation

from mdio.schemas.v1.templates.template_registry import TemplateRegistry

def test_segy_to_mdio_v1() -> None:
    pref_path = "/DATA/export_masked/3d_stack"

    segy_to_mdio_v1(
        input= StorageLocation(f"{pref_path}.sgy"),
        output= StorageLocation(f"{pref_path}_tmp.mdio"),
        segy_spec= get_segy_standard(1.0),
        mdio_template= TemplateRegistry().get("PostStack3DTime"))

    pass
