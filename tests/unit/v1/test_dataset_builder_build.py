# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""

from datetime import UTC, datetime
import pytest

from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset import DatasetInfo, DatasetMetadata
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata

def test_build() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL

    builder.add_dimension("x", 100)
    builder.add_dimension("y", 100)

    builder.build()

def test_play() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    # builder = MDIODatasetBuilder("test_dataset")

    u = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))
    
    u_dict = u.model_dump(mode="json", by_alias=True)
    vm = VariableMetadata(**u_dict)
    vm_dict = vm.dict()
    vm_json = vm.model_dump(mode="json", by_alias=True)

#     vm = VariableMetadata(u)
    
#     dInfo = DatasetInfo(
#             name="My Dataset",
#             api_version="1.0.0",
#             created_on= datetime.now(UTC)),
    
#     d = dict(dInfo.model_dump(mode="json", by_alias=True))
 
#     meta = DatasetMetadata(d)
#     print(dInfo.dict())

#     i = 0