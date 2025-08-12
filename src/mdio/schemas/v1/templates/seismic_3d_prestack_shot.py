"""Seismic3DPreStackShotTemplate MDIO v1 dataset templates."""

from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.units import AllUnits


class Seismic3DPreStackShotTemplate(AbstractDatasetTemplate):
    """Seismic Shot pre-stack 3D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ["energy_source_point_num", "cable", "channel"]  # Custom coordinates for shot gathers
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"]
        self._var_chunk_shape = [1, 1, 512, 4096]

    @property
    def _name(self) -> str:
        return f"PreStackShotGathers3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "3D",
                "ensembleType": "shot",
                "processingStage": "pre-stack",
            }
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=[name],
                data_type=ScalarType.INT32,
                metadata_info=None,
            )

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "gun",
            dimensions=["energy_source_point_num", "cable", "channel"],
            data_type=ScalarType.UINT8,
            metadata_info=[AllUnits(units_v1=None)],
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=["energy_source_point_num", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=["energy_source_point_num", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=["energy_source_point_num", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=["energy_source_point_num", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
