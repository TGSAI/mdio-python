"""Seismic3DPreStackCocaTemplate MDIO v1 dataset templates."""

from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.units import AllUnits


class Seismic3DPreStackCocaTemplate(AbstractDatasetTemplate):
    """Seismic Shot pre-stack 3D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ["inline", "crossline", "offset", "azimuth"]
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["cdp_x", "cdp_y"]
        self._var_chunk_shape = [8, 8, 32, 1, 1024]

    @property
    def _name(self) -> str:
        return f"PreStackCocaGathers3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "3D",
                "ensembleType": "cdp_coca",
                "processingStage": "pre-stack",
            }
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "inline",
            dimensions=["inline"],
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "crossline",
            dimensions=["crossline"],
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "offset",
            dimensions=["offset"],
            data_type=ScalarType.INT32,
            metadata_info=[self._horizontal_coord_unit],
        )
        angle_unit = AllUnits(units_v1={"angle": "deg"})
        self._builder.add_coordinate(
            "azimuth",
            dimensions=["azimuth"],
            data_type=ScalarType.FLOAT32,
            metadata_info=[angle_unit],
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=[self.trace_domain],
            data_type=ScalarType.INT32,
        )

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=["inline", "crossline"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=["inline", "crossline"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
