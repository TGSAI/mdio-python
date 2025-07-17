from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.units import AllUnits


class Seismic3DPreStackShotTemplate(AbstractDatasetTemplate):
    """
    Seismic Shot pre-stack 3D time or depth Dataset template.
    """

    def __init__(self, domain: str):
        super().__init__(domain=domain)
        
        self._coord_dim_names = [] # Custom coordinate definition for shot gathers
        self._dim_names = ["shot_point", "cable", "channel", self._trace_domain]
        self._coord_names = ["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"]
        self._var_name = "AmplitudeShot"
        self._var_chunk_shape = [1, 1, 512, 4096]

    def _get_name(self) -> str:
        return f"PreStackShotGathers3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(attributes={
            "surveyDimensionality": "3D",
            "ensembleType": "shot",
            "processingStage": "pre-stack",
        })

    def _add_coordinates(self) -> None:

        self._builder.add_coordinate("gun",
                                    dimensions=["shot_point"],
                                    data_type=ScalarType.UINT8,
                                    metadata_info=[AllUnits(units_v1=None)])
        self._builder.add_coordinate("shot-x",
                                    dimensions=["shot_point"],
                                    data_type=ScalarType.FLOAT64,
                                    metadata_info=[self._coord_units[0]])
        self._builder.add_coordinate("shot-y",
                                    dimensions=["shot_point"],
                                    data_type=ScalarType.FLOAT64,
                                    metadata_info=[self._coord_units[1]])
        self._builder.add_coordinate("receiver-x",
                                    dimensions=["shot_point", "cable", "channel"],
                                    data_type=ScalarType.FLOAT64,
                                    metadata_info=[self._coord_units[0]])
        self._builder.add_coordinate("receiver-y",
                                    dimensions=["shot_point", "cable", "channel"],
                                    data_type=ScalarType.FLOAT64,
                                    metadata_info=[self._coord_units[1]])
