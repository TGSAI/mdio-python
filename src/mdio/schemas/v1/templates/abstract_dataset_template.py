from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mdio.schemas import compressors
from mdio.schemas.chunk_grid import RegularChunkGrid, RegularChunkShape
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata, UserAttributes
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.units import AllUnits

class AbstractDatasetTemplate(ABC):
    """
    Abstract base class that defines the template method for Dataset building factory.
    
    The template method defines the skeleton of the data processing algorithm,
    while allowing subclasses to override specific steps.
    """
    
    def __init__(self, domain: str = "") -> None:
        """
        Initializes the template factory with the necessary parameters. 
        """
        # Template attributes to be overridden by subclasses
        # Domain of the seismic data, e.g. "time" or "depth"
        self._trace_domain = domain.lower()
        # Names of all coordinate dimensions in the dataset
        # e.g. ["cdp"] for 2D post-stack depth  
        # e.g. ["inline", "crossline"] for 3D post-stack
        # e.g. ["inline", "crossline"] for 3D pre-stack CDP gathers 
        # Note: For pre-stack Shot gathers, the coordinates are defined differently
        #       and are not directly tied to _coord_dim_names.
        self._coord_dim_names = []
        # Names of all dimensions in the dataset
        # e.g. ["cdp", "depth"] for 2D post-stack depth
        # e.g. ["inline", "crossline", "depth"] for 3D post-stack depth
        # e.g. ["inline", "crossline", "offset", "depth"] for 3D pre-stack depth CPD gathers
        # e.g. ["shot_point", "cable", "channel", "time"] for 3D pre-stack time Shot gathers
        self._dim_names = []
        # Names of all coordinates in the dataset
        # e.g. ["cdp-x", "cdp-y"] for 2D post-stack depth
        # e.g. ["cdp-x", "cdp-y"] for 3D post-stack depth
        # e.g. ["cdp-x", "cdp-y"] for 3D pre-stack CPD depth
        # e.g. ["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"] for 3D pre-stack time Shot gathers
        self._coord_names = []
        # Name of the variable in the dataset
        # e.g. "StackedAmplitude" for 2D post-stack depth
        # e.g. "StackedAmplitude" for 3D post-stack depth
        # e.g. "AmplitudeCDP" for 3D pre-stack CPD depth
        # e.g. "AmplitudeShot" for 3D pre-stack time Shot gathers
        self._var_name = ""
        # Chunk shape for the variable in the dataset
        # e.g. [1024, 1024] for 2D post-stack depth
        # e.g. [128, 128, 128] for 3D post-stack depth
        # e.g. [1, 1, 512, 4096] for 3D pre-stack CPD depth
        # e.g. [1, 1, 512, 4096] for 3D pre-stack time Shot gathers
        self._var_chunk_shape = []
        
        # Variables instantiated when build_dataset() is called
        self._builder: MDIODatasetBuilder = None
        # Sizes of the dimensions in the dataset, to be set when build_dataset() is called
        self._dim_sizes = []
        # Horizontal units for the coordinates (e.g, "m", "ft"), to be set when build_dataset() is called
        self._coord_units = []

    def build_dataset(self, 
                      name: str, 
                      sizes: list[int],
                      coord_units: list[AllUnits] ) -> Dataset:
        """
        Template method that builds the dataset.
        Args:
            name (str): The name of the dataset.
            sizes (list[int]): The sizes of the dimensions.
            coord_units (list[AllUnits]): The units for the coordinates.
        """
        self._dim_sizes = sizes
        self._coord_units = coord_units

        self._builder = MDIODatasetBuilder(
            name=name,
            attributes=self._load_dataset_attributes()
        )
        self._add_dimensions()
        self._add_coordinates()
        self._add_variables()
        return self._builder.build()
    
    def get_name(self) -> str:
        """
        Returns the name of the template.
        """
        return self._get_name()
        

    @abstractmethod
    def _get_name(self) -> str:
        """
        Abstract method to get the name of the template.
        Must be implemented by subclasses.

        Returns:
            str: The name of the template
        """
        pass

    @abstractmethod
    def _load_dataset_attributes(self) -> UserAttributes:
        """
        Abstract method to load dataset attributes.
        Must be implemented by subclasses.

        Returns:
            UserAttributes: The dataset attributes
        """            
        pass
    

    def _add_dimensions(self) -> None:
        """
        A virtual method that can be overwritten by subclasses to add custom dimensions.
        Uses the class field 'builder' to add dimensions to the dataset.
        """
        for i in range(len(self._dim_names)):
            self._builder.add_dimension(self._dim_names[i], self._dim_sizes[i])


    def _add_coordinates(self) -> None:
        """
        A virtual method that can be overwritten by subclasses to add custom coordinates.
        Uses the class field 'builder' to add coordinates to the dataset.
        """
        for i in range(len(self._coord_names)):
            self._builder.add_coordinate(self._coord_names[i],
                             dimensions=self._coord_dim_names,
                             data_type=ScalarType.FLOAT64,
                             metadata_info=[self._coord_units])

    
    def _add_variables(self) -> None:
        """
        A virtual method that can be overwritten by subclasses to add custom variables.
        Uses the class field 'builder' to add variables to the dataset.
        """
        self._builder.add_variable(
            name=self._var_name,
            dimensions=self._dim_names,
            data_type=ScalarType.FLOAT32,
            compressor=compressors.Blosc(algorithm=compressors.BloscAlgorithm.ZSTD),
            coordinates=self._coord_names,
            metadata_info=[
                ChunkGridMetadata(
                    chunk_grid=RegularChunkGrid(
                        configuration=RegularChunkShape(chunk_shape=self._var_chunk_shape))
                )
            ])
    