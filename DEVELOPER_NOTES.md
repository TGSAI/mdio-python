# Developer Notes

## MDIO v1 scope of work

### TASK 1: Creation an empty MDIO v1 dataset with metadata defined using the v1 schema   
#### DESCRIPTION
In the v0, the following code was used to create an empty dataset:  

```Python
grid = Grid([Dimension("inline",...), Dimension("crossline", ...), Dimension("depth", ...)])
variable = MDIOVariableConfig("stack_amplitude", ...)
create_conf = MDIOCreateConfig(path="demo.mdio", grid=grid, variables=[variable])
create_empty(config=create_conf)
```

In the v1 it is replaced with the following API, which uses v1 schema:

```Python
builder = MDIODatasetBuilder(...)
builder.add_dimension("inline", ...)
builder.add_dimension("crossline",...)
builder.add_dimension("depth", ...)
builder.add_coordinate("cdp_x",...)
builder.add_coordinate("cdp_y",...)
builder.add_variable("stack_amplitude",...)
builder.to_mdio(store="demo.mdio")
```

#### DEFINITION OF DONE
* The resulting v1 MDIO control `demo.mdio` file structure must be identical between Python and C++
* Code coverage 90%
* Code documentation will be updated:
    * API doc strings are reviewed
    * docs/tutorials/creation.ipynb - current version describes v0 API. Should be updated with v1 API
    * docs/api_reference.md - will be updated with new API

#### ASSUMPTIONS
We expect that the following v0 workflows to keep working with this change
* Populating MDIOs
* Updating File and Checking with MDIOReader
* Write to SEG-Y

## Overall API design and implementation
We will have only a strongly-typed (see pydantic) API. For example:

```Python
VariableMetadataList: TypeAlias = list[AllUnits | UserAttributes | ChunkGridMetadata | StatisticsMetadata]
def add_dimension(
    self,
    name: str,
    size: int,
    long_name: str = None,
    data_type: ScalarType | StructuredType = ScalarType.INT32,
    metadata_info: VariableMetadataList | None = None,
) -> "MDIODatasetBuilder":
```

Which will be used as following:

```Python
builder.add_dimension(
    "length",
    size=100,
    data_type=ScalarType.FLOAT32,
    metadata_info=[
        AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT)),
        UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"}),
        ChunkGridMetadata(
            chunk_grid=RegularChunkGrid(
                configuration=RegularChunkShape(
                    chunk_shape=[20]))),
        StatisticsMetadata(stats_v1=SummaryStatistics(
            count=100,
            sum=1215.1,
            sumSquares=125.12,
            min=5.61,
            max=10.84,
            histogram=CenteredBinHistogram(
                binCenters=[1, 2], 
                counts=[10, 15])))
    ]
)
```
### Notes
* When a coordinate or a variable is created, their schema allows to store their dimensions either as
    * list of dimensions name `list[str]`, where the names refer to the dimensions defined in the builder._dimensions 
    * list of named dimensions `list[NamedDimension]`, which duplicate the dimensions defined in the builder._dimensions
    * Mixture of the two above `list[NamedDimension | str]` 

    which approach should be used?  
    
    `RESOLUTION: We will be using the first approach.`

    **IMPORTANT: For binary compatibility, We need to ensure that the C++ code follows the same logic**

* Metadata population from a dictionary in add_coordinate() and add_variable() will not be supported to ensure that the API is strongly-typed. If it is needed, such conversion should be done as a separate step:
    ```Python
    def make_variable_metadata_list_from_dict(metadata: dict[str, Any]) -> VariableMetadataList:
        # Implementation goes here
    def make_coordinate_metadata_list_from_dict(metadata: dict[str, Any]) -> CoordinateMetadataList:
        # Implementation goes here
    ```
    `RESOLUTION: The approach confirmed.`

## Schema V1 questions

* add_dimension(): Can a dimension with the same name be added multiple times. Options:
   * Allowed: the second request is ignored (current implementation)
   * Not Allowed: should it raise an error?

    `RESOLUTION: The dimensions with the same name are not allowed`
* The pydantic attribute names are different from the v1 schema attributes names.  What are the repercussions?
    ```
    'statsV1' <-> 'stats_v1'
    'unitsV1' <-> 'units_v1'
    'chunkGrid' <-> 'chunk_grid'
    ```
    `Under investigation`
* Should histogram (e.g., SummaryStatistics) have a `histogram_type` attribute?

  `Under investigation`
* Units
  * Why 'ftUS' is not supported by the schema? U.S. survey foot vs the International Foot:  
    *"The U.S. survey foot is defined as 1200/3937 meters, while the international foot is defined as exactly 0.3048 meters.
    https://www.axiomint.com/survey-foot-versus-international-foot-whats-the-difference/ 
    "The REAL issue is when ... applied to State Plane coordinates in the N2,000,000 and E6,000,000 range! 
    This ... moves a State Plane coordinate position 4 feet by 12 feet."*
  * Why there are no dimensionless unis (for seismic amplitudes, inlines, etc.)

  `Under investigation`

## Design suggestions
* Should we rename add_dimension to add_dimension_variable (or similar) to indicate that we not just providing the dimension name, but also creating the dimension variable

    `RESOLUTION: Shorter names are preferable for public API. The function behavior will be described in the docs`

