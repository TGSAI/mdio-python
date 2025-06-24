# Developer Notes
What are the goals for MDIO v1:

## Overall API design and implementation
1. Do we want to have a strongly-typed (see pydantic) or dynamic-typed (see dictionary args) API?  
  For example
```Python
    # Strongly typed
    builder.add_dimension(
        "length",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata=[
            AllUnits(units_v1=LengthUnitModel(
                length=LengthUnitEnum.FOOT)),
            UserAttributes(
                attributes={"MGA": 51, "UnitSystem": "Imperial"}),
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

    # dynamically-typed
    builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={
            "unitsV1": {"length": "m"},
            "attributes": {"MGA": 51},
            "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [20]}},
            "statsV1": {
                "count": 100,
                "sum": 1215.1,
                "sumSquares": 125.12,
                "min": 5.61,
                "max": 10.84,
                "histogram": {"binCenters": [1, 2], "counts": [10, 15]},
            },
        },
    )
```
2. How extensive the handling of the edge cases and the invalid arguments should be? This affects the ammount of validation code that needs to be written    
   For example,
    * Should we validate in the code that the units list contain a single item for the dimensions units    
      or should we expect the developer to always pass a single-item list?
    * Should we test the statistics for count > 0 or dim(binCenter) == dim(count) in the case above?

## V1 Schema questions
1. Why do we allow default / empty names?
2. Adding a dimension with the same name multiple times: is it allowed or should it raise an error?
   * It is currently allowed: the second request is ignored
   * Adding a dimension with the same name, but the different size currently throws an error
3. Why do we allow methods with dictionary parameters (non-strongly-typed)?
4. For the add_dimension():
   * Can AllUnits / UserAttributes / ChunkGridMetadata / StatisticsMetadata be repeated in the metadata list?
   * For units, chunkGrid, statsV1 dict should we validate structure of the data passed in?
   * Do we validate the unit string supplied in dictionary parameters? What what if someone supplies ftUS instead of ft?
   * Are multiple dimension attributes allowed (I assume yes)?
5. It is not clear, how RectilinearChunkGrid can be mapped to a single dimension
   ```RectilinearChunkGrid(configuration=RectilinearChunkShape(chunk_shape=[[2,3,4],[2,3,4]]))```
6. StatisticsMetadata accepts list[SummaryStatistics]. what does this mean and does it need to be tested?
7. The pydentic attribute names are different from the v1 schema attributes names.
    'statsV1' <-> 'stats_v1', 'unitsV1' <-> 'units_v1', 'chunkGrid' <-> 'chunk_grid'
    Tgus, we will pass `units_v1` if we use the typesafe API and `'unitsV1` if we use dictionary API
8. Can we add two variables with the same name?
9. Why histogram (e.g., SummaryStatistics) does not have a `histogram_type` attribute?
10. Why 'ftUS' is not supported by the schema?
    Units: what foot does the MDIO uses: the U.S. survey foot or the International Foot?
    The U.S. survey foot is defined as 1200/3937 meters, while the international foot is defined as exactly 0.3048 meters.
    https://www.axiomint.com/survey-foot-versus-international-foot-whats-the-difference/ 
    "The REAL issue is when ... applied to State Plane coordinates in the N2,000,000 and E6,000,000 range! 
    This ... moves a State Plane coordinate position 4 feet by 12 feet.

## Unclear
* Did we have a notion of the fixed increment for inline & xline annotations?
* How is rotation of East/North axes relatively to inline/xline axes is handled
* How is right-handed and left-handed surveys are handled? 
* add_variable - should dimensions argument be required??
  
(src/mdio/schemas/v1/dataset_builder.py)
## Design suggestions
1. Instead of trying to track the state, should we just return a wrapper/pimpl class with the permitted methods?
2. Should we rename add_dimension to add_dimension_variable / add_dimension_annotation to indicate that we not just 
   providing the dimension name, but also creating the dimension variable
4. add_variable - should we call it `append_variable`. add implies that either name or index must be provided.

## Under constructions
* TODO: ??? refactor _BuilderState to make inner class ???
* TODO: Need an example of EdgeDefinedHistogram for add_dimension with histogram

## Bugs
1. I assume we do not want attribute.attribute in the contract (see docs\tutorials\builder.ipynb)
   'metadata': {'unitsV1': {'length': 'm'}, 'attributes': {'attributes': {'MGA': 51}}}

https://osdu.pages.opengroup.org/platform/domain-data-mgmt-services/seismic/open-vds/vds/specification/Metadata.html