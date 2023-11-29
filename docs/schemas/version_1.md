```{eval-rst}
:tocdepth: 3
```

# Version 1

## Data Models

```{eval-rst}
.. automodule:: mdio.schemas.v1.dataset
    :members: Dataset
    :inherited-members: BaseModel
```

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v1.variable.Variable
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.schemas.v1.variable.Coordinate
    :inherited-members: BaseModel
```

## Data Types

```{eval-rst}
.. autopydantic_model:: mdio.schemas.base.scalar.StructuredType

.. autopydantic_model:: mdio.schemas.base.scalar.StructuredField
```

## Metadata

```{eval-rst}
.. automodule:: mdio.schemas.base.metadata
    :members: UserAttributes
```

```{eval-rst}
.. automodule:: mdio.schemas.base.dimension
    :members:
    :undoc-members:
```

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v1.units.AllUnits

.. autopydantic_model:: mdio.schemas.v1.units.CoordinateUnits
```

```{eval-rst}
.. automodule:: mdio.schemas.v1.units
    :members: LengthUnitModel,
              TimeUnitModel,
              AngleUnitModel,
              DensityUnitModel,
              SpeedUnitModel,
              FrequencyUnitModel
```

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v1.stats.StatisticsMetadata

.. autopydantic_model:: mdio.schemas.v1.stats.SummaryStatistics

.. autopydantic_model:: mdio.schemas.v1.stats.EdgeDefinedHistogram
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.schemas.v1.stats.CenteredBinHistogram
    :inherited-members: BaseModel
```

## Enumerations

```{eval-rst}
.. autoclass:: mdio.schemas.base.scalar.ScalarType()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.schemas.v1.units.AngleUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.schemas.v1.units.DensityUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.schemas.v1.units.FrequencyUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.schemas.v1.units.LengthUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.schemas.v1.units.SpeedUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.schemas.v1.units.TimeUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource
```
