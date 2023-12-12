```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.schemas.v1.dataset

```

# MDIO v1

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Intro

```{eval-rst}
.. autosummary:: Dataset
.. autosummary:: DatasetMetadata
```

## Reference

:::{dropdown} Dataset
:open:

```{eval-rst}
.. autopydantic_model:: Dataset
    :inherited-members: BaseModel

.. autopydantic_model:: DatasetMetadata
    :inherited-members: BaseModel
```

:::
:::{dropdown} Variable

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v1.variable.Variable
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.schemas.v1.variable.Coordinate
    :inherited-members: BaseModel

.. automodule:: mdio.schemas.metadata
    :members: UserAttributes

.. autopydantic_model:: mdio.schemas.v1.variable.VariableMetadata
    :inherited-members: BaseModel
```

:::

:::{dropdown} Units

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v1.units.AllUnits
```

```{eval-rst}
.. automodule:: mdio.schemas.v1.units
    :members: LengthUnitModel,
              TimeUnitModel,
              AngleUnitModel,
              DensityUnitModel,
              SpeedUnitModel,
              FrequencyUnitModel,
              VoltageUnitModel
```

:::

:::{dropdown} Stats

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v1.stats.StatisticsMetadata

.. autopydantic_model:: mdio.schemas.v1.stats.SummaryStatistics

.. autopydantic_model:: mdio.schemas.v1.stats.EdgeDefinedHistogram
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.schemas.v1.stats.CenteredBinHistogram
    :inherited-members: BaseModel
```

:::

:::{dropdown} Enums

```{eval-rst}
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

.. autoclass:: mdio.schemas.v1.units.VoltageUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource
```

:::
