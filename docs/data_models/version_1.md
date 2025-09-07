```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.builder.schemas.v1.dataset

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
.. autopydantic_model:: mdio.builder.schemas.v1.variable.Variable
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.builder.schemas.v1.variable.Coordinate
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.builder.schemas.v1.variable.CoordinateMetadata
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.builder.schemas.v1.variable.VariableMetadata
    :inherited-members: BaseModel
```

:::

:::{dropdown} Units

```{eval-rst}
.. automodule:: mdio.builder.schemas.v1.units
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
.. autopydantic_model:: mdio.builder.schemas.v1.stats.SummaryStatistics

.. autopydantic_model:: mdio.builder.schemas.v1.stats.EdgeDefinedHistogram
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.builder.schemas.v1.stats.CenteredBinHistogram
    :inherited-members: BaseModel
```

:::

:::{dropdown} Enums

```{eval-rst}
.. autoclass:: mdio.builder.schemas.v1.units.AngleUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.builder.schemas.v1.units.DensityUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.builder.schemas.v1.units.FrequencyUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.builder.schemas.v1.units.LengthUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.builder.schemas.v1.units.SpeedUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.builder.schemas.v1.units.TimeUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mdio.builder.schemas.v1.units.VoltageUnitEnum()
    :members:
    :undoc-members:
    :member-order: bysource
```

:::
