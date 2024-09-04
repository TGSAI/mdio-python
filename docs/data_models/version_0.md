```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.schema.v0.dataset

```

# MDIO v0

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Intro

```{eval-rst}
.. autosummary::
   DatasetModelV0
   VariableModelV0
   DatasetMetadataModelV0
   DimensionModelV0
```

## Reference

:::{dropdown} Dataset
:open:

```{eval-rst}
.. autopydantic_model:: DatasetModelV0
    :inherited-members: BaseModel
.. autopydantic_model:: DatasetMetadataModelV0
    :inherited-members: BaseModel
.. autopydantic_model:: DimensionModelV0
    :inherited-members: BaseModel
```

:::

:::{dropdown} Variable
:open:

```{eval-rst}
.. autopydantic_model:: VariableModelV0
    :inherited-members: BaseModel
```

:::
