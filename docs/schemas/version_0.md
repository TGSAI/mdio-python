# Version 0

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Dataset

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v0.dataset.Dataset
    :inherited-members: BaseModel

.. autopydantic_model:: mdio.schemas.v0.variable.Variable
    :inherited-members: BaseModel
```

## Metadata

```{eval-rst}
.. autopydantic_model:: mdio.schemas.v0.variable.VariableMetadata
    :inherited-members: BaseModel
```
