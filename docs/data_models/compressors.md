```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.schemas.compressors

```

# Compressors

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Dataset Compression

```{eval-rst}
.. autosummary::
   Blosc
   ZFP
```

## Reference

:::
:::{dropdown} Blosc
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: Blosc

----------

.. autoclass:: BloscAlgorithm()
    :members:
    :undoc-members:
    :member-order: bysource

----------

.. autoclass:: BloscShuffle()
    :members:
    :undoc-members:
    :member-order: bysource
```

:::

:::{dropdown} ZFP
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: ZFP

----------

.. autoclass:: ZFPMode()
    :members:
    :undoc-members:
    :member-order: bysource
```

:::
