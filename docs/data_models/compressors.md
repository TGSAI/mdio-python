```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.schemas.compressors

```

# Compressors

## Dataset Compression

```{eval-rst}
.. autosummary::
   Blosc
   ZFP
```

## Reference

```{eval-rst}
.. autoclass:: Compressors
```

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
