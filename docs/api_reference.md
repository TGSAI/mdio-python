# API Reference

## Data Converters

### Seismic Data

````{note}
By default, the SEG-Y ingestion tool uses Python's multiprocessing
to speed up parsing the data. This almost always requires a `__main__`
guard on any other Python code that is executed directly like
`python file.py`. When running inside Jupyter, this is **NOT** needed.

```python
if __name__ == "__main__":
    segy_to_mdio(...)
```

When the CLI is invoked, this is already handled.

See the official `multiprocessing` documentation
[here](https://docs.python.org/3/library/multiprocessing.html#the-process-class)
and
[here](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming).
````

```{eval-rst}
.. automodule:: mdio.converters.segy
   :members:
   :exclude-members: grid_density_qc, parse_index_types, get_compressor, populate_dim_coordinates, populate_non_dim_coordinates

.. automodule:: mdio.converters.mdio
   :members:
```

## Core Functionality

### Dimensions

```{eval-rst}
.. automodule:: mdio.core.dimension
   :members:
```

## Optimization

```{eval-rst}
.. automodule:: mdio.optimize.access_pattern
   :members:
```
