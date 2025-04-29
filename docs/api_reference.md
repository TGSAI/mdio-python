# API Reference

## Readers / Writers

```{eval-rst}
.. automodule:: mdio.api.accessor
   :members:
```

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
   :exclude-members: grid_density_qc, parse_index_types, get_compressor

.. automodule:: mdio.converters.mdio
   :members:

.. automodule:: mdio.converters.numpy
   :members:
```

## Convenience Functions

```{eval-rst}
.. automodule:: mdio.api.convenience
   :members:
   :exclude-members: create_rechunk_plan, write_rechunked_values
```

## Core Functionality

### Dimensions

```{eval-rst}
.. automodule:: mdio.core.dimension
   :members:
```

### Creation

```{eval-rst}
.. automodule:: mdio.core.factory
   :members:
```

### Data I/O

```{eval-rst}
.. automodule:: mdio.core.serialization
   :members:
```
