# MDIO

[![PyPI](https://img.shields.io/pypi/v/mdio.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/mdio.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/mdio)][python version]
[![License](https://img.shields.io/pypi/l/mdio)][license]

[![Read the documentation at https://mdio.readthedocs.io/](https://img.shields.io/readthedocs/mdio/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/TGSAI/mdio/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/TGSAI/mdio/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/mdio/
[status]: https://pypi.org/project/mdio/
[python version]: https://pypi.org/project/mdio
[read the docs]: https://mdio.readthedocs.io/
[tests]: https://github.com/TGSAI/mdio/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/TGSAI/mdio
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

**_"MDIO"_** is a library to work with large multi-dimensional energy datasets.
The primary motivation behind **MDIO** is to represent multi-dimensional
time series data in a format that makes it easier to use in resource assesment,
machine learning, and data processing workflows.

## Features

**Shared Features**

- Abstractions for common energy data types (see below).
- Cloud native chunked storage based on [Zarr][zarr] and [fsspec][fsspec].
- Lossy and lossless data compression using [Blosc][blosc] and [ZFP][zfp].
- Distributed reads and writes using [Dask][dask].
- Powerful command-line-interface (CLI) based on [Click][click]

**Domain Specific Features**

- Oil & Gas Data
  - Import and export 2D - 5D seismic data types stored in SEG-Y.
  - Import seismic interpretation, horizon, data (experimental).
  - Optimized chunking logic for various seismic types.
- Wind Resource Assessment
  - Numerical weather prediction models with arbitrary metadata.
  - Optimized chunking logic for time-series analysis and mapping.
  - [Xarray][xarray] interface.

## Requirements

- TODO

## Installation

You can install _MDIO_ via [pip] from [PyPI]:

```console
$ pip install mdio
```

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [Apache 2.0 license][license],
_MDIO_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/TGSAI/mdio-python/issues
[pip]: https://pip.pypa.io/
[dask]: https://www.dask.org/
[zarr]: https://zarr.dev/
[fsspec]: https://filesystem-spec.readthedocs.io/en/latest/
[blosc]: https://www.blosc.org/
[zfp]: https://computing.llnl.gov/projects/zfp
[xarray]: https://xarray.dev/
[click]: https://palletsprojects.com/p/click/

<!-- github-only -->

[license]: https://github.com/TGSAI/mdio-python/blob/main/LICENSE
[contributor guide]: https://github.com/TGSAI/mdio-python/blob/main/CONTRIBUTING.md
[command-line reference]: https://mdio-python.readthedocs.io/en/latest/usage.html
