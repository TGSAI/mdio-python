# Installation

There are different ways to install MDIO:

- Install the latest release via [`pip`](#using-pip-and-virtualenv) or [`conda`](#using-conda).
- Building package [from source](#building-from-source).

```{note}
We strongly recommend using a virtual environment `venv` or `conda`
to avoid potential conflicts with other Python packages.
```

## Using `pip` and `virtualenv`

Install the 64-bit version of Python 3 from https://www.python.org.

Then we can create a `venv` and install _MDIO_.

```shell
$ python -m venv mdio-venv
$ mdio-venv/Scripts/activate
$ pip install -U multidimio
```

To check if installation was successful see [checking installation](#checking-installation).

You can also install some optional dependencies (extras) like this:

```shell
$ pip install multidimio[distributed]
$ pip install multidimio[cloud]
$ pip install multidimio[lossy]
```

`distributed` installs [Dask][dask] for parallel, distributed processing.\
`cloud` installs [fsspec][fsspec] backed I/O libraries for [AWS' S3][s3fs],
[Google's GCS][gcsfs], and [Azure ABS][adlfs].\
`lossy` will install the [ZFPY][zfp] library for lossy chunk compression.

[dask]: https://www.dask.org/
[fsspec]: https://filesystem-spec.readthedocs.io/en/latest/
[s3fs]: https://s3fs.readthedocs.io/
[gcsfs]: https://gcsfs.readthedocs.io/
[adlfs]: https://github.com/fsspec/adlfs
[zfp]: https://computing.llnl.gov/projects/zfp

## Using `conda`

MDIO can also be installed in a `conda` environment.

```{note}
_MDIO_ is hosted in the `conda-forge` channel. Make sure to always provide the
`-c conda-forge` when running `conda install` or else it won't be able to find
the package.
```

We first run the following to create and activate an environment:

```shell
$ conda create -n mdio-env
$ conda activate mdio-env
```

Then we can to install with `conda`:

```shell
$ conda install -c conda-forge multidimio
```

The above command will install MDIO into your `conda` environment.

```{note}
_MDIO_ extras must be installed separately when using `conda`.
```

## Checking Installation

After installing MDIO, run the following:

```shell
$ python -c "import mdio; print(mdio.__version__)"
```

You should see the version of MDIO printed to the screen.

## Building from Source

All dependencies of _MDIO_ are Python packages, so the build process is very simple.
To install from source, we need to clone the repo first and then install locally via `pip`.

```shell
$ git clone https://github.com/TGSAI/mdio-python.git
$ cd mdio-python
$ pip install .
```

We can also install the extras in a similar way, for example:

```shell
$ pip install .[cloud]
```

If you want an editable version of _MDIO_ then we could install it with the command below.
This does allow you to make code changes on the fly.

```shell
$ pip install --editable .[cloud]
```

To check if installation was successful see [checking installation](#checking-installation).
