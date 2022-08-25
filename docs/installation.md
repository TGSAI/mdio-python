# Install Instructions

There are different ways to install MDIO:

- Install the latest release.
- Building package from source.

```{note}
We strongly recommend using a virtual environment `venv` or `conda`
to avoid potential conflicts with other Python packages.
```

## Using `pip` and `virtualenv`.

Install the 64-bit version of Python 3 from https://www.python.org.

```shell
$ python -m venv mdio-venv
$ mdio-venv\Scripts\activate
$ pip install -U multidimio
```

## Using `conda`

MDIO can also be installed in a `conda` environment.

```{warning}
Native `conda` installation for MDIO is work-in-progress. Due to the bundled
packages with the [Anaconda](https://www.anaconda.com/products/distribution)
distrtibution, installing MDIO using `pip` into the `conda` environment will
cause your environment to break.

In the meantime, please use the
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution which
doesn't come with bundled packages.
```

We first run the following to create and activate an environment:

```shell
$ conda create -n mdio-env
$ conda activate mdio-env
```

## Checking Installation

After installing MDIO, run the following:

```shell
python -c import mdio; print(mdio.__version__)"
```

You should see the version of MDIO printed to the screen.

## Extras

You can also install some optional dependencies (extras) like this:

```shell
pip install multidimio[distributed]
pip install multidimio[cloud]
pip install multidimio[lossy]
```

`distributed` installs [Dask][dask] for parallel, distributed processing.\
`cloud` installs [fsspec][fsspec] backed I/O libraries for [AWS' S3][s3fs],
[Google's GCS][gcsfs], and [Azure ABS][adlfs].\
`lossy` will install the [ZFPY][zfp] library for lossy chunk compression.
