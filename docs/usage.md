# Usage

## Examples

The following example shows how to ingest a 3D seismic stack into
MDIO format. Only one lossless copy will be made.

There are many more options, please see the [usage](#usage).

```shell
mdio segy import \
  -i path_to_segy_file.segy \
  -o path_to_mdio_file.mdio \
  -loc 181,185 \
  -names inline,crossline
```

To export the same file back to SEG-Y format, the following command
should be executed.

```shell
mdio segy export \
  -i path_to_mdio_file.mdio \
  -o path_to_segy_file.segy
```

## CLI Reference

MDIO provides a convenient command-line-interface (CLI) to do
various tasks.

For each command / subcommand you can provide `--help` argument to
get information about usage.

```{eval-rst}
.. click:: mdio.__main__:main
    :prog: mdio
    :nested: full
```
