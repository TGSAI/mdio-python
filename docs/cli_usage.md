# CLI Usage

## Ingestion and Export

The following example shows how to minimally ingest a 3D seismic stack into
a local **_MDIO_** file. Only one lossless copy will be made.

There are many more options, please see the [CLI Reference](#cli-reference).

```shell
$ mdio segy import \
    path_to_segy_file.segy \
    path_to_mdio_file.mdio \
    -loc 181,185 \
    -names inline,crossline
```

To export the same file back to SEG-Y format, the following command
should be executed.

```shell
$ mdio segy export \
    path_to_mdio_file.mdio \
    path_to_segy_file.segy
```

## Cloud Connection Strings

**_MDIO_** supports I/O on major cloud service providers. The cloud I/O capabilities are
supported using the [fsspec](https://filesystem-spec.readthedocs.io/) and its specialized
version for:

- Amazon Web Services (AWS S3) - [s3fs](https://s3fs.readthedocs.io)
- Google Cloud Provider (GCP GCS) - [gcsfs](https://gcsfs.readthedocs.io)
- Microsoft Azure (Datalake Gen2) - [adlfs](https://github.com/fsspec/adlfs)

Any other file-system supported by `fsspec` will also be supported by **_MDIO_**. However,
we will focus on the major providers here.

The protocols that help choose a backend (i.e. `s3://`, `gs://`, or `az://`) can be passed
prepended to the **_MDIO_** path.

The connection string can be passed to the command-line-interface (CLI) using the
`-storage-{input,output, --storage-options-{input,output}` flag as a JSON string or the Python API with
the `storage_options_{input,output}` keyword argument as a Python dictionary.

````{warning}
On Windows clients, JSON strings are passed to the CLI with a special escape character.

For instance a JSON string:
```json
{"key": "my_super_private_key", "secret": "my_super_private_secret"}
```
must be passed with an escape character `\` for inner quotes as:
```shell
"{\"key\": \"my_super_private_key\", \"secret\": \"my_super_private_secret\"}"
```
whereas, on Linux bash this works just fine:
```shell
'{"key": "my_super_private_key", "secret": "my_super_private_secret"}'
```
If this done incorrectly, you will get an invalid JSON string error from the CLI.
````

### Amazon Web Services

Credentials can be automatically fetched from pre-authenticated AWS CLI.
See [here](https://s3fs.readthedocs.io/en/latest/index.html#credentials) for the order `s3fs`
checks them. If it is not pre-authenticated, you need to pass `--storage-options-{input,output}`.

**Prefix:**  
`s3://`

**Storage Options:**  
`key`: The auth key from AWS  
`secret`: The auth secret from AWS

Using UNIX:

```shell
mdio segy import \
  path/to/my.segy \
  s3://bucket/prefix/my.mdio \
  --header-locations 189,193 \
  --storage-options-output '{"key": "my_super_private_key", "secret": "my_super_private_secret"}'
```

Using Windows (note the extra escape characters `\`):

```console
mdio segy import \
  path/to/my.segy \
  s3://bucket/prefix/my.mdio \
  --header-locations 189,193 \
  --storage-options-output "{\"key\": \"my_super_private_key\", \"secret\": \"my_super_private_secret\"}"
```

### Google Cloud Provider

Credentials can be automatically fetched from pre-authenticated `gcloud` CLI.
See [here](https://gcsfs.readthedocs.io/en/latest/#credentials) for the order `gcsfs`
checks them. If it is not pre-authenticated, you need to pass `--storage-options-{input-output}`.

GCP uses [service accounts](https://cloud.google.com/iam/docs/service-accounts) to pass
authentication information to APIs.

**Prefix:**  
`gs://` or `gcs://`

**Storage Options:**  
`token`: The service account JSON value as string, or local path to JSON

Using a service account:

```shell
mdio segy import \
  path/to/my.segy \
  gs://bucket/prefix/my.mdio \
  --header-locations 189,193 \
  --storage-options-output '{"token": "~/.config/gcloud/application_default_credentials.json"}'
```

Using browser to populate authentication:

```shell
mdio segy import \
  path/to/my.segy \
  gs://bucket/prefix/my.mdio \
  --header-locations 189,193 \
  --storage-options-output '{"token": "browser"}'
```

### Microsoft Azure

There are various ways to authenticate with Azure Data Lake (ADL).
See [here](https://github.com/fsspec/adlfs#details) for some details.
If ADL is not pre-authenticated, you need to pass `--storage-options-{input,output}`.

**Prefix:**  
`az://` or `abfs://`

**Storage Options:**  
`account_name`: Azure Data Lake storage account name  
`account_key`: Azure Data Lake storage account access key

```shell
mdio segy import \
  path/to/my.segy \
  az://bucket/prefix/my.mdio \
  --header-locations 189,193 \
  --storage-options-output '{"account_name": "myaccount", "account_key": "my_super_private_key"}'
```

### Advanced Cloud Features

There are additional functions provided by `fsspec`. These are advanced features and we refer
the user to read `fsspec` [documentation](https://filesystem-spec.readthedocs.io/en/latest/features.html).
Some useful examples are:

- Caching Files Locally
- Remote Write Caching
- File Buffering and random access
- Mount anything with FUSE

#### Buffered Reads in Ingestion

MDIO v0.8.2 introduces the `MDIO__IMPORT__CLOUD_NATIVE` environment variable to optimize
SEG-Y header scans by balancing bandwidth usage with read latency through buffered reads.

**When to Use:** This variable is most effective in high-throughput environments like cloud-based ingestion
systems but can also improve performance for mechanical drives or slow connections.

**How to Enable:** Set the variable to one of `{"True", "1", "true"}`. For example:

```console
$ export MDIO__IMPORT__CLOUD_NATIVE="true"
```

**How It Works:** Buffered reads minimize millions of remote requests during SEG-Y header scans:

- **Cloud Environments:** Ideal for high-throughput connections between cloud ingestion
  machines and object stores.
- **Slow Connections:** Bandwidth is the bottleneck, may be faster without it.
- **Local Reads:** May benefit mechanical drives; SSDs typically perform fine without it.

While buffered reads process the file twice, the tradeoff improves ingestion performance and
reduces object-store request costs.

#### Chaining `fsspec` Protocols

When combining advanced protocols like `simplecache` and using a remote store like `s3` the
URL can be chained like `simplecache::s3://bucket/prefix/file.mdio`. When doing this the
`--storage-options-{input,output}` argument must explicitly state parameters for the cloud backend and the
extra protocol. For the above example it would look like this:

```json
{
  "s3": {
    "key": "my_super_private_key",
    "secret": "my_super_private_secret"
  },
  "simplecache": {
    "cache_storage": "/custom/temp/storage/path"
  }
}
```

In one line:

```json
{"s3": {"key": "my_super_private_key", "secret": "my_super_private_secret"}, "simplecache": {"cache_storage": "/custom/temp/storage/path"}
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
