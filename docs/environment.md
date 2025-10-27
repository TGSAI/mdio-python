# Environment Variables

MDIO can be configured using environment variables to customize behavior for import, export,
and validation operations. These variables provide runtime control without requiring code changes.

## CPU and Performance

### `MDIO__EXPORT__CPU_COUNT`

**Type:** Integer  
**Default:** Number of logical CPUs available

Controls the number of CPUs used during SEG-Y export operations. Adjust this to balance
performance with system resource availability.

```shell
$ export MDIO__EXPORT__CPU_COUNT=8
$ mdio segy export input.mdio output.segy
```

### `MDIO__IMPORT__CPU_COUNT`

**Type:** Integer  
**Default:** Number of logical CPUs available

Controls the number of CPUs used during SEG-Y import operations. Higher values can
significantly speed up ingestion of large datasets.

```shell
$ export MDIO__IMPORT__CPU_COUNT=16
$ mdio segy import input.segy output.mdio --header-locations 189,193
```

## Grid Validation

### `MDIO__GRID__SPARSITY_RATIO_WARN`

**Type:** Float  
**Default:** 2.0

Sparsity ratio threshold that triggers warnings during grid validation. The sparsity ratio
measures how sparse the trace grid is compared to a dense grid. Values above this threshold
will log warnings but won't prevent operations.

```shell
$ export MDIO__GRID__SPARSITY_RATIO_WARN=3.0
```

### `MDIO__GRID__SPARSITY_RATIO_LIMIT`

**Type:** Float  
**Default:** 10.0

Sparsity ratio threshold that triggers errors and prevents operations. Use this to enforce
quality standards and prevent ingestion of excessively sparse datasets that may indicate
data quality issues.

```shell
$ export MDIO__GRID__SPARSITY_RATIO_LIMIT=15.0
```

## SEG-Y Processing

### `MDIO__IMPORT__SAVE_SEGY_FILE_HEADER`

**Type:** Boolean  
**Default:** false  
**Accepted values:** `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

When enabled, preserves the original SEG-Y textual file header during import.
This is useful for maintaining full SEG-Y standard compliance and preserving survey metadata.

```shell
$ export MDIO__IMPORT__SAVE_SEGY_FILE_HEADER=true
$ mdio segy import input.segy output.mdio --header-locations 189,193
```

````

### `MDIO__IMPORT__CLOUD_NATIVE`

**Type:** Boolean
**Default:** false
**Accepted values:** `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

Enables buffered reads during SEG-Y header scans to optimize performance when reading from or
writing to cloud object storage (S3, GCS, Azure). This mode balances bandwidth usage with read
latency by processing the file twice: first to determine optimal buffering, then to perform the
actual ingestion.

```{note}
This variable is designed for cloud storage I/O, regardless of where the compute is running.
````

**When to use:**

- Reading from cloud storage (e.g., `s3://bucket/input.segy`)
- Writing to cloud storage (e.g., `gs://bucket/output.mdio`)

**When to skip:**

- Local file paths on fast storage
- Very slow network connections where bandwidth is the primary bottleneck

```shell
$ export MDIO__IMPORT__CLOUD_NATIVE=true
$ mdio segy import s3://bucket/input.segy output.mdio --header-locations 189,193
```

## Development and Testing

### `MDIO_IGNORE_CHECKS`

**Type:** Boolean  
**Default:** false  
**Accepted values:** `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

Bypasses validation checks during MDIO operations. This is primarily intended for development,
testing, or debugging scenarios where you need to work with non-standard data.

```{warning}
Disabling checks can lead to corrupted output or unexpected behavior. Only use this
when you understand the implications and are working in a controlled environment.
```

```shell
$ export MDIO_IGNORE_CHECKS=true
$ mdio segy import input.segy output.mdio --header-locations 189,193
```

## Deprecated Features

### `MDIO__IMPORT__RAW_HEADERS`

**Type:** Boolean  
**Default:** false  
**Accepted values:** `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

```{warning}
This is a deprecated feature and is expected to be removed without warning in a future release.
```

## Configuration Best Practices

### Setting Multiple Variables

You can configure multiple environment variables at once:

```shell
# Set for current session
export MDIO__IMPORT__CPU_COUNT=16
export MDIO__GRID__SPARSITY_RATIO_LIMIT=15.0
export MDIO__IMPORT__CLOUD_NATIVE=true

# Run MDIO commands
mdio segy import input.segy output.mdio --header-locations 189,193
```

### Persistent Configuration

To make environment variables permanent, add them to your shell profile:

**Bash/Zsh:**

```shell
# Add to ~/.bashrc or ~/.zshrc
export MDIO__IMPORT__CPU_COUNT=16
export MDIO__IMPORT__CLOUD_NATIVE=true
```

**Windows:**

```console
# Set permanently in PowerShell (run as Administrator)
[System.Environment]::SetEnvironmentVariable('MDIO__IMPORT__CPU_COUNT', '16', 'User')
```

### Project-Specific Configuration

For project-specific settings, use a `.env` file with tools like `python-dotenv`:

```python
# example_import.py
from dotenv import load_dotenv
import mdio

load_dotenv()  # Load environment variables from .env file
# Your MDIO operations here
```

```shell
# .env file
MDIO__IMPORT__CPU_COUNT=16
MDIO__GRID__SPARSITY_RATIO_LIMIT=15.0
MDIO__IMPORT__CLOUD_NATIVE=true
```
