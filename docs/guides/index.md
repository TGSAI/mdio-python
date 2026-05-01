# Guides

Welcome to the MDIO guides. This section provides in-depth documentation on advanced features and specialized workflows.

## Topics

```{toctree}
:maxdepth: 1
:titlesonly:

grid_overrides
obn_data_import
```

## Overview

### Grid Overrides

Grid overrides are transformations applied during SEG-Y import that modify how trace headers are interpreted and indexed. They handle complex acquisition geometries like multi-gun acquisition with interleaved shot points.

See [Grid Overrides](grid_overrides.md) for documentation.

### OBN Data Import

Ocean Bottom Node (OBN) data has unique characteristics requiring specialized handling. The OBN guide covers:

- The `ObnReceiverGathers3D` template
- Required grid overrides for OBN
- Component synthesis for single-component data

See [OBN Data Import](obn_data_import.md) for the complete guide.
