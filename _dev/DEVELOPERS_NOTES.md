# Wring empty XArray / Zarr to a local storage

src/mdio/schemas/v1/dataset_serializer.py

## Issues encountered

1. FIXED: Non-zero size of the serialized data files
2. FIXED: Not clear how to properly set `compressor`, `dimension_separator`, and `fill_value`
3. FIXED: For image_inline chunks[2] are somehow different?

4. `fill_value` for StructuredType is set to null, but "AAAAAAAAAAAAAAAA" is expected

## TO DO:

- Add more unit tests for internal functions
- Add a trest comparing expected and actual .zmetadata for the serialized dataset
