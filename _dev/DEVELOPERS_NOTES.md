# Wring empty XArray / Zarr to a local storage

src/mdio/schemas/v1/dataset_serializer.py

## Issues encountered

1. Non-zero size of the serialized data files
2. Not clear how to properly set `compressor`, `dimension_separator`, and `fill_value`
    * Should `fill_value` be a part f the model?
3. For image_inline chunks[2] are somehow different?