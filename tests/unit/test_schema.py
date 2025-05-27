"""Test the schema for the v1 dataset."""

from mdio.schemas.v1 import Dataset as V1Dataset

TEST_SCHEMA = {
    "metadata": {
        "name": "test_dataset",
        "apiVersion": "1.0.0",
        "createdOn": "2023-01-01T00:00:00Z",
    },
    "variables": [
        {
            "name": "actual_variable",
            "dataType": "float32",
            "dimensions": ["dim0", "dim1"],
            "compressor": {"name": "blosc", "level": 3},
            "coordinates": ["coord"],
            "metadata": {
                "chunkGrid": {
                    "name": "regular",
                    "configuration": {"chunkShape": [10, 20]},
                },
            },
        },
        {
            "name": "coord",
            "dataType": "float32",
            "dimensions": ["dim0", "dim1"],
            "metadata": {
                "chunkGrid": {
                    "name": "regular",
                    "configuration": {"chunkShape": [10, 20]},
                },
                "unitsV1": {"length": "m"},
            },
        },
        {
            "name": "dim0",
            "dataType": "int32",
            "dimensions": [{"name": "dim0", "size": 100}],
        },
        {
            "name": "dim1",
            "dataType": "int32",
            "dimensions": [{"name": "dim1", "size": 200}],
        },
    ],
}


def test_dataset_schema_validation() -> None:
    """Test that the dataset schema validates correctly."""
    V1Dataset.model_validate(TEST_SCHEMA)
