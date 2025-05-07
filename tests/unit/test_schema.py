"""Test the schema for the v1 dataset."""

from mdio.schemas.v1 import Dataset as V1Dataset

TEST_SCHEMA = {
    "metadata": {
        "name": "test_dataset",
        "api_version": "1.0.0",
        "created_on": "2023-01-01T00:00:00Z",
    },
    "variables": [
        {
            "name": "actual_variable",
            "data_type": "float32",
            "dimensions": ["dim0", "dim1"],
            "compressor": {"name": "blosc", "level": 3},
            "coordinates": ["coord"],
            "metadata": {
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [10, 20]},
                },
            },
        },
        {
            "name": "coord",
            "data_type": "float32",
            "dimensions": ["dim0", "dim1"],
            "metadata": {
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [10, 20]},
                },
                "units_v1": {"length": "m"},
            },
        },
        {
            "name": "dim0",
            "data_type": "int32",
            "dimensions": [{"name": "dim0", "size": 100}],
        },
        {
            "name": "dim1",
            "data_type": "int32",
            "dimensions": [{"name": "dim1", "size": 200}],
        },
    ],
}


def test_dataset_schema_validation() -> None:
    """Test that the dataset schema validates correctly."""
    V1Dataset.model_validate(TEST_SCHEMA)
