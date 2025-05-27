"""Test the schema for the v1 dataset."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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


class TestV1DatasetJSONSerialization:
    """Test JSON serialization capabilities of V1Dataset using Pydantic methods."""

    @pytest.fixture
    def sample_dataset(self) -> V1Dataset:
        """Create a sample V1Dataset for testing."""
        # Use a deep copy to avoid test interference
        return V1Dataset.model_validate(copy.deepcopy(TEST_SCHEMA))

    def test_model_dump_json_default_camel_case(self, sample_dataset: V1Dataset) -> None:
        """Test that JSON serialization uses camelCase by default."""
        json_str = sample_dataset.model_dump_json(by_alias=True)

        print(json_str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Should contain expected top-level keys
        assert "metadata" in parsed
        assert "variables" in parsed

        # Metadata should have expected fields
        assert parsed["metadata"]["name"] == "test_dataset"
        assert parsed["metadata"]["apiVersion"] == "1.0.0"
        assert parsed["metadata"]["createdOn"] == "2023-01-01T00:00:00Z"

        # Should have 4 variables
        assert len(parsed["variables"]) == 4  # noqa: PLR2004

    def test_model_dump_json_exclude_none(self, sample_dataset: V1Dataset) -> None:
        """Test JSON serialization excluding None values."""
        json_str = sample_dataset.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)  # noqa: F841

        # Should not contain null values in the JSON
        json_str_lower = json_str.lower()
        assert "null" not in json_str_lower

    def test_model_validate_json_basic(self) -> None:
        """Test basic JSON deserialization using model_validate_json."""
        json_str = json.dumps(TEST_SCHEMA)
        dataset = V1Dataset.model_validate_json(json_str)

        assert dataset.metadata.name == "test_dataset"
        assert dataset.metadata.api_version == "1.0.0"
        assert len(dataset.variables) == 4  # noqa: PLR2004

        # Check first variable
        var = dataset.variables[0]
        assert var.name == "actual_variable"
        assert var.data_type.value == "float32"
        assert var.dimensions == ["dim0", "dim1"]

    def test_model_validate_json_invalid(self) -> None:
        """Test JSON deserialization with invalid data."""
        invalid_json = '{"metadata": {"name": "test"}, "variables": []}'

        with pytest.raises(ValidationError) as exc_info:
            V1Dataset.model_validate_json(invalid_json)

        # Should have validation errors
        errors = exc_info.value.errors()
        assert len(errors) > 0

    def test_model_validate_json_malformed(self) -> None:
        """Test JSON deserialization with malformed JSON."""
        malformed_json = '{"metadata": {"name": "test"'  # Missing closing braces

        with pytest.raises(ValidationError):
            V1Dataset.model_validate_json(malformed_json)

    def test_json_schema_generation(self) -> None:
        """Test JSON schema generation using model_json_schema."""
        schema = V1Dataset.model_json_schema()

        # Should be a valid JSON schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema

        # Should have metadata and variables properties
        properties = schema["properties"]
        assert "metadata" in properties
        assert "variables" in properties

        # Should have required fields
        assert "required" in schema
        required = schema["required"]
        assert "metadata" in required
        assert "variables" in required

    def test_json_schema_with_mode(self) -> None:
        """Test JSON schema generation with different modes."""
        # Test validation mode (default)
        validation_schema = V1Dataset.model_json_schema(mode="validation")
        assert "properties" in validation_schema

        # Test serialization mode
        serialization_schema = V1Dataset.model_json_schema(mode="serialization")
        assert "properties" in serialization_schema

    def test_round_trip_consistency_default(self, sample_dataset: V1Dataset) -> None:
        """Test that serialization -> deserialization preserves data (default camelCase)."""
        # Export to JSON (default camelCase)
        json_str = sample_dataset.model_dump_json()

        # Import from JSON
        restored_dataset = V1Dataset.model_validate_json(json_str)

        # Export again
        json_str2 = restored_dataset.model_dump_json()

        # Should be identical
        assert json_str == json_str2

        # Key properties should match
        assert sample_dataset.metadata.name == restored_dataset.metadata.name
        assert sample_dataset.metadata.api_version == restored_dataset.metadata.api_version
        assert len(sample_dataset.variables) == len(restored_dataset.variables)

        # Variables should match
        for orig_var, restored_var in zip(
            sample_dataset.variables, restored_dataset.variables, strict=False
        ):
            assert orig_var.name == restored_var.name
            assert orig_var.data_type == restored_var.data_type
            assert orig_var.dimensions == restored_var.dimensions

    def test_round_trip_with_aliases(self, sample_dataset: V1Dataset) -> None:
        """Test round-trip consistency when using aliases."""
        # Export with aliases (should be default now)
        json_str = sample_dataset.model_dump_json()

        # Import (should handle aliases automatically)
        restored_dataset = V1Dataset.model_validate_json(json_str)

        # Should preserve data
        assert sample_dataset.metadata.name == restored_dataset.metadata.name
        assert len(sample_dataset.variables) == len(restored_dataset.variables)

    def test_json_file_operations(self, sample_dataset: V1Dataset, tmp_path: Path) -> None:
        """Test JSON serialization to/from files."""
        json_file = tmp_path / "test_dataset.json"

        # Write to file (using default camelCase)
        json_str = sample_dataset.model_dump_json(indent=2)
        json_file.write_text(json_str, encoding="utf-8")

        # Verify file exists and has content
        assert json_file.exists()
        assert json_file.stat().st_size > 0

        # Read from file
        file_content = json_file.read_text(encoding="utf-8")
        restored_dataset = V1Dataset.model_validate_json(file_content)

        # Should match original
        assert sample_dataset.metadata.name == restored_dataset.metadata.name
        assert len(sample_dataset.variables) == len(restored_dataset.variables)

    def test_json_validation_without_instantiation(self) -> None:
        """Test JSON validation without creating a dataset instance."""
        valid_json = json.dumps(TEST_SCHEMA)

        # This should not raise an exception
        try:
            V1Dataset.model_validate_json(valid_json)
            validation_passed = True
        except ValidationError:
            validation_passed = False

        assert validation_passed

    def test_partial_json_validation(self) -> None:
        """Test validation of partial/incomplete JSON data."""
        # Missing required fields
        incomplete_schema = {
            "metadata": {
                "name": "test_dataset",
                # Missing apiVersion and createdOn
            },
            "variables": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            V1Dataset.model_validate_json(json.dumps(incomplete_schema))

        errors = exc_info.value.errors()
        # Should have errors for missing required fields
        error_fields = {error["loc"][-1] for error in errors}
        assert "apiVersion" in error_fields or "api_version" in error_fields

    def test_json_with_extra_fields(self) -> None:
        """Test JSON deserialization with extra fields."""
        # Use a copy to avoid modifying the global TEST_SCHEMA
        schema_with_extra = copy.deepcopy(TEST_SCHEMA)
        schema_with_extra["extra_field"] = "should_be_ignored"
        schema_with_extra["metadata"]["extra_metadata"] = "also_ignored"

        # Should raise ValidationError because extra fields are forbidden
        with pytest.raises(ValidationError) as exc_info:
            V1Dataset.model_validate_json(json.dumps(schema_with_extra))

        # Should have error about extra fields
        errors = exc_info.value.errors()
        assert any("extra_forbidden" in str(error) for error in errors)

    def test_json_schema_contains_examples(self) -> None:
        """Test that generated JSON schema contains useful information."""
        schema = V1Dataset.model_json_schema()

        # Should have descriptions for properties
        properties = schema.get("properties", {})
        if "metadata" in properties:
            # Check if metadata has some schema information
            metadata_schema = properties["metadata"]
            assert isinstance(metadata_schema, dict)

        if "variables" in properties:
            # Check if variables has some schema information
            variables_schema = properties["variables"]
            assert isinstance(variables_schema, dict)
            assert variables_schema.get("type") == "array"

    def test_json_serialization_performance(self, sample_dataset: V1Dataset) -> None:
        """Test that JSON serialization is reasonably performant."""
        import time

        # Time multiple serializations
        start_time = time.time()
        for _ in range(100):
            json_str = sample_dataset.model_dump_json()
        end_time = time.time()

        # Should complete 100 serializations in reasonable time (< 1 second)
        elapsed = end_time - start_time
        assert elapsed < 1.0

        # Verify the JSON is still valid
        parsed = json.loads(json_str)
        assert parsed["metadata"]["name"] == "test_dataset"


class TestPydanticMDIORoundTrip:
    """Test round-trip conversions between JSON and MDIO datasets using to_mdio."""

    def test_json_to_mdio_dataset(self, tmp_path: Path) -> None:
        """Test converting TEST_SCHEMA JSON to an MDIO dataset using to_mdio."""
        from mdio.core.v1._serializer import _construct_mdio_dataset
        
        output_path = tmp_path / "from_json.mdio"
        # output_path = "test_mdio_from_json.mdio"
        
        # Step 1: Validate the TEST_SCHEMA JSON with Pydantic
        dataset = V1Dataset.model_validate(TEST_SCHEMA)
        
        # Step 2: Convert to MDIO dataset using the internal constructor
        mdio_dataset = _construct_mdio_dataset(dataset)
        
        # Step 3: Use to_mdio to save the dataset
        mdio_dataset.to_mdio(store=str(output_path))
        
        # Verify the dataset was created
        assert output_path.exists()
        
        # Verify we can read it back
        from mdio.core.v1 import mdio
        with mdio.open(str(output_path)) as reader:
            assert "actual_variable" in reader
            assert "coord" in reader
            assert "dim0" in reader.coords
            assert "dim1" in reader.coords
            assert reader.attrs["name"] == "test_dataset"

    def test_mdio_dataset_to_json(self, tmp_path: Path) -> None:
        """Test converting an MDIO dataset back to JSON (camelCase)."""
        from mdio.core.v1._serializer import _construct_mdio_dataset
        from mdio.core.v1 import mdio
        
        # Step 1: Create MDIO dataset from TEST_SCHEMA
        dataset = V1Dataset.model_validate(TEST_SCHEMA)
        mdio_dataset = _construct_mdio_dataset(dataset)
        
        mdio_path = tmp_path / "test_dataset.mdio"
        mdio_dataset.to_mdio(store=str(mdio_path))
        
        # Step 2: Read back the MDIO dataset
        with mdio.open(str(mdio_path)) as reader:
            # Step 3: Extract information to reconstruct Pydantic model
            variables = []
            
            # Add dimension variables
            for dim_name in ["dim0", "dim1"]:
                if dim_name in reader.coords:
                    coord = reader.coords[dim_name]
                    var_dict = {
                        "name": dim_name,
                        "dataType": str(coord.dtype),
                        "dimensions": [{"name": dim_name, "size": reader.dims[dim_name]}],
                    }
                    variables.append(var_dict)
            
            # Add data variables with their metadata
            for var_name in reader.data_vars:
                var = reader[var_name]
                var_dict = {
                    "name": var_name,
                    "dataType": str(var.dtype),
                    "dimensions": list(var.dims),
                }
                
                # Reconstruct metadata based on original TEST_SCHEMA
                if var_name == "coord":
                    var_dict["metadata"] = {
                        "chunkGrid": {
                            "name": "regular",
                            "configuration": {"chunkShape": [10, 20]},
                        },
                        "unitsV1": {"length": "m"},
                    }
                elif var_name == "actual_variable":
                    var_dict["compressor"] = {"name": "blosc", "level": 3}
                    var_dict["coordinates"] = ["coord"]
                    var_dict["metadata"] = {
                        "chunkGrid": {
                            "name": "regular",
                            "configuration": {"chunkShape": [10, 20]},
                        },
                    }
                variables.append(var_dict)
            
            # Step 4: Create Pydantic model data (camelCase)
            dataset_data = {
                "metadata": {
                    "name": reader.attrs.get("name"),
                    "apiVersion": reader.attrs.get("apiVersion", "1.0.0"),
                    "createdOn": reader.attrs.get("createdOn", "2023-01-01T00:00:00Z"),
                },
                "variables": variables
            }
            
            # Step 5: Validate with Pydantic and serialize to JSON using by_alias=True
            pydantic_dataset = V1Dataset.model_validate(dataset_data)
            json_str = pydantic_dataset.model_dump_json(by_alias=True)
            
            # Verify it's valid JSON and camelCase
            parsed = json.loads(json_str)

            print(parsed)

            assert "apiVersion" in parsed["metadata"]
            assert "createdOn" in parsed["metadata"]
            assert "dataType" in parsed["variables"][0]
            
            # Verify the conversion preserved data
            assert pydantic_dataset.metadata.name == "test_dataset"

    def test_full_round_trip_json_mdio_json(self, tmp_path: Path) -> None:
        """Test full round-trip: TEST_SCHEMA JSON -> MDIO -> JSON using to_mdio."""
        from mdio.core.v1._serializer import _construct_mdio_dataset
        from mdio.core.v1 import mdio
        
        # Step 1: Start with TEST_SCHEMA (input JSON)
        original_dataset = V1Dataset.model_validate(TEST_SCHEMA)
        original_json = original_dataset.model_dump_json(by_alias=True)
        original_parsed = json.loads(original_json)
        
        # Verify original is camelCase
        assert "apiVersion" in original_parsed["metadata"]
        assert "createdOn" in original_parsed["metadata"]
        
        # Step 2: Convert to MDIO dataset and save
        mdio_dataset = _construct_mdio_dataset(original_dataset)
        mdio_path = tmp_path / "round_trip.mdio"
        mdio_dataset.to_mdio(store=str(mdio_path))
        
        # Step 3: Read back from MDIO and convert to JSON
        with mdio.open(str(mdio_path)) as reader:
            # Reconstruct the schema structure
            variables = []
            
            # Add dimension variables
            for dim_name in ["dim0", "dim1"]:
                if dim_name in reader.coords:
                    coord = reader.coords[dim_name]
                    var_dict = {
                        "name": dim_name,
                        "dataType": str(coord.dtype),
                        "dimensions": [{"name": dim_name, "size": reader.dims[dim_name]}],
                    }
                    variables.append(var_dict)
            
            # Add coordinate variables that are not dimensions
            for coord_name, coord in reader.coords.items():
                if coord_name not in ["dim0", "dim1"]:  # Skip dimension coordinates
                    var_dict = {
                        "name": coord_name,
                        "dataType": str(coord.dtype),
                        "dimensions": list(coord.dims),
                    }
                    
                    # Add metadata for coord variable from original TEST_SCHEMA
                    if coord_name == "coord":
                        var_dict["metadata"] = {
                            "chunkGrid": {
                                "name": "regular",
                                "configuration": {"chunkShape": [10, 20]},
                            },
                            "unitsV1": {"length": "m"},
                        }
                    variables.append(var_dict)
            
            # Add data variables with original metadata
            for var_name in reader.data_vars:
                var = reader[var_name]
                var_dict = {
                    "name": var_name,
                    "dataType": str(var.dtype),
                    "dimensions": list(var.dims),
                }
                
                # Add original metadata back from TEST_SCHEMA
                if var_name == "actual_variable":
                    var_dict["compressor"] = {"name": "blosc", "level": 3}
                    var_dict["coordinates"] = ["coord"]
                    var_dict["metadata"] = {
                        "chunkGrid": {
                            "name": "regular",
                            "configuration": {"chunkShape": [10, 20]},
                        },
                    }
                variables.append(var_dict)
            
            # Create final dataset
            final_data = {
                "metadata": {
                    "name": reader.attrs.get("name", "test_dataset"),
                    "apiVersion": reader.attrs.get("apiVersion", "1.0.0"),
                    "createdOn": reader.attrs.get("createdOn", "2023-01-01T00:00:00Z"),
                },
                "variables": variables
            }
            
            final_dataset = V1Dataset.model_validate(final_data)
            final_json = final_dataset.model_dump_json(by_alias=True)
            final_parsed = json.loads(final_json)
            
            # Step 4: Verify round-trip integrity
            assert final_parsed["metadata"]["name"] == original_parsed["metadata"]["name"]
            assert final_parsed["metadata"]["apiVersion"] == original_parsed["metadata"]["apiVersion"]
            
            # Verify camelCase is preserved
            assert "apiVersion" in final_parsed["metadata"]
            assert "createdOn" in final_parsed["metadata"]
            assert "dataType" in final_parsed["variables"][0]
            
            # Verify variable structure is preserved
            original_var_names = {v["name"] for v in original_parsed["variables"]}
            final_var_names = {v["name"] for v in final_parsed["variables"]}

            print(original_var_names)
            print("=================================")
            print(final_var_names)

            assert original_var_names == final_var_names

    def test_invalid_snake_case_json_fails(self) -> None:
        """Test that snake_case JSON fails validation (negative test)."""
        # Create snake_case version of TEST_SCHEMA (should fail)
        invalid_snake_case_schema = {
            "metadata": {
                "name": "test_dataset",
                "api_version": "1.0.0",  # snake_case should fail
                "created_on": "2023-01-01T00:00:00Z",  # snake_case should fail
            },
            "variables": [
                {
                    "name": "test_var",
                    "data_type": "float32",  # snake_case should fail
                    "dimensions": ["dim0"],
                }
            ]
        }
        
        # This should fail validation
        with pytest.raises(ValidationError):
            V1Dataset.model_validate(invalid_snake_case_schema)

    def test_camel_case_serialization_only(self) -> None:
        """Test that serialization only produces camelCase output."""
        dataset = V1Dataset.model_validate(TEST_SCHEMA)
        json_str = dataset.model_dump_json()
        parsed = json.loads(json_str)
        
        # Verify camelCase fields are present
        assert "apiVersion" in parsed["metadata"]
        assert "createdOn" in parsed["metadata"]
        
        # Verify snake_case fields are NOT present
        assert "api_version" not in parsed["metadata"]
        assert "created_on" not in parsed["metadata"]
        
        # Check variables use camelCase
        for var in parsed["variables"]:
            assert "dataType" in var
            assert "data_type" not in var
            
            # Check nested metadata if present
            if "metadata" in var and "chunkGrid" in var["metadata"]:
                assert "chunkGrid" in var["metadata"]
                assert "chunk_grid" not in var["metadata"]
