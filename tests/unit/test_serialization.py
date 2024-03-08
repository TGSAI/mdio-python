"""Check lower-level serialization functions."""

from inspect import Parameter
from inspect import Signature
from typing import Any

import pytest

from mdio.core.serialization import Serializer
from mdio.core.serialization import get_deserializer
from mdio.core.serialization import get_serializer


@pytest.mark.parametrize(
    "mappable, json_str",
    [
        ({"a": 5}, '{"a": 5}'),
        ({"a": 5, "b": [-1, 5]}, '{"a": 5, "b": [-1, 5]}'),
        ({"b": 5, "a": [-1, 5]}, '{"b": 5, "a": [-1, 5]}'),
        ({"b": 5, "a": [-11], "c": "5"}, '{"b": 5, "a": [-11], "c": "5"}'),
        ({"k": {"l": [1, 5], "m": "v"}}, '{"k": {"l": [1, 5], "m": "v"}}'),
    ],
)
class TestJSON:
    """JSON conversion and back."""

    def test_json_serialize(self, mappable: dict[str, Any], json_str: str) -> None:
        """Dictionary to JSON."""
        serializer = get_serializer("json")
        assert serializer(mappable) == json_str

    def test_json_deserialize(self, mappable: dict[str, Any], json_str: str) -> None:
        """JSON to dictionary."""
        deserializer = get_deserializer("json")
        assert deserializer(json_str) == mappable


@pytest.mark.parametrize(
    "mappable, yaml_str",
    [
        ({"a": 5}, "a: 5\n"),
        ({"a": 5, "b": [-1, 5]}, "a: 5\nb:\n- -1\n- 5\n"),
        ({"b": 5, "a": [-1, 5]}, "b: 5\na:\n- -1\n- 5\n"),
        ({"b": 5, "a": [-11], "c": "5"}, "b: 5\na:\n- -11\nc: '5'\n"),
        ({"k": {"l": [1, 5], "m": "v"}}, "k:\n  l:\n  - 1\n  - 5\n  m: v\n"),
    ],
)
class TestYAML:
    """YAML conversion and back."""

    def test_yaml_serialize(self, mappable: dict[str, Any], yaml_str: str) -> None:
        """Dictionary to YAML."""
        serializer = get_serializer("yaml")
        assert serializer(mappable) == yaml_str

    def test_yaml_deserialize(self, mappable: dict[str, Any], yaml_str: str) -> None:
        """YAML to dictionary."""
        deserializer = get_deserializer("yaml")
        assert deserializer(yaml_str) == mappable


class TestExceptions:
    """Test if exceptions are raised properly."""

    def test_unsupported_format_serializer(self) -> None:
        """Unknown serializer format."""
        with pytest.raises(ValueError, match="Unsupported serializer format"):
            get_serializer("foo")

    def test_unsupported_format_deserializer(self) -> None:
        """Unknown deserializer format."""
        with pytest.raises(ValueError, match="Unsupported deserializer format"):
            get_deserializer("bar")

    def test_missing_key(self) -> None:
        """Raise if required key is missing."""
        mock_signature = Signature(
            [
                Parameter("param1", Parameter.POSITIONAL_ONLY),
                Parameter("param2", Parameter.POSITIONAL_ONLY),
            ]
        )

        exact_input = {"param1": 1, "param2": 2}
        extra_inputs = {"param1": 1, "param2": 2, "extra_param": 5}

        assert exact_input == Serializer.validate_payload(exact_input, mock_signature)
        assert exact_input == Serializer.validate_payload(extra_inputs, mock_signature)

        missing_key = {"wrong_param1": 1, "param2": 2}
        with pytest.raises(KeyError):
            Serializer.validate_payload(missing_key, mock_signature)
