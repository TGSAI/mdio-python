"""(De)serialization factory design pattern.

Current support for JSON and YAML.
"""

import json
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from inspect import Signature

import yaml


class Serializer(ABC):
    """Serializer base class.

    Here we define the interface for any serializer implementation.

    Args:
        stream_format: Stream format. Must be in {"JSON", "YAML"}.
    """

    def __init__(self, stream_format: str) -> None:
        self.format = stream_format
        self.serialize_func = get_serializer(stream_format)
        self.deserialize_func = get_deserializer(stream_format)

    @abstractmethod
    def serialize(self, payload: dict) -> str:
        """Abstract method for serialize."""

    @abstractmethod
    def deserialize(self, stream: str) -> dict:
        """Abstract method for deserialize."""

    @staticmethod
    def validate_payload(payload: dict, signature: Signature) -> dict:
        """Validate if required keys exist in the payload for a function signature."""
        observed = set(payload)
        expected = set(signature.parameters)

        if not expected.issubset(observed):
            msg = f"Key mismatch: {observed}, expected {expected}"
            raise KeyError(msg)

        if len(observed) != len(expected):
            print(f"Ignoring extra key: {observed - expected}")
            payload = {key: payload[key] for key in expected}

        return payload


def get_serializer(stream_format: str) -> Callable:
    """Get serializer based on format."""
    stream_format = stream_format.upper()
    if stream_format == "JSON":
        return _serialize_to_json
    if stream_format == "YAML":
        return _serialize_to_yaml
    msg = f"Unsupported serializer for format: {stream_format}"
    raise ValueError(msg)


def get_deserializer(stream_format: str) -> Callable:
    """Get deserializer based on format."""
    stream_format = stream_format.upper()
    if stream_format == "JSON":
        return _deserialize_json
    if stream_format == "YAML":
        return _deserialize_yaml
    msg = f"Unsupported deserializer for format: {stream_format}"
    raise ValueError(msg)


def _serialize_to_json(payload: dict) -> str:
    """Convert dictionary to JSON string."""
    return json.dumps(payload)


def _serialize_to_yaml(payload: dict) -> str:
    """Convert dictionary to YAML string."""
    return yaml.dump(payload, sort_keys=False)


def _deserialize_json(stream: str) -> dict:
    """Convert JSON string to dictionary."""
    return json.loads(stream)


def _deserialize_yaml(stream: str) -> dict:
    """Convert YAML string to dictionary."""
    return yaml.safe_load(stream)
