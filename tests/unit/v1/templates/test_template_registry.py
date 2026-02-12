"""Tests for the singleton TemplateRegistry implementation."""

import copy
import pickle
import threading
import time

import pytest

from mdio.builder.template_registry import TemplateRegistry
from mdio.builder.template_registry import get_template
from mdio.builder.template_registry import get_template_registry
from mdio.builder.template_registry import is_template_registered
from mdio.builder.template_registry import list_templates
from mdio.builder.template_registry import register_template
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain

EXPECTED_DEFAULT_TEMPLATE_NAMES = [
    "PostStack2DTime",
    "PostStack2DDepth",
    "PostStack3DTime",
    "PostStack3DDepth",
    "CdpOffsetGathers2DTime",
    "CdpAngleGathers2DDepth",
    "CdpOffsetGathers2DTime",
    "CdpAngleGathers2DDepth",
    "CdpOffsetGathers3DTime",
    "CdpAngleGathers3DTime",
    "CdpOffsetGathers3DDepth",
    "CdpAngleGathers3DDepth",
    "CocaGathers3DTime",
    "CocaGathers3DDepth",
    "StreamerShotGathers2D",
    "StreamerShotGathers3D",
    "StreamerFieldRecords3D",
    "ObnReceiverGathers3D",
    "ShotReceiverLineGathers3D",
]


class MockDatasetTemplate(AbstractDatasetTemplate):
    """Mock template for testing."""

    def __init__(self, name: str, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)
        self.template_name = name

    @property
    def _name(self) -> str:
        return self.template_name

    def _load_dataset_attributes(self) -> None:
        return None  # pragma: no cover - Mock implementation


def _assert_default_templates(template_names: list[str]) -> None:
    assert len(template_names) == len(EXPECTED_DEFAULT_TEMPLATE_NAMES)
    for name in EXPECTED_DEFAULT_TEMPLATE_NAMES:
        assert name in template_names


class TestTemplateRegistrySingleton:
    """Test cases for TemplateRegistry singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        TemplateRegistry._reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if TemplateRegistry._instance:
            TemplateRegistry._instance.clear()
        TemplateRegistry._reset_instance()

    def test_singleton_same_instance(self) -> None:
        """Test that multiple instantiations return the same instance."""
        registry1 = TemplateRegistry()
        registry2 = TemplateRegistry()
        registry3 = TemplateRegistry.get_instance()

        assert registry1 is registry2
        assert registry2 is registry3
        assert id(registry1) == id(registry2) == id(registry3)

    def test_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe during creation."""
        instances = []
        errors = []

        def create_instance() -> None:
            instance = TemplateRegistry()
            instances.append(instance)
            time.sleep(0.001)  # Small delay to increase contention

        # Create multiple threads trying to create instances
        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(errors) == 0
        assert len(instances) == 10
        assert all(instance is instances[0] for instance in instances)

    def test_initialization_only_once(self) -> None:
        """Test that internal state is initialized only once."""
        registry1 = TemplateRegistry()
        template1 = MockDatasetTemplate("test_template")
        registry1.register(template1)

        # Create another instance - should have same templates
        registry2 = TemplateRegistry()

        assert registry1 is registry2
        assert registry2.is_registered("test_template")
        # Templates are returned as deep copies, not the original
        retrieved = registry2.get("test_template")
        assert retrieved is not template1
        assert retrieved.name == template1.name

    def test_register_template(self) -> None:
        """Test template registration."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test")

        name = registry.register(template)

        assert name == "test"  # Should be the template name
        assert registry.is_registered("test")
        # Templates are returned as deep copies, not the original
        retrieved = registry.get("test")
        assert retrieved is not template
        assert retrieved.name == template.name

    def test_register_duplicate_template(self) -> None:
        """Test error when registering duplicate template."""
        registry = TemplateRegistry()
        template1 = MockDatasetTemplate("duplicate")
        template2 = MockDatasetTemplate("duplicate")

        registry.register(template1)

        with pytest.raises(ValueError, match="Template 'duplicate' is already registered"):
            registry.register(template2)

    def test_get_nonexistent_template(self) -> None:
        """Test error when getting non-existent template."""
        registry = TemplateRegistry()

        with pytest.raises(KeyError, match="Template 'nonexistent' is not registered"):
            registry.get("nonexistent")

    def test_unregister_template(self) -> None:
        """Test template unregistration."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test_template")

        registry.register(template)
        assert registry.is_registered("test_template")

        registry.unregister("test_template")
        assert not registry.is_registered("test_template")

    def test_unregister_nonexistent_template(self) -> None:
        """Test error when unregistering non-existent template."""
        registry = TemplateRegistry()

        with pytest.raises(KeyError, match="Template 'nonexistent' is not registered"):
            registry.unregister("nonexistent")

    def test_get_returns_deep_copy(self) -> None:
        """Test that get() returns a deep copy, not the original."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test_deepcopy")
        registry.register(template)

        # Get the template twice
        retrieved1 = registry.get("test_deepcopy")
        retrieved2 = registry.get("test_deepcopy")

        # Each retrieval should return a different object
        assert retrieved1 is not retrieved2
        assert retrieved1 is not template
        assert retrieved2 is not template

        # But they should have the same name
        assert retrieved1.name == template.name
        assert retrieved2.name == template.name

    def test_template_modifications_are_independent(self) -> None:
        """Test that modifications to one template don't affect others."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test_modify")
        template._units = {"original": "value"}
        registry.register(template)

        # Get two copies
        copy1 = registry.get("test_modify")
        copy2 = registry.get("test_modify")

        # Modify the first copy
        copy1._units["copy1_key"] = "copy1_value"

        # Modify the second copy differently
        copy2._units["copy2_key"] = "copy2_value"

        # Get a third copy
        copy3 = registry.get("test_modify")

        # Each copy should have only its own modifications
        assert "copy1_key" in copy1._units
        assert "copy1_key" not in copy2._units
        assert "copy1_key" not in copy3._units

        assert "copy2_key" in copy2._units
        assert "copy2_key" not in copy1._units
        assert "copy2_key" not in copy3._units

        # All should have the original value
        assert copy1._units["original"] == "value"
        assert copy2._units["original"] == "value"
        assert copy3._units["original"] == "value"

    def test_list_all_templates(self) -> None:
        """Test listing all registered templates."""
        registry = TemplateRegistry()

        # Default templates are always installed
        templates = list_templates()
        _assert_default_templates(templates)

        # Add some templates
        template1 = MockDatasetTemplate("Template_One")
        template2 = MockDatasetTemplate("Template_Two")

        registry.register(template1)
        registry.register(template2)

        templates = registry.list_all_templates()
        assert len(templates) == 19 + 2  # 19 default + 2 custom
        assert "Template_One" in templates
        assert "Template_Two" in templates

    def test_clear_templates(self) -> None:
        """Test clearing all templates."""
        registry = TemplateRegistry()

        # Default templates are always installed
        templates = list_templates()
        assert len(templates) == 19

        # Add some templates
        template1 = MockDatasetTemplate("Template1")
        template2 = MockDatasetTemplate("Template2")

        registry.register(template1)
        registry.register(template2)

        assert len(registry.list_all_templates()) == 19 + 2  # 19 default + 2 custom

        # Clear all
        registry.clear()

        assert len(registry.list_all_templates()) == 0
        assert not registry.is_registered("Template1")
        assert not registry.is_registered("Template2")
        # default templates are also cleared
        for template_name in EXPECTED_DEFAULT_TEMPLATE_NAMES:
            assert not registry.is_registered(template_name)

    def test_reset_instance(self) -> None:
        """Test resetting the singleton instance."""
        registry1 = TemplateRegistry()
        template = MockDatasetTemplate("test")
        registry1.register(template)

        # Reset the instance
        TemplateRegistry._reset_instance()

        # New instance should be different and contain default templates only
        registry2 = TemplateRegistry()

        assert registry1 is not registry2
        assert not registry2.is_registered("test")

        # default templates are registered
        assert len(registry2.list_all_templates()) == len(EXPECTED_DEFAULT_TEMPLATE_NAMES)
        for template_name in EXPECTED_DEFAULT_TEMPLATE_NAMES:
            assert registry2.is_registered(template_name)

    def test_template_is_deepcopyable(self) -> None:
        """Test that retrieved templates can be deep copied."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test_deepcopy_method")
        template._units = {"test": "value"}
        registry.register(template)

        # Get a template
        retrieved = registry.get("test_deepcopy_method")

        # Deep copy it
        copied = copy.deepcopy(retrieved)

        # Should be different objects
        assert copied is not retrieved

        # Modify the copy
        copied._units["new_key"] = "new_value"

        # Original retrieved template should be unchanged
        assert "new_key" not in retrieved._units
        assert "new_key" in copied._units

    def test_template_is_pickleable(self) -> None:
        """Test that retrieved templates can be pickled and unpickled."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test_pickle")
        template._units = {"test": "value"}
        registry.register(template)

        # Get a template
        retrieved = registry.get("test_pickle")

        # Pickle and unpickle it
        pickled = pickle.dumps(retrieved)
        # You should only ever unpickle with trusted data!
        unpickled = pickle.loads(pickled)  # noqa: S301

        # Should be different objects
        assert unpickled is not retrieved

        # Should have the same data
        assert unpickled.name == retrieved.name
        assert unpickled._units == retrieved._units

        # Modify unpickled
        unpickled._units["new_key"] = "new_value"

        # Original should be unchanged
        assert "new_key" not in retrieved._units
        assert "new_key" in unpickled._units


class TestGlobalFunctions:
    """Test cases for global convenience functions."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        TemplateRegistry._reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if TemplateRegistry._instance:
            TemplateRegistry._instance.clear()
        TemplateRegistry._reset_instance()

    def test_get_template_registry(self) -> None:
        """Test global registry getter."""
        registry1 = get_template_registry()
        registry2 = get_template_registry()
        direct_registry = TemplateRegistry()

        assert registry1 is registry2
        assert registry1 is direct_registry

    def test_register_template_global(self) -> None:
        """Test global template registration."""
        template = MockDatasetTemplate("global_test")

        name = register_template(template)

        assert name == "global_test"
        assert is_template_registered("global_test")
        # Templates are returned as deep copies, not the original
        retrieved = get_template("global_test")
        assert retrieved is not template
        assert retrieved.name == template.name

    def test_list_templates_global(self) -> None:
        """Test global template listing."""
        # Default templates are always installed
        templates = list_templates()
        _assert_default_templates(templates)

        template1 = MockDatasetTemplate("template1")
        template2 = MockDatasetTemplate("template2")

        register_template(template1)
        register_template(template2)

        templates = list_templates()
        assert len(templates) == 21  # 19 default + 2 custom
        assert "template1" in templates
        assert "template2" in templates


class TestConcurrentAccess:
    """Test concurrent access to the singleton registry."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        TemplateRegistry._reset_instance()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if TemplateRegistry._instance:
            TemplateRegistry._instance.clear()
        TemplateRegistry._reset_instance()

    def test_concurrent_registration(self) -> None:
        """Test concurrent template registration."""
        registry = TemplateRegistry()
        results = []
        errors = []

        def register_template_worker(template_id: int) -> None:
            template = MockDatasetTemplate(f"template_{template_id}")
            name = registry.register(template)
            results.append((template_id, name))
            time.sleep(0.001)  # Small delay

        # Create multiple threads registering different templates
        threads = [threading.Thread(target=register_template_worker, args=(i,)) for i in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All registrations should succeed
        assert len(errors) == 0
        assert len(results) == 10
        # Including default templates
        assert len(registry.list_all_templates()) == 29  # 19 default + 10 registered

        # Check all templates are registered
        for i in range(10):
            assert registry.is_registered(f"template_{i}")

    def test_concurrent_access_mixed_operations(self) -> None:
        """Test concurrent mixed operations (register, get, list)."""
        registry = TemplateRegistry()

        # Pre-register some templates
        for i in range(5):
            template = MockDatasetTemplate(f"initial_{i}")
            registry.register(template)

        results = []
        errors = []

        def mixed_operations_worker(worker_id: int) -> None:
            operations_results = []

            # Get existing template
            if worker_id % 2 == 0:
                template = registry.get("initial_0")
                operations_results.append(("get", template.template_name))

            # Register new template
            if worker_id % 3 == 0:
                new_template = MockDatasetTemplate(f"worker_{worker_id}")
                name = registry.register(new_template)
                operations_results.append(("register", name))

            # List templates
            templates = registry.list_all_templates()
            operations_results.append(("list", len(templates)))

            results.append((worker_id, operations_results))

        # Run concurrent operations
        threads = [threading.Thread(target=mixed_operations_worker, args=(i,)) for i in range(15)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Check that operations completed without errors
        assert len(errors) == 0
        assert len(results) == 15

        # Verify final state is consistent
        final_templates = registry.list_all_templates()
        assert len(final_templates) >= 5  # At least the initial templates
