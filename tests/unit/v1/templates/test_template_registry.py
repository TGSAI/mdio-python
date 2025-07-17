"""Tests for the singleton TemplateRegistry implementation."""

import pytest
import threading
import time
from unittest.mock import Mock

from mdio.schemas.v1.templates.template_registry import (
    TemplateRegistry,
    get_template_registry,
    register_template,
    get_template,
    is_template_registered,
    list_templates
)
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class MockDatasetTemplate(AbstractDatasetTemplate):
    """Mock template for testing."""
    
    def __init__(self, name: str):
        super().__init__()
        self.template_name = name
    
    def _get_name(self) -> str:
        return self.template_name
    
    def _load_dataset_attributes(self):
        return None  # Mock implementation
    
    def create_dataset(self, *args, **kwargs):
        return f"Mock dataset created by {self.template_name}"


class TestTemplateRegistrySingleton:
    """Test cases for TemplateRegistry singleton pattern."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        TemplateRegistry.reset_instance()
    
    def teardown_method(self):
        """Clean up after each test."""
        if TemplateRegistry._instance:
            TemplateRegistry._instance.clear()
        TemplateRegistry.reset_instance()
    
    def test_singleton_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        registry1 = TemplateRegistry()
        registry2 = TemplateRegistry()
        registry3 = TemplateRegistry.get_instance()
        
        assert registry1 is registry2
        assert registry2 is registry3
        assert id(registry1) == id(registry2) == id(registry3)
    
    def test_singleton_thread_safety(self):
        """Test that singleton is thread-safe during creation."""
        instances = []
        errors = []
        
        def create_instance():
            try:
                instance = TemplateRegistry()
                instances.append(instance)
                time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
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
    
    def test_initialization_only_once(self):
        """Test that internal state is initialized only once."""
        registry1 = TemplateRegistry()
        template1 = MockDatasetTemplate("test_template")
        registry1.register(template1)
        
        # Create another instance - should have same templates
        registry2 = TemplateRegistry()
        
        assert registry1 is registry2
        assert registry2.is_registered("test_template")
        assert registry2.get("test_template") is template1
    
    def test_register_template(self):
        """Test template registration."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test")
        
        name = registry.register(template)
        
        assert name == "test"  # Should be the template name
        assert registry.is_registered("test")
        assert registry.get("test") is template
    
    def test_register_duplicate_template(self):
        """Test error when registering duplicate template."""
        registry = TemplateRegistry()
        template1 = MockDatasetTemplate("duplicate")
        template2 = MockDatasetTemplate("duplicate")
        
        registry.register(template1)
        
        with pytest.raises(ValueError, match="Template 'duplicate' is already registered"):
            registry.register(template2)
    
    def test_get_nonexistent_template(self):
        """Test error when getting non-existent template."""
        registry = TemplateRegistry()
        
        with pytest.raises(KeyError, match="Template 'nonexistent' is not registered"):
            registry.get("nonexistent")
    
    def test_unregister_template(self):
        """Test template unregistration."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("test_template")
        
        registry.register(template)
        assert registry.is_registered("test_template")
        
        registry.unregister("test_template")
        assert not registry.is_registered("test_template")
    
    def test_unregister_nonexistent_template(self):
        """Test error when unregistering non-existent template."""
        registry = TemplateRegistry()
        
        with pytest.raises(KeyError, match="Template 'nonexistent' is not registered"):
            registry.unregister("nonexistent")
    
    def test_case_insensitive_operations(self):
        """Test that all operations are case-insensitive."""
        registry = TemplateRegistry()
        template = MockDatasetTemplate("Test_Template")
        
        # Register template
        registry.register(template)
        
        # All these should work (case insensitive)
        assert registry.is_registered("test_template")
        assert registry.is_registered("TEST_TEMPLATE")
        assert registry.is_registered("Test_Template")
        
        # Get with different cases
        assert registry.get("test_template") is template
        assert registry.get("TEST_TEMPLATE") is template
        assert registry.get("Test_Template") is template
        
        # Unregister with different case
        registry.unregister("TEST_TEMPLATE")
        assert not registry.is_registered("test_template")
    
    def test_list_all_templates(self):
        """Test listing all registered templates."""
        registry = TemplateRegistry()
        
        # Initially empty
        assert registry.list_all_templates() == []
        
        # Add some templates
        template1 = MockDatasetTemplate("Template_One")
        template2 = MockDatasetTemplate("Template_Two")
        
        registry.register(template1)
        registry.register(template2)
        
        templates = registry.list_all_templates()
        assert len(templates) == 2
        assert "template_one" in templates
        assert "template_two" in templates
    
    def test_clear_templates(self):
        """Test clearing all templates."""
        registry = TemplateRegistry()
        
        # Add some templates
        template1 = MockDatasetTemplate("template1")
        template2 = MockDatasetTemplate("template2")
        
        registry.register(template1)
        registry.register(template2)
        
        assert len(registry.list_all_templates()) == 2
        
        # Clear all
        registry.clear()
        
        assert len(registry.list_all_templates()) == 0
        assert not registry.is_registered("template1")
        assert not registry.is_registered("template2")
    
    def test_reset_instance(self):
        """Test resetting the singleton instance."""
        registry1 = TemplateRegistry()
        template = MockDatasetTemplate("test")
        registry1.register(template)
        
        # Reset the instance
        TemplateRegistry.reset_instance()
        
        # New instance should be different and empty
        registry2 = TemplateRegistry()
        
        assert registry1 is not registry2
        assert not registry2.is_registered("test")
        assert len(registry2.list_all_templates()) == 0


class TestGlobalFunctions:
    """Test cases for global convenience functions."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        TemplateRegistry.reset_instance()
    
    def teardown_method(self):
        """Clean up after each test."""
        if TemplateRegistry._instance:
            TemplateRegistry._instance.clear()
        TemplateRegistry.reset_instance()
    
    def test_get_template_registry(self):
        """Test global registry getter."""
        registry1 = get_template_registry()
        registry2 = get_template_registry()
        direct_registry = TemplateRegistry()
        
        assert registry1 is registry2
        assert registry1 is direct_registry
    
    def test_register_template_global(self):
        """Test global template registration."""
        template = MockDatasetTemplate("global_test")

        name = register_template(template)
        
        assert name == "global_test"
        assert is_template_registered("global_test")
        assert get_template("global_test") is template
    
    def test_list_templates_global(self):
        """Test global template listing."""
        template1 = MockDatasetTemplate("template1")
        template2 = MockDatasetTemplate("template2")
        
        register_template(template1)
        register_template(template2)
        
        templates = list_templates()
        assert len(templates) == 2
        assert "template1" in templates
        assert "template2" in templates


class TestConcurrentAccess:
    """Test concurrent access to the singleton registry."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        TemplateRegistry.reset_instance()
    
    def teardown_method(self):
        """Clean up after each test."""
        if TemplateRegistry._instance:
            TemplateRegistry._instance.clear()
        TemplateRegistry.reset_instance()
    
    def test_concurrent_registration(self):
        """Test concurrent template registration."""
        registry = TemplateRegistry()
        results = []
        errors = []
        
        def register_template_worker(template_id):
            try:
                template = MockDatasetTemplate(f"template_{template_id}")
                name = registry.register(template)
                results.append((template_id, name))
                time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append((template_id, e))
        
        # Create multiple threads registering different templates
        threads = [threading.Thread(target=register_template_worker, args=(i,)) 
                  for i in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All registrations should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert len(registry.list_all_templates()) == 10
        
        # Check all templates are registered
        for i in range(10):
            assert registry.is_registered(f"template_{i}")
    
    def test_concurrent_access_mixed_operations(self):
        """Test concurrent mixed operations (register, get, list)."""
        registry = TemplateRegistry()
        
        # Pre-register some templates
        for i in range(5):
            template = MockDatasetTemplate(f"initial_{i}")
            registry.register(template)
        
        results = []
        errors = []
        
        def mixed_operations_worker(worker_id):
            try:
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
                
            except Exception as e:
                errors.append((worker_id, e))
        
        # Run concurrent operations
        threads = [threading.Thread(target=mixed_operations_worker, args=(i,)) 
                  for i in range(15)]
        
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
