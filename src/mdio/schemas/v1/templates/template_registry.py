"""Template registry for MDIO v1 dataset templates."""

import threading
from typing import Optional

from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class TemplateRegistry:
    """A thread-safe singleton registry for dataset templates."""

    _instance: Optional["TemplateRegistry"] = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls) -> "TemplateRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._templates: dict[str, AbstractDatasetTemplate] = {}
                    self._registry_lock = threading.RLock()
                    TemplateRegistry._initialized = True

    def register(self, instance: AbstractDatasetTemplate) -> str:
        """Register a template instance.

        This method registers a template instance and returns its name.
        If the template name is already registered, it raises a ValueError.
        """
        with self._registry_lock:
            name = instance.get_name().lower()
            if name in self._templates:
                err = f"Template '{name}' is already registered."
                raise ValueError(err)
            self._templates[name] = instance
        return name

    def get(self, template_name: str) -> AbstractDatasetTemplate:
        """Get a registered template instance."""
        with self._registry_lock:
            name = template_name.lower()
            if name not in self._templates:
                err = f"Template '{name}' is not registered."
                raise KeyError(err)
            return self._templates[name]

    def unregister(self, template_name: str) -> None:
        """Unregister a template instance."""
        with self._registry_lock:
            name = template_name.lower()
            if name not in self._templates:
                err_msg = f"Template '{name}' is not registered."
                raise KeyError(err_msg)
            del self._templates[name]

    def is_registered(self, template_name: str) -> bool:
        """Check if a template instance is registered."""
        with self._registry_lock:
            name = template_name.lower()
            return name in self._templates

    def list_all_templates(self) -> list[str]:
        """Get all registered template names."""
        with self._registry_lock:
            return list(self._templates.keys())

    def clear(self) -> None:
        """Clear all registered templates (useful for testing)."""
        with self._registry_lock:
            self._templates.clear()

    @classmethod
    def get_instance(cls) -> "TemplateRegistry":
        """Get the singleton instance (alternative to constructor)."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False


# Global convenience functions
def get_template_registry() -> TemplateRegistry:
    """Get the global template registry instance."""
    return TemplateRegistry.get_instance()


def register_template(template: AbstractDatasetTemplate) -> str:
    """Register a template in the global registry."""
    return get_template_registry().register(template)


def get_template(name: str) -> AbstractDatasetTemplate:
    """Get a template from the global registry."""
    return get_template_registry().get(name)


def is_template_registered(name: str) -> bool:
    """Check if a template is registered in the global registry."""
    return get_template_registry().is_registered(name)


def list_templates() -> list[str]:
    """List all registered template names."""
    return get_template_registry().list_all_templates()
