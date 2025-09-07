"""Template registry for MDIO v1 dataset templates."""

import threading
from typing import Optional

from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.builder.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
from mdio.builder.templates.seismic_2d_prestack_cdp import Seismic2DPreStackCDPTemplate
from mdio.builder.templates.seismic_2d_prestack_shot import Seismic2DPreStackShotTemplate
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.builder.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate
from mdio.builder.templates.seismic_3d_prestack_coca import Seismic3DPreStackCocaTemplate
from mdio.builder.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate


class TemplateRegistry:
    """A thread-safe singleton registry for dataset templates."""

    _instance: Optional["TemplateRegistry"] = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls) -> "TemplateRegistry":
        """Create or return the singleton instance.

        Uses double-checked locking pattern to ensure thread safety.

        Returns:
            The singleton instance of TemplateRegistry.
        """
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
                    self._register_default_templates()
                    TemplateRegistry._initialized = True

    def register(self, instance: AbstractDatasetTemplate) -> str:
        """Register a template instance by its name.

        Args:
            instance: An instance of template to register.

        Returns:
            The name of the registered template.

        Raises:
            ValueError: If the template name is already registered.
        """
        with self._registry_lock:
            name = instance.name
            if name in self._templates:
                err = f"Template '{name}' is already registered."
                raise ValueError(err)
            self._templates[name] = instance
        return name

    def _register_default_templates(self) -> None:
        """Register default templates if needed.

        Subclasses can override this method to register default templates.
        """
        # Post-Stack Data
        self.register(Seismic2DPostStackTemplate("time"))
        self.register(Seismic2DPostStackTemplate("depth"))
        self.register(Seismic3DPostStackTemplate("time"))
        self.register(Seismic3DPostStackTemplate("depth"))

        # CDP/CMP Ordered Data
        for data_domain in ("time", "depth"):
            for gather_domain in ("offset", "angle"):
                self.register(Seismic3DPreStackCDPTemplate(data_domain, gather_domain))
                self.register(Seismic2DPreStackCDPTemplate(data_domain, gather_domain))

        self.register(Seismic3DPreStackCocaTemplate("time"))
        self.register(Seismic3DPreStackCocaTemplate("depth"))

        # Field (shot) data
        self.register(Seismic2DPreStackShotTemplate("time"))
        self.register(Seismic3DPreStackShotTemplate("time"))

    def get(self, template_name: str) -> AbstractDatasetTemplate:
        """Get a template from the registry by its name.

        Args:
            template_name: The name of the template to retrieve.

        Returns:
            The template instance if found.

        Raises:
            KeyError: If the template is not registered.
        """
        with self._registry_lock:
            name = template_name
            if name not in self._templates:
                err = f"Template '{name}' is not registered."
                raise KeyError(err)
            return self._templates[name]

    def unregister(self, template_name: str) -> None:
        """Unregister a template from the registry.

        Args:
            template_name: The name of the template to unregister.

        Raises:
            KeyError: If the template is not registered.
        """
        with self._registry_lock:
            name = template_name
            if name not in self._templates:
                err_msg = f"Template '{name}' is not registered."
                raise KeyError(err_msg)
            del self._templates[name]

    def is_registered(self, template_name: str) -> bool:
        """Check if a template is registered in the registry.

        Args:
            template_name: The name of the template to check.

        Returns:
            True if the template is registered, False otherwise.
        """
        with self._registry_lock:
            name = template_name
            return name in self._templates

    def list_all_templates(self) -> list[str]:
        """Get all registered template names.

        Returns:
            A list of all registered template names.
        """
        with self._registry_lock:
            return list(self._templates.keys())

    def clear(self) -> None:
        """Clear all registered templates (useful for testing)."""
        with self._registry_lock:
            self._templates.clear()

    @classmethod
    def get_instance(cls) -> "TemplateRegistry":
        """Get the singleton instance (alternative to constructor).

        Returns:
            The singleton instance of TemplateRegistry.
        """
        return cls()

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False


# Global convenience functions
def get_template_registry() -> TemplateRegistry:
    """Get the global template registry instance.

    Returns:
        The singleton instance of TemplateRegistry.
    """
    return TemplateRegistry.get_instance()


def register_template(template: AbstractDatasetTemplate) -> str:
    """Register a template in the global registry.

    Args:
        template: An instance of AbstractDatasetTemplate to register.

    Returns:
        The name of the registered template.
    """
    return get_template_registry().register(template)


def get_template(name: str) -> AbstractDatasetTemplate:
    """Get a template from the global registry.

    Args:
        name: The name of the template to retrieve.

    Returns:
        The template instance if found.
    """
    return get_template_registry().get(name)


def is_template_registered(name: str) -> bool:
    """Check if a template is registered in the global registry.

    Args:
        name: The name of the template to check.

    Returns:
        True if the template is registered, False otherwise.
    """
    return get_template_registry().is_registered(name)


def list_templates() -> list[str]:
    """List all registered template names.

    Returns:
        A list of all registered template names.
    """
    return get_template_registry().list_all_templates()
