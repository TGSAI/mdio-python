"""Template registry for MDIO dataset templates.

This module provides a tiny, thread-safe singleton registry to discover and fetch predefined dataset
templates used by the MDIO builder.

Key points

- Global, thread-safe singleton (safe to use across threads)
- Fetching a template returns a deep-copied instance you can modify freely
- Comes pre-populated with common seismic templates

Use the top-level helpers for convenience: ``get_template``, ``list_templates``, ``register_template``,
``is_template_registered``, ``get_template_registry``.
"""

from __future__ import annotations

import copy
import html
import threading
from typing import TYPE_CHECKING

from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
from mdio.builder.templates.seismic_2d_prestack_cdp import Seismic2DPreStackCDPTemplate
from mdio.builder.templates.seismic_2d_prestack_shot import Seismic2DPreStackShotTemplate
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.builder.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate
from mdio.builder.templates.seismic_3d_prestack_coca import Seismic3DPreStackCocaTemplate
from mdio.builder.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate


__all__ = [
    "TemplateRegistry",
    "get_template_registry",
    "register_template",
    "get_template",
    "is_template_registered",
    "list_templates",
]


class TemplateRegistry:
    """Thread-safe singleton registry for dataset templates.

    The registry stores template instances by their public name and returns a deep-copied instance on
    every retrieval, so callers can safely mutate the returned object without affecting the registry
    or other callers.

    Thread-safety

    - Creation uses double-checked locking to guarantee a single instance.
    - Registry operations are protected by an internal re-entrant lock.

    Typical usage

    - Use the module helpers: ``get_template``, ``list_templates``, ``register_template``,
      and ``is_template_registered``.
    - Alternatively, use ``TemplateRegistry.get_instance()`` for direct access.
    """

    _instance: TemplateRegistry | None = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    def __new__(cls) -> TemplateRegistry:
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
                    self._registry_lock: threading.RLock = threading.RLock()
                    self._register_default_templates()
                    TemplateRegistry._initialized = True

    def register(self, instance: AbstractDatasetTemplate) -> str:
        """Register a template instance under its public name.

        A deep copy of the provided instance is stored internally to avoid accidental side effects
        from later external mutations.

        Args:
            instance: Template instance to register.

        Returns:
            The public name of the registered template.

        Raises:
            ValueError: If a template with the same name is already registered.
        """
        with self._registry_lock:
            name = instance.name
            if name in self._templates:
                msg = f"Template '{name}' is already registered."
                raise ValueError(msg)
            self._templates[name] = copy.deepcopy(instance)
        return name

    def _register_default_templates(self) -> None:
        """Register built-in seismic templates.

        Subclasses may override this hook to extend or change the defaults.
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
        """Get an instance of a template from the registry by its name.

        Each call returns a fresh, independent copy of the template that can be modified without affecting
        the original template or other copies.

        Args:
            template_name: The name of the template to retrieve.

        Returns:
            An instance of the template if found.

        Raises:
            KeyError: If the template is not registered.
        """
        with self._registry_lock:
            if template_name not in self._templates:
                msg = f"Template '{template_name}' is not registered."
                raise KeyError(msg)
            return copy.deepcopy(self._templates[template_name])

    def unregister(self, template_name: str) -> None:
        """Unregister a template from the registry.

        Args:
            template_name: The name of the template to unregister.

        Raises:
            KeyError: If the template is not registered.
        """
        with self._registry_lock:
            if template_name not in self._templates:
                msg = f"Template '{template_name}' is not registered."
                raise KeyError(msg)
            del self._templates[template_name]

    def is_registered(self, template_name: str) -> bool:
        """Check if a template is registered in the registry.

        Args:
            template_name: The name of the template to check.

        Returns:
            True if the template is registered, False otherwise.
        """
        with self._registry_lock:
            return template_name in self._templates

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
    def get_instance(cls) -> TemplateRegistry:
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

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        template_names = sorted(self._templates.keys())
        return f"TemplateRegistry(templates={template_names})"

    def _repr_html_(self) -> str:
        """Return an HTML representation of the registry for Jupyter notebooks."""
        template_rows = ""
        td_style = (
            "padding: 10px 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
            "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
            "font-size: 14px; line-height: 1.4;"
        )
        for name in sorted(self._templates.keys()):
            template = self._templates[name]
            template_class = template.__class__.__name__
            data_domain = getattr(template, "_data_domain", "—")
            template_rows += (
                f"<tr><td style='{td_style}'>{html.escape(str(name))}</td>"
                f"<td style='{td_style}'>{html.escape(str(template_class))}</td>"
                f"<td style='{td_style}'>{html.escape(str(data_domain))}</td></tr>"
            )

        box_style = (
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "border: 1px solid rgba(128, 128, 128, 0.3); "
            "border-radius: 8px; padding: 16px; max-width: 100%; box-sizing: border-box; "
            "background: rgba(255, 255, 255, 0.02);"
        )
        header_style = (
            "padding: 12px 16px; margin: -16px -16px 16px -16px; "
            "border-bottom: 2px solid rgba(128, 128, 128, 0.3); "
            "background: rgba(128, 128, 128, 0.05); border-radius: 8px 8px 0 0;"
        )
        no_templates = '<tr><td colspan="3" style="padding: 10px; opacity: 0.5; text-align: center;">No templates registered</td></tr>'  # noqa: E501

        return f"""
        <div style="{box_style}" role="region" aria-labelledby="registry-header">
            <header style="{header_style}" id="registry-header">
                <h3 style="font-size: 1.1em; margin: 0;">TemplateRegistry</h3>
                <span style="margin-left: 15px; opacity: 0.7;">({len(self._templates)} templates)</span>
            </header>
            <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="registry-header">
                <thead>
                    <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                        <th style="text-align: left; padding: 10px; font-weight: 600;" role="columnheader" scope="col">Template Name</th>
                        <th style="text-align: left; padding: 10px; font-weight: 600;" role="columnheader" scope="col">Class</th>
                        <th style="text-align: left; padding: 10px; font-weight: 600;" role="columnheader" scope="col">Domain</th>
                    </tr>
                </thead>
                <tbody role="rowgroup">
                    {template_rows if template_rows else no_templates}
                </tbody>
            </table>
        </div>
        """


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
    """Get an instance of a template from the global registry.

    Each call returns a fresh, independent instance of the template that can be modified without affecting
    the original template or other copies.

    Args:
        name: The name of the template to retrieve.

    Returns:
        An instance of the template if found.
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

    The order is implementation-defined and may change; do not rely on it for stable sorting.

    Returns:
        A list of all registered template names.
    """
    return get_template_registry().list_all_templates()
