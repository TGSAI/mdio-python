# Template Registry

A thread-safe singleton registry for managing dataset templates in MDIO applications.

## Overview

The `TemplateRegistry` implements the singleton pattern to ensure there's only one instance managing all dataset templates throughout the application lifecycle. This provides a centralized registry for template management with thread-safe operations.

**Important:** When you retrieve a template from the registry using `get()`, you receive a **deep copy** of the template. This means each retrieved template is an independent, modifiable instance that can be customized for your specific use case without affecting the original template or other retrieved copies.

## Features

- **Singleton Pattern**: Ensures only one registry instance exists
- **Thread Safety**: All operations are thread-safe using locks
- **Global Access**: Convenient global functions for common operations
- **Deep Copy on Retrieval**: Each `get()` call returns an independent copy that can be modified without side effects
- **Pickleable & Deepcopyable**: Retrieved templates are fully pickleable and deepcopyable
- **Default Templates**: The registry is instantiated with the default set of templates:
  - **Time Domain Templates**:
    - PostStack2DTime
    - PostStack3DTime
    - PreStackCdpOffsetGathers2DTime
    - PreStackCdpAngleGathers2DTime
    - PreStackCdpOffsetGathers3DTime
    - PreStackCdpAngleGathers3DTime
    - PreStackCocaGathers3DTime
    - PreStackShotGathers2DTime
    - PreStackShotGathers3DTime
  - **Depth Domain Templates**:
    - PostStack2DDepth
    - PostStack3DDepth
    - PreStackCdpOffsetGathers2DDepth
    - PreStackCdpAngleGathers2DDepth
    - PreStackCdpOffsetGathers3DDepth
    - PreStackCdpAngleGathers3DDepth
    - PreStackCocaGathers3DDepth

## Usage

### Basic Usage

```python
from mdio.builder.template_registry import TemplateRegistry

# Get the singleton instance
registry = TemplateRegistry()

# Or use the class method
registry = TemplateRegistry.get_instance()

# Register a template
template = MyDatasetTemplate()
template_name = registry.register(template)
print(f"Registered template named {template_name}")

# Retrieve a template using a well-known name
template = registry.get("my_template")
# Retrieve a template using the name returned when the template was registered
template = registry.get(template_name)

# Check if template exists
if registry.is_registered("my_template"):
    print("Template is registered")

# List all templates
template_names = registry.list_all_templates()
```

### Global Functions

For convenience, you can use global functions that operate on the singleton instance:

```python
from mdio.builder.template_registry import (
    register_template,
    get_template,
    is_template_registered,
    list_templates
)

# Register a template globally
register_template(Seismic3DTemplate())

# Get a template
template = get_template("seismic_3d")

# Check registration
if is_template_registered("seismic_3d"):
    print("Template available")

# List all registered templates
templates = list_templates()
```

### Template Independence and Modification

Each call to `get()` returns an independent deep copy of the template, allowing you to customize templates without affecting other instances:

```python
from mdio.builder.template_registry import get_template

# Get two instances of the same template
template1 = get_template("PostStack3DTime")
template2 = get_template("PostStack3DTime")

# They are different objects
assert template1 is not template2

# Modify the first template
template1.add_units({"amplitude": "meters"})

# The second template is unaffected
# template2 does not have the custom units from template1

# Get a fresh copy - also unaffected by previous modifications
template3 = get_template("PostStack3DTime")
```

This design ensures:

- **Thread Safety**: Multiple threads can retrieve and modify templates independently
- **Isolation**: Changes to one template instance don't affect others
- **Reusability**: You can always get a fresh, unmodified template from the registry
- **Pickling Support**: Each template instance can be pickled for distributed computing

### Multiple Instantiation

The singleton pattern ensures all instantiations of the registry return the same object:

```python
registry1 = TemplateRegistry()
registry2 = TemplateRegistry()
registry3 = TemplateRegistry.get_instance()

# All variables point to the same instance
assert registry1 is registry2 is registry3
```

## API Reference

```{eval-rst}
.. automodule:: mdio.builder.template_registry
   :members:
```

## Thread Safety

All operations on the registry are thread-safe:

```python
import threading

def register_templates():
    registry = TemplateRegistry()
    for i in range(10):
        template = MyTemplate(f"template_{i}")
        registry.register(template)

# Multiple threads can safely access the registry
threads = [threading.Thread(target=register_templates) for _ in range(5)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

## Best Practices

1. **Use Global Functions**: For simple operations, prefer the global convenience functions
2. **Register Early**: Register all templates during application startup
3. **Thread Safety**: The registry is thread-safe, but individual templates may not be
4. **Testing Isolation**: Always reset the singleton in test setup/teardown

## Example: Complete Template Management

```python
from mdio.builder.template_registry import TemplateRegistry, register_template, list_templates
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.builder.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate


def setup_templates():
    """Register MDIO templates at runtime.
    Custom templates can be created in external projects and added without modifying the MDIO library code
    """
    # Use parametrized templates
    template_name = register_template(Seismic3DPostStackTemplate("time"))
    print(f"Registered template named {template_name}")

    template_name = register_template(Seismic3DPostStackTemplate("depth"))
    print(f"Registered template named {template_name}")

    template_name = register_template(Seismic3DPreStackCDPTemplate("time", "offset"))
    print(f"Registered template named {template_name}")

    print(f"Registered templates: {list_templates()}")


# Application startup
setup_templates()

# Later in the application
template = TemplateRegistry().get("PostStack3DDepth")
dataset = template.build_dataset(name="Seismic 3d m/m/ft",
                                  sizes=(256, 512, 384))
```

## Error Handling

The registry provides clear error messages:

```python
# Template not registered
try:
    template = get_template("nonexistent")
except KeyError as e:
    print(f"Error: {e}")  # "Template 'nonexistent' is not registered."

# Duplicate registration
try:
    # Create two templates with the same name
    class DuplicateTemplate(Seismic3DPostStackTemplate):
        def __init__(self):
            super().__init__("time")

        @property
        def _name(self) -> str:
            return "duplicate"

    template1 = DuplicateTemplate()
    template2 = DuplicateTemplate()

    register_template(template1)
    register_template(template2)  # This will raise ValueError
except ValueError as e:
    print(f"Error: {e}")  # "Template 'duplicate' is already registered."
```
