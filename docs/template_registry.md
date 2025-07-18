# Template Registry Singleton

A thread-safe singleton registry for managing dataset templates in MDIO applications.

## Overview

The `TemplateRegistry` implements the singleton pattern to ensure there's only one instance managing all dataset templates throughout the application lifecycle. This provides a centralized registry for template management with thread-safe operations.

## Features

- **Singleton Pattern**: Ensures only one registry instance exists
- **Thread Safety**: All operations are thread-safe using locks
- **Global Access**: Convenient global functions for common operations
- **Advanced Support**: Reset functionality for environment re-usability.
- **Default Templates**: The registry is instantiated with the default set of templates:
  - PostStack2DTime
  - PostStack3DTime
  - PreStackCdpGathers3DTime
  - PreStackShotGathers3DTime
  - PostStack2DDepth
  - PostStack3DDepth
  - PreStackCdpGathers3DDepth
  - PreStackShotGathers3DDepth

## Usage

### Basic Usage

```python
from mdio.schemas.v1.templates.template_registry import TemplateRegistry

# Get the singleton instance
registry = TemplateRegistry()

# Or use the class method
registry = TemplateRegistry.get_instance()

# Register a template
template = MyDatasetTemplate()
template_name=registry.register(template)
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
from mdio.schemas.v1.templates.template_registry import (
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

### Multiple Instantiation

The singleton pattern ensures all instantiations return the same object:

```python
registry1 = TemplateRegistry()
registry2 = TemplateRegistry()
registry3 = TemplateRegistry.get_instance()

# All variables point to the same instance
assert registry1 is registry2 is registry3
```

## API Reference

### Core Methods

#### `register(instance: AbstractDatasetTemplate) -> str`

Registers a template instance and returns its normalized name.

- **Parameters:**
  - `instance`: Template instance implementing `AbstractDatasetTemplate`
- **Returns:** The template name
- **Raises:** `ValueError` if template name is already registered

#### `get(template_name: str) -> AbstractDatasetTemplate`

Retrieves a registered template by name.

- **Parameters:**
  - `template_name`: Name of the template (case-insensitive)
- **Returns:** The registered template instance
- **Raises:** `KeyError` if template is not registered

#### `unregister(template_name: str) -> None`

Removes a template from the registry.

- **Parameters:**
  - `template_name`: Name of the template to remove
- **Raises:** `KeyError` if template is not registered

#### `is_registered(template_name: str) -> bool`

Checks if a template is registered.

- **Parameters:**
  - `template_name`: Name of the template to check
- **Returns:** `True` if template is registered, `False` otherwise

#### `list_all_templates() -> List[str]`

Returns a list of all registered template names.

- **Returns:** List of template names

#### `clear() -> None`

Removes all registered templates. Useful for testing.

### Class Methods

#### `get_instance() -> TemplateRegistry`

Alternative way to get the singleton instance.

- **Returns:** The singleton registry instance

### Global Functions

#### `get_template_registry() -> TemplateRegistry`

Returns the global singleton registry instance.

#### `register_template(template: AbstractDatasetTemplate) -> str`

Registers a template in the global registry.

#### `get_template(name: str) -> AbstractDatasetTemplate`

Gets a template from the global registry.

#### `is_template_registered(name: str) -> bool`

Checks if a template is registered in the global registry.

#### `list_templates() -> List[str]`

Lists all templates in the global registry.

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
from mdio.schemas.v1.templates.template_registry import TemplateRegistry
from mdio.schemas.v1.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.schemas.v1.templates.seismic_3d_prestack_time import Seismic3DPostStackTimeTemplate
from mdio.schemas.v1.templates.seismic_3d_prestack import Seismic3DPreStackTemplate

def setup_templates():
    """Register MDIO templates runtime.
    Custom templates can be created in external projects and added without modifying the MDIO library code
    """
    # Use strongly-typed template
    template_name = TemplateRegistry.register(Seismic3DPostStackTimeTemplate())
    print(f"Registered template named {template_name}")
    # Use parametrized template
    template_name = TemplateRegistry.register(Seismic3DPostStackTemplate("Depth"))
    print(f"Registered template named {template_name}")
    template_name = TemplateRegistry.register(Seismic3DPreStackTemplate())
    print(f"Registered template named {template_name}")

    print(f"Registered templates: {list_templates()}")

# Application startup
setup_standard_templates()

# Later in the application
template = TemplateRegistry().get_template("PostStack3DDepth")
dataset = template.create_dataset(name="Seismic 3d m/m/ft",
                                  sizes = [256, 512, 384]
                                  coord_units = [
                                    AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER)),
                                    AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER)),
                                    AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))]
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
    register_template("duplicate", template1)
    register_template("duplicate", template2)
except ValueError as e:
    print(f"Error: {e}")  # "Template 'duplicate' is already registered."
```
