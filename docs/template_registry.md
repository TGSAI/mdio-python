# Template Registry

A simple, thread-safe place to discover and fetch dataset templates for MDIO.

## Why use it

- One place to find all available templates
- Safe to use across threads and the whole app (singleton)
- Every fetch gives you your own editable copy (no side effects)
- Comes preloaded with common seismic templates

```{note}
Fetching a template with `get_template()` returns a deep copy. Editing it will not change the
registry or anyone else’s copy.
```

## Quick start

```python
from mdio.builder.template_registry import get_template, list_templates

# See what's available
print(list_templates())
# e.g. ["Seismic2DPostStackTime", "Seismic3DPostStackDepth", ...]

# Grab a template by name
template = get_template("Seismic3DPostStackTime")

# Customize your copy (safe)
template.add_units({"amplitude": "unitless"})
```

## Common tasks

### Fetch a template you can edit

```python
from mdio.builder.template_registry import get_template

template = get_template("Seismic2DPostStackDepth")
# Use/modify template freely — it’s your copy
```

### List available templates

```python
from mdio.builder.template_registry import list_templates

names = list_templates()
for name in names:
    print(name)
```

### Check if a template exists

```python
from mdio.builder.template_registry import is_template_registered

if is_template_registered("Seismic3DPostStackTime"):
    ...  # safe to fetch
```

### Register your own template (optional)

If you have a custom template class, register an instance so others can fetch it by name:

```python
from typing import Any
from mdio.builder.template_registry import register_template
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class MyTemplate(AbstractDatasetTemplate):
    def __init__(self, domain: SeismicDataDomain = "time"):
        super().__init__(domain)

    @property
    def _name(self) -> str:
        # The public name becomes something like "MyTemplateTime"
        return f"MyTemplate{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "2D", "gatherType": "custom"}


# Make it available globally
registered_name = register_template(MyTemplate("time"))
print(registered_name)  # "MyTemplateTime"
```

```{tip}
Use `list_templates()` to discover the exact names to pass to `get_template()`.
```

## Troubleshooting

- KeyError: “Template 'XYZ' is not registered.”
  - The name is wrong or not registered yet.
  - Call `list_templates()` to see valid names, or `is_template_registered(name)` to check first.

## FAQ

- Do I need to create a TemplateRegistry instance?
  No. Use the global helpers: `get_template`, `list_templates`, `register_template`, and `is_template_registered`.
- Are templates shared between callers or threads?
  No. Each `get_template()` call returns a deep-copied instance that is safe to modify independently.

## API reference

```{eval-rst}
.. automodule:: mdio.builder.template_registry
   :members:
```
