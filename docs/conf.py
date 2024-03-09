"""Sphinx configuration."""

# -- Project information -----------------------------------------------------

project = "MDIO"
author = "TGS"
copyright = "2023, TGS"  # noqa: A001

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autosectionlabel",
    "sphinx_click",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_design",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    "jupyter_execute",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

pygments_style = "vs"
pygments_dark_style = "material"

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "groupwise"
autoclass_content = "class"
autosectionlabel_prefix_document = True

autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_swap_name_and_alias = True
autodoc_pydantic_field_show_alias = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False

html_theme = "furo"

myst_number_code_blocks = ["python"]
myst_heading_anchors = 2
myst_words_per_minute = 80
myst_enable_extensions = [
    "colon_fence",
    "linkify",
    "replacements",
    "smartquotes",
    "attrs_inline",
]

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_line_continuation_character = "\\"
copybutton_prompt_is_regexp = True

nb_execution_mode = "off"
