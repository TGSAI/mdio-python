"""Sphinx configuration."""
project = "MDIO"
author = "TGS"
copyright = "2022, TGS"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
    "sphinx_copybutton",
]
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "groupwise"
html_theme = "furo"
autoclass_content = "both"

myst_heading_anchors = 2

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
