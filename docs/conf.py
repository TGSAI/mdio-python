"""Sphinx configuration."""
project = "MDIO"
author = "TGS"
copyright = "2023, TGS"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_click",
    "sphinx_copybutton",
    "myst_nb",
]

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "groupwise"
autoclass_content = "both"
autosectionlabel_prefix_document = True

html_theme = "furo"

myst_number_code_blocks = ["python"]
myst_heading_anchors = 2
myst_enable_extensions = [
    "linkify",
    "replacements",
    "smartquotes",
]

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_line_continuation_character = "\\"
copybutton_prompt_is_regexp = True

nb_execution_mode = "off"
