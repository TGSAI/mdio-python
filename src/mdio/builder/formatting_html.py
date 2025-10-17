"""HTML formatting utilities for MDIO builder classes."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdio.builder.dataset_builder import MDIODatasetBuilder
    from mdio.builder.template_registry import TemplateRegistry
    from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate


# Common CSS styles shared across HTML representations
BOX_STYLE = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
    "border: 1px solid rgba(128, 128, 128, 0.3); "
    "border-radius: 8px; padding: 16px; max-width: 100%; box-sizing: border-box; "
    "background: rgba(255, 255, 255, 0.02);"
)

HEADER_STYLE = (
    "padding: 12px 16px; margin: -16px -16px 16px -16px; "
    "border-bottom: 2px solid rgba(128, 128, 128, 0.3); "
    "background: rgba(128, 128, 128, 0.05); border-radius: 8px 8px 0 0;"
)

TD_STYLE_BASE = (
    "padding: 10px 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
    "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', "
    "Consolas, 'Courier New', monospace; font-size: 14px; line-height: 1.4;"
)

TD_STYLE_LEFT = f"{TD_STYLE_BASE} text-align: left;"
TD_STYLE_CENTER = f"{TD_STYLE_BASE} text-align: center;"

SUMMARY_STYLE_BASE = (
    "cursor: pointer; font-weight: 600; padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
)

SUMMARY_STYLE_FIRST = f"{SUMMARY_STYLE_BASE} margin-bottom: 8px;"
SUMMARY_STYLE_SUBSEQUENT = f"{SUMMARY_STYLE_BASE} margin: 16px 0 8px 0;"


def _make_empty_row(colspan: int, message: str) -> str:
    """Create an empty data row for tables."""
    return f'<tr><td colspan="{colspan}" style="padding: 8px; opacity: 0.5; text-align: left;">{message}</td></tr>'


def _make_table_row(*cells: object, center_last: bool = False) -> str:
    """Create an HTML table row from cell values."""
    if not cells:
        return ""

    cell_html = "".join(f'<td style="{TD_STYLE_LEFT}">{html.escape(str(cell))}</td>' for cell in cells[:-1])

    # Last cell can be centered if needed
    last_style = TD_STYLE_CENTER if center_last else TD_STYLE_LEFT
    cell_html += f'<td style="{last_style}">{html.escape(str(cells[-1]))}</td>'

    return f"<tr>{cell_html}</tr>"


def _make_html_container(header_title: str, content: str, header_id: str = "header") -> str:
    """Create the outer HTML container with header."""
    escaped_title = html.escape(str(header_title))
    return f"""
    <div style="{BOX_STYLE}" role="region" aria-labelledby="{header_id}">
        <header style="{HEADER_STYLE}" id="{header_id}">
            <h3 style="font-size: 1.1em; margin: 0;">{escaped_title}</h3>
        </header>
        {content}
    </div>
    """


def _make_metadata_section(metadata_items: list[tuple[str, str]]) -> str:
    """Create a metadata display section."""
    items_html = "".join(f"<strong>{key}:</strong> {value}<br>" for key, value in metadata_items)
    return f'<div style="margin-bottom: 15px;">{items_html}</div>'


def _make_details_section(
    summary_text: str,
    table_html: str,
    section_id: str,
    expanded: bool = True,
    is_first: bool = False,
) -> str:
    """Create a collapsible details section with a table."""
    expanded_attr = 'open aria-expanded="true"' if expanded else 'aria-expanded="false"'
    style = SUMMARY_STYLE_FIRST if is_first else SUMMARY_STYLE_SUBSEQUENT
    summary_id = f"{section_id}-summary"
    table_id = f"{section_id}-table"

    return f"""
        <details {expanded_attr}>
            <summary style="{style}" aria-controls="{table_id}"
                     id="{summary_id}">{summary_text}</summary>
            <div style="margin-left: 20px;">
                {table_html}
            </div>
        </details>
        """


def _make_table_header(headers: list[tuple[str, str]]) -> str:
    """Create HTML table header from list of (name, alignment) tuples."""
    header_html = "".join(
        f'<th style="{align}; padding: 8px; font-weight: 600;" role="columnheader" scope="col">{name}</th>'
        for name, align in headers
    )
    return f"""
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);"
                            role="row">
                            {header_html}
                        </tr>
                    </thead>"""


def _make_table(headers: list[tuple[str, str]], rows: str, empty_row: str, table_id: str) -> str:
    """Create an HTML table with headers and rows."""
    header_html = _make_table_header(headers)
    body_content = rows if rows else empty_row

    return f"""
                <table style="width: 100%; border-collapse: collapse;"
                       role="table" aria-labelledby="{table_id}">
                    {header_html}
                    <tbody role="rowgroup">
                        {body_content}
                    </tbody>
                </table>
                """


def dataset_builder_repr_html(builder: MDIODatasetBuilder) -> str:
    """Return an HTML representation of the builder for Jupyter notebooks."""
    # Generate table rows
    dim_rows = "".join(_make_table_row(dim.name, dim.size) for dim in builder._dimensions)
    coord_rows = "".join(
        _make_table_row(
            coord.name,
            ", ".join(d.name for d in coord.dimensions),
            coord.data_type,
        )
        for coord in builder._coordinates
    )
    var_rows = "".join(
        _make_table_row(var.name, ", ".join(d.name for d in var.dimensions), var.data_type)
        for var in builder._variables
    )

    # Metadata section
    created_str = builder._metadata.created_on.strftime("%Y-%m-%d %H:%M:%S UTC")
    metadata_items = [
        ("Name", html.escape(str(builder._metadata.name))),
        ("State", html.escape(str(builder._state.name))),
        ("API Version", html.escape(str(builder._metadata.api_version))),
        ("Created", html.escape(created_str)),
    ]
    metadata_html = _make_metadata_section(metadata_items)

    # Tables
    dim_table = _make_table(
        [("Name", "text-align: left"), ("Size", "text-align: left")],
        dim_rows,
        _make_empty_row(2, "No dimensions added"),
        "builder-dimensions-summary",
    )
    coord_table = _make_table(
        [
            ("Name", "text-align: left"),
            ("Dimensions", "text-align: left"),
            ("Type", "text-align: left"),
        ],
        coord_rows,
        _make_empty_row(3, "No coordinates added"),
        "builder-coordinates-summary",
    )
    var_table = _make_table(
        [
            ("Name", "text-align: left"),
            ("Dimensions", "text-align: left"),
            ("Type", "text-align: left"),
        ],
        var_rows,
        _make_empty_row(3, "No variables added"),
        "builder-variables-summary",
    )

    # Details sections
    dimensions_section = _make_details_section(
        f"▸ Dimensions ({len(builder._dimensions)})",
        dim_table,
        "builder-dimensions",
        is_first=True,
    )
    coordinates_section = _make_details_section(
        f"▸ Coordinates ({len(builder._coordinates)})",
        coord_table,
        "builder-coordinates",
    )
    variables_section = _make_details_section(
        f"▸ Variables ({len(builder._variables)})",
        var_table,
        "builder-variables",
    )

    content = metadata_html + dimensions_section + coordinates_section + variables_section
    return _make_html_container("MDIODatasetBuilder", content, "builder-header")


def template_repr_html(template: AbstractDatasetTemplate) -> str:
    """Return an HTML representation of the template for Jupyter notebooks."""
    # Generate dimension rows with special center alignment for last column
    dim_rows = ""
    if template._dim_names:
        for i, name in enumerate(template._dim_names):
            size = template._dim_sizes[i] if i < len(template._dim_sizes) else "Not set"
            is_spatial = "✓" if name in template._spatial_dim_names else ""
            dim_rows += _make_table_row(name, size, is_spatial, center_last=True)

    # Generate coordinate rows
    all_coords = list(template._physical_coord_names) + list(template._logical_coord_names)
    coord_rows = "".join(
        _make_table_row(
            coord,
            "Physical" if coord in template._physical_coord_names else "Logical",
            template._units.get(coord).name if template._units.get(coord) else "—",
        )
        for coord in all_coords
    )

    # Generate unit rows
    unit_rows = "".join(_make_table_row(key, unit.name) for key, unit in template._units.items())

    # Metadata section
    chunk_shape = html.escape(str(template._var_chunk_shape)) if template._var_chunk_shape else "Not set"
    metadata_items = [
        ("Template Name", html.escape(str(template.name))),
        ("Data Domain", html.escape(str(template._data_domain))),
        ("Default Variable", html.escape(str(template._default_variable_name))),
        ("Chunk Shape", chunk_shape),
    ]
    metadata_html = _make_metadata_section(metadata_items)

    # Tables
    dim_table = _make_table(
        [
            ("Name", "text-align: left"),
            ("Size", "text-align: left"),
            ("Spatial", "text-align: center"),
        ],
        dim_rows,
        _make_empty_row(3, "No dimensions defined"),
        "dimensions-summary",
    )
    coord_table = _make_table(
        [
            ("Name", "text-align: left"),
            ("Type", "text-align: left"),
            ("Units", "text-align: left"),
        ],
        coord_rows,
        _make_empty_row(3, "No coordinates defined"),
        "coordinates-summary",
    )
    units_table = _make_table(
        [("Key", "text-align: left"), ("Unit", "text-align: left")],
        unit_rows,
        _make_empty_row(2, "No units defined"),
        "units-summary",
    )

    # Details sections
    dimensions_section = _make_details_section(
        f"▸ Dimensions ({len(template._dim_names)})",
        dim_table,
        "dimensions",
        is_first=True,
    )
    coordinates_section = _make_details_section(
        f"▸ Coordinates ({len(all_coords)})",
        coord_table,
        "coordinates",
    )
    units_section = _make_details_section(
        f"▸ Units ({len(template._units)})",
        units_table,
        "units",
        expanded=False,
    )

    content = metadata_html + dimensions_section + coordinates_section + units_section
    return _make_html_container(template.__class__.__name__, content, "template-header")


def template_registry_repr_html(registry: TemplateRegistry) -> str:
    """Return an HTML representation of the template registry for Jupyter notebooks."""
    # Generate table rows
    template_rows = "".join(
        _make_table_row(name, template.__class__.__name__, getattr(template, "_data_domain", "—"))
        for name, template in sorted(registry._templates.items())
    )

    # Create table
    table_html = _make_table(
        [
            ("Template Name", "text-align: left"),
            ("Class", "text-align: left"),
            ("Domain", "text-align: left"),
        ],
        template_rows,
        _make_empty_row(3, "No templates registered"),
        "registry-header",
    )

    # Wrap in container with subtitle
    escaped_count = html.escape(str(len(registry._templates)))
    content = f"""
        <span style="margin-left: 15px; opacity: 0.7;">({escaped_count} templates)</span>
        {table_html}
    """

    return _make_html_container("TemplateRegistry", content, "registry-header")
