"""HTML formatting utilities for MDIO builder classes."""

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

TD_STYLE_LEFT = (
    "padding: 10px 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
    "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
    "font-size: 14px; line-height: 1.4;"
)

TD_STYLE_CENTER = (
    "padding: 10px 8px; text-align: center; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
    "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
    "font-size: 14px; line-height: 1.4;"
)

SUMMARY_STYLE = (
    "cursor: pointer; font-weight: 600; margin-bottom: 8px; "
    "padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
)

SUMMARY_STYLE_2 = (
    "cursor: pointer; font-weight: 600; margin: 16px 0 8px 0; "
    "padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
)


def _make_table_row(*cells: object) -> str:
    """Create an HTML table row from cell values."""
    cell_html = "".join(f"<td style='{TD_STYLE_LEFT}'>{html.escape(str(cell))}</td>" for cell in cells)
    return f"<tr>{cell_html}</tr>"


def _make_html_container(header_title: str, content: str, header_id: str = "header") -> str:
    """Create the outer HTML container with header."""
    return f"""
    <div style="{BOX_STYLE}" role="region" aria-labelledby="{header_id}">
        <header style="{HEADER_STYLE}" id="{header_id}">
            <h3 style="font-size: 1.1em; margin: 0;">{html.escape(str(header_title))}</h3>
        </header>
        {content}
    </div>
    """


def _make_metadata_section(metadata_items: list[tuple[str, str]]) -> str:
    """Create a metadata display section."""
    items_html = "".join(f"<strong>{key}:</strong> {value}<br>" for key, value in metadata_items)
    return f'<div style="margin-bottom: 15px;">{items_html}</div>'


def _make_details_section(  # noqa: PLR0913
    summary_text: str,
    table_html: str,
    summary_id: str,
    table_id: str,
    expanded: bool = True,
    summary_style: str = "SUMMARY_STYLE",
) -> str:
    """Create a collapsible details section with a table."""
    expanded_attr = 'open aria-expanded="true"' if expanded else 'aria-expanded="false"'
    style = SUMMARY_STYLE if summary_style == "SUMMARY_STYLE" else SUMMARY_STYLE_2
    return f"""
        <details {expanded_attr}>
            <summary style="{style}" aria-controls="{table_id}"
                     id="{summary_id}">{summary_text}</summary>
            <div style="margin-left: 20px;">
                {table_html}
            </div>
        </details>
        """


def _make_table(headers: list[tuple[str, str]], rows: str, no_data_row: str, table_id: str) -> str:
    """Create an HTML table with headers and rows."""
    header_html = "".join(
        f'<th style="{align}; padding: 8px; font-weight: 600;" role="columnheader" scope="col">{name}</th>'
        for name, align in headers
    )
    return f"""
                <table style="width: 100%; border-collapse: collapse;" role="table"
                       aria-labelledby="{table_id}">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            {header_html}
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {rows if rows else no_data_row}
                    </tbody>
                </table>
                """


def dataset_builder_repr_html(builder: "MDIODatasetBuilder") -> str:
    """Return an HTML representation of the builder for Jupyter notebooks."""
    # Generate table rows
    dim_rows = "".join(_make_table_row(dim.name, dim.size) for dim in builder._dimensions)
    coord_rows = "".join(
        _make_table_row(coord.name, ", ".join(d.name for d in coord.dimensions), coord.data_type)
        for coord in builder._coordinates
    )
    var_rows = "".join(
        _make_table_row(var.name, ", ".join(d.name for d in var.dimensions), var.data_type)
        for var in builder._variables
    )

    # No data messages
    no_dims = '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions added</td></tr>'  # noqa: E501
    no_coords = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates added</td></tr>'  # noqa: E501
    )
    no_vars = '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No variables added</td></tr>'  # noqa: E501

    # Metadata section
    metadata_items = [
        ("Name", html.escape(str(builder._metadata.name))),
        ("State", html.escape(str(builder._state.name))),
        ("API Version", html.escape(str(builder._metadata.api_version))),
        ("Created", html.escape(str(builder._metadata.created_on.strftime("%Y-%m-%d %H:%M:%S UTC")))),
    ]
    metadata_html = _make_metadata_section(metadata_items)

    # Tables
    dim_table = _make_table(
        [("Name", "text-align: left"), ("Size", "text-align: left")], dim_rows, no_dims, "builder-dimensions-summary"
    )
    coord_table = _make_table(
        [("Name", "text-align: left"), ("Dimensions", "text-align: left"), ("Type", "text-align: left")],
        coord_rows,
        no_coords,
        "builder-coordinates-summary",
    )
    var_table = _make_table(
        [("Name", "text-align: left"), ("Dimensions", "text-align: left"), ("Type", "text-align: left")],
        var_rows,
        no_vars,
        "builder-variables-summary",
    )

    # Details sections
    dimensions_section = _make_details_section(
        f"▸ Dimensions ({len(builder._dimensions)})",
        dim_table,
        "builder-dimensions-summary",
        "builder-dimensions-table",
        expanded=True,
    )
    coordinates_section = _make_details_section(
        f"▸ Coordinates ({len(builder._coordinates)})",
        coord_table,
        "builder-coordinates-summary",
        "builder-coordinates-table",
        expanded=True,
        summary_style="SUMMARY_STYLE_2",
    )
    variables_section = _make_details_section(
        f"▸ Variables ({len(builder._variables)})",
        var_table,
        "builder-variables-summary",
        "builder-variables-table",
        expanded=True,
        summary_style="SUMMARY_STYLE_2",
    )

    content = metadata_html + dimensions_section + coordinates_section + variables_section
    return _make_html_container("MDIODatasetBuilder", content, "builder-header")


def template_repr_html(template: "AbstractDatasetTemplate") -> str:
    """Return an HTML representation of the template for Jupyter notebooks."""
    # Generate table rows
    dim_rows = ""
    if template._dim_names:
        for i, name in enumerate(template._dim_names):
            size = template._dim_sizes[i] if i < len(template._dim_sizes) else "Not set"
            is_spatial = "✓" if name in template._spatial_dim_names else ""
            dim_rows += (
                f"<tr><td style='{TD_STYLE_LEFT}'>{html.escape(str(name))}</td>"
                f"<td style='{TD_STYLE_LEFT}'>{html.escape(str(size))}</td>"
                f"<td style='{TD_STYLE_CENTER}'>{html.escape(is_spatial)}</td></tr>"
            )

    all_coords = list(template._physical_coord_names) + list(template._logical_coord_names)
    coord_rows = "".join(
        _make_table_row(
            coord,
            "Physical" if coord in template._physical_coord_names else "Logical",
            template._units.get(coord).name if template._units.get(coord) else "—",
        )
        for coord in all_coords
    )

    unit_rows = "".join(_make_table_row(key, unit.name) for key, unit in template._units.items())

    # No data messages
    no_dims = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions defined</td></tr>'  # noqa: E501
    )
    no_coords = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates defined</td></tr>'  # noqa: E501
    )
    no_units = '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No units defined</td></tr>'  # noqa: E501

    # Metadata section
    chunk_shape = html.escape(str(template._var_chunk_shape)) if template._var_chunk_shape else "Not set"
    metadata_items = [
        ("Template Name", html.escape(str(template.name))),
        ("Data Domain", html.escape(str(template._data_domain))),
        ("Default Variable", html.escape(str(template._default_variable_name))),
        ("Chunk Shape", chunk_shape),
    ]
    metadata_html = _make_metadata_section(metadata_items)

    # Tables - dimensions table needs special handling for center alignment
    dim_table_html = f"""
                <table style="width: 100%; border-collapse: collapse;" role="table"
                       aria-labelledby="dimensions-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader"
                                scope="col">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader"
                                scope="col">Size</th>
                            <th style="text-align: center; padding: 8px; font-weight: 600;" role="columnheader"
                                scope="col">Spatial</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {dim_rows if dim_rows else no_dims}
                    </tbody>
                </table>
                """

    coord_table = _make_table(
        [("Name", "text-align: left"), ("Type", "text-align: left"), ("Units", "text-align: left")],
        coord_rows,
        no_coords,
        "coordinates-summary",
    )
    units_table = _make_table(
        [("Key", "text-align: left"), ("Unit", "text-align: left")], unit_rows, no_units, "units-summary"
    )

    # Details sections
    dimensions_section = _make_details_section(
        f"▸ Dimensions ({len(template._dim_names)})",
        dim_table_html,
        "dimensions-summary",
        "dimensions-table",
        expanded=True,
    )
    coordinates_section = _make_details_section(
        f"▸ Coordinates ({len(all_coords)})",
        coord_table,
        "coordinates-summary",
        "coordinates-table",
        expanded=True,
        summary_style="SUMMARY_STYLE_2",
    )
    units_section = _make_details_section(
        f"▸ Units ({len(template._units)})",
        units_table,
        "units-summary",
        "units-table",
        expanded=False,
        summary_style="SUMMARY_STYLE_2",
    )

    content = metadata_html + dimensions_section + coordinates_section + units_section
    return _make_html_container(template.__class__.__name__, content, "template-header")


def template_registry_repr_html(registry: "TemplateRegistry") -> str:
    """Return an HTML representation of the template registry for Jupyter notebooks."""
    # Generate table rows
    template_rows = "".join(
        _make_table_row(name, template.__class__.__name__, getattr(template, "_data_domain", "—"))
        for name, template in sorted(registry._templates.items())
    )

    # No data message
    no_templates = (
        '<tr><td colspan="3" style="padding: 10px; opacity: 0.5; text-align: center;">No templates registered</td></tr>'  # noqa: E501
    )

    # Special header with subtitle
    header_html = f"""
    <div style="{BOX_STYLE}" role="region" aria-labelledby="registry-header">
        <header style="{HEADER_STYLE}" id="registry-header">
            <h3 style="font-size: 1.1em; margin: 0;">TemplateRegistry</h3>
            <span style="margin-left: 15px; opacity: 0.7;">({len(registry._templates)} templates)</span>
        </header>
    """

    # Table
    table_html = f"""
        <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="registry-header">
            <thead>
                <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                    <th style="text-align: left; padding: 10px; font-weight: 600;" role="columnheader"
                        scope="col">Template Name</th>
                    <th style="text-align: left; padding: 10px; font-weight: 600;" role="columnheader"
                        scope="col">Class</th>
                    <th style="text-align: left; padding: 10px; font-weight: 600;" role="columnheader"
                        scope="col">Domain</th>
                </tr>
            </thead>
            <tbody role="rowgroup">
                {template_rows if template_rows else no_templates}
            </tbody>
        </table>
    </div>
    """

    return header_html + table_html
