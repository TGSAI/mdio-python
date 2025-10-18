"""HTML formatting utilities for MDIO builder classes."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mdio.builder.schemas.v1.units import AllUnitModel

if TYPE_CHECKING:
    from mdio.builder.dataset_builder import MDIODatasetBuilder
    from mdio.builder.template_registry import TemplateRegistry
    from mdio.builder.templates.base import AbstractDatasetTemplate


@dataclass(frozen=True)
class CSSStyles:
    """Centralized CSS styles for HTML rendering."""

    box: str = (
        "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        "border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 6px; "
        "padding: 12px; max-width: 100%; box-sizing: border-box; "
        "background: rgba(255, 255, 255, 0.02);"
    )
    header: str = (
        "padding: 8px 12px; margin: -12px -12px 10px -12px; "
        "border-bottom: 1px solid rgba(128, 128, 128, 0.3); "
        "background: rgba(128, 128, 128, 0.05); border-radius: 6px 6px 0 0;"
    )
    td_base: str = (
        "padding: 6px 6px; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
        "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', "
        "Consolas, 'Courier New', monospace; font-size: 12px; line-height: 1.3; "
        "color: var(--color-foreground-primary, currentColor);"
    )
    td_left: str = f"{td_base} text-align: left;"
    td_center: str = f"{td_base} text-align: center;"
    td_right: str = f"{td_base} text-align: right;"
    summary_base: str = (
        "cursor: pointer; font-weight: 600; padding: 6px 8px; border-radius: 4px; transition: background-color 0.2s;"
    )
    summary_first: str = f"{summary_base} margin-bottom: 6px;"
    summary_subsequent: str = f"{summary_base} margin: 10px 0 6px 0;"


class TableBuilder:
    """Utility class for building HTML tables."""

    def __init__(self, headers: list[tuple[str, str]], table_id: str):
        self.headers = headers
        self.table_id = table_id
        self.rows: list[str] = []

    def add_row(self, *cells: object, aligns: list[str] | None = None, center_last: bool = False) -> None:
        """Add a table row with specified cell values and alignments."""
        if not cells:
            return

        cell_html_parts: list[str] = []
        css = CSSStyles()

        if aligns:
            for i, cell in enumerate(cells):
                align = aligns[i] if i < len(aligns) else "left"
                style = css.td_left if align == "left" else css.td_center if align == "center" else css.td_right
                cell_html_parts.append(f'  <td style="{style}">{html.escape(str(cell))}</td>')
        else:
            cell_html_parts.extend(
                [f'  <td style="{css.td_left}">{html.escape(str(cell))}</td>' for cell in cells[:-1]]
            )
            last_style = css.td_center if center_last else css.td_left
            cell_html_parts.append(f'  <td style="{last_style}">{html.escape(str(cells[-1]))}</td>')

        self.rows.append("\n<tr>\n" + "\n".join(cell_html_parts) + "\n</tr>\n")

    def add_empty_row(self, message: str) -> None:
        """Add an empty row with a message spanning all columns."""
        self.rows.append(
            f"\n<tr>\n"
            f'  <td colspan="{len(self.headers)}" style="padding: 8px; opacity: 0.5; text-align: left;">'
            f"{message}</td>\n"
            f"</tr>\n"
        )

    def build(self) -> str:
        """Build the complete HTML table."""
        header_html = "\n".join(
            (
                f'                            <th style="{align}; padding: 6px; font-weight: 600; '
                'color: var(--color-foreground-primary, currentColor);" '
                f'role="columnheader" scope="col">{name}</th>'
            )
            for name, align in self.headers
        )
        header_section = (
            f"                    <thead>\n"
            f'                        <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.4);" '
            f'role="row">\n'
            f"{header_html}\n"
            f"                        </tr>\n"
            f"                    </thead>"
        )
        body_content = "".join(self.rows) if self.rows else self.add_empty_row("No data available")
        return (
            f"\n                <style>\n"
            f"                    #{self.table_id} tbody tr:nth-child(odd) {{ "
            f"background: rgba(128, 128, 128, 0.03); }}\n"
            f"                    #{self.table_id} tbody tr:hover {{ "
            f"background: rgba(128, 128, 128, 0.06); }}\n"
            f"                </style>\n"
            f'                <table id="{self.table_id}" style="width: 100%; border-collapse: collapse;" '
            f'role="table" aria-labelledby="{self.table_id}">\n'
            f"{header_section}\n"
            f'                    <tbody role="rowgroup">\n'
            f"                        {body_content}\n"
            f"                    </tbody>\n"
            f"                </table>\n"
        )


def make_html_container(header_title: str, content: str, header_id: str = "header") -> str:
    """Create an HTML container with a header."""
    css = CSSStyles()
    escaped_title = html.escape(str(header_title))
    return (
        f'\n    <div style="{css.box}" role="region" aria-labelledby="{header_id}">\n'
        f'        <header style="{css.header}" id="{header_id}">\n'
        f'            <h3 style="font-size: 1.1em; margin: 0;">{escaped_title}</h3>\n'
        f"        </header>\n"
        f"        {content}\n"
        f"    </div>\n"
    )


def make_metadata_section(metadata_items: list[tuple[str, str]]) -> str:
    """Create a metadata display section."""
    items_html = "\n".join(f"  <strong>{key}:</strong> {value}<br>" for key, value in metadata_items)
    return f'<div style="margin-bottom: 10px;">\n{items_html}\n</div>'


def make_details_section(
    summary_text: str,
    table_html: str,
    section_id: str,
    expanded: bool = True,
    is_first: bool = False,
) -> str:
    """Create a collapsible details section with a table."""
    css = CSSStyles()
    expanded_attr = 'open aria-expanded="true"' if expanded else 'aria-expanded="false"'
    style = css.summary_first if is_first else css.summary_subsequent
    summary_id = f"{section_id}-summary"
    table_id = f"{section_id}-table"
    return (
        f"\n        <details {expanded_attr}>\n"
        f'            <summary style="{style}" aria-controls="{table_id}" '
        f'id="{summary_id}">{summary_text}</summary>\n'
        f'            <div style="margin-left: 12px;">\n'
        f"                {table_html}\n"
        f"            </div>\n"
        f"        </details>\n"
    )


def format_unit_for_display(unit_model: AllUnitModel | None) -> str:
    """Return a human-friendly unit string from a unit model or enum."""
    if unit_model is None or not isinstance(unit_model, AllUnitModel):
        return "—"
    field_name = list(unit_model.__fields__)[0]
    return str(getattr(unit_model, field_name).value)


def format_int_or_dash(value: object) -> str:
    """Format integers with thousands separators; return an em dash for None/empty markers."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if value in (None, "—", "Not set"):
        return "—"
    return str(value)


def dataset_builder_repr_html(builder: MDIODatasetBuilder) -> str:
    """Return an HTML representation of the builder for Jupyter notebooks."""
    # Build dimension table
    dim_table = TableBuilder(
        headers=[
            ("Name", "text-align: left"),
            ("Size", "text-align: left"),
        ],
        table_id="builder-dimensions-summary",
    )
    for dim in builder._dimensions:
        dim_table.add_row(dim.name, dim.size)

    # Build coordinates table
    coord_table = TableBuilder(
        headers=[
            ("Name", "text-align: left"),
            ("Dimensions", "text-align: left"),
            ("Type", "text-align: left"),
        ],
        table_id="builder-coordinates-summary",
    )
    for coord in builder._coordinates:
        coord_table.add_row(coord.name, ", ".join(d.name for d in coord.dimensions), coord.data_type)

    # Build variables table
    var_table = TableBuilder(
        headers=[
            ("Name", "text-align: left"),
            ("Dimensions", "text-align: left"),
            ("Type", "text-align: left"),
        ],
        table_id="builder-variables-summary",
    )
    for var in builder._variables:
        var_table.add_row(var.name, ", ".join(d.name for d in var.dimensions), var.data_type)

    # Metadata section
    created_str = builder._metadata.created_on.strftime("%Y-%m-%d %H:%M:%S UTC")
    metadata_items = [
        ("Name", html.escape(str(builder._metadata.name))),
        ("State", html.escape(str(builder._state.name))),
        ("API Version", html.escape(str(builder._metadata.api_version))),
        ("Created", html.escape(created_str)),
    ]
    metadata_html = make_metadata_section(metadata_items)

    # Details sections
    dimensions_section = make_details_section(
        f"Dimensions ({len(builder._dimensions)})",
        dim_table.build(),
        "builder-dimensions",
        is_first=True,
    )
    coordinates_section = make_details_section(
        f"Coordinates ({len(builder._coordinates)})",
        coord_table.build(),
        "builder-coordinates",
    )
    variables_section = make_details_section(
        f"Variables ({len(builder._variables)})",
        var_table.build(),
        "builder-variables",
    )

    content = metadata_html + dimensions_section + coordinates_section + variables_section
    return make_html_container("MDIODatasetBuilder", content, "builder-header")


def template_repr_html(template: AbstractDatasetTemplate) -> str:
    """Return an HTML representation of the template for Jupyter notebooks."""
    # Build dimension table
    dim_table = TableBuilder(
        headers=[
            ("Name", "text-align: left"),
            ("Size", "text-align: right"),
            ("Chunk Sizes", "text-align: right"),
            ("Units", "text-align: right"),
            ("Spatial", "text-align: center"),
        ],
        table_id="dimensions-summary",
    )
    for i, name in enumerate(template.dimension_names):
        # Guard against templates not yet built (sizes/chunks may be empty)
        size_val = template._dim_sizes[i] if i < len(template._dim_sizes) else "—"
        chunk_val = template.full_chunk_shape[i] if i < len(template.full_chunk_shape) else "—"
        unit_str = format_unit_for_display(template.get_unit_by_key(name))
        is_spatial = "✓" if name in template.spatial_dimension_names else ""
        dim_table.add_row(
            name,
            format_int_or_dash(size_val),
            format_int_or_dash(chunk_val),
            unit_str,
            is_spatial,
            aligns=["left", "right", "right", "right", "center"],
        )

    # Build coordinates table
    all_coords = template.coordinate_names
    coord_table = TableBuilder(
        headers=[
            ("Name", "text-align: left"),
            ("Type", "text-align: left"),
            ("Units", "text-align: left"),
        ],
        table_id="coordinates-summary",
    )
    for coord in all_coords:
        coord_table.add_row(
            coord,
            "Physical" if coord in template.physical_coordinate_names else "Logical",
            format_unit_for_display(template.get_unit_by_key(coord)),
        )

    # Metadata section
    default_variable_name = getattr(template, "_default_variable_name", "")
    default_var_units = format_unit_for_display(template.get_unit_by_key(default_variable_name))
    metadata_items = [
        ("Template Name", str(template.name)),
        ("Data Domain", template._data_domain),
        ("Default Variable", template.default_variable_name),
        ("Default Variable Units", default_var_units),
    ]
    metadata_html = make_metadata_section(metadata_items)

    # Details sections
    dimensions_section = make_details_section(
        f"Dimensions ({len(template.dimension_names)})",
        dim_table.build(),
        "dimensions",
        is_first=True,
    )
    coordinates_section = make_details_section(
        f"Coordinates ({len(all_coords)})",
        coord_table.build(),
        "coordinates",
    )

    content = metadata_html + dimensions_section + coordinates_section
    return make_html_container(template.__class__.__name__, content, "template-header")


def template_registry_repr_html(registry: TemplateRegistry) -> str:
    """Return an HTML representation of the template registry for Jupyter notebooks."""
    registered_templates = registry.list_all_templates()
    n_templates = len(registered_templates)

    # Build template table with count next to column header
    table = TableBuilder(
        headers=[
            (
                f'Template <span style="opacity: 0.6; font-weight: 500;">({n_templates})</span>',
                "text-align: left",
            ),
            ("Default Var", "text-align: center"),
            ("Dimensions", "text-align: left"),
            ("Chunk Sizes", "text-align: left"),
            ("Coords", "text-align: left"),
        ],
        table_id="registry-header",
    )

    for name in sorted(registered_templates):
        template = registry.get(name)
        default_var = template._default_variable_name
        dim_names_str = ", ".join(template.dimension_names)
        coords_names_str = ", ".join(template.coordinate_names)
        chunk_str = "×".join(str(cs) for cs in template.full_chunk_shape)
        table.add_row(
            name,
            default_var,
            dim_names_str,
            chunk_str,
            coords_names_str,
            aligns=["left", "center", "left", "left", "left"],
        )

    content = table.build()
    return make_html_container("TemplateRegistry", content, "registry-header")
