"""HTML formatting utilities for MDIO builder classes."""

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdio.builder.dataset_builder import MDIODatasetBuilder
    from mdio.builder.template_registry import TemplateRegistry
    from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate


def dataset_builder_repr_html(builder: "MDIODatasetBuilder") -> str:
    """Return an HTML representation of the builder for Jupyter notebooks."""
    dim_rows = ""
    td_style = (
        "padding: 10px 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
        "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
        "font-size: 14px; line-height: 1.4;"
    )
    for dim in builder._dimensions:
        dim_rows += f"<tr><td style='{td_style}'>{html.escape(str(dim.name))}</td><td style='{td_style}'>{html.escape(str(dim.size))}</td></tr>"

    coord_rows = ""
    for coord in builder._coordinates:
        dims_str = ", ".join(html.escape(str(d.name)) for d in coord.dimensions)
        coord_rows += (
            f"<tr><td style='{td_style}'>{html.escape(str(coord.name))}</td>"
            f"<td style='{td_style}'>{html.escape(dims_str)}</td>"
            f"<td style='{td_style}'>{html.escape(str(coord.data_type))}</td></tr>"
        )

    var_rows = ""
    for var in builder._variables:
        dims_str = ", ".join(html.escape(str(d.name)) for d in var.dimensions)
        var_rows += (
            f"<tr><td style='{td_style}'>{html.escape(str(var.name))}</td>"
            f"<td style='{td_style}'>{html.escape(dims_str)}</td>"
            f"<td style='{td_style}'>{html.escape(str(var.data_type))}</td></tr>"
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
    summary_style = (
        "cursor: pointer; font-weight: 600; margin-bottom: 8px; "
        "padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
    )
    summary_style_2 = (
        "cursor: pointer; font-weight: 600; margin: 16px 0 8px 0; "
        "padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
    )

    no_dims = '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions added</td></tr>'  # noqa: E501
    no_coords = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates added</td></tr>'  # noqa: E501
    )
    no_vars = '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No variables added</td></tr>'  # noqa: E501

    return f"""
    <div style="{box_style}" role="region" aria-labelledby="builder-header">
        <header style="{header_style}" id="builder-header">
            <h3 style="font-size: 1.1em; margin: 0;">MDIODatasetBuilder</h3>
        </header>
        <div style="margin-bottom: 15px;">
            <strong>Name:</strong> {html.escape(str(builder._metadata.name))}<br>
            <strong>State:</strong> {html.escape(str(builder._state.name))}<br>
            <strong>API Version:</strong> {html.escape(str(builder._metadata.api_version))}<br>
            <strong>Created:</strong> {html.escape(str(builder._metadata.created_on.strftime("%Y-%m-%d %H:%M:%S UTC")))}
        </div>
        <details open aria-expanded="true">
            <summary style="{summary_style}" aria-controls="builder-dimensions-table" id="builder-dimensions-summary">▸ Dimensions ({len(builder._dimensions)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="builder-dimensions-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Size</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {dim_rows if dim_rows else no_dims}
                    </tbody>
                </table>
            </div>
        </details>
        <details open aria-expanded="true">
            <summary style="{summary_style_2}" aria-controls="builder-coordinates-table" id="builder-coordinates-summary">▸ Coordinates ({len(builder._coordinates)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="builder-coordinates-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Dimensions</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Type</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {coord_rows if coord_rows else no_coords}
                    </tbody>
                </table>
            </div>
        </details>
        <details open aria-expanded="true">
            <summary style="{summary_style_2}" aria-controls="builder-variables-table" id="builder-variables-summary">▸ Variables ({len(builder._variables)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="builder-variables-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Dimensions</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Type</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {var_rows if var_rows else no_vars}
                    </tbody>
                </table>
            </div>
        </details>
    </div>
    """


def template_repr_html(template: "AbstractDatasetTemplate") -> str:
    """Return an HTML representation of the template for Jupyter notebooks."""
    # Format dimensions
    dim_rows = ""
    td_style_left = (
        "padding: 10px 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
        "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
        "font-size: 14px; line-height: 1.4;"
    )
    td_style_center = (
        "padding: 10px 8px; text-align: center; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
        "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
        "font-size: 14px; line-height: 1.4;"
    )
    if template._dim_names:
        for i, name in enumerate(template._dim_names):
            size = template._dim_sizes[i] if i < len(template._dim_sizes) else "Not set"
            is_spatial = "✓" if name in template._spatial_dim_names else ""
            dim_rows += (
                f"<tr><td style='{td_style_left}'>{html.escape(str(name))}</td>"
                f"<td style='{td_style_left}'>{html.escape(str(size))}</td>"
                f"<td style='{td_style_center}'>{html.escape(is_spatial)}</td></tr>"
            )

    # Format coordinates
    coord_rows = ""
    all_coords = list(template._physical_coord_names) + list(template._logical_coord_names)
    for coord in all_coords:
        coord_type = "Physical" if coord in template._physical_coord_names else "Logical"
        unit = template._units.get(coord, None)
        unit_str = html.escape(str(unit.name)) if unit else "—"
        coord_rows += (
            f"<tr><td style='{td_style_left}'>{html.escape(str(coord))}</td>"
            f"<td style='{td_style_left}'>{html.escape(coord_type)}</td>"
            f"<td style='{td_style_left}'>{unit_str}</td></tr>"
        )

    # Format units
    unit_rows = ""
    for key, unit in template._units.items():
        unit_rows += f"<tr><td style='{td_style_left}'>{html.escape(str(key))}</td><td style='{td_style_left}'>{html.escape(str(unit.name))}</td></tr>"

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
    summary_style = (
        "cursor: pointer; font-weight: 600; margin-bottom: 8px; "
        "padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
    )
    summary_style_2 = (
        "cursor: pointer; font-weight: 600; margin: 16px 0 8px 0; "
        "padding: 8px 12px; border-radius: 4px; transition: background-color 0.2s;"
    )

    no_dims = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions defined</td></tr>'  # noqa: E501
    )
    no_coords = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates defined</td></tr>'  # noqa: E501
    )
    no_units = '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No units defined</td></tr>'  # noqa: E501

    return f"""
    <div style="{box_style}" role="region" aria-labelledby="template-header">
        <header style="{header_style}" id="template-header">
            <h3 style="font-size: 1.1em; margin: 0;">{html.escape(str(template.__class__.__name__))}</h3>
        </header>
        <div style="margin-bottom: 15px;">
            <strong>Template Name:</strong> {html.escape(str(template.name))}<br>
            <strong>Data Domain:</strong> {html.escape(str(template._data_domain))}<br>
            <strong>Default Variable:</strong> {html.escape(str(template._default_variable_name))}<br>
            <strong>Chunk Shape:</strong> {html.escape(str(template._var_chunk_shape)) if template._var_chunk_shape else "Not set"}
        </div>
        <details open aria-expanded="true">
            <summary style="{summary_style}" aria-controls="dimensions-table" id="dimensions-summary">▸ Dimensions ({len(template._dim_names)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="dimensions-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Size</th>
                            <th style="text-align: center; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Spatial</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {dim_rows if dim_rows else no_dims}
                    </tbody>
                </table>
            </div>
        </details>
        <details open aria-expanded="true">
            <summary style="{summary_style_2}" aria-controls="coordinates-table" id="coordinates-summary">▸ Coordinates ({len(all_coords)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="coordinates-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Type</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Units</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {coord_rows if coord_rows else no_coords}
                    </tbody>
                </table>
            </div>
        </details>
        <details aria-expanded="false">
            <summary style="{summary_style_2}" aria-controls="units-table" id="units-summary">▸ Units ({len(template._units)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;" role="table" aria-labelledby="units-summary">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);" role="row">
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Key</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;" role="columnheader" scope="col">Unit</th>
                        </tr>
                    </thead>
                    <tbody role="rowgroup">
                        {unit_rows if unit_rows else no_units}
                    </tbody>
                </table>
            </div>
        </details>
    </div>
    """



def template_registry_repr_html(registry: "TemplateRegistry") -> str:
    """Return an HTML representation of the template registry for Jupyter notebooks."""
    template_rows = ""
    td_style = (
        "padding: 10px 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2); "
        "font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; "
        "font-size: 14px; line-height: 1.4;"
    )
    for name in sorted(registry._templates.keys()):
        template = registry._templates[name]
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
            <span style="margin-left: 15px; opacity: 0.7;">({len(registry._templates)} templates)</span>
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
