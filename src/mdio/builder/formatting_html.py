"""HTML formatting utilities for MDIO builder classes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdio.builder.dataset_builder import MDIODatasetBuilder
    from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate


def dataset_builder_repr_html(builder: "MDIODatasetBuilder") -> str:
    """Return an HTML representation of the builder for Jupyter notebooks."""
    dim_rows = ""
    td_style = "padding: 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
    for dim in builder._dimensions:
        dim_rows += f"<tr><td style='{td_style}'>{dim.name}</td><td style='{td_style}'>{dim.size}</td></tr>"

    coord_rows = ""
    for coord in builder._coordinates:
        dims_str = ", ".join(d.name for d in coord.dimensions)
        coord_rows += (
            f"<tr><td style='{td_style}'>{coord.name}</td>"
            f"<td style='{td_style}'>{dims_str}</td>"
            f"<td style='{td_style}'>{coord.data_type}</td></tr>"
        )

    var_rows = ""
    for var in builder._variables:
        dims_str = ", ".join(d.name for d in var.dimensions)
        var_rows += (
            f"<tr><td style='{td_style}'>{var.name}</td>"
            f"<td style='{td_style}'>{dims_str}</td>"
            f"<td style='{td_style}'>{var.data_type}</td></tr>"
        )

    box_style = (
        "font-family: monospace; border: 1px solid rgba(128, 128, 128, 0.3); "
        "border-radius: 5px; padding: 15px; max-width: 1000px;"
    )
    header_style = "padding: 10px; margin: -15px -15px 15px -15px; border-bottom: 2px solid rgba(128, 128, 128, 0.3);"
    summary_style = "cursor: pointer; font-weight: bold; margin-bottom: 8px;"
    summary_style_2 = "cursor: pointer; font-weight: bold; margin: 15px 0 8px 0;"

    no_dims = '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions added</td></tr>'  # noqa: E501
    no_coords = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates added</td></tr>'  # noqa: E501
    )
    no_vars = '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No variables added</td></tr>'  # noqa: E501

    return f"""
    <div style="{box_style}">
        <div style="{header_style}">
            <strong style="font-size: 1.1em;">MDIODatasetBuilder</strong>
        </div>
        <div style="margin-bottom: 15px;">
            <strong>Name:</strong> {builder._metadata.name}<br>
            <strong>State:</strong> {builder._state.name}<br>
            <strong>API Version:</strong> {builder._metadata.api_version}<br>
            <strong>Created:</strong> {builder._metadata.created_on.strftime("%Y-%m-%d %H:%M:%S UTC")}
        </div>
        <details open>
            <summary style="{summary_style}">▸ Dimensions ({len(builder._dimensions)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Size</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dim_rows if dim_rows else no_dims}
                    </tbody>
                </table>
            </div>
        </details>
        <details open>
            <summary style="{summary_style_2}">▸ Coordinates ({len(builder._coordinates)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Dimensions</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        {coord_rows if coord_rows else no_coords}
                    </tbody>
                </table>
            </div>
        </details>
        <details open>
            <summary style="{summary_style_2}">▸ Variables ({len(builder._variables)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Dimensions</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Type</th>
                        </tr>
                    </thead>
                    <tbody>
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
    td_style_left = "padding: 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
    td_style_center = "padding: 8px; text-align: center; border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
    if template._dim_names:
        for i, name in enumerate(template._dim_names):
            size = template._dim_sizes[i] if i < len(template._dim_sizes) else "Not set"
            is_spatial = "✓" if name in template._spatial_dim_names else ""
            dim_rows += (
                f"<tr><td style='{td_style_left}'>{name}</td>"
                f"<td style='{td_style_left}'>{size}</td>"
                f"<td style='{td_style_center}'>{is_spatial}</td></tr>"
            )

    # Format coordinates
    coord_rows = ""
    all_coords = list(template._physical_coord_names) + list(template._logical_coord_names)
    for coord in all_coords:
        coord_type = "Physical" if coord in template._physical_coord_names else "Logical"
        unit = template._units.get(coord, None)
        unit_str = f"{unit.name}" if unit else "—"
        coord_rows += (
            f"<tr><td style='{td_style_left}'>{coord}</td>"
            f"<td style='{td_style_left}'>{coord_type}</td>"
            f"<td style='{td_style_left}'>{unit_str}</td></tr>"
        )

    # Format units
    unit_rows = ""
    for key, unit in template._units.items():
        unit_rows += f"<tr><td style='{td_style_left}'>{key}</td><td style='{td_style_left}'>{unit.name}</td></tr>"

    box_style = (
        "font-family: monospace; border: 1px solid rgba(128, 128, 128, 0.3); "
        "border-radius: 5px; padding: 15px; max-width: 1000px;"
    )
    header_style = "padding: 10px; margin: -15px -15px 15px -15px; border-bottom: 2px solid rgba(128, 128, 128, 0.3);"
    summary_style = "cursor: pointer; font-weight: bold; margin-bottom: 8px;"
    summary_style_2 = "cursor: pointer; font-weight: bold; margin: 15px 0 8px 0;"

    no_dims = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions defined</td></tr>'  # noqa: E501
    )
    no_coords = (
        '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates defined</td></tr>'  # noqa: E501
    )
    no_units = '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No units defined</td></tr>'  # noqa: E501

    return f"""
    <div style="{box_style}">
        <div style="{header_style}">
            <strong style="font-size: 1.1em;">{template.__class__.__name__}</strong>
        </div>
        <div style="margin-bottom: 15px;">
            <strong>Template Name:</strong> {template.name}<br>
            <strong>Data Domain:</strong> {template._data_domain}<br>
            <strong>Default Variable:</strong> {template._default_variable_name}<br>
            <strong>Chunk Shape:</strong> {template._var_chunk_shape if template._var_chunk_shape else "Not set"}
        </div>
        <details open>
            <summary style="{summary_style}">▸ Dimensions ({len(template._dim_names)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Size</th>
                            <th style="text-align: center; padding: 8px; font-weight: 600;">Spatial</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dim_rows if dim_rows else no_dims}
                    </tbody>
                </table>
            </div>
        </details>
        <details open>
            <summary style="{summary_style_2}">▸ Coordinates ({len(all_coords)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Type</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Units</th>
                        </tr>
                    </thead>
                    <tbody>
                        {coord_rows if coord_rows else no_coords}
                    </tbody>
                </table>
            </div>
        </details>
        <details>
            <summary style="{summary_style_2}">▸ Units ({len(template._units)})</summary>
            <div style="margin-left: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Key</th>
                            <th style="text-align: left; padding: 8px; font-weight: 600;">Unit</th>
                        </tr>
                    </thead>
                    <tbody>
                        {unit_rows if unit_rows else no_units}
                    </tbody>
                </table>
            </div>
        </details>
    </div>
    """
